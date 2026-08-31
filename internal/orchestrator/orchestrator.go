// Package orchestrator is the local client's EMBEDDED ORCHESTRATOR (#455; the record-plane
// owner of every worker this daemon runs): the scheduling role of the ONE long-lived
// Cozy daemon `cozy run list` starts (cl-001). Since the 2026-08-25 re-landing (#436) the
// WORKER hosts the protocol and this side DIALS it: each spawned worker binds its own
// local socket, and this owner claims it (Claim -> ClaimAck -> snapshot -> SnapshotAck)
// before any dispatch. It is the authority for everything the runtime deliberately is
// not: which request runs, which attempt ordinal exists, which devices a process may
// see, and which terminal/result becomes visible.
//
// What it does NOT do, structurally rather than by policy: it never executes a model,
// never chooses a placement or a plan, never authorizes a reader from inside the
// runtime, and never mints a credential. A local grant is a CAS root plus an output
// directory — there is no token field set anywhere in this package, which is what makes
// "no fake cloud tokens" a fact a reader can check rather than a promise.
//
// The one law this package is written around: an accepted attempt is an OPEN OBLIGATION,
// and no next ordinal may be minted until it has a terminal. Remote supervisors replay
// their worker-local ledger; for a dead local Runtime child, this records authority writes
// ABANDONED itself. Runtime is never asked to remember what died with it.
package orchestrator

import (
	"context"
	"fmt"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Options is the frozen input to one Cozy daemon. Every field is decided by the
// entrypoint; nothing in this package reads the environment.
type Options struct {
	Cfg    config.Config
	Layout home.Layout
	Store  *records.Store
	// Yield is the GPU yield policy: smart | always | never.
	Yield string
	Log   io.Writer

	// Packages resolves `org/name` to the spec that makes its worker resident. It is the
	// START-OR-SELECT half of the one execution path (cl-010): a request whose binding no
	// live worker advertises MAKES one, so a cold invocation and a warm one traverse the
	// same states and differ only in latency. Without it a cold request queues for
	// capacity that nothing would ever create.
	Packages Launcher

	// Rentals resolves an attached-worker id (`cozy rental new`'s persisted triple) to its
	// dial spec. Wired by the entrypoint; nil = this daemon attaches no remote workers.
	Rentals func(id string) (*RemoteTarget, *exit.Error)
	// ObserveRental persists the remote worker's ClaimAck readback. Selection intent is
	// not hardware evidence: a rented worker is not dispatchable until this callback has
	// durably joined its actual accelerator and worker identity to the rental.
	ObserveRental func(RentalObservation) *exit.Error
	// RentalClaimProof signs the exact worker/boot/TLS leaf Creator is about to claim.
	RentalClaimProof RentalClaimProofSource
	// RentalPackageSet signs Creator's logical package/model download authority.
	RentalPackageSet RentalPackageSetSource
	// RentalFleet renders the one fleet burn line after reconciling every local
	// rental with Tensorhub. AcquireManagedRental durably assigns one --rental
	// request; ReleaseManagedRental tears down an idle Creator-managed pod.
	RentalFleet          func() (string, *exit.Error)
	AcquireManagedRental func(records.Request) (string, string, *exit.Error)
	ReleaseManagedRental func(string) (string, *exit.Error)
	// ConfigDigest is the local evaluated-config identity. Environment identity
	// comes only from the exact selected PlacementSet.
	ConfigDigest string
	MaxOutputMiB int64
}

type RentalClaimProofSource func(*WorkerConnection, uint64) ([]byte, *exit.Error)
type RentalPackageSetSource func(*WorkerConnection, []*pb.DownloadPackageRef,
	[]*pb.DownloadModelRef) ([]byte, []byte, *exit.Error)

type RentalObservation struct {
	RentalID               string
	Accelerator            string
	DeviceCount            int
	Backend                string
	DriverVersion          string
	BackendVersion         string
	DeviceMemoryTotalBytes uint64
	WorkerInstance         string
	WorkerID               string
	WorkerBootID           string
}

// Launcher resolves a package ref along the two boundaries #484 split: the
// platform-neutral desired placement and the local target-environment launch. A connected
// worker asks only for ResolvePlacement; it must never force this host to materialize or
// execute the target environment merely to author a remote plan.
type Launcher interface {
	ResolvePlacement(pkg string) (DesiredPlacement, *exit.Error)
	Resolve(pkg string) (WorkerLaunchSpec, *exit.Error)
	// ResolveInstall relaunches the immutable local install a durable request resolved
	// before entering the queue, so a changed pin cannot change accepted work.
	ResolveInstall(installID string) (WorkerLaunchSpec, *exit.Error)
	// ResolveJob is the JOB lane's half: `org/name` plus a job function to the spec that
	// makes THAT job's worker resident. It is a separate method rather than a flag
	// because the two produce different Directives and different worker slots — the
	// mode is a fact about the worker, and a resolver that returned "either" would push
	// the choice into the orchestrator, which resolves nothing.
	ResolveJob(pkg, function string) (WorkerLaunchSpec, *exit.Error)
	ResolveJobInstall(installID, function string) (WorkerLaunchSpec, *exit.Error)
}

type LogicalPackage struct {
	Package       string
	Release       string
	ReleaseDigest string
	Function      string
	Outputs       []string
	PlanID        string
	Models        []ModelRef
}

type LogicalJob struct {
	Package         string
	Release         string
	ReleaseDigest   string
	Function        string
	DescriptorID    string
	Outputs         []string
	ArtifactOutputs []ArtifactOutput
	GPUCount        int64
}

type ModelRef = records.ModelRef

// Orchestrator is the Cozy daemon's scheduling role.
type Orchestrator struct {
	opt Options

	// done closes when the daemon is closing; Serve blocks on it (the owner DIALS
	// workers, so there is no server here to run, #436). closeOnce makes Close
	// idempotent — harnesses close defensively and twice is not an event.
	done      chan struct{}
	closeOnce sync.Once

	// drainMu serializes the dispatch queue's drain. It is separate from `mu` because a
	// drain dispatches — it talks to the store and to a session — and holding the state
	// lock across that would serialize every report behind one 4.8 GiB fill.
	drainMu  sync.Mutex
	mu       sync.Mutex
	sessions map[string]*session // by worker_boot_id (the live claimed stream per worker)
	workers  map[string]*worker  // by instance_id
	waits    map[string]*wait    // by request#attempt
	// offers are seats reserved for emitted offers that have not yet produced the causal
	// Accepted or pre-execution Refused frame. Reports cannot reopen these seats.
	offers map[string]*dispatchReservation
	// mediaCleaning prevents overlapping retries of one durable cleanup obligation. It
	// contains only calls in flight; success is recorded on the attempt row.
	mediaCleaning map[string]bool
	// pending is the dispatch queue: requests that have no ready worker YET. A requeue
	// with nowhere to go WAITS for capacity instead of evaporating — the alternative is
	// a request that quietly stops existing because a worker was still loading.
	pending []string
	// closing is set by Close: a worker stopped during shutdown must not make the queue
	// ask for a replacement, because the daemon that would run it is going away.
	closing bool
	// starting names the packages a select-or-start is already making resident. One
	// launch per package: three cold requests for one package must not spawn three
	// workers and three device grants for a card that serves one attempt at a time.
	starting map[string]bool
	// ensuring is the per-instance creation fence beneath every caller, including child
	// recovery. `starting` serializes queue policy; this prevents two callers that already
	// chose the same deterministic slot from spawning two processes into it.
	ensuring         map[string]chan struct{}
	revision         uint64 // hub-owned, monotonic; every Directive bumps it
	residentRevision uint64 // local serving-worker arrival order; tie-breaks never-used LRU rows
	lastUseRevision  uint64 // successful local serving dispatch order
	events           []string

	// frames is the LOSSY live lane's fanout (stream.go). The durable lane is rows in
	// the records authority; these two are the whole event surface cl-006 serves.
	frames *fanout
}

type wait struct {
	accepted chan struct{}
	closed   chan struct{}
	once     sync.Once
	onceA    sync.Once
	refs     int
	err      *exit.Error
}

// Open builds the orchestrator. No listener binds here: the owner DIALS each worker's
// own socket (#436); a second Cozy daemon on one root fails on the daemon lock instead.
func Open(opt Options) (*Orchestrator, *exit.Error) {
	if opt.Log == nil {
		opt.Log = io.Discard
	}
	if opt.Yield == "" {
		opt.Yield = "smart"
	}
	c := &Orchestrator{
		opt:           opt,
		done:          make(chan struct{}),
		sessions:      map[string]*session{},
		workers:       map[string]*worker{},
		waits:         map[string]*wait{},
		offers:        map[string]*dispatchReservation{},
		mediaCleaning: map[string]bool{},
		starting:      map[string]bool{},
		ensuring:      map[string]chan struct{}{},
		frames:        newFanout(),
	}
	// The retirement watch samples on the worker report cadence. The cadence is a
	// SAMPLING resolution, never a verdict: every verdict it acts on is the worker's own
	// report (a latched fault, a declared wedge) or the absence of reports the worker
	// owes on that same cadence.
	go c.retirementLoop()
	return c, nil
}

// Serve blocks until Close. The owner dials workers; there is no server to run (#436),
// and the blocking shape is kept so entrypoints stay one-line callers.
func (c *Orchestrator) Serve() error {
	<-c.done
	return nil
}

// Close stops every worker this daemon owns. A worker outliving its launcher is exactly
// the class of bug the birth identity exists to catch, so `down` stops what it started
// rather than orphaning it.
func (c *Orchestrator) Close(grace time.Duration) {
	c.mu.Lock()
	c.closing = true
	c.mu.Unlock()
	// Workers drain IN PARALLEL: each gets the same grace, and the whole close costs one
	// grace window, not one per worker — a serial loop here could outlive the deadline
	// its own caller was waiting under (#449).
	var wg sync.WaitGroup
	for _, w := range c.workerList() {
		wg.Add(1)
		go func(worker *worker) {
			defer wg.Done()
			c.shutdownWorker(worker, grace)
		}(w)
	}
	wg.Wait()
	c.closeOnce.Do(func() { close(c.done) })
}

func (c *Orchestrator) workerList() []*worker {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]*worker, 0, len(c.workers))
	for _, w := range c.workers {
		out = append(out, w)
	}
	return out
}

func (c *Orchestrator) logf(format string, args ...any) {
	line := fmt.Sprintf("[orchestrator %s] ", time.Now().UTC().Format("15:04:05.000")) +
		fmt.Sprintf(format, args...)
	fmt.Fprintln(c.opt.Log, line)
	c.mu.Lock()
	c.events = append(c.events, line)
	if len(c.events) > 512 {
		c.events = c.events[len(c.events)-512:]
	}
	c.mu.Unlock()
}

// Events is the orchestrator's own recent activity, for status rendering. Runtime events
// reach a client through here and never bypass the orchestrator.
func (c *Orchestrator) Events() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.events...)
}

// Store is the lifecycle authority this orchestrator writes. cl-006's API reads requests,
// attempts, outputs and durable events through it — the same rows, never a second copy.
func (c *Orchestrator) Store() *records.Store { return c.opt.Store }

// Layout is the local root this orchestrator grants into.
func (c *Orchestrator) Layout() home.Layout { return c.opt.Layout }

// emit appends ONE durable lifecycle event. A failure to append is logged and never
// fatal: the authority's own row is the fact, and the stream is its announcement.
func (c *Orchestrator) emit(requestID, eventType string, attempt uint64, payload map[string]any) {
	if e := c.opt.Store.AppendEvent(requestID, eventType, int64(attempt), payload); e != nil {
		c.logf("event %s for %s NOT appended: %s", eventType, requestID, e.Message)
	}
}

// nextRevision mints the Directive revision. The hub owns it; it is monotonic, and a
// changed body always carries a new one.
func (c *Orchestrator) nextRevision() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.revision++
	return c.revision
}

func key(requestID string, attempt uint64) string {
	return fmt.Sprintf("%s#%d", requestID, attempt)
}

// acquireWait retains only live callers. Terminal/acceptance signals never create a map
// entry of their own, so fire-and-forget requests cannot accumulate one wait forever.
func (c *Orchestrator) acquireWait(k string) (*wait, func()) {
	c.mu.Lock()
	w, ok := c.waits[k]
	if !ok {
		w = &wait{accepted: make(chan struct{}), closed: make(chan struct{})}
		c.waits[k] = w
	}
	w.refs++
	c.mu.Unlock()
	return w, func() {
		c.mu.Lock()
		w.refs--
		if w.refs == 0 && c.waits[k] == w {
			delete(c.waits, k)
		}
		c.mu.Unlock()
	}
}

func (c *Orchestrator) signalAccepted(k string) {
	c.mu.Lock()
	w := c.waits[k]
	c.mu.Unlock()
	if w != nil {
		w.markAccepted()
	}
}

func (c *Orchestrator) signalClosed(k string, e *exit.Error) {
	c.mu.Lock()
	w := c.waits[k]
	c.mu.Unlock()
	if w != nil {
		w.markClosed(e)
	}
}

// enqueue parks a request until some worker advertises its binding as ready.
func (c *Orchestrator) enqueue(requestID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, id := range c.pending {
		if id == requestID {
			return
		}
	}
	c.pending = append(c.pending, requestID)
}

// drain dispatches everything the newly-ready capacity can now take. Called when a
// worker reports READY, which is the only event that can change the answer.
//
// ONE DRAIN AT A TIME. A worker's READY report and `selectOrStart`'s own post-WaitReady
// drain both fire within milliseconds of the same fact, and two concurrent drains read the
// same queue snapshot: the durable ordinal law is what refuses the duplicate, but doing the
// work twice and relying on a refusal is not a design. The lock makes the second drain read
// a queue the first one has already emptied.
func (c *Orchestrator) drain() {
	c.drainMu.Lock()
	defer c.drainMu.Unlock()
	c.mu.Lock()
	queued := append([]string(nil), c.pending...)
	c.mu.Unlock()
	for _, id := range queued {
		req, e := c.opt.Store.RequestRow(id)
		if e != nil || req == nil {
			c.forget(id)
			continue
		}
		attempt, e := c.dispatch(*req)
		if e != nil {
			// NO CAPACITY and AN ORDINAL THE LAW WILL NOT MINT YET are both "wait"; every
			// other refusal is the request's ANSWER, and leaving it queued would be the
			// failure mode this queue exists to prevent. Found live by cl-004's
			// publication-escape arm: a request whose GRANT can never be built (a
			// destination outside its own publication root) queued forever, because
			// `dispatch` refuses AFTER `pick` succeeded and the only branch here was
			// `continue`.
			if e.Code == exit.Unavailable || e.Code == exit.Conflict {
				continue
			}
			c.failQueued(id, e)
			continue
		}
		c.forget(id)
		c.logf("%s left the dispatch queue as attempt %d", id, attempt)
	}
}

// QueuePosition is where a waiting request sits in the dispatch queue, counted from 1.
// Zero means it is not waiting — either it never queued or it already has an attempt.
// The queue is a slice appended in submission order and drained in that order, so the
// position is the real thing rather than an estimate (cr-019: many queued jobs against
// one worker drain FIFO, and a client can watch it happen).
func (c *Orchestrator) QueuePosition(requestID string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, id := range c.pending {
		if id == requestID {
			return i + 1
		}
	}
	return 0
}

// reviveQueue re-asks select-or-start for the HEAD of the dispatch queue. It runs when
// the answer to "does the capacity this request needs exist?" has just changed: a worker's
// process went, or a launch finished.
//
// It exists because `selectOrStart` returns early twice over — while a launch is in
// FLIGHT, and while any worker for the slot is resident — and neither early return leaves
// anything behind to ask again. cl-004's live run met both: six queued jobs sat forever
// behind a launch that was for a DIFFERENT plan, and again behind a worker that had been
// `kill -9`ed moments after they queued.
//
// THE HEAD, and only the head. Reviving every waiting request would let two requests
// needing different plans stop each other's worker in turn; the FIFO head is the one that
// gets capacity next, so it is the one whose need is asked about. Nothing is dispatched
// here — `drain` is still the one placement path.
func (c *Orchestrator) reviveQueue() {
	c.mu.Lock()
	closing, head := c.closing, ""
	if len(c.pending) > 0 {
		head = c.pending[0]
	}
	c.mu.Unlock()
	if closing || head == "" {
		return
	}
	req, e := c.opt.Store.RequestRow(head)
	if e != nil || req == nil {
		return
	}
	c.selectOrStart(*req)
}

// recoverWorker settles the local process death from Creator's existing records
// authority. Runtime is a disposable execution child: it owns neither a journal nor a
// recovery decision. An unoffered assignment returns to the queue; an offer that may have
// crossed the process boundary closes as ABANDONED and earns a fresh ordinal. The old
// ordinal is never executed again.
func (c *Orchestrator) recoverWorker(spec WorkerLaunchSpec) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing {
		return
	}
	open, e := c.opt.Store.OpenAttemptsOf(spec.InstanceID())
	if e != nil {
		c.logf("cannot read the open attempts of %s: %s", spec.InstanceID(), e.Message)
	}
	if len(open) == 0 {
		c.reviveQueue()
		return
	}
	c.logf("worker %s died owing %d attempt(s); Creator is settling its local authority",
		spec.InstanceID(), len(open))
	for _, attempt := range open {
		c.settleLocalProcessDeath(attempt)
	}
	// Restore the same slot only after every old ordinal is durably closed or aborted.
	// The fresh Runtime child receives only new work; it reconstructs nothing.
	if _, _, e := c.EnsureWorker(spec); e != nil {
		c.logf("the local slot %s could not be restarted after death settlement: %s",
			spec.InstanceID(), e.Message)
	}
	c.reviveQueue()
}

func (c *Orchestrator) settleLocalProcessDeath(attempt records.Attempt) {
	if attempt.State == "preparing" {
		c.settleDispatch(attempt.RequestID, uint64(attempt.Attempt), false)
		if e := c.opt.Store.AbortDispatch(attempt.RequestID, attempt.Attempt,
			attempt.SessionID, "local Runtime exited before the offer boundary"); e != nil {
			c.logf("local Runtime death could not abort %s#%d: %s",
				attempt.RequestID, attempt.Attempt, e.Message)
			return
		}
		c.enqueue(attempt.RequestID)
		return
	}
	if attempt.State == "terminal" {
		if e := c.opt.Store.Closed(attempt.RequestID, attempt.Attempt); e != nil {
			c.logf("local Runtime death could not close %s#%d: %s",
				attempt.RequestID, attempt.Attempt, e.Message)
			return
		}
		req, e := c.opt.Store.RequestRow(attempt.RequestID)
		if e == nil && req != nil {
			c.afterAck(*req, attempt, nil)
		}
		return
	}
	if attempt.State != "offered" && attempt.State != "accepted" &&
		attempt.State != "recovered_open" {
		c.logf("local Runtime death left %s#%d in unexpected state %s",
			attempt.RequestID, attempt.Attempt, attempt.State)
		return
	}
	specDigest, err := canonical.Raw(attempt.InvocationDigest)
	if err != nil {
		c.logf("local Runtime death cannot settle %s#%d: malformed invocation digest",
			attempt.RequestID, attempt.Attempt)
		return
	}
	message := "local Runtime exited; its execution context is gone and this ordinal will not run again"
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{
		RequestId: attempt.RequestID, AttemptOrdinal: uint64(attempt.Attempt),
		InvocationSpecDigest: attempt.InvocationDigest,
		Status:               pb.OutcomeStatus_OUTCOME_STATUS_ABANDONED,
		SafeMessage:          message,
		Cause: &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_EXECUTOR_INVALIDATED,
			Origin: pb.CauseOrigin_CAUSE_ORIGIN_RECORD_OWNER, Detail: message},
		ExecutionStarted: attempt.State != "offered",
	})
	if err != nil {
		c.logf("local Runtime death cannot author %s#%d outcome: %s",
			attempt.RequestID, attempt.Attempt, err)
		return
	}
	// onOutcome remains the one terminal validator/transaction. The synthetic local
	// session has one buffered slot solely so its now-meaningless worker ACK can cross the
	// existing closure boundary without another execution-specific store path.
	s := &session{ctx: context.Background(), bootID: attempt.SessionID,
		instanceID: attempt.InstanceID, out: make(chan *pb.RecordOwnerFrame, 1)}
	c.onOutcome(s, &pb.AttemptOutcome{
		RequestId: attempt.RequestID, AttemptOrdinal: uint64(attempt.Attempt),
		InvocationSpecDigest: specDigest,
		OutcomeId:            records.NewID("out"), OutcomeDigest: digest, OutcomeCanonicalBytes: body,
	})
}

// NoProgressReports is how many SUCCESSIVE worker reports must both declare a wedge and
// show no movement on any reported axis before the worker is retired on the no-progress
// ground. A COUNT of the worker's own reports, never a duration (decisions #613): the
// judgment "nothing is moving" is made by the process that can see movement — the worker's
// liveness monitor, which is itself clock-free — and this only asks that the declared
// verdict persist rather than flicker. A legitimately slow fill never meets it, because a
// worker making progress does not declare a wedge.
const NoProgressReports = 15

// retirementLoop samples the retirement grounds on the worker report cadence, for as long
// as this daemon lives. A ticker is needed because ground 2 is the ABSENCE of reports —
// a dead worker produces no frame for a frame handler to run on. The tick itself decides
// nothing; every ground below is the worker's own word or its silence.
func (c *Orchestrator) retirementLoop() {
	tick := time.NewTicker(ReportCadence)
	defer tick.Stop()
	for {
		select {
		case <-c.done:
			return
		case <-tick.C:
			c.checkRetirement()
		}
	}
}

// checkRetirement replaces a worker that either blocks the head of the queue or owes an
// active attempt, and has ANSWERED that it cannot progress. `StallGrace` and its timer are DELETED, not
// resized (cl-025, decisions #613): no wall-clock number here may race a legitimate
// workload, because any constant sized to one workload kills the next — H3's cold fill
// runs minutes by construction and was killed at 90 s forever. A worker is retired on
// exactly three grounds, each an observation rather than a schedule:
//
//  1. WORKER-DECLARED FAILURE — a FAILED axis is the worker saying "I cannot" and is
//     acted on immediately. Fault rows only explain an axis: BINDING_DEGRADED explicitly
//     coexists with service. A claim this owner REFUSED is the same terminal class.
//  2. LIVENESS DEATH — the worker owes an ObservedWorkerState every `ReportCadence` and
//     has missed `SilentReports` of them. A count of missed heartbeats, never a guess
//     about how long a load takes: a worker filling for an hour reports on every one.
//  3. WORKER-DECLARED NO-PROGRESS — the worker's own liveness monitor declared a subject
//     WEDGED (its verdict is clock-free: consecutive observations of an unmoved monotone
//     position), and that declaration has persisted across `NoProgressReports` successive
//     reports in which no reported axis moved either.
//
// A worker that is merely SLOW — materializing, activating, filling — is none of these,
// however long it takes, and nothing here may kill it.
func (c *Orchestrator) checkRetirement() {
	c.mu.Lock()
	head, closing := "", c.closing
	if len(c.pending) > 0 {
		head = c.pending[0]
	}
	c.mu.Unlock()
	if closing {
		return
	}
	type candidate struct {
		worker *worker
		state  string
	}
	snapshot := func(eligible func(*worker) bool) []candidate {
		c.mu.Lock()
		defer c.mu.Unlock()
		out := []candidate{}
		for _, w := range c.workers {
			if w.exited || w.stopping || !eligible(w) {
				continue
			}
			if retirementGround(w, nil) != "" {
				out = append(out, candidate{worker: w,
					state: fmt.Sprintf("%s/%s",
						trimEnum(pb.MaterializationState_name[int32(w.materialization)], "MATERIALIZATION_STATE_"),
						trimEnum(pb.ServingState_name[int32(w.serving)], "SERVING_STATE_"))})
			}
		}
		sort.Slice(out, func(i, j int) bool {
			return out[i].worker.instanceID < out[j].worker.instanceID
		})
		return out
	}
	retire := func(victim candidate, subject string, attempts []records.Attempt) bool {
		w := victim.worker
		c.mu.Lock()
		current := c.workers[w.instanceID] == w && !w.exited && !w.stopping
		ground := retirementGround(w, attempts)
		c.mu.Unlock()
		if !current || ground == "" {
			return false
		}
		c.logf("worker %s is %s while %s depends on it; retiring it: %s",
			w.instanceID, victim.state, subject, ground)
		if e := c.retireWorker(w); e != nil {
			c.logf("worker %s could not be told to retire %s (%s); the stop follows",
				w.instanceID, w.placementID, e.Message)
		}
		if !c.shutdownWorker(w, StopGrace) {
			return false
		}
		go c.recoverWorker(w.spec)
		return true
	}

	// An accepted attempt is its own obligation. A completely unrelated queue head must
	// neither mask its wedge nor become the subject printed in its retirement evidence.
	// Only a worker that DECLARED a wedged subject needs its open attempts read to decide
	// whether the wedge is on accepted work; the other grounds never depend on them.
	active := snapshot(func(w *worker) bool {
		return len(w.wedgedSubjects) > 0 || retirementGround(w, []records.Attempt{}) != ""
	})
	for _, victim := range active {
		open, e := c.opt.Store.OpenAttemptsOf(victim.worker.instanceID)
		if e != nil || len(open) == 0 {
			continue
		}
		if retire(victim, open[0].RequestID, open) {
			return
		}
	}

	if head == "" {
		return
	}
	req, e := c.opt.Store.RequestRow(head)
	if e != nil || req == nil {
		return
	}
	c.mu.Lock()
	for _, w := range c.workers {
		if w.exited || w.stopping || !staged(w, req.PlanID) {
			continue
		}
		if w.dispatchableFor(req.PlanID) ||
			w.spec.IsJob() && w.dispatchable[req.PlanID] {
			// Dispatchable. The queue is waiting on placement, not on this worker.
			c.mu.Unlock()
			return
		}
	}
	c.mu.Unlock()
	queued := snapshot(func(w *worker) bool {
		return staged(w, req.PlanID) && !w.dispatchableFor(req.PlanID) &&
			!(w.spec.IsJob() && w.dispatchable[req.PlanID])
	})
	for _, victim := range queued {
		if retire(victim, head, nil) {
			return
		}
	}
}

func retirementGround(w *worker, attempts []records.Attempt) string {
	if w.refusal != nil {
		return fmt.Sprintf("this owner refused its claim (%s: %s)",
			w.refusal.ErrName(), w.refusal.Message)
	}
	if w.faulted {
		return fmt.Sprintf("it reported a FAILED worker/placement axis: %s", w.fault)
	}
	quiet := time.Duration(0)
	if !w.lastReport.IsZero() {
		quiet = time.Since(w.lastReport)
	} else if !w.spawned.IsZero() {
		quiet = time.Since(w.spawned)
	}
	if quiet > SilentReports*ReportCadence {
		return fmt.Sprintf("it reported no observed state for %s — %d missed reports of %s",
			quiet.Round(time.Second), SilentReports, ReportCadence)
	}
	wedgeApplies := w.wedged
	if attempts != nil {
		wedgeApplies = false
		for _, attempt := range attempts {
			if w.wedgedSubjects[fmt.Sprintf("%s#%d", attempt.RequestID, attempt.Attempt)] {
				wedgeApplies = true
				break
			}
		}
	}
	if wedgeApplies && w.noProgress >= NoProgressReports {
		return fmt.Sprintf("it declared itself WEDGED and %d successive reports moved no axis",
			w.noProgress)
	}
	return ""
}

// queueDepth is how many requests are waiting for capacity right now.
func (c *Orchestrator) queueDepth() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.pending)
}

// CancelQueued settles a request that is WAITING and has no attempt to cancel. It leaves
// the queue and is settled canceled — a client that asked for a cancel is owed an answer,
// and "it will start later anyway" is not one.
func (c *Orchestrator) CancelQueued(requestID string) *exit.Error {
	payload := map[string]any{
		"status": "CANCELED", "cause": "CLIENT_CANCELED",
		"error_type": "CLIENT_CANCELED",
		"error":      "canceled from the dispatch queue before any attempt was dispatched",
		"outputs":    []any{}, "requeuing": false,
	}
	applied, e := c.opt.Store.CancelQueuedRequest(requestID, payload)
	if e != nil {
		return e
	}
	if !applied {
		return nil
	}
	c.forget(requestID)
	c.frames.forget(requestID)
	c.logf("%s left the dispatch queue: canceled before any attempt", requestID)
	c.signalClosed(requestWaitKey(requestID),
		exit.New(exit.Canceled, "%s was canceled before any attempt was dispatched", requestID))
	return nil
}

func (c *Orchestrator) forget(requestID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := c.pending[:0]
	for _, id := range c.pending {
		if id != requestID {
			out = append(out, id)
		}
	}
	c.pending = out
}

// requestWaitKey names the request-level wait, which may span several attempt ordinals.
func requestWaitKey(requestID string) string { return "request:" + requestID }

func (w *wait) markAccepted() { w.onceA.Do(func() { close(w.accepted) }) }
func (w *wait) markClosed(e *exit.Error) {
	w.once.Do(func() {
		w.err = e
		close(w.closed)
	})
}
