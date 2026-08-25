// Package coord is the LocalCoordinator: the scheduling role of the ONE long-lived
// LocalService `cozy up` starts (cl-001). It is the worker protocol's SERVER — the
// cozy-runtime supervisor dials IT, per the dial-toward-a-stable-address law — and the
// authority for everything the runtime deliberately is not: which request runs, which
// attempt ordinal exists, which devices a process may see, and which terminal/result
// becomes visible.
//
// What it does NOT do, structurally rather than by policy: it never executes a model,
// never chooses a placement or a plan, never authorizes a reader from inside the
// runtime, and never mints a credential. A local grant is a CAS root plus an output
// directory — there is no token field set anywhere in this package, which is what makes
// "no fake cloud tokens" a fact a reader can check rather than a promise.
//
// The one law this package is written around (worker-protocol/02 §6.2): the
// `recovered_attempts` a new session reports on Register are OPEN OBLIGATIONS, and no
// next ordinal for those request ids may be minted until each is closed by its own
// journaled terminal. A new session_id never manufactures absence.
package coord

import (
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"

	"github.com/cozy-creator/cozy-creator-v2/internal/config"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Options is the frozen input to one LocalService. Every field is decided by the
// entrypoint; nothing in this package reads the environment.
type Options struct {
	Cfg    config.Config
	Layout home.Layout
	Store  *records.Store
	// Socket is the worker-protocol unix socket the supervisor dials. A stable address
	// the worker dials TOWARD; the coordinator never dials a worker.
	Socket string
	// Yield is the GPU yield policy: smart | always | never.
	Yield string
	Log   io.Writer

	// Endpoints resolves `org/name` to the spec that makes its worker resident. It is the
	// START-OR-SELECT half of the one execution path (cl-010): a request whose binding no
	// live worker advertises MAKES one, so a cold invocation and a warm one traverse the
	// same states and differ only in latency. Without it a cold request queues for
	// capacity that nothing would ever create.
	Endpoints Launcher

	// ImageDigest and ConfigDigest ride INSIDE every ExecutionSpec document: the exact
	// execution environment (class b) and the evaluated-config document's identity
	// (class a, cr-003's). They are frozen per service, never per request — a request
	// cannot choose the environment it is admitted under.
	ImageDigest  string
	ConfigDigest string
	GrantTTL     time.Duration
	MaxOutputMiB int64
}

// Launcher resolves an endpoint ref to the spec that starts its worker. The coordinator
// holds it to SELECT-OR-START and for nothing else: it never resolves a name itself, and
// the object that does is the LOCAL module's install-generation resolver (cl-010) or, on
// a pod, cl-014's.
type Launcher interface {
	Resolve(endpoint string) (EndpointSpec, *exit.Error)
	// ResolveJob is the JOB lane's half: `org/name` plus a job function to the spec that
	// makes THAT job's worker resident. It is a separate method rather than a flag
	// because the two produce different Directives and different worker slots — the
	// mode is a fact about the worker, and a resolver that returned "either" would push
	// the choice into the coordinator, which resolves nothing.
	ResolveJob(endpoint, function string) (EndpointSpec, *exit.Error)
}

// Coordinator is the LocalService's scheduling role.
type Coordinator struct {
	opt Options

	grpc     *grpc.Server
	listener net.Listener

	// drainMu serializes the dispatch queue's drain. It is separate from `mu` because a
	// drain dispatches — it talks to the store and to a session — and holding the state
	// lock across that would serialize every report behind one 4.8 GiB fill.
	drainMu  sync.Mutex
	mu       sync.Mutex
	sessions map[string]*session // by session_id
	workers  map[string]*worker  // by instance_id
	waits    map[string]*wait    // by request#attempt
	// pending is the dispatch queue: requests that have no ready worker YET. A requeue
	// with nowhere to go WAITS for capacity instead of evaporating — the alternative is
	// a request that quietly stops existing because a worker was still loading.
	pending []string
	// closing is set by Close: a worker stopped during shutdown must not make the queue
	// ask for a replacement, because the service that would run it is going away.
	closing bool
	// starting names the endpoints a select-or-start is already making resident. One
	// launch per endpoint: three cold requests for one endpoint must not spawn three
	// workers and three device grants for a card that serves one attempt at a time.
	starting map[string]bool
	// stalls is when each RESIDENT-but-undispatchable worker first failed to serve the
	// head of the queue. It is the stall watchdog's only state (see checkStall).
	stalls   map[string]time.Time
	revision uint64 // hub-owned, monotonic; every Directive bumps it
	events   []string

	// frames is the LOSSY live lane's fanout (stream.go). The durable lane is rows in
	// the records authority; these two are the whole event surface cl-006 serves.
	frames *fanout
}

type wait struct {
	accepted chan struct{}
	closed   chan struct{}
	once     sync.Once
	onceA    sync.Once
	err      *exit.Error
}

// Open builds the coordinator and binds its unix socket. Binding is where a second
// LocalService on one root would fail, and it happens before any worker exists.
func Open(opt Options) (*Coordinator, *exit.Error) {
	if opt.Log == nil {
		opt.Log = io.Discard
	}
	if opt.Yield == "" {
		opt.Yield = "smart"
	}
	c := &Coordinator{
		opt:      opt,
		sessions: map[string]*session{},
		workers:  map[string]*worker{},
		waits:    map[string]*wait{},
		starting: map[string]bool{},
		stalls:   map[string]time.Time{},
		frames:   newFanout(),
	}
	// A stale socket file is a leftover, never evidence: the service lock already proved
	// no live owner exists on this root, so removing it is safe and required.
	_ = os.Remove(opt.Socket)
	ln, err := net.Listen("unix", opt.Socket)
	if err != nil {
		return nil, exit.New(exit.Conflict, "cannot bind the worker socket %s: %s", opt.Socket, err).
			WithRemedy("another LocalService may own this root; `cozy status`")
	}
	c.listener = ln
	c.grpc = grpc.NewServer(grpc.Creds(unixPeer{}))
	pb.RegisterWorkerServer(c.grpc, &hub{c: c})
	return c, nil
}

// Serve runs until Close. The coordinator is the SERVER; it never dials a worker.
func (c *Coordinator) Serve() error { return c.grpc.Serve(c.listener) }

// Close stops every worker this service owns, then the server. A worker outliving its
// launcher is exactly the class of bug the birth identity exists to catch, so `down`
// stops what it started rather than orphaning it.
func (c *Coordinator) Close(grace time.Duration) {
	c.mu.Lock()
	c.closing = true
	c.mu.Unlock()
	for _, w := range c.workerList() {
		c.StopWorker(w.instanceID, grace)
	}
	c.grpc.Stop()
	_ = os.Remove(c.opt.Socket)
}

func (c *Coordinator) workerList() []*worker {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]*worker, 0, len(c.workers))
	for _, w := range c.workers {
		out = append(out, w)
	}
	return out
}

func (c *Coordinator) logf(format string, args ...any) {
	line := fmt.Sprintf("[coord %s] ", time.Now().UTC().Format("15:04:05.000")) +
		fmt.Sprintf(format, args...)
	fmt.Fprintln(c.opt.Log, line)
	c.mu.Lock()
	c.events = append(c.events, line)
	if len(c.events) > 512 {
		c.events = c.events[len(c.events)-512:]
	}
	c.mu.Unlock()
}

// Events is the coordinator's own recent activity, for status rendering. Runtime events
// reach a client through here and never bypass the coordinator.
func (c *Coordinator) Events() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.events...)
}

// Store is the lifecycle authority this coordinator writes. cl-006's API reads requests,
// attempts, outputs and durable events through it — the same rows, never a second copy.
func (c *Coordinator) Store() *records.Store { return c.opt.Store }

// Layout is the local root this coordinator grants into.
func (c *Coordinator) Layout() home.Layout { return c.opt.Layout }

// emit appends ONE durable lifecycle event. A failure to append is logged and never
// fatal: the authority's own row is the fact, and the stream is its announcement.
func (c *Coordinator) emit(requestID, eventType string, attempt uint64, payload map[string]any) {
	if e := c.opt.Store.AppendEvent(requestID, eventType, int64(attempt), payload); e != nil {
		c.logf("event %s for %s NOT appended: %s", eventType, requestID, e.Message)
	}
}

// nextRevision mints the Directive revision. The hub owns it; it is monotonic, and a
// changed body always carries a new one.
func (c *Coordinator) nextRevision() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.revision++
	return c.revision
}

func key(requestID string, attempt uint64) string {
	return fmt.Sprintf("%s#%d", requestID, attempt)
}

func (c *Coordinator) waitFor(k string) *wait {
	c.mu.Lock()
	defer c.mu.Unlock()
	w, ok := c.waits[k]
	if !ok {
		w = &wait{accepted: make(chan struct{}), closed: make(chan struct{})}
		c.waits[k] = w
	}
	return w
}

// enqueue parks a request until some worker advertises its binding as ready.
func (c *Coordinator) enqueue(requestID string) {
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
func (c *Coordinator) drain() {
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
func (c *Coordinator) QueuePosition(requestID string) int {
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
func (c *Coordinator) reviveQueue() {
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

// recoverWorker is what happens when a worker's PROCESS dies. Two different things are
// owed, and only one of them is the queue's:
//
//   - the attempts that worker was RUNNING owe a terminal, and the only thing that can
//     produce one is the supervisor's own journal replayed by a worker in the SAME SLOT
//     (worker-protocol/02 §6.2). So the slot is started again — its root, and therefore
//     its journal, is deliberately reused — and the recovered attempt arrives on Register
//     as an open obligation exactly as the law says.
//   - anything merely WAITING needs capacity asked for, which is `reviveQueue`.
//
// Without the first half a `kill -9` of a job worker left its in-flight attempt with no
// terminal forever: the request was not queued (it had an ordinal), so nothing looked at
// it again. Found live by cl-004's kill arm, which only converged when a LATER submission
// happened to restart the same slot.
func (c *Coordinator) recoverWorker(spec EndpointSpec) {
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
	c.logf("worker %s died owing %d terminal(s); restarting the slot so its journal replays",
		spec.InstanceID(), len(open))
	if _, e := c.StartWorker(spec); e != nil {
		// The slot cannot come back. The attempts it holds are unsettleable, and saying so
		// beats leaving a client on a stream that will never close.
		c.logf("the slot %s could not be restarted (%s); its %d open attempt(s) cannot settle",
			spec.InstanceID(), e.Message, len(open))
	}
	c.reviveQueue()
}

// StallGrace is how long a RESIDENT worker may go on not being dispatchable for the
// request at the head of its own queue before it is replaced. It is deliberately the same
// clock `WaitReady` gives a worker reporting ERROR: a worker that has not become able to
// serve its queue in a minute and a half has answered, whatever state it is reporting.
const StallGrace = 90 * time.Second

// checkStall replaces a worker that is resident, alive, staged for the head of the queue,
// and STILL not dispatchable. It runs on every Report, because a Report is the only thing
// a stuck worker keeps producing.
//
// The condition is real and was met live: `kill -9` of a job executor mid-attempt left the
// supervisor rebuilding, its Report pinned at INTAKE_STATE_LOADING with jobs_available=0,
// and the queue behind it waiting forever. `drain` could not help — it only runs on READY,
// which is exactly what never came. Nothing here decides WHY a worker is stuck: the
// coordinator's business is that a request has been owed capacity for too long by a
// process it started, and the one thing it owns is whether that process keeps the slot.
func (c *Coordinator) checkStall() {
	c.mu.Lock()
	head := ""
	if len(c.pending) > 0 {
		head = c.pending[0]
	}
	c.mu.Unlock()
	if head == "" {
		return
	}
	req, e := c.opt.Store.RequestRow(head)
	if e != nil || req == nil {
		return
	}
	c.mu.Lock()
	var victim *worker
	for _, w := range c.workers {
		if w.exited || !staged(w, req.PlanID) {
			continue
		}
		if w.intake == pb.IntakeState_INTAKE_STATE_READY && w.ready[req.PlanID] {
			// Dispatchable. The queue is waiting on placement, not on this worker.
			delete(c.stalls, w.instanceID)
			c.mu.Unlock()
			return
		}
		victim = w
	}
	if victim == nil {
		c.mu.Unlock()
		return
	}
	first, seen := c.stalls[victim.instanceID]
	if !seen {
		c.stalls[victim.instanceID] = time.Now()
		c.mu.Unlock()
		return
	}
	stuck, intake := time.Since(first), pb.IntakeState_name[int32(victim.intake)]
	instance := victim.instanceID
	c.mu.Unlock()
	if stuck < StallGrace {
		return
	}
	c.logf("worker %s has been %s and undispatchable for %s while %s waits; replacing it",
		instance, intake, stuck.Round(time.Second), head)
	c.mu.Lock()
	delete(c.stalls, instance)
	c.mu.Unlock()
	// StopWorker's own reviveQueue asks for the replacement, so the request the watchdog
	// fired for is the one whose need gets re-asked.
	c.StopWorker(instance, 20*time.Second)
}

// queueDepth is how many requests are waiting for capacity right now.
func (c *Coordinator) queueDepth() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.pending)
}

// CancelQueued settles a request that is WAITING and has no attempt to cancel. It leaves
// the queue and is settled canceled — a client that asked for a cancel is owed an answer,
// and "it will start later anyway" is not one.
func (c *Coordinator) CancelQueued(requestID string) {
	c.forget(requestID)
	if e := c.opt.Store.SettleRequest(requestID, "canceled"); e != nil {
		c.logf("%s could not be settled canceled: %s", requestID, e.Message)
		return
	}
	c.frames.forget(requestID)
	c.emit(requestID, "request.canceled", 0, map[string]any{
		"status": "CANCELED", "cause": "CLIENT_CANCELED",
		"error_type": "CLIENT_CANCELED",
		"error":      "canceled from the dispatch queue before any attempt was dispatched",
		"outputs":    []any{}, "requeuing": false,
	})
	c.logf("%s left the dispatch queue: canceled before any attempt", requestID)
	c.waitRequest(requestID).markClosed(
		exit.New(exit.Canceled, "%s was canceled before any attempt was dispatched", requestID))
}

func (c *Coordinator) forget(requestID string) {
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

// waitRequest is the REQUEST-level wait: it closes when the request settles, which may
// be several attempts after the one a caller first saw.
func (c *Coordinator) waitRequest(requestID string) *wait { return c.waitFor("request:" + requestID) }

func (w *wait) markAccepted() { w.onceA.Do(func() { close(w.accepted) }) }
func (w *wait) markClosed(e *exit.Error) {
	w.once.Do(func() {
		w.err = e
		close(w.closed)
	})
}
