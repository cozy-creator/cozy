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
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Options is the frozen input to one Cozy daemon. Every field is decided by the
// entrypoint; nothing in this package reads the environment.
type Options struct {
	// StartMachineExecution transfers and observes an execution owned by Runtime.
	// It never offers an attempt through this legacy cross-machine dispatcher.
	StartMachineExecution func(records.Request) *exit.Error
	// ReclaimInstall delegates unpinned snapshot cleanup to the existing package owner.
	ReclaimInstall func(string) *exit.Error
	Cfg            config.Config
	Layout         home.Layout
	Store          *records.Store
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
	// RentalPrepareFacts fetches the hub-known release facts one
	// PreparePackageSetCall carries on fields 3-6 (wire 31, xs-019) for
	// dispatching one package release to this rental. Callback for the same
	// reason RentalPackageSet is: the orchestrator holds no Tensorhub client.
	RentalPrepareFacts RentalPrepareFactsSource
	// ReportReleaseDefect relays a package-interface-falsifying pod refusal to the hub
	// (th-106). Callback for the same reason RentalPackageSet is: the
	// orchestrator holds no Tensorhub client. Fire-and-forget; the hub's defect
	// tombstone is idempotent.
	ReportReleaseDefect ReleaseDefectReporter
	// RentalFleet renders the one fleet burn line after reconciling every local
	// rental with Tensorhub. AcquireManagedRental is the placement decision for a --rental
	// request no rental holds a placement for (placement-economics.md): it pins the
	// request to an attached ready rental or BUYS a pod (owner ruling 2026-09-03:
	// --rental is permission AND intent to spend) and records what it chose over what.
	// ReleaseManagedRental observes a rental as a request pinned to it settles and tears
	// it down once nothing is left on it.
	RentalFleet          func() (string, *exit.Error)
	AcquireManagedRental func(req records.Request) (PlacementDecision, string, *exit.Error)
	ReleaseManagedRental func(string) (string, *exit.Error)
	// ReleaseRetainedRental is explicit owner abandonment, independent of idle policy.
	ReleaseRetainedRental func(string) (string, *exit.Error)
	ModelTransfers        ModelTransferOwner
	MaxOutputMiB          int64
}

type RentalClaimProofSource func(*WorkerConnection, uint64) ([]byte, *exit.Error)

// ReleaseDefect is one defect report: the exact release the pod refused and the
// rental it was refused on. The signed-delegation chain of authority that used to
// travel with it is deleted (owner ruling 2026-09-03), so the hub authorizes the
// report by rental OWNERSHIP — provenance is no longer proven.
type ReleaseDefect struct {
	Package, Release string
	RentalID         string
	Code, Detail     string
}

// ReleaseDefectReporter posts one defect report; failures are logged, never
// retried here — a republished falsifying release will be refused again.
type ReleaseDefectReporter func(report ReleaseDefect)

// RentalPackageSetSource authors the desired download set for one selection. It takes
// no worker connection: the document names content only and binds no rental, worker or
// boot (owner ruling 2026-09-03).
type RentalPackageSetSource func([]*pb.DownloadPackageRef,
	[]*pb.DownloadModelRef) ([]byte, *exit.Error)

// PrepareFacts is the hub-known half of one PreparePackageSetCall: the release
// facts (fields 3-6) the record owner fetched for this exact package release on
// this rental. The pod host relays them verbatim to the Runtime's preparation,
// which refuses a call without them.
type PrepareFacts struct {
	Application        string
	ModelSlotPaths     []string
	ImageInventory     *pb.ImageInventory
	LockedRequirements []byte
}

// RentalPrepareFactsSource answers one package release's facts for one rental.
// Every call returns a fresh value; the orchestrator sends it on the wire as is.
type RentalPrepareFactsSource func(context.Context, *WorkerConnection,
	*pb.DownloadPackageRef) (PrepareFacts, *exit.Error)

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

type ModelTransferOwner interface {
	MaterializeLocal(context.Context, string, records.ModelTransferIntent) ([]ModelRef, *exit.Error)
	RefreshRemoteSource(context.Context, records.ModelTransferIntent) ([]ModelSourceCapability, *exit.Error)
	Finalize(context.Context, string, ModelTransferMover) *exit.Error
	AbandonModelTransferPublications(context.Context, string) *exit.Error
	PassThrough(context.Context, string, records.ModelTransferIntent) *exit.Error
	SyncCheckpoints(context.Context, string, CheckpointHost) *exit.Error
	RestoreWeightsCheckpoint(context.Context, string, CheckpointHost, *pb.WeightsCheckpointSubject) (*pb.CheckpointRef, *exit.Error)
	RestoreSourceCheckpoints(context.Context, string, CheckpointHost) *exit.Error
	ReleaseCheckpoints(context.Context, string) *exit.Error
}

type ModelTransferMover func(context.Context, records.ModelTransferWeights,
	WeightsGrantMinter) *exit.Error

// Launcher resolves a package ref along the two boundaries #484 split: the
// platform-neutral desired placement and the local target-environment launch. A connected
// worker asks only for ResolvePlacement; it must never force this host to materialize or
// execute the target environment merely to author a remote plan.
type Launcher interface {
	ValidateExecutionCapture(records.Request) *exit.Error
	ResolvePlacement(pkg string) (DesiredPlacement, *exit.Error)
	Resolve(pkg string) (WorkerLaunchSpec, *exit.Error)
	// ResolveInstall relaunches the immutable local install a durable request resolved
	// before entering the queue, so a changed pin cannot change accepted work.
	ResolveInstall(installID string, models []ModelRef) (WorkerLaunchSpec, *exit.Error)
	// ResolveJob is the JOB lane's half: `org/name` plus a job function to the spec that
	// makes THAT job's worker resident. It is a separate method rather than a flag
	// because the two produce different Directives and different worker slots — the
	// mode is a fact about the worker, and a resolver that returned "either" would push
	// the choice into the orchestrator, which resolves nothing.
	ResolveJob(pkg, function string) (WorkerLaunchSpec, *exit.Error)
	ResolveJobInstall(installID, function string) (WorkerLaunchSpec, *exit.Error)
	LocalRevision(installID, digest string) (localpackage.Revision, *exit.Error)
}

type LogicalPackage struct {
	Package          string
	Release          string
	Function         string
	Outputs          []string
	PlanID           string
	Models           []ModelRef
	NeedsAccelerator bool
}

type LogicalJob struct {
	Package          string
	Release          string
	Function         string
	DescriptorID     string
	Outputs          []string
	WeightsOutputs   []WeightsOutput
	NeedsAccelerator bool
	Models           []ModelRef
	ProducerParams   []string
}

type ModelRef = records.ModelRef

// PlacementDecision is the capacity decision's whole record (cl-165, placement-economics.md):
// the tier it ran under, the config it read, every throughput row it used, and every
// candidate — attached ready rental or purchasable product — with its rate, expected
// time, cost, score and verdict. It is the `request.placement` event. A record naming
// only its winner cannot be audited: on 2026-09-04 a `--rental-only` run bought a second
// pod while a ready rental sat idle, and the durable record could not say why.
type PlacementDecision struct {
	Tier         string `json:"tier"`
	ConfigDigest string `json:"config_digest"`
	// Ladder is the owner's fit map the candidates were sized by (`H100=fp8 > *=bf16`),
	// one entry per slot when slots differ; none when the request binds no ladder.
	// Override is the lane the request pinned itself — an explicit `model.<param>=…/lane`
	// (cl-170) — which is not a rung and asserts no fit.
	Ladder     []string              `json:"ladder,omitempty"`
	Override   string                `json:"override,omitempty"`
	Throughput []hub.ModelThroughput `json:"throughput"`
	Candidates []PlacementCandidate  `json:"candidates"`
	// RentalID is the machine chosen; empty on a decision to wait for one attaching.
	RentalID string `json:"rental,omitempty"`
	Bought   bool   `json:"bought"`
	// Models is the request's selection pinned to the chosen machine's rung; nil when
	// the request was already exact.
	Models []ModelRef `json:"-"`
}

// PlacementCandidate is one machine the run could be placed on, as the choice saw it: an
// attached ready rental (Rental set) or a purchasable product.
type PlacementCandidate struct {
	Rental  string `json:"rental,omitempty"`
	Machine string `json:"machine,omitempty"`
	SKU     string `json:"sku"`
	// Rung is the 1-based ladder rung the machine fell under — 0 under an explicit lane,
	// which is not a rung — and Lane the lane it pins; Fit says how the device was sized
	// against that lane (cl-168, cl-170).
	Rung int    `json:"rung,omitempty"`
	Lane string `json:"lane,omitempty"`
	Fit  string `json:"fit,omitempty"`
	// Ahead is the attempts a new request waits behind on an attached rental.
	Ahead int `json:"ahead,omitempty"`
	// RateUSDMicrosPerHour is what the renter pays: the live catalog's price plus storage.
	RateUSDMicrosPerHour int64 `json:"rate_usd_micros_per_hour"`
	// Measured says a throughput row exists for (lane, sku). TimeS, CostUSDMicros and
	// Score follow placement-economics.md and are zero on an unmeasured candidate.
	Measured      bool    `json:"measured"`
	TimeS         float64 `json:"time_s,omitempty"`
	CostUSDMicros int64   `json:"cost_usd_micros,omitempty"`
	Score         float64 `json:"score,omitempty"`
	// Verdict is one of the constants below; empty only while the choice is still open.
	Verdict string     `json:"verdict"`
	Models  []ModelRef `json:"-"`
}

// Verdicts a placement candidate carries once the choice is made.
const (
	VerdictChosen     = "chosen"
	VerdictSlower     = "slower"
	VerdictDearer     = "dearer"
	VerdictUnmeasured = "unmeasured"
	// VerdictAttaching: a fitting rental the fleet holds that cannot take the request
	// YET but is provably on its way — anywhere from `pending_acquisition` through a
	// `ready` pod whose worker has not attached. The request waits for it (cl-170) and
	// is never failed on it (cl-185).
	VerdictAttaching = "attaching"
	// VerdictNoRung: no rung of the binding ladder names this machine's accelerator.
	VerdictNoRung = "no_rung"
	// VerdictNoStock: the hub refused the buy for want of inventory, and the choice repeated.
	VerdictNoStock = "no_stock"
	// VerdictExcluded is a prefix; the reason follows (`excluded:vram_short: …`).
	VerdictExcluded = "excluded:"
)

// Attached says the candidate is a rental this daemon already holds.
func (c PlacementCandidate) Attached() bool { return c.Rental != "" }

// Name is how a reader knows the candidate: the machine word, or the product.
func (c PlacementCandidate) Name() string {
	if c.Machine != "" {
		return c.Machine
	}
	if c.Rental != "" {
		return c.Rental
	}
	return c.SKU
}

// Chosen is the candidate the decision placed the run on.
func (d PlacementDecision) Chosen() (PlacementCandidate, bool) {
	for _, c := range d.Candidates {
		if c.Verdict == VerdictChosen {
			return c, true
		}
	}
	return PlacementCandidate{}, false
}

// Unexplained names a candidate passed over with NO verdict, and is empty when the
// record is sound. It is the invariant th-151 was filed to check: choosing a dearer or
// slower machine is allowed, but passing one over with nothing said is a chooser defect.
func (d PlacementDecision) Unexplained() string {
	for _, c := range d.Candidates {
		if c.Verdict == "" {
			return c.Name()
		}
	}
	return ""
}

// Waiting is the rental the decision waits on: a fitting one whose worker has not attached.
func (d PlacementDecision) Waiting() (PlacementCandidate, bool) {
	for _, c := range d.Candidates {
		if c.Verdict == VerdictAttaching {
			return c, true
		}
	}
	return PlacementCandidate{}, false
}

// Line is the one human sentence `cozy run` prints for the decision.
func (d PlacementDecision) Line() string {
	c, ok := d.Chosen()
	if !ok {
		if w, waiting := d.Waiting(); waiting {
			return fmt.Sprintf("placement: wait for %s to attach — %s", w.describe(), d.Tier)
		}
		return "placement: nothing chosen"
	}
	line := "placement: buy " + c.describe()
	if c.Attached() {
		line = "placement: reuse " + c.describe()
	}
	switch {
	case c.Measured:
		return fmt.Sprintf("%s — %s, %.0f s, $%.2f", line, d.Tier, c.TimeS, float64(c.CostUSDMicros)/1e6)
	case c.Rung > 0:
		return fmt.Sprintf("%s — %s, unmeasured (rung %d)", line, d.Tier, c.Rung)
	}
	return fmt.Sprintf("%s — %s, unmeasured", line, d.Tier)
}

// describe is the candidate as a line names it: `morgiana (h100-80, fp8)` for a rental,
// `h100-80 (fp8)` for a product.
func (c PlacementCandidate) describe() string {
	name, detail := c.SKU, []string{}
	if c.Attached() {
		name, detail = c.Name(), append(detail, c.SKU)
	}
	if c.Lane != "" {
		detail = append(detail, c.Lane)
	}
	if len(detail) == 0 {
		return name
	}
	return name + " (" + strings.Join(detail, ", ") + ")"
}

// verdicts renders every candidate with its verdict, for the daemon log.
func (d PlacementDecision) verdicts() string {
	parts := make([]string, 0, len(d.Candidates))
	for _, c := range d.Candidates {
		parts = append(parts, c.Name()+" "+c.Verdict)
	}
	return strings.Join(parts, ", ")
}

// payload is the record as the `request.placement` event carries it, plus the line.
func (d PlacementDecision) payload() map[string]any {
	raw, _ := json.Marshal(d)
	out := map[string]any{}
	_ = json.Unmarshal(raw, &out)
	out["line"] = d.Line()
	return out
}

// Exclusion reasons an attached rental can carry behind VerdictExcluded. They are
// properties of the FLEET at the moment of the decision, not of the request, so a
// reader can tell "nothing was available" from "nothing was eligible" (cl-132).
const (
	// ExcludedProtocol: new directives use the current generated wire contract.
	ExcludedProtocol = "protocol_unsupported"
	// ExcludedSpent: a completed managed job rental is retained custody, not capacity.
	ExcludedSpent = "managed_job_spent"
	// ExcludedModeConflict: the rental's worker already holds the other half of the
	// `oneof mode` — a job where a serving set is wanted, or the reverse.
	ExcludedModeConflict = "mode_conflict"
	// ExcludedWrongClass: CPU rental cannot satisfy an accelerator requirement.
	ExcludedWrongClass = "wrong_class"
	// ExcludedNotReady is spelled with the hub's own state word appended: the rental is
	// FINISHED — its acquisition failed, or it is being or has been given back — so no
	// request may wait for it. A rental merely on its way is VerdictAttaching, not this
	// (cl-185).
	ExcludedNotReady = "not_ready"
	// ExcludedAttaching is spelled with the machine appended: a fitting rental is
	// attaching, and nothing is chosen or bought past it (cl-170).
	ExcludedAttaching = "attaching"
	// ExcludedVRAMShort is spelled with the need and the device's memory appended: what
	// the pinned lanes need does not fit (cl-168, cl-170).
	ExcludedVRAMShort = "vram_short"
	// ExcludedBaseMismatch is spelled with the pod's own reason appended: the release's
	// requirements contradict the product's base image.
	ExcludedBaseMismatch = "base_mismatch"
	// ExcludedWidthUndeclared is spelled with the product's width and the degrees the
	// package declares: a machine wider than one card serves a placement as ONE group of
	// that degree, and a package whose author declared no such degree cannot be sharded
	// across it. The worker would refuse `device_group_unsupported` on arrival, so the
	// exclusion belongs here, before the pod is paid for (cl-179).
	ExcludedWidthUndeclared = "width_undeclared"
)

// RentalStanding is what this owner knows live about one ready rental a placement could
// go to: the reason its worker cannot take this request's MODE, or the attempts the
// worker holds and has been offered — what a new request waits behind, with the
// requests still queued for it. DesiredWorkerState is a full-replace `oneof mode` — a
// JobDirective or a serving placement set — so a pod worker cannot host both at once:
// converging a job onto a worker with a serving desire (or the reverse) would unload the
// other tenant mid-flight. An unattached rental has no mode yet and holds nothing.
func (c *Orchestrator) RentalStanding(id string, job bool) (reason string, held int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[rentalInstanceID(id)]
	if w == nil {
		return "", 0
	}
	if !w.supportsCurrentProtocol() {
		return ExcludedProtocol, 0
	}
	if w.exited || w.stopping {
		return "", 0
	}
	serving := len(w.desiredPackages) > 0 || w.desiredLocal != nil ||
		w.desiredUnpublishedPlacement != nil || len(w.observedRemote) > 0
	if (job && serving || !job && w.spec.IsJob()) && !c.idleRentalWorkerLocked(w) {
		return ExcludedModeConflict, 0
	}
	if c.sessions[w.bootID] == nil {
		return "", 0
	}
	return "", w.held + w.seats.reserved
}

// A completed job does not consume the rental forever. This fence uses observed
// custody plus every owner-side reservation, so a delayed idle report cannot hide
// an offer that has already crossed the dispatch boundary. Callers hold c.mu.
func (c *Orchestrator) idleRentalWorkerLocked(w *worker) bool {
	if w == nil || w.spec.Connection == nil || w.exited || w.stopping ||
		!w.snapshotAcknowledged || w.lastReport.IsZero() || w.held != 0 ||
		w.unacked != 0 || w.seats.reserved != 0 || w.reservedJobs != 0 {
		return false
	}
	for _, offer := range c.offers {
		if offer.worker == w {
			return false
		}
	}
	attempts, problem := c.opt.Store.OpenAttemptsOf(w.instanceID)
	return problem == nil && len(attempts) == 0
}

// Orchestrator is the Cozy daemon's scheduling role.
type Orchestrator struct {
	artifactSequence atomic.Uint64
	artifactPending  sync.Map
	opt              Options

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
	mediaCleaning    map[string]bool
	retainedCleaning map[string]*retainedCleanup
	// outputExporting serializes retries of one durable local --out obligation.
	outputExporting map[string]bool
	// pending is the dispatch queue: requests that have no ready worker YET. A requeue
	// with nowhere to go WAITS for capacity instead of evaporating — the alternative is
	// a request that quietly stops existing because a worker was still loading.
	pending []string
	// parked is what the drain knows about each queued request it skipped, by id
	// (route.go `parking`): the lanes it competes for and the overtake budget that turns
	// into a claim. An entry lives exactly as long as its request is in `pending`.
	parked map[string]*parking
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
	// phases is the preparation-phase lane (phase.go): what a request is doing before
	// its first attempt exists. Live-only and observational, exactly like `frames`.
	phases *phases
	// transferWake is a lossy nudge over durable request-attached transfer rows.
	transferWake         map[string]chan struct{}
	transferRunning      map[string]bool
	sourcePauseRunning   map[string]bool
	transferDispatching  map[string]bool
	transferCancels      map[string]context.CancelFunc
	transferProgressSeq  map[string]uint64
	sourcePrepareReplies map[string]uint64
	sourcePrepareBlocked map[string]sourcePreparationBackoff
	checkpointUploads    map[string]*checkpointUpload
	// localTransfers is command-scoped, lossy progress over Creator's durable request
	// row and sealed revision. A restart simply replays exact chunks from those authorities.
	localTransfers   map[string]*localTransfer
	childWatches     map[string]*session
	operationLookups map[string]bool // one native lookup in flight per request; not a result cache
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
		opt:                  opt,
		done:                 make(chan struct{}),
		sessions:             map[string]*session{},
		workers:              map[string]*worker{},
		waits:                map[string]*wait{},
		offers:               map[string]*dispatchReservation{},
		mediaCleaning:        map[string]bool{},
		outputExporting:      map[string]bool{},
		starting:             map[string]bool{},
		parked:               map[string]*parking{},
		ensuring:             map[string]chan struct{}{},
		frames:               newFanout(),
		phases:               newPhases(),
		transferWake:         make(map[string]chan struct{}),
		transferRunning:      make(map[string]bool),
		sourcePauseRunning:   make(map[string]bool),
		transferDispatching:  make(map[string]bool),
		transferCancels:      make(map[string]context.CancelFunc),
		transferProgressSeq:  make(map[string]uint64),
		sourcePrepareReplies: make(map[string]uint64),
		sourcePrepareBlocked: make(map[string]sourcePreparationBackoff),
		checkpointUploads:    make(map[string]*checkpointUpload),
		localTransfers:       make(map[string]*localTransfer),
		childWatches:         make(map[string]*session),
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
	for _, cleanup := range c.retainedCleaning {
		cleanup.cancel()
	}
	starting := make([]chan struct{}, 0, len(c.ensuring))
	for _, done := range c.ensuring {
		starting = append(starting, done)
	}
	c.mu.Unlock()
	// A start already crossed the ownership gate. Let it publish its process
	// before taking the shutdown census; later starts refuse under closing.
	for _, done := range starting {
		<-done
	}
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

// nextRevision mints the Directive revision. The record owner owns it; it is monotonic, and a
// changed body always carries a new one.
func (c *Orchestrator) nextRevision() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.nextRevisionLocked()
}

func (c *Orchestrator) nextRevisionLocked() uint64 {
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
func (c *Orchestrator) enqueue(requestID string) bool {
	if link, problem := c.opt.Store.MachineExecution(requestID); problem != nil || link != nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	// Cancellation and asynchronous activation race at this boundary. Read the
	// durable state while holding the same lock CancelQueued uses to remove an id:
	// either activation appends first and cancel removes it, or activation observes
	// the absorbing terminal and appends nothing.
	row, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || row == nil || (row.State != "submitted" && row.State != "queued") {
		return false
	}
	for _, id := range c.pending {
		if id == requestID {
			return true
		}
	}
	c.pending = append(c.pending, requestID)
	return true
}

// drain dispatches everything the ready capacity can now take. Called when a worker
// reports DISPATCHABLE (or job capacity), which is the only event that can change the
// answer.
//
// EVERY QUEUED REQUEST IS ASKED AGAINST ITS OWN CANDIDATES (cl-099). The queue is one
// FIFO over requests whose candidate lanes differ — a request pinned to a rental, a
// local one, a job whose model transfer is still materializing on the pod — and a head
// that cannot go now must not stop a request behind it whose lane is idle. Found live on
// the owner's box: a `--rental` job sat an hour in source materialization at the head and
// a local `paul/anima/generate` with the 4070 idle waited eight minutes behind it until
// its own timeout. So: a request with no dispatchable candidate right now PARKS — keeps
// its ordinal and position — and the next is tried. FIFO stays the law where two requests
// compete for one lane: a parked request is overtaken on its lanes at most its position
// worth of times, then claims them (`parking`), and `route` skips a claimed lane for
// everything behind the claimant.
//
// ONE DRAIN AT A TIME. A worker's report and `selectOrStart`'s own post-launch drain both
// fire within milliseconds of the same fact, and two concurrent drains read the same queue
// snapshot: the durable ordinal law is what refuses the duplicate, but doing the work
// twice and relying on a refusal is not a design. The lock makes the second drain read a
// queue the first one has already emptied.
func (c *Orchestrator) drain() {
	c.drainMu.Lock()
	defer c.drainMu.Unlock()
	c.mu.Lock()
	queued := append([]string(nil), c.pending...)
	c.mu.Unlock()
	// A named rental is an explicit serial queue. If its head cannot dispatch yet
	// (for example because its selected lane is still warming), later requests on
	// that same rental must not overtake it merely because they select a different
	// already-ready lane. Automatic requests remain work-conserving across rentals.
	blockedRentals := map[string]*WaitingRun{}
	for position, id := range queued {
		req, e := c.opt.Store.RequestRow(id)
		if e != nil || req == nil {
			c.forget(id)
			continue
		}
		if req.State != "submitted" && req.State != "queued" {
			c.forget(id)
			continue
		}
		if rentalID := req.RequestedRental; rentalID != "" && blockedRentals[rentalID] != nil && !c.activeChild(*req) {
			c.park(*req, position, waitFacts{cause: WaitQueueAhead, waitingFor: blockedRentals[rentalID]},
				"an earlier request is waiting on this rental")
			continue
		}
		if req.ModelTransfer != nil {
			transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
			if problem == nil && transfer != nil && transfer.State != "materialized" {
				// The transfer's own dispatch runs outside drainMu: its source lands on the
				// worker before the ordinal exists. Meanwhile the request holds its place
				// and its lanes like any other parked request.
				c.kickQueuedTransferDispatch(*req)
				c.park(*req, position, waitFacts{cause: WaitModelTransfer},
					"model transfer "+transfer.State+" on the selected worker")
				continue
			}
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
				if req.RequestedRental != "" {
					blockedRentals[req.RequestedRental] = &WaitingRun{Number: req.Number, RequestID: req.ID}
				}
				c.park(*req, position, waitFacts{}, e.Message)
				continue
			}
			c.failQueued(id, e, "")
			continue
		}
		c.forget(id)
		c.logf("%s left the dispatch queue as attempt %d", id, attempt)
	}
}

// park records that the drain skipped a queued request and why. The lanes it competes
// for are read fresh every time (a worker may have appeared); the budget is fixed at the
// first parking. Only a CHANGE is logged and emitted (`request.parked`): the drain runs
// on every worker report, and a parked request that is still parked is not news.
// A caller that knows the blocking condition passes it; an empty waitFacts means "a
// capacity refusal" and the condition is read from the same routing that names the lanes.
func (c *Orchestrator) park(req records.Request, position int, facts waitFacts, reason string) bool {
	c.mu.Lock()
	if !c.queued(req.ID) {
		c.mu.Unlock()
		return false
	}
	p := c.parked[req.ID]
	if p == nil {
		p = &parking{budget: position}
		c.parked[req.ID] = p
	}
	r := c.route(req)
	if facts.cause == "" {
		facts = c.classifyCapacityWait(req, r)
	}
	p.wait = facts
	lanes := r.lanes
	sort.Slice(lanes, func(i, j int) bool { return lanes[i].String() < lanes[j].String() })
	p.lanes = lanes
	blocking := ""
	if facts.waitingFor != nil {
		blocking = facts.waitingFor.RequestID
	}
	state := fmt.Sprintf("%s|%s|%s|%s|%t|%s", reason, facts.cause, facts.on, laneStrings(lanes), p.claims(), blocking)
	changed := state != p.logged
	p.logged = state
	overtaken, budget, claims := p.overtaken, p.budget, p.claims()
	c.mu.Unlock()
	if !changed {
		return false
	}
	c.logf("%s PARKED at queue position %d (lanes %s, overtaken %d of %d, claims=%t): %s",
		req.ID, position+1, orNone(laneStrings(lanes)), overtaken, budget, claims, reason)
	rows := make([]string, 0, len(lanes))
	for _, l := range lanes {
		rows = append(rows, l.String())
	}
	c.emit(req.ID, "request.parked", 0, facts.decorate(map[string]any{
		"reason": reason, "position": position + 1, "lanes": rows,
		"overtaken": overtaken, "budget": budget, "claims": claims,
	}, req))
	return true
}

// queued answers whether a request is still in the dispatch queue. Callers hold c.mu.
func (c *Orchestrator) queued(requestID string) bool {
	for _, id := range c.pending {
		if id == requestID {
			return true
		}
	}
	return false
}

func (c *Orchestrator) kickQueuedTransferDispatch(req records.Request) {
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil || (current.State != "submitted" && current.State != "queued") {
		return
	}
	req = *current
	c.mu.Lock()
	if c.transferDispatching[req.ID] {
		c.mu.Unlock()
		return
	}
	c.transferDispatching[req.ID] = true
	c.mu.Unlock()
	go func() {
		defer func() {
			c.mu.Lock()
			delete(c.transferDispatching, req.ID)
			c.mu.Unlock()
		}()
		attempt, problem := c.dispatch(req)
		if problem == nil {
			c.forget(req.ID)
			c.logf("%s left the dispatch queue as attempt %d", req.ID, attempt)
			go c.drain()
			return
		}
		if problem.Code != exit.Unavailable && (problem.Code != exit.Conflict || req.RetainWork) {
			c.failQueued(req.ID, problem, "")
			go c.drain()
			return
		}
		current, readProblem := c.opt.Store.RequestRow(req.ID)
		if readProblem == nil && current != nil && (current.State == "submitted" || current.State == "queued") {
			c.selectOrStart(*current)
			time.AfterFunc(2*time.Second, func() { c.kickQueuedTransferDispatch(*current) })
		}
	}()
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

// QueueState describes the request's machine queue. The dispatcher visits one global
// list, but requests pinned to another machine do not occupy this machine's queue.
// Read both numbers from the same pending snapshot and exclude settled/open attempts.
func (c *Orchestrator) QueueState(requestID string) (position, depth int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	owed, problem := c.opt.Store.Owed()
	if problem != nil {
		return 0, 0
	}
	venues := make(map[string]string, len(owed))
	for _, request := range owed {
		venue := request.Worker
		if venue == "" {
			venue = request.RequestedRental
		}
		if venue == "" && request.RentalRequired {
			venue = "unassigned-rental"
		}
		venues[request.ID] = venue
	}
	venue, found := venues[requestID]
	if !found {
		return 0, 0
	}
	for _, id := range c.pending {
		if other, waiting := venues[id]; !waiting || other != venue {
			continue
		}
		depth++
		if id == requestID {
			position = depth
		}
	}
	return position, depth
}

// reviveQueue re-asks select-or-start for the HEAD of the dispatch queue — the head per
// MACHINE. It runs when the answer to "does the capacity this request needs exist?" has
// just changed: a worker's process went, or a launch finished.
//
// It exists because `selectOrStart` returns early twice over — while a launch is in
// FLIGHT, and while any worker for the slot is resident — and neither early return leaves
// anything behind to ask again. cl-004's live run met both: six queued jobs sat forever
// behind a launch that was for a DIFFERENT plan, and again behind a worker that had been
// `kill -9`ed moments after they queued.
//
// ONE HEAD PER MACHINE. Reviving every waiting request would let two requests needing
// different plans on one card stop each other's worker in turn, so on each machine only
// the first request queued for it is asked. A request pinned to a rental and a local one
// do not share a card (cl-099): the first of each pin is revived, and neither waits on
// the other's residency. Nothing is dispatched here — `drain` is still the one placement
// path.
func (c *Orchestrator) reviveQueue() {
	c.mu.Lock()
	closing, queued := c.closing, append([]string(nil), c.pending...)
	c.mu.Unlock()
	if closing {
		return
	}
	asked := map[string]bool{}
	for _, id := range queued {
		req, e := c.opt.Store.RequestRow(id)
		if e != nil || req == nil {
			continue
		}
		machine := req.Worker
		if machine == "" {
			machine = req.RequestedRental
		}
		if asked[machine] {
			continue
		}
		asked[machine] = true
		c.selectOrStart(*req)
	}
}

// WakeQueue re-asks durable queued work after an external capacity observation changes. It is
// intentionally edge-triggered by the fleet observer; the queue remains the authority and
// duplicate wakes cannot mint duplicate attempts.
func (c *Orchestrator) WakeQueue() {
	c.reviveQueue()
	go c.drain()
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
		if req, read := c.opt.Store.RequestRow(attempt.RequestID); read == nil && req != nil &&
			req.ModelTransfer != nil && req.State == "finalizing" {
			c.kickRecoveredLocalTransfer(attempt.RequestID, attempt.Attempt)
			return
		}
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

// retirementLoop samples typed worker/refusal grounds. The ticker schedules observation;
// elapsed time and a count of unchanged reports grant no retirement authority.
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
//
// Stream/process exit is observed by the worker owner elsewhere. A worker that is merely
// silent or slow is not failed or replaced by this loop.
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
			if retirementGround(w) != "" {
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
	retire := func(victim candidate, subject string) bool {
		w := victim.worker
		c.mu.Lock()
		current := c.workers[w.instanceID] == w && !w.exited && !w.stopping
		ground := retirementGround(w)
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

	active := snapshot(func(w *worker) bool {
		return retirementGround(w) != ""
	})
	for _, victim := range active {
		open, e := c.opt.Store.OpenAttemptsOf(victim.worker.instanceID)
		if e != nil || len(open) == 0 {
			continue
		}
		if retire(victim, open[0].RequestID) {
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
		if retire(victim, head) {
			return
		}
	}
}

func retirementGround(w *worker) string {
	if w.refusal != nil {
		return fmt.Sprintf("this owner refused its claim (%s: %s)",
			w.refusal.ErrName(), w.refusal.Message)
	}
	if w.spec.Connection == nil && w.desiredRefusal != nil {
		return fmt.Sprintf("it refused the desired placement (%s: %s)",
			w.desiredRefusal.ErrName(), w.desiredRefusal.Message)
	}
	if w.faulted {
		return fmt.Sprintf("it reported a FAILED worker/placement axis: %s", w.fault)
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
// and "it will start later anyway" is not one. `actor` is who asked (cl-108): every
// cancellation is attributed in its own durable terminal, never an anonymous verdict.
func (c *Orchestrator) CancelQueued(requestID, actor string) *exit.Error {
	if actor == "" {
		actor = "an unnamed client"
	}
	payload := map[string]any{
		"status": "CANCELED", "cause": "CLIENT_CANCELED",
		"error_type": "CLIENT_CANCELED", "actor": actor,
		"error": fmt.Sprintf(
			"canceled by %s from the dispatch queue before any attempt was dispatched", actor),
		"outputs": []any{}, "requeuing": false,
	}
	applied, e := c.opt.Store.CancelQueuedRequest(requestID, payload)
	if e != nil {
		return e
	}
	if !applied {
		return nil
	}
	// CancelQueuedRequest committed the request terminal and its event together. Settle
	// the independent --out obligation too: attempt zero can never produce publishable
	// bytes, and `pending` must not outlive an absorbing request terminal.
	c.RetryOutputExport(requestID)
	row, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || row == nil {
		if problem != nil {
			return problem
		}
		return exit.Internalf("canceled request %s cannot be read back", requestID)
	}
	abortProblem := c.cancelLocalTransfer(requestID)
	c.forget(requestID)
	if row.ModelTransfer != nil {
		c.forgetTransferProgress(requestID)
		c.signalTransfer(requestID)
		c.kickCheckpointUpload(requestID)
	} else {
		c.frames.forget(requestID)
	}
	c.logf("%s left the dispatch queue: canceled before any attempt", requestID)
	c.signalClosed(requestWaitKey(requestID),
		exit.New(exit.Canceled, "%s was canceled before any attempt was dispatched", requestID))
	c.cleanupRequestAssets(*row)
	// If this was the FIFO head, the next request inherits the scheduling question now;
	// it must not wait for an unrelated worker report merely because the old head left.
	c.reviveQueue()
	if problem := c.releaseManagedNow(*row); problem != nil {
		return problem
	}
	return abortProblem
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
	delete(c.parked, requestID)
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
