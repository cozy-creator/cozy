package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/plan"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Binding is the record owner's resolution of one `entrypoint_binding_plan_id`. For a
// wholly weightless local package, RuntimePlan is the installed runtime's exact subject
// and Runtime owns the private bytes. The modeled/connected path still carries Record so
// it can stage or deliver those bytes explicitly.
//
// `Record` is the WHOLE older modeled record; which half is IDENTITY and which half is
// this machine's RESOLUTION is declared once in internal/plan (#506a). RuntimePlan never
// revives that format: its id, digest and length are consumed exactly as Runtime reports.
type Binding struct {
	Entrypoint string         `json:"entrypoint"`
	Record     map[string]any `json:"record"`
	// RuntimePlan is the exact ArtifactSubject emitted by the installed runtime for a
	// wholly weightless package. Runtime owns and privately stages those canonical
	// bytes; this record owner consumes only their measured identity before spawn.
	RuntimePlan *BindingPlanSubject `json:"runtime_plan,omitempty"`
	planID      string
	// Outputs names this entrypoint's RESULT FIELD PATHS (`image`, `preview`,
	// `detail.thumb`). It is not part of the binding record's canonical bytes — it is
	// the LAUNCHER's knowledge of the entrypoint's declared result shape, which cl-006
	// needs so a client is not forced to restate it on every submission. cr-003's
	// descriptor is where it comes from once an installed generation exists.
	Outputs []string `json:"outputs,omitempty"`
}

// BindingPlanSubject is the wire identity of one canonical plan document.
// Runtime-owned and remote plans both carry identity only. Local Runtime stages its
// private plan; a rented worker resolves signed package intent directly from Tensorhub.
// Cozy validates remote plan bytes in the control snapshot but never relays them.
type BindingPlanSubject struct {
	SubjectID string `json:"subject_id"`
	Kind      string `json:"kind"`
	Digest    string `json:"digest"`
	Length    uint64 `json:"length"`
}

// PlanID is the binding's identity. Runtime-owned plans carry it exactly; an older modeled
// record derives it from the canonical bytes of its IDENTITY half (#506a).
func (b *Binding) PlanID() (string, *exit.Error) {
	if b.RuntimePlan != nil {
		if _, err := canonical.Raw(b.RuntimePlan.SubjectID); err != nil {
			return "", exit.Internalf("binding-plan subject id is malformed: %s", err)
		}
		return b.RuntimePlan.SubjectID, nil
	}
	if b.planID != "" {
		return b.planID, nil
	}
	id, e := plan.ID(b.Record)
	if e != nil {
		return "", e
	}
	b.planID = id
	return id, nil
}

// Staged renders an older local modeled record exactly as it lands on a spawned worker's
// disk. Runtime-owned and remote grant subjects deliberately have no bytes on this path.
func (b *Binding) Staged() (string, []byte, *exit.Error) {
	if b.RuntimePlan != nil {
		return "", nil, exit.Named(exit.Structural, "binding_plan_bytes_unavailable",
			"binding plan %s is materialized by its runtime or direct worker acquisition", b.RuntimePlan.SubjectID)
	}
	id, e := b.PlanID()
	if e != nil {
		return "", nil, e
	}
	data, e := plan.Render(b.Record, id)
	if e != nil {
		return "", nil, e
	}
	return id, data, nil
}

// artifactSubject returns the exact wire subject carried by Runtime or Tensorhub.
// Older modeled local records obtain their subject from Staged's exact bytes instead.
func (b *Binding) artifactSubject() (*pb.ArtifactSubject, *exit.Error) {
	if b.RuntimePlan.Kind != "plan" {
		return nil, exit.Internalf("binding-plan subject kind is %q, want plan", b.RuntimePlan.Kind)
	}
	digest, err := canonical.Raw(b.RuntimePlan.Digest)
	if err != nil {
		return nil, exit.Internalf("binding-plan subject has a malformed digest: %s", err)
	}
	id, e := b.PlanID()
	if e != nil {
		return nil, e
	}
	if b.RuntimePlan.Length == 0 {
		return nil, exit.Internalf("binding-plan subject %s has zero length", id)
	}
	return &pb.ArtifactSubject{
		Digest: digest, SubjectId: id, Kind: b.RuntimePlan.Kind, Length: b.RuntimePlan.Length,
	}, nil
}

// THE THREE OBJECTS THAT USED TO BE ONE (#484). `PackageSpec` conflated three
// independent facts — how a worker PROCESS comes to exist, WHAT that worker is asked to
// host, and HOW this owner reaches a worker it did not spawn — and the conflation is
// exactly what rev-2 makes unstateable: a placement set is a full-replace document a
// RecordOwner converges a LIVE worker onto, so "the placement" cannot be a field of "the
// launch". Splitting them is what lets `ConvergePlacementSet` exist at all.

// WarmupPolicy is the boot warm pass, as an enum rather than a `NoWarm bool` (#484). A
// negated boolean can only ever spell two things and reads backwards at every call site;
// the enum says which policy is in force and leaves room for the ones cl-003 found it
// needs (a per-entrypoint warm shape).
type WarmupPolicy string

const (
	// WarmupOnBoot pays the first-call tax before a user's first request. It is the
	// serving default, so it is also the zero value's meaning.
	WarmupOnBoot WarmupPolicy = "on_boot"
	// WarmupNone skips the pass. cl-003 found the two cases that want it: an entrypoint
	// whose warm shape does not fit degrades its binding at boot for nothing, and the warm
	// pass is a BIT-LEVEL input to the first real image.
	WarmupNone WarmupPolicy = "none"
)

// Or fills in the serving default. A spec that never mentions warmup warms.
func (p WarmupPolicy) Or() WarmupPolicy {
	if p == "" {
		return WarmupOnBoot
	}
	return p
}

// DesiredPlacement is ONE assignment this owner wants a worker to host — the local half of
// rev-2's `Placement` (placement_id -> PlacementSpec, #481). It carries the identity
// facts, and nothing about how a process is started.
type DesiredPlacement struct {
	Package          string `json:"package"`            // org/name — the slot this placement serves under
	PackageReleaseID string `json:"package_release_id"` // PROVENANCE: what the spec was resolved FROM
	// InstallID is the install this placement was resolved from ("" = an uninstalled dev
	// tree). Was `Generation`, which named a protocol word this side does not own (#484).
	InstallID string `json:"install_id"`
	// PackageDescriptorDigest is the cr-003 descriptor's own identity, as the install VERIFIED
	// it in the generation's own venv. It rides the PlacementSpec because a placement is
	// named by the bytes it serves, and the descriptor is one of them.
	PackageDescriptorDigest string     `json:"package_descriptor_digest"`
	Bindings                []*Binding `json:"bindings"`
	// Remote control truth comes only from Tensorhub's persisted acquisition
	// snapshot. Local placements retain their independent install-derived path
	// and leave these fields empty.
	EnvironmentSpecDigest             string `json:"environment_spec_digest,omitempty"`
	InstalledEnvironmentReceiptDigest string `json:"installed_environment_receipt_digest,omitempty"`
	ExactPlacementSetDigest           string `json:"exact_placement_set_digest,omitempty"`
	ExactPlacementSetBytes            []byte `json:"exact_placement_set_bytes,omitempty"`
	// PlacementRevision is Tensorhub's monotonic revision for a private rental. It is
	// the exact DesiredWorkerState.revision Cozy relays for remote placements, so Hub
	// can compare desired, accepted, and converged without guessing a local daemon counter.
	PlacementRevision    uint64 `json:"placement_revision,omitempty"`
	PlacementIDValue     string `json:"placement_id,omitempty"`
	ModelObjectSetDigest string `json:"model_object_set_digest,omitempty"`
	ModelObjectSetLength uint64 `json:"model_object_set_length,omitempty"`
	// Hidden names the entrypoints this placement deliberately does NOT serve (#572d).
	// Recorded so an operator reading a placement can tell "no binding was staged" from
	// "a binding was staged and broke".
	Hidden []string `json:"hidden,omitempty"`
	// Jobs is the JOB-mode declaration (cl-004). A worker is in exactly ONE mode — the
	// DesiredWorkerState's own oneof says which — so a placement carries bindings or jobs,
	// never both, and `IsJob` is read from it rather than re-derived from what happens to
	// be populated later.
	Jobs []*JobPlan `json:"jobs,omitempty"`
}

// WorkerConnection is the dial identity for a worker this daemon did not spawn. The
// worker server certificate is pinned exactly, and Creator signs Claim with the
// per-rental Ed25519 key whose public half Tensorhub provisioned into the pod.
type WorkerConnection struct {
	RentalID     string `json:"rental_id"`
	Addr         string `json:"addr"`
	CACert       string `json:"ca_cert"` // path to the worker's pinned PEM
	WorkerID     string `json:"worker_id"`
	WorkerBootID string `json:"worker_boot_id"`
	// Media is the pod's BYTE PLANE (cl-014, ruled #506b): the co-resident media server
	// this owner uploads inputs and binding records to and downloads outputs from. It is
	// the pod's own listener with its own keys, not a second use of the control leg —
	// which is why it is a spec of its own rather than a port on this one.
	//
	// Nil is not "no bytes needed": a connected worker with no media plane cannot be fed,
	// and `connectWorker` refuses typed rather than falling back to owner-local paths the
	// pod cannot reach. That fallback is exactly the defect #493.3 caught.
	Media *media.Spec `json:"media,omitempty"`
}

// RemoteTarget joins one dial triple to the exact attempt-bound placement the
// provisioned worker already staged. Connection is authority; Placement is
// immutable execution meaning.
type RemoteTarget struct {
	Connection *WorkerConnection
	Placement  DesiredPlacement
}

// WorkerLaunchSpec is everything this owner needs to make one worker exist and host one
// placement. The caller (cl-010's `start`, or cl-001's live driver) resolves it from the
// install; the orchestrator itself resolves nothing about Python.
type WorkerLaunchSpec struct {
	Python   string       `json:"python"` // the interpreter inside the package's own venv
	Args     []string     `json:"args"`
	Dir      string       `json:"dir"`
	Imposed  []string     `json:"imposed"` // exact env values the launcher imposes, never inherited
	Devices  []string     `json:"devices"` // the device envelope this process may SEE
	GraceSec float64      `json:"grace_sec"`
	Warmup   WarmupPolicy `json:"warmup,omitempty"`
	// Placement is what this worker is launched to host. LAUNCH CLAMPS THE SET TO ONE
	// (worker-protocol header): a longer set is a typed refusal at the worker, so this
	// side names one placement rather than pretending to a generality it cannot deliver.
	Placement DesiredPlacement `json:"placement"`
	// Connection attaches an ALREADY-RUNNING worker instead of spawning one: the owner
	// pins CACert and signs Claim with its per-rental key. Nil = the ordinary local spawn.
	Connection *WorkerConnection `json:"connection,omitempty"`
}

// WorkerChange is what an EnsureWorker call actually DID (#484). `Resident bool` could
// only answer "was it already there", which collapses two different answers into one:
// nothing happened, and a live worker gained a placement it was not hosting. A caller that
// has to tell an idempotent no-op from a convergence cannot read it off a boolean.
type WorkerChange string

const (
	ChangeNone           WorkerChange = "none"
	ChangeWorkerStarted  WorkerChange = "worker_started"
	ChangePlacementAdded WorkerChange = "placement_added"
)

// pinnedPackage is the package name a request PINNED to a rental resolves its slot
// under. One attached worker per rental id: a second request naming the same rental finds
// the worker already conversing instead of attaching a second control stream to it, and a
// request naming no rental never lands in a rented worker's slot.
func pinnedPackage(pkg, rental string) string {
	if rental == "" {
		return pkg
	}
	return pkg + "@" + rental
}

// IsJob answers the worker's mode.
func (p DesiredPlacement) IsJob() bool { return len(p.Jobs) > 0 }

// RuntimeStagesBindings reports the one local launch mode in which the installed runtime
// owns every canonical binding-plan byte. A partial set is never a launch mode: older
// modeled local placements retain their explicit staging path.
func (p DesiredPlacement) RuntimeStagesBindings() bool {
	if len(p.Bindings) == 0 {
		return false
	}
	for _, b := range p.Bindings {
		if b.RuntimePlan == nil {
			return false
		}
	}
	return true
}

// IsJob answers the worker's mode, from the one placement it hosts.
func (s WorkerLaunchSpec) IsJob() bool { return s.Placement.IsJob() }

// InstanceID is the package's local worker SLOT identity, and it is deliberately STABLE
// across supervisor restarts: `worker_instance_id` names one provisioned instance
// lifetime, and outcome replay across a restart is authorized by that identity (02 §2). A
// slot keeps its journal root, which is what lets a restarted worker report its held
// attempts at all. A NEW install is a genuinely new instance and gets a new id.
func (p DesiredPlacement) InstanceID() string {
	slot := "slot/" + p.Package + "/" + p.InstallID
	if p.IsJob() {
		// A JOB worker is its own slot: one worker per (package, install, job function),
		// because a JobDirective names ONE job and a worker is in one mode. It is also what
		// lets a serving worker and a job worker of the same package coexist instead of
		// fighting over one instance id.
		slot += "/job/" + p.Jobs[0].Function
	}
	sum := sha256.Sum256([]byte(slot))
	return "ins-" + hex.EncodeToString(sum[:12])
}

func (s WorkerLaunchSpec) InstanceID() string {
	if s.Connection != nil && s.Connection.RentalID != "" {
		sum := sha256.Sum256([]byte("rental/" + s.Connection.RentalID))
		return "ins-" + hex.EncodeToString(sum[:12])
	}
	return s.Placement.InstanceID()
}

// PlacementID is the RecordOwner-minted routing + journal key for the one placement this
// slot hosts (#481; renamed from deployment_id — tensorhub's "deployment" is an id-less
// semantic tuple and the collision was real). Routing, NEVER identity: an InvocationSpec
// digest excludes it, so the same invocation is the same work wherever it routes.
func (p DesiredPlacement) PlacementID() string {
	if p.PlacementIDValue != "" {
		return p.PlacementIDValue
	}
	return "plc-" + strings.TrimPrefix(p.InstanceID(), "ins-")
}

type worker struct {
	instanceID string
	spec       WorkerLaunchSpec
	cmd        *exec.Cmd
	logPath    string
	home       string
	planIDs    []string
	// subjects is each staged binding-plan record as an ArtifactSubject — the exact bytes
	// a PlacementSpec names, sorted by digest. Kept from the staging pass so the placement
	// document names what was actually written rather than re-deriving it later.
	subjects []*pb.ArtifactSubject
	// media is the pod's byte plane, dialled once at connect. Nil for a locally spawned
	// worker: it shares this host's filesystem, so its grant IS a path and there is
	// nothing to transport.
	media *media.Client
	// cancelControl closes the active owner stream for an attached worker. Local
	// processes also exit through their process waiter; remote workers have no local
	// process to signal, so retirement must explicitly close this connection.
	cancelControl func()
	// stopping makes one teardown the sole owner of this worker's process, control stream,
	// durable row, and recovery decision. Other callers wait for stopped instead of racing
	// a replacement under the same deterministic instance id.
	stopping    bool
	stopped     chan struct{}
	attachDone  chan struct{}
	processDone chan struct{}

	// remoteInstance is the instance identity an ATTACHED pod's worker declared for
	// itself. A pod is a machine this host never spawned, so it names its own worker the
	// way it mints its own boot id; what this host may hold it to is that the name does
	// not CHANGE under it. Empty for a locally spawned worker, whose identity is this
	// launcher's by construction.
	remoteInstance string
	remoteWorkerID string

	// what the worker itself reported; the orchestrator echoes, never invents
	exited bool
	// refusal is a claim-time verdict this owner reached about the thing at the other end —
	// a pod serving a different release than the one this host pinned, say. It is kept so a
	// waiter gets the ANSWER instead of waiting out the silence window for a worker this
	// owner has already decided not to talk to (#505's carried-not-verified gap).
	refusal *exit.Error
	// exitCode is the process's own disposition. RECYCLE is not a death (cr-009): a
	// run-once job worker exits with it the moment its terminal is acknowledged, and
	// reading that as "the worker died" turns a completed job into a failed request.
	exitCode int
	pid      int // the process THIS orchestrator started; the only one that may register
	// bootstrap is the per-spawn credential the worker verifies at Claim (#463's flip:
	// the OWNER presents it as proof; the worker checks constant-time). Minted for every
	// spawn on every platform — the flipped direction has no SO_PEERCRED to lean on.
	bootstrap secret.Value
	// placementID is the RecordOwner-minted routing key for the ONE placement this worker
	// hosts (#481). Routing, never identity.
	placementID string
	bootID      string

	// THE OBSERVED STATE, on rev-2's own axes. Nothing here is inferred: every field is
	// the worker's own last word, and the two axes exist because ONE enum cannot say
	// "staged on disk but offline" — the exact state an outgoing spec holds under
	// fallback-retention (#473/#482).
	phase           pb.WorkerPhase          // machine lifecycle, out of the placement enum
	materialization pb.MaterializationState // axis 1: what is on disk
	serving         pb.ServingState         // axis 2: what it will take
	generation      uint64                  // THIS placement's executor generation
	dispatchable    map[string]bool         // dispatchable_plan_ids
	materializable  map[string]bool         // DISJOINT from dispatchable
	specDigest      []byte                  // what the placement actually HOLDS right now
	fallbackPin     []byte                  // the predecessor kept for restore; empty = replacement PAUSED

	// THE ONE ADMISSION FENCE (#472e/#482/#486c). Per-placement credits are DELETED: N
	// counters over ONE serialized device advertise N x the real capacity. Capacity is a
	// WORKER property; dispatchability is a PLACEMENT property.
	admission    pb.AdmissionState
	admissionGen uint64 // echoed on every offer; a stale echo refuses deterministically
	// reportedSlots is the worker's last available_attempt_slots. reservedSlots is the
	// owner's reservation from choosing a worker until its offer is accepted or refused.
	// Keeping them separate prevents a Report racing preparation or send from reopening it.
	reportedSlots int
	reservedSlots int
	slots         int // reportedSlots - reservedSlots, clamped at zero
	// unacked is how many outcomes this owner HOLDS without having acked. The worker
	// counts them against its own available_attempt_slots (#480d), so an owner that stops
	// acking starves its own admission — boundedness is structural, and this is the number
	// that makes it VISIBLE here instead of only on the worker.
	unacked int

	// acceptedRevision is what the worker durably ACCEPTED; convergedRevision advances
	// only when its observed state SATISFIES that intent (#473). `applied_revision` is
	// retired as dishonest — it advanced on acceptance, so a reader learned only that its
	// own message arrived. converged < accepted is the normal, readable state of a
	// convergence in progress or a latched failure, never an error.
	acceptedRevision  uint64
	convergedRevision uint64
	acceptedSetDigest []byte
	// The job lane uses the same reported-versus-pre-offer split as serving. jobsAvail is
	// the effective number dispatch reads.
	reportedJobs int
	reservedJobs int
	jobsAvail    int
	// revision is the desired-state revision THIS owner last issued, with the placement
	// set it issued. The set travels as bytes, so the owner keeps the bytes it authored:
	// a worker's accepted digest is compared against these, never re-canonicalized.
	revision             uint64
	artifactRevision     uint64
	delegationID         string
	authorizationExpires uint64
	acquisition          PlacementAcquisitionFacts
	setDigest            []byte
	setBytes             []byte
	// The worker's last diagnostic fault and the first report that carried it. The FAILED
	// axes, not this text, decide terminality: BINDING_DEGRADED may coexist with service.
	fault      string
	faulted    bool
	errorSince time.Time
	// lastReport is when this worker last said ANYTHING. The ObservedWorkerState cadence
	// is a protocol fact rather than a choice made here, which is what makes silence
	// measurable: a worker that is loading for twenty minutes still reports every
	// period, so "has not reported" is a stall and never merely "is slow".
	lastReport time.Time
	// spawned starts the silence clock BEFORE the first report: a worker that never
	// claims at all owes its first one on the same cadence, and a wait with no clock
	// until the first frame is a wait that cannot end (found by the flip: a pre-flip
	// runtime wheel that cannot host left the readiness wait spinning forever).
	// For an ATTACHED worker it is zero until its first ClaimAck: the dial / plan delivery
	// phase is bounded by its own measured progress, not by the report cadence.
	spawned time.Time
	// LRU is dispatch-based, not report-based: reports say the worker lives, while an
	// accepted attempt says a user actually used it. Never-used workers fall back to
	// residentRevision so two cold holders still have a deterministic oldest member.
	residentRevision uint64
	lastUseRevision  uint64

	// THE NO-PROGRESS GROUND'S STATE (cl-025). progressSig is a signature over every
	// axis the last report carried — states, revisions, plan sets, the activity lane's
	// high-water sequence — so "no axis moved" is a comparison of two reports, never a
	// clock. wedged is whether the worker's own liveness monitor currently declares a
	// WEDGED subject (its verdict rides the activity lane), and noProgress counts the
	// SUCCESSIVE reports that both declared it and moved nothing. Any movement resets.
	progressSig    string
	wedged         bool
	wedgedSubjects map[string]bool
	noProgress     int
}

// dispatchableFor is the ROUTING GATE, and it is two questions with two owners (#482).
// DISPATCHABILITY is a PLACEMENT property: the serving axis says DISPATCHABLE and the
// placement advertises this plan. ADMISSION is a WORKER property: the fence is OPEN and a
// seat is free. A placement that is STAGED but OFFLINE is not capacity, however much of it
// is on disk.
func (w *worker) dispatchableFor(planID string) bool {
	return w.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE && w.dispatchable[planID]
}

// admissible answers the worker-level half. CLOSED is STRUCTURAL (pre-snapshot-barrier,
// draining, mid-cutover); OPEN with zero seats is TRANSIENT saturation — both refuse under
// CAUSE_CODE_NO_CAPACITY at the worker, and an owner backs off differently for each
// (#486c), which is why they are two fields here and not one.
func (w *worker) admissible() bool {
	return w.admission == pb.AdmissionState_ADMISSION_STATE_OPEN && w.slots > 0
}

func (w *worker) observeSlots(n int) {
	w.reportedSlots = n
	w.slots = max(0, n-w.reservedSlots)
}

func (w *worker) observeJobs(n int) {
	w.reportedJobs = n
	w.jobsAvail = max(0, n-w.reservedJobs)
}

// EnsureWorker makes one worker exist hosting one placement, and SAYS WHICH OF THE THREE
// THINGS IT DID (#484). It is idempotent by construction rather than by a caller's check:
// a slot already hosting this placement is `none`, a slot that exists without it converges
// and is `placement_added`, and only an absent slot is spawned or connected.
//
// The three answers are not decoration. `cozy invoke run` and POST /v1/local/workers both need
// to tell an idempotent no-op from a real convergence, and a boolean `resident` could only
// tell them "it was there", which is true of both.
func (c *Orchestrator) EnsureWorker(spec WorkerLaunchSpec) (string, WorkerChange, *exit.Error) {
	instanceID := spec.InstanceID()
	var live *worker
	var mine chan struct{}
	for {
		c.mu.Lock()
		if inFlight := c.ensuring[instanceID]; inFlight != nil {
			c.mu.Unlock()
			<-inFlight
			continue
		}
		live = c.workers[instanceID]
		if live != nil && live.stopping {
			stopped := live.stopped
			c.mu.Unlock()
			<-stopped
			continue
		}
		if live != nil && live.exited {
			live = nil
		}
		if live == nil {
			mine = make(chan struct{})
			c.ensuring[instanceID] = mine
		}
		c.mu.Unlock()
		break
	}
	if live != nil {
		// The worker is here. Does it already host what is wanted? The placement's
		// identity for this purpose is its plan set, which is what the desired set names.
		if hostsPlans(live, spec.Placement) {
			return instanceID, ChangeNone, nil
		}
		if live.spec.Connection != nil {
			return instanceID, ChangeNone, exit.Named(exit.Conflict,
				"rental.placement_revision_required",
				"rental %s is claimed with another desired placement",
				live.spec.Connection.RentalID).
				WithRemedy("use `cozy rental list %s --package-ref ...`; remote desired state always follows its active grant",
					live.spec.Connection.RentalID)
		}
		if e := c.ConvergePlacementSet(instanceID, []DesiredPlacement{spec.Placement}); e != nil {
			return instanceID, ChangeNone, e
		}
		return instanceID, ChangePlacementAdded, nil
	}
	defer func() {
		c.mu.Lock()
		if c.ensuring[instanceID] == mine {
			delete(c.ensuring, instanceID)
			close(mine)
		}
		c.mu.Unlock()
	}()
	if spec.Connection != nil {
		id, e := c.connectWorker(spec)
		return id, ChangeWorkerStarted, e
	}
	id, e := c.spawnWorker(spec)
	originalHolders := map[string]bool{}
	expectedHolders := 0
	if e != nil && e.ErrName() == "device_envelope_held" {
		holders, problem := c.opt.Store.DeviceHolders(spec.Devices)
		if problem != nil {
			return "", ChangeNone, problem
		}
		for _, holder := range holders {
			originalHolders[holder] = true
		}
		expectedHolders = len(holders)
	}
	// A multi-device envelope may be held by one idle serving worker per device. Each
	// pass re-reads the ledger and may evict exactly one observed LRU holder; the bound is
	// the requested envelope size, so concurrent/new evidence escapes instead of turning
	// pressure handling into an unbounded machine drain.
	for evictions := 0; e != nil && e.ErrName() == "device_envelope_held" &&
		evictions < len(spec.Devices); evictions++ {
		evicted, evictionProblem := c.evictLRUIdleDeviceHolder(
			spec.Devices, originalHolders, expectedHolders)
		if evictionProblem != nil {
			return "", ChangeNone, evictionProblem
		}
		if !evicted {
			break
		}
		expectedHolders--
		id, e = c.spawnWorker(spec)
	}
	return id, ChangeWorkerStarted, e
}

// evictLRUIdleDeviceHolder releases one local device envelope under launch pressure.
// Every holder conflicting with the requested envelope must be an idle local serving
// worker; an active, remote, job, unknown, offered, reserved, or unacked holder makes the
// conflict ineligible and preserves all workers. A changed holder set is concurrent/new
// evidence and is never folded into the original pressure decision. The selected holder
// is claimed under the orchestrator lock before teardown, so dispatch cannot race into it.
func (c *Orchestrator) evictLRUIdleDeviceHolder(devices []string,
	original map[string]bool, expected int) (bool, *exit.Error) {
	holders, problem := c.opt.Store.DeviceHolders(devices)
	if problem != nil || len(holders) == 0 {
		return false, problem
	}
	if len(holders) != expected {
		return false, nil
	}
	for _, holder := range holders {
		if !original[holder] {
			return false, nil
		}
	}
	active, problem := c.opt.Store.ActiveRequests()
	if problem != nil {
		return false, problem
	}

	c.mu.Lock()
	eligible := make([]*worker, 0, len(holders))
	for _, instanceID := range holders {
		w := c.workers[instanceID]
		if !c.idleLocalServingWorkerLocked(w, active) {
			c.mu.Unlock()
			return false, nil
		}
		eligible = append(eligible, w)
	}
	sort.Slice(eligible, func(i, j int) bool { return lessRecentlyUsed(eligible[i], eligible[j]) })
	victim := eligible[0]
	victim.stopping = true
	instanceID, pkg := victim.instanceID, victim.spec.Placement.Package
	lastUse, resident := victim.lastUseRevision, victim.residentRevision
	c.mu.Unlock()

	c.logf("device pressure: evicting LRU idle local worker %s (%s, last_use=%d, resident=%d)",
		instanceID, pkg, lastUse, resident)
	if !c.stopClaimedWorker(victim, StopGrace) {
		return false, exit.New(exit.Conflict,
			"idle device holder %s changed before it could be evicted", instanceID)
	}
	c.reviveQueue()
	return true, nil
}

func lessRecentlyUsed(a, b *worker) bool {
	if (a.lastUseRevision == 0) != (b.lastUseRevision == 0) {
		return a.lastUseRevision == 0
	}
	if a.lastUseRevision != b.lastUseRevision {
		return a.lastUseRevision < b.lastUseRevision
	}
	if a.residentRevision != b.residentRevision {
		return a.residentRevision < b.residentRevision
	}
	return a.instanceID < b.instanceID
}

// EnsureRental makes an already-provisioned rental's worker resident without invoking
// an package. This is the explicit paid-run preflight: the control claim must persist
// actual hardware readback before a request can be submitted to the pod.
func (c *Orchestrator) EnsureRental(id string) (string, string, WorkerChange, *exit.Error) {
	if c.opt.Rentals == nil {
		return "", "", ChangeNone, exit.Unavailablef("this Cozy daemon attaches no rented workers")
	}
	target, problem := c.opt.Rentals(id)
	if problem != nil {
		return "", "", ChangeNone, problem
	}
	if target == nil || target.Connection == nil {
		return "", "", ChangeNone, exit.Internalf("rental %s resolved no connected worker", id)
	}
	// The rental IS the slot, exactly as dispatch pins it: the same `org/name@<rental>`
	// name, so a probe and a later pinned run share ONE connected worker.
	spec := WorkerLaunchSpec{Placement: target.Placement, Connection: target.Connection}
	spec.Placement.Package = pinnedPackage(spec.Placement.Package, id)
	instance, change, problem := c.EnsureWorker(spec)
	return instance, spec.Placement.Package, change, problem
}

// hostsPlans answers whether a live worker was launched holding exactly the plans this
// placement names. It reads what the LAUNCHER staged, not what the worker has got around
// to advertising: a worker still materializing already holds the placement.
func hostsPlans(w *worker, p DesiredPlacement) bool {
	want := map[string]bool{}
	for _, b := range p.Bindings {
		id, e := b.PlanID()
		if e != nil {
			return false
		}
		want[id] = true
	}
	for _, j := range p.Jobs {
		want[j.DescriptorID] = true
	}
	if len(want) != len(w.planIDs) {
		return false
	}
	for _, id := range w.planIDs {
		if !want[id] {
			return false
		}
	}
	return true
}

// spawnWorker journals the device grant, resolves exact binding subjects, and spawns the
// worker. It stages older modeled records; weightless canonical bytes are staged by the
// runtime's explicit launch mode. The grant is journaled BEFORE the process exists: a process that was never
// granted an envelope cannot appear, and two concurrent starts cannot both consume one.
func (c *Orchestrator) spawnWorker(spec WorkerLaunchSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	// The slot's root is REUSED on purpose: the worker journal under it is what a
	// restarted worker replays as `recovered_attempts`. Wiping it would manufacture the
	// absence this protocol refuses to manufacture.
	root := c.opt.Layout.WorkerDir(instanceID)
	workerHome := filepath.Join(root, "home")
	if err := os.MkdirAll(filepath.Join(workerHome, "binding-plans"), 0o755); err != nil {
		return "", exit.Internalf("cannot create the worker home %s: %s", workerHome, err)
	}

	var planIDs []string
	var subjects []*pb.ArtifactSubject
	for _, b := range spec.Placement.Bindings {
		var id string
		var subject *pb.ArtifactSubject
		if b.RuntimePlan != nil {
			var e *exit.Error
			subject, e = b.artifactSubject()
			if e != nil {
				return "", e
			}
			id = subject.SubjectId
		} else {
			var data []byte
			var e *exit.Error
			id, data, e = b.Staged()
			if e != nil {
				return "", e
			}
			if _, e := plan.Stage(filepath.Join(workerHome, "binding-plans"), b.Record); e != nil {
				return "", e
			}
			subject = subjectOf(id, data)
		}
		planIDs = append(planIDs, id)
		subjects = append(subjects, subject)
	}
	sort.Strings(planIDs) // the wire field is sorted lexicographic ascending
	sortSubjects(subjects)
	if spec.IsJob() {
		if e := stageJobPlans(workerHome, spec.Placement.Jobs); e != nil {
			return "", e
		}
		for _, p := range spec.Placement.Jobs {
			planIDs = append(planIDs, p.DescriptorID)
		}
	}

	logPath := filepath.Join(root, "worker.log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return "", exit.Internalf("cannot open the worker log %s: %s", logPath, err)
	}

	// The GRANT IS JOURNALED FIRST, before any process exists. An admission that
	// refuses here means nothing was started, which is why the refusal has no cleanup.
	if e := c.opt.Store.SpawnWorker(records.WorkerProcess{
		InstanceID:       instanceID,
		Package:          spec.Placement.Package,
		Generation:       spec.Placement.InstallID,
		PackageReleaseID: spec.Placement.PackageReleaseID,
		WorkerID:         "local",
		Devices:          spec.Devices,
	}); e != nil {
		logFile.Close()
		return "", e
	}

	// `cozy-runtime serve`'s CLOSED flag set — the launch facts nothing can discover for a
	// orchestrator. It is the runtime's PUBLIC launch grammar, so this is the only spelling
	// the orchestrator knows: `--socket`/`--out` are the path grants the verb translates
	// into its worker loop.
	// THE WORKER BINDS ITS OWN SOCKET (#436): a unix path under its run root (loopback
	// `127.0.0.1:0` on Windows — proc shims pick it), published to run/control.addr for
	// this owner to dial. `--socket` is the runtime serve verb's listen grant.
	listen := filepath.Join(root, "run", "control.sock")
	if bootstrapRequired {
		// Windows has no unix socket worth having: the worker binds an ephemeral
		// loopback port and publishes the real address (cr-020 item 4).
		listen = "127.0.0.1:0"
	}
	if err := os.MkdirAll(filepath.Join(root, "run"), 0o755); err != nil {
		return "", exit.Internalf("cannot create the worker run root: %s", err)
	}
	_ = os.Remove(filepath.Join(root, "run", "control.addr"))
	_ = os.Remove(filepath.Join(root, "run", "control.sock"))
	args := append([]string{}, spec.Args...)
	args = append(args,
		"--socket", listen,
		"--out", filepath.Join(root, "run"),
		"--instance-id", instanceID,
		"--release-id", spec.Placement.PackageReleaseID,
		"--devices", strings.Join(spec.Devices, ","),
		"--grace", strconv.FormatFloat(graceOr(spec.GraceSec), 'f', -1, 64),
	)
	if spec.Warmup.Or() == WarmupNone {
		args = append(args, "--no-warm")
	}
	cmd := exec.Command(spec.Python, args...)
	cmd.Dir = spec.Dir
	// The child's whole environment: the allowlist plus the values THIS launcher
	// imposes. COZY_HOME points the worker at its own staged records and nothing else.
	imposed := append([]string{
		"COZY_HOME=" + workerHome,
		"CUDA_VISIBLE_DEVICES=" + strings.Join(spec.Devices, ","),
	}, spec.Imposed...)
	// The launcher mints a PER-SPAWN bootstrap credential and hands it over through the
	// environment — the one channel only this child inherits. The flip made it the
	// controller-authority proof on EVERY platform (#463): this owner presents it as
	// Claim.proof and the worker verifies constant-time; the unix socket's filesystem
	// authority is belt, this is suspenders.
	bootstrap := secret.Mint()
	imposed = append(imposed, secret.EnvEntry("COZY_BOOTSTRAP_CREDENTIAL", bootstrap))
	cmd.Env = c.opt.Cfg.Child(imposed...)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	setProcessGroup(cmd)

	w := newWorker(instanceID, spec)
	w.cmd, w.logPath, w.home = cmd, logPath, workerHome
	w.planIDs, w.subjects, w.bootstrap = planIDs, subjects, bootstrap
	w.processDone = make(chan struct{})
	// Registered BEFORE the process can dial: a worker that registers faster than its
	// launcher can record it would be refused as an instance nobody spawned.
	c.mu.Lock()
	c.residentRevision++
	w.residentRevision = c.residentRevision
	c.workers[instanceID] = w
	c.mu.Unlock()

	if err := cmd.Start(); err != nil {
		c.mu.Lock()
		delete(c.workers, instanceID)
		c.mu.Unlock()
		_ = c.opt.Store.CloseWorker(instanceID)
		logFile.Close()
		return "", exit.Internalf("cannot start the package worker: %s", err)
	}
	// The group is established at fork on Unix and must be ATTACHED after start on Windows,
	// where the child is created suspended and the job adopts it before it runs. Adoption
	// FAILS CLOSED (#449): a worker that cannot be contained is ended before it executes,
	// and the spawn refuses exactly as if the process had never started.
	if err := adoptProcessGroup(cmd); err != nil {
		_ = cmd.Process.Kill()
		go func() { _ = cmd.Wait(); logFile.Close() }()
		c.mu.Lock()
		delete(c.workers, instanceID)
		c.mu.Unlock()
		_ = c.opt.Store.CloseWorker(instanceID)
		return "", exit.Internalf("cannot contain the package worker: %s", err)
	}
	c.mu.Lock()
	w.pid = cmd.Process.Pid
	c.mu.Unlock()
	if e := c.opt.Store.WorkerStarted(instanceID, cmd.Process.Pid, birthOf(cmd.Process.Pid)); e != nil {
		_ = killGroup(cmd.Process.Pid, syscall.SIGKILL)
		_ = cmd.Wait()
		logFile.Close()
		releaseGroup(cmd.Process.Pid)
		close(w.processDone)
		close(w.attachDone)
		c.mu.Lock()
		if c.workers[instanceID] == w {
			delete(c.workers, instanceID)
		}
		c.mu.Unlock()
		_ = c.opt.Store.CloseWorker(instanceID)
		return "", e
	}
	c.logf("worker %s spawned pid=%d devices=[%s] plans=%d",
		instanceID, cmd.Process.Pid, strings.Join(spec.Devices, ","), len(planIDs))
	// THE OWNER DIALS (#436): one goroutine owns this worker's claim conversation for
	// the life of its process, re-claiming across stream drops.
	go c.attach(w)

	// A worker row outliving its process is exactly the sidecar bug this design refuses:
	// the row holds a device grant, so it dies with the process that held it.
	go func() {
		defer close(w.processDone)
		err := cmd.Wait()
		logFile.Close()
		// The process is reaped: retire its containment handle so a reused pid can
		// never meet a stale job (Windows; a no-op where the group is a kernel fact).
		releaseGroup(cmd.Process.Pid)
		c.mu.Lock()
		current, live := c.workers[instanceID]
		mine := live && current == w
		controlledStop := mine && w.stopping
		if mine {
			w.exitCode = cmd.ProcessState.ExitCode()
			// The entry STAYS, marked exited: a caller waiting on readiness needs to
			// learn the worker is gone and where its log is, and a row that vanishes
			// silently is the same lie as a row that outlives its process.
			w.exited = true
			if !controlledStop {
				w.stopping = true // spontaneous exit owns cleanup until its row is closed
			}
			if w.bootID != "" {
				delete(c.sessions, w.bootID)
			}
		}
		cancelControl := w.cancelControl
		c.mu.Unlock()
		if cancelControl != nil {
			cancelControl()
		}
		if mine {
			if controlledStop {
				return // the teardown that set stopping owns the row and recovery decision
			}
			<-w.attachDone
			closeProblem := c.opt.Store.CloseWorker(instanceID)
			c.mu.Lock()
			if c.workers[instanceID] == w {
				delete(c.workers, instanceID)
			}
			close(w.stopped)
			closing := c.closing
			c.mu.Unlock()
			if closeProblem != nil {
				c.logf("worker %s exited (%v) but its durable row could not close: %s",
					instanceID, err, closeProblem.Message)
				return
			}
			c.logf("worker %s exited (%v); its device grant is released — log %s",
				instanceID, err, logPath)
			// The queue's answer to "does capacity exist" just changed, and so has the
			// fate of anything this worker was RUNNING.
			if !closing {
				go c.recoverWorker(w.spec)
			}
		}
	}()
	return instanceID, nil
}

// newWorker is the one place a worker's observed state starts, and it starts UNKNOWN:
// every enum at its UNSPECIFIED zero, every map empty, zero seats. A worker this owner has
// not heard from is not dispatchable, and the gates read that off these values rather than
// off a separate "have we heard from it" flag that could disagree with them.
func newWorker(instanceID string, spec WorkerLaunchSpec) *worker {
	w := &worker{
		instanceID:     instanceID,
		spec:           spec,
		placementID:    spec.Placement.PlacementID(),
		dispatchable:   map[string]bool{},
		materializable: map[string]bool{},
		stopped:        make(chan struct{}),
		attachDone:     make(chan struct{}),
	}
	if spec.Connection == nil {
		w.spawned = time.Now()
	}
	return w
}

// connectWorker registers an ALREADY-RUNNING worker (a rented pod's TLS leg, cl-015):
// no spawn, no device grant (the pod's card is the pod's), no birth identity — the
// conversation is the same claim the local path runs, dialed at the rental's address
// with the pinned server cert and signed Creator ClaimProof (#445/proto-013).
// The media plane remains the invocation byte path, but it is NOT a package distribution
// path. Binding plans are ordinary artifact-grant subjects now: Tensorhub supplies their
// locations and the worker verifies their digests while materializing PlacementSet/2.
func (c *Orchestrator) connectWorker(spec WorkerLaunchSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	if spec.Connection.Media == nil {
		return "", exit.Named(exit.Unavailable, "rental_no_media_plane",
			"rental %s pins a control address and no media plane, and a pod that cannot be "+
				"handed bytes cannot be served", spec.Placement.Package).
			WithRemedy("a rented pod runs its worker and a co-resident media server " +
				"(cl-014); this host will not fall back to granting paths on its own disk, " +
				"because the pod cannot reach them").
			WithNext("cozy rental list")
	}
	// THE SAME SILENCE BUDGET THE CONTROL LEG LIVES UNDER. One pod, two listeners, one
	// standard for "has not answered": a byte plane that misses what eight report periods
	// cost is judged exactly as a worker that misses eight reports.
	byteplane, e := media.Dial(*spec.Connection.Media, SilentReports*ReportCadence,
		int64(c.maxOutputBytes()))
	if e != nil {
		return "", e
	}
	if e := byteplane.Health(); e != nil {
		return "", e
	}
	planIDs, subjects, e := remoteBindingSubjects(spec.Placement)
	if e != nil {
		return "", e
	}
	// AttachWorker, not SpawnWorker: the device-envelope admission arbitrates THIS host's
	// cards, and the pod's card is the pod's. There is no grant to journal and none to
	// release, which is also why nothing here has a pid or a birth identity to record.
	if e := c.opt.Store.AttachWorker(records.WorkerProcess{
		InstanceID:       instanceID,
		Package:          spec.Placement.Package,
		Generation:       spec.Placement.InstallID,
		PackageReleaseID: spec.Placement.PackageReleaseID,
		WorkerID:         "remote",
	}); e != nil {
		return "", e
	}
	w := newWorker(instanceID, spec)
	w.logPath = "(connected worker: its log lives on the pod)"
	w.planIDs, w.subjects, w.media = planIDs, subjects, byteplane
	c.mu.Lock()
	c.workers[instanceID] = w
	c.mu.Unlock()
	c.logf("worker %s CONNECTED at %s (media %s) plans=%d",
		instanceID, spec.Connection.Addr, byteplane.Addr(), len(planIDs))
	go c.attach(w)
	return instanceID, nil
}

func remoteBindingSubjects(placement DesiredPlacement) ([]string, []*pb.ArtifactSubject, *exit.Error) {
	var planIDs []string
	var subjects []*pb.ArtifactSubject
	for _, b := range placement.Bindings {
		if b.RuntimePlan == nil {
			return nil, nil, exit.Named(exit.Structural, "remote_binding_plan_subject_missing",
				"remote binding %s carries no exact artifact subject", b.Entrypoint)
		}
		subject, e := b.artifactSubject()
		if e != nil {
			return nil, nil, e
		}
		planIDs = append(planIDs, subject.SubjectId)
		subjects = append(subjects, subject)
	}
	sort.Strings(planIDs)
	sortSubjects(subjects)
	return planIDs, subjects, nil
}

// subjectOf names the exact bytes of one staged binding-plan record, as the
// `ArtifactSubject` a PlacementSpec carries. The subject_id is the plan id (a `sha256:`
// document identity) and the digest is over the record as it lands on the worker's disk:
// the id fences MEANING, the digest fences the BYTES, and they are deliberately not the
// same number.
func subjectOf(planID string, data []byte) *pb.ArtifactSubject {
	return &pb.ArtifactSubject{
		Digest:    canonical.Digest(data),
		SubjectId: planID,
		Kind:      "plan",
		Length:    uint64(len(data)),
	}
}

// sortSubjects puts the list in the order the wire declares — by digest, ascending.
func sortSubjects(subjects []*pb.ArtifactSubject) {
	sort.Slice(subjects, func(i, j int) bool {
		return bytes.Compare(subjects[i].Digest, subjects[j].Digest) < 0
	})
}

func graceOr(v float64) float64 {
	if v <= 0 {
		return 3
	}
	return v
}

// ReportCadence is the worker's OWN ObservedWorkerState period, a protocol fact rather
// than a number chosen here: the worker reports durably on this cadence so an owner can
// always see a desired state it issued that has not converged.
const ReportCadence = 2 * time.Second

// SilentReports is how many of those periods may pass with NOTHING arriving before the
// worker is called silent. A COUNT of missed heartbeats, which is why it is not a guess
// about how long a load takes: a worker resident-loading a 20 GB binding for half an hour
// reports on every one of them, and only a worker that has stopped talking misses them.
const SilentReports = 8

// EnsurePlacementReady blocks until the placement's SERVING AXIS says DISPATCHABLE for
// this plan — a real activation completed, never merely "connected" and never merely
// "materialized". The two axes are why this can now be said precisely: a placement that is
// STAGED on disk and OFFLINE is not ready, and under the retired single enum it was
// indistinguishable from one that was still fetching.
//
// It waits on OBSERVATIONS and on nothing else. It used to carry a `timeout` — 30 minutes
// from `dispatch`, 10 from `cozy warm`, two different ceilings on the same cold start —
// and that number was a ceiling on how large a model may be, not a bound on anything that
// had gone wrong. Every way this can actually fail is already visible: the worker EXITS,
// reports a FAILED axis, or goes SILENT. A worker that is
// materializing is none of those, however long it takes.
func (c *Orchestrator) EnsurePlacementReady(instanceID, planID string) *exit.Error {
	silent := SilentReports * ReportCadence
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		ok := w != nil && !w.exited && w.dispatchableFor(planID)
		gone := w == nil || w.exited
		logPath, fault, code := "", "", 0
		workerFaulted := false
		quiet := time.Duration(0)
		unstaged, holds := false, ""
		var refused *exit.Error
		if w != nil {
			logPath, fault, code, refused = w.logPath, w.fault, w.exitCode, w.refusal
			workerFaulted = w.faulted
			if !w.lastReport.IsZero() {
				quiet = time.Since(w.lastReport)
			} else if !w.spawned.IsZero() {
				quiet = time.Since(w.spawned)
			}
			if !w.exited && !staged(w, planID) {
				unstaged, holds = true, strings.Join(w.planIDs, ", ")
			}
		}
		c.mu.Unlock()
		if ok {
			return nil
		}
		// THIS OWNER'S OWN VERDICT COMES FIRST. A worker whose claim was refused here is
		// not slow and not silent — this side has decided not to converse with it — and
		// waiting out eight missed report periods to say "it is stalled" would report a
		// network symptom for an identity fact this process already established.
		if refused != nil {
			return refused
		}
		// A PLAN THE LAUNCHER NEVER STAGED CAN NEVER BECOME DISPATCHABLE (cl-022's
		// corollary guard). The live case: submit, then an install --force moves the
		// active pin before the launch goroutine runs — selectOrStart replaces the stale
		// worker with one launched for the NEW pin, and this wait would watch it for the
		// OLD plan forever, holding `starting[slot]` and wedging the package slot until
		// restart. A structural mismatch is an answer, not a longer wait.
		if unstaged {
			return exit.Named(exit.Conflict, "plan_not_staged",
				"worker %s was launched holding [%s] and will never advertise %s: the "+
					"binding this request froze is not one its worker was staged with",
				instanceID, holds, planID).
				WithRemedy("the active pin moved after this request was accepted; " +
					"re-submit against the current install")
		}
		if gone {
			if code == RecycleExit {
				// NOT A DEATH. The worker finished its bounded attempt and recycled; the
				// request this wait was started for is either settled already or back on
				// the queue, and its exit has ALREADY asked the queue for a replacement.
				// Calling this a failure settled a requeued job as `failed` after ONE of
				// its three budgeted attempts (observed live).
				return exit.Named(exit.Unavailable, "worker_recycled",
					"the job worker recycled (exit %d) after its bounded attempt", RecycleExit)
			}
			return exit.New(exit.Failed, "the package worker exited before reporting ready").
				WithRemedy("its log is %s", logPath)
		}
		if workerFaulted {
			// FAILED/fault is the worker's settled typed answer, not a timer start.
			// Transient work remains MATERIALIZING/ACTIVATING and keeps reporting progress.
			return workerError(fault).WithRemedy("its log is %s", logPath)
		}
		if quiet > silent {
			// STALLED, and said as an observation rather than as an elapsed time: this
			// worker owes a Report every `ReportCadence` and has missed `SilentReports`
			// of them. A slow load is not this — a loading worker keeps reporting.
			return exit.Named(exit.Failed, "worker_silent",
				"the package worker has reported no observed state for %s, which is %d missed "+
					"periods of %s: it is stalled, not slow", quiet.Round(time.Second),
				SilentReports, ReportCadence).WithRemedy("its log is %s", logPath)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// RecycleExit is the run-once COMPLETION disposition (cr-009, v1 rc 75). A job worker
// exits with it after its terminal is acknowledged: the process ending is the successful
// end of the work, and it is deliberately distinct from any death.
const RecycleExit = 75

// workerError is the orchestrator's PROJECTION over a worker's fault reason. The reasons
// are the runtime's neutral vocabulary; deciding what a user should do about one is this
// side's job, exactly as it is for a terminal's (status, cause).
func workerError(fault string) *exit.Error {
	said := fault
	if said == "" {
		said = "no reason reported"
	}
	if strings.Contains(fault, "shortfall") || strings.Contains(fault, "capacity") {
		return exit.New(exit.Capacity,
			"the package worker cannot make its binding resident on this device: %s", said)
	}
	return exit.New(exit.Failed,
		"the package worker's placement reported a terminal fault and cannot become "+
			"dispatchable: %s", said)
}

func (c *Orchestrator) WorkerLog(instanceID string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if w := c.workers[instanceID]; w != nil {
		return w.logPath
	}
	return ""
}

// WorkerFacts is what status renders: the protocol's own identities and its own state
// vocabulary. There is no local synonym tuple.
//
// `intake_state` and `readiness_epoch` are GONE, not renamed (#482/#486b): IntakeState is
// retired with both of its uses, and one fence beats two to keep consistent. What replaces
// them is what the wire actually says — a machine phase, two placement axes, and the
// worker-level admission fence — and `applied_revision` splits into the two facts it was
// pretending to be (#473).
type WorkerFacts struct {
	InstanceID                 string   `json:"instance_id"`
	RentalID                   string   `json:"rental_id,omitempty"`
	Package                    string   `json:"package"`
	PackageReleaseID           string   `json:"package_release_id"`
	BootID                     string   `json:"worker_boot_id"`
	PlacementID                string   `json:"placement_id"`
	PlacementSpecDigest        string   `json:"placement_spec_digest"`
	RetainedFallbackSpecDigest string   `json:"retained_fallback_spec_digest"`
	PID                        int      `json:"pid"`
	Generation                 uint64   `json:"executor_generation"`
	Exited                     bool     `json:"exited"`
	Devices                    []string `json:"devices"`

	Phase           string `json:"worker_phase"`
	Materialization string `json:"materialization"`
	Serving         string `json:"serving"`
	// Dispatchable is the placement's own `dispatchable_plan_ids`; Materializable is the
	// DISJOINT set it could take but has not activated.
	Dispatchable   []string `json:"dispatchable_plan_ids"`
	Materializable []string `json:"materializable_plan_ids"`

	Admission           string `json:"admission_state"`
	AdmissionGeneration uint64 `json:"admission_generation"`
	AvailableSlots      int    `json:"available_attempt_slots"`
	// UnackedOutcomes is what this owner holds without having acked. The worker counts
	// them against its own free seats, so a rising number here is an owner starving its
	// own admission — which is the point of making boundedness structural (#480d).
	UnackedOutcomes int `json:"unacked_outcomes"`

	// ACCEPTANCE AND CONVERGENCE ARE TWO FACTS. converged < accepted is the normal,
	// readable state of a convergence in progress or a latched failure, never an error.
	DesiredRevision            uint64                    `json:"desired_state_revision"`
	AcceptedRevision           uint64                    `json:"accepted_desired_state_revision"`
	ConvergedRevision          uint64                    `json:"converged_revision"`
	AcceptedPlacementSetDigest string                    `json:"accepted_placement_set_digest"`
	ArtifactRevision           uint64                    `json:"artifact_revision"`
	DelegationID               string                    `json:"artifact_delegation_id"`
	AuthorizationExpiresAtUnix uint64                    `json:"artifact_authorization_expires_at_unix"`
	Acquisition                PlacementAcquisitionFacts `json:"acquisition"`

	// QuietMS makes missed protocol reports visible. ErrorForMS is diagnostic history
	// only: a typed FAILED axis is acted on immediately, never after a timer.
	QuietMS    int64  `json:"quiet_ms"`
	ErrorForMS int64  `json:"error_for_ms"`
	Fault      string `json:"fault"`
	// Refusal is THIS OWNER'S OWN VERDICT about the thing at the other end — a foreign
	// instance identity or an unpinned release. It
	// is deliberately NOT `Fault`: fault rows explain worker state, while FAILED axes decide
	// terminality. A refusal is this owner's settled verdict and waits on no error clock.
	Refusal string `json:"refusal"`
}

func (c *Orchestrator) Worker(instanceID string) *WorkerFacts {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[instanceID]
	if w == nil {
		return nil
	}
	f := factsOf(w)
	return &f
}

// Workers is every worker this daemon currently owns — the LOCAL extension module's
// listing (cl-006) and `cozy invoke list`'s source.
func (c *Orchestrator) Workers() []WorkerFacts {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]WorkerFacts, 0, len(c.workers))
	for _, w := range c.workers {
		out = append(out, factsOf(w))
	}
	return out
}

func factsOf(w *worker) WorkerFacts {
	f := WorkerFacts{
		InstanceID: w.instanceID, Package: w.spec.Placement.Package,
		PackageReleaseID: w.spec.Placement.PackageReleaseID, BootID: w.bootID,
		PlacementID: w.placementID, Generation: w.generation,
		Exited: w.exited, Devices: w.spec.Devices,

		Phase:           trimEnum(pb.WorkerPhase_name[int32(w.phase)], "WORKER_PHASE_"),
		Materialization: trimEnum(pb.MaterializationState_name[int32(w.materialization)], "MATERIALIZATION_STATE_"),
		Serving:         trimEnum(pb.ServingState_name[int32(w.serving)], "SERVING_STATE_"),
		Dispatchable:    keysOf(w.dispatchable),
		Materializable:  keysOf(w.materializable),

		Admission:           trimEnum(pb.AdmissionState_name[int32(w.admission)], "ADMISSION_STATE_"),
		AdmissionGeneration: w.admissionGen,
		AvailableSlots:      w.slots,
		UnackedOutcomes:     w.unacked,

		DesiredRevision:            w.revision,
		AcceptedRevision:           w.acceptedRevision,
		ConvergedRevision:          w.convergedRevision,
		ArtifactRevision:           w.artifactRevision,
		DelegationID:               w.delegationID,
		AuthorizationExpiresAtUnix: w.authorizationExpires,
		Acquisition:                w.acquisition,

		Fault: w.fault,
	}
	if w.spec.Connection != nil {
		f.RentalID = w.spec.Connection.RentalID
	}
	f.PlacementSpecDigest, _ = canonical.Spell(w.specDigest)
	f.RetainedFallbackSpecDigest, _ = canonical.Spell(w.fallbackPin)
	f.AcceptedPlacementSetDigest, _ = canonical.Spell(w.acceptedSetDigest)
	// THIS OWNER'S OWN VERDICT IS A FACT ABOUT THE WORKER, so it is reported as one — in
	// its own field. A claim this side refused is the answer a poller of
	// `/v1/local/workers` needs; without it the only observable was a readiness wait that
	// timed out, which reads as "slow" for something that has already been decided.
	if w.refusal != nil {
		f.Refusal = w.refusal.ErrName() + ": " + w.refusal.Message
	}
	if !w.lastReport.IsZero() {
		f.QuietMS = time.Since(w.lastReport).Milliseconds()
	}
	if !w.errorSince.IsZero() {
		f.ErrorForMS = time.Since(w.errorSince).Milliseconds()
	}
	if w.cmd != nil && w.cmd.Process != nil {
		f.PID = w.cmd.Process.Pid
	}
	return f
}

type AcquisitionLegFacts struct {
	StartedNS       uint64 `json:"started_monotonic_ns"`
	EndedNS         uint64 `json:"ended_monotonic_ns"`
	DownloadedBytes uint64 `json:"downloaded_bytes"`
	ReusedBytes     uint64 `json:"reused_bytes"`
}

type PlacementAcquisitionFacts struct {
	Package AcquisitionLegFacts `json:"package"`
	Model   AcquisitionLegFacts `json:"model"`
}

// trimEnum renders a protocol enum by its own name, minus the type prefix proto3's
// package-level value scoping forces onto it. The NUMBERS are normative; this is for a
// person reading `cozy invoke list`.
func trimEnum(name, prefix string) string {
	if name == "" {
		return "UNSPECIFIED"
	}
	return strings.TrimPrefix(name, prefix)
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k, ok := range m {
		if ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// StopGrace is how long a worker has between SIGTERM and SIGKILL. It is a CANCELLATION
// BUDGET, not a wait for work to finish: the supervisor's own cooperative-cancel budget
// plus the terminal transaction that follows it, which is what a worker still needs to do
// after it is told to stop. It is deliberately ONE number — this operation used to be
// spelled 20 s in the stall watchdog, 20 s for a stale worker, 20 s for one that failed to
// become ready, and 30 s on the API route, with nothing anywhere saying why the same
// SIGTERM deserved three budgets.
const StopGrace = 30 * time.Second

// retireWorker is the generation-fenced retirement used by the stall path. A recovered
// worker intentionally reuses instance and placement ids; only the exact object whose
// observations established the retirement ground may receive the empty desired set.
func (c *Orchestrator) retireWorker(w *worker) *exit.Error {
	c.mu.Lock()
	if c.workers[w.instanceID] != w || w.exited || w.stopping {
		c.mu.Unlock()
		return exit.New(exit.NotFound, "worker %s is no longer the observed generation", w.instanceID)
	}
	s := c.sessions[w.bootID]
	c.mu.Unlock()
	if s == nil {
		return exit.Unavailablef("worker %s holds no claimed control stream to retire", w.instanceID)
	}
	c.logf("retiring placement %s from %s: the desired set becomes empty and it drains",
		w.placementID, w.instanceID)
	return c.converge(s, w, nil)
}

// ShutdownWorker drains and stops the WHOLE worker process group — the only yield
// mechanism there is. An attempt is never killed to improve queue latency, and no suspend
// path exists anywhere in this package. The wait for the process to go is `alive(pid)`,
// the kernel's own answer, polled until `StopGrace` is spent.
func (c *Orchestrator) ShutdownWorker(instanceID string, grace time.Duration) {
	c.mu.Lock()
	w := c.workers[instanceID]
	c.mu.Unlock()
	c.shutdownWorker(w, grace)
}

// shutdownWorker stops exactly the worker object the caller observed. Instance ids are
// deterministic and reused for journal recovery, so an id-only teardown can otherwise close
// the replacement's row after the old process exits. The caller that wins stopping owns all
// four acts: close control, reap process, close the row, remove the in-memory worker.
func (c *Orchestrator) shutdownWorker(w *worker, grace time.Duration) bool {
	if w == nil {
		return false
	}
	c.mu.Lock()
	if c.workers[w.instanceID] != w {
		c.mu.Unlock()
		return false
	}
	if w.stopping {
		stopped := w.stopped
		c.mu.Unlock()
		<-stopped
		return false
	}
	w.stopping = true
	c.mu.Unlock()
	return c.stopClaimedWorker(w, grace)
}

// stopClaimedWorker completes teardown after the caller has atomically withdrawn the
// worker from selection by setting stopping. Keeping the claim separate lets `unload`
// stop only a worker it proved idle without reopening a dispatch race between proof and
// teardown.
func (c *Orchestrator) stopClaimedWorker(w *worker, grace time.Duration) bool {
	c.mu.Lock()
	cancelControl := w.cancelControl
	attachDone, processDone := w.attachDone, w.processDone
	pid, exited := 0, w.exited
	if w.cmd != nil && w.cmd.Process != nil {
		pid = w.cmd.Process.Pid
	}
	c.mu.Unlock()
	if cancelControl != nil {
		cancelControl()
	}
	if pid != 0 && !exited {
		// The cooperative tier: SIGTERM to the group, CTRL_BREAK to the job's console
		// group on Windows. A failure here is loud but not an escalation by itself —
		// the bounded wait below is what separates asking from insisting.
		if err := killGroup(pid, syscall.SIGTERM); err != nil {
			c.logf("worker %s: the cooperative stop could not be delivered (%s); "+
				"the forced tier follows the grace window", w.instanceID, err)
		}
		timer := time.NewTimer(grace)
		select {
		case <-processDone:
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
		case <-timer.C:
			_ = killGroup(pid, syscall.SIGKILL)
			<-processDone // process reaping is an observed fact, not another timeout
		}
	} else if processDone != nil {
		<-processDone
	}
	if attachDone != nil {
		<-attachDone
	}
	closeProblem := c.opt.Store.CloseWorker(w.instanceID)
	c.mu.Lock()
	if c.workers[w.instanceID] == w {
		w.exited = true
		delete(c.workers, w.instanceID)
		if w.bootID != "" {
			delete(c.sessions, w.bootID)
		}
	}
	close(w.stopped)
	c.mu.Unlock()
	if closeProblem != nil {
		c.logf("worker %s stopped but its durable row could not close: %s",
			w.instanceID, closeProblem.Message)
	} else {
		c.logf("worker %s stopped; its device grant is released", w.instanceID)
	}
	return true
}

// UnloadIdleLocalWorkers stops every definitely-idle local serving worker. Remote
// workers are paid resources owned by the rental lifecycle, job workers are run-once,
// and any active request, outstanding offer, reservation, or unacked terminal keeps a
// worker alive. Process exit is the reliable release of its GPU-resident model.
func (c *Orchestrator) UnloadIdleLocalWorkers() ([]WorkerFacts, *exit.Error) {
	active, e := c.opt.Store.ActiveRequests()
	if e != nil {
		return nil, e
	}

	c.mu.Lock()
	candidates := make([]*worker, 0, len(c.workers))
	for _, w := range c.workers {
		if w.spec.Connection == nil && !w.spec.IsJob() {
			candidates = append(candidates, w)
		}
	}
	c.mu.Unlock()
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].instanceID < candidates[j].instanceID
	})

	stopped := make([]WorkerFacts, 0, len(candidates))
	for _, w := range candidates {
		c.mu.Lock()
		if !c.idleLocalWorkerLocked(w, active) {
			c.mu.Unlock()
			continue
		}
		facts := factsOf(w)
		w.stopping = true
		c.mu.Unlock()
		if c.stopClaimedWorker(w, StopGrace) {
			stopped = append(stopped, facts)
		}
	}
	if len(stopped) > 0 {
		c.reviveQueue()
	}
	return stopped, nil
}

func (c *Orchestrator) idleLocalWorkerLocked(w *worker, active []records.Request) bool {
	if w == nil || c.workers[w.instanceID] != w || w.exited || w.stopping ||
		w.spec.Connection != nil || w.spec.IsJob() || w.reservedSlots != 0 || w.unacked != 0 {
		return false
	}
	for _, reservation := range c.offers {
		if reservation.worker == w {
			return false
		}
	}
	for _, req := range active {
		if req.Worker == "" && staged(w, req.PlanID) {
			return false
		}
	}
	return true
}

func (c *Orchestrator) idleLocalServingWorkerLocked(w *worker, active []records.Request) bool {
	return c.idleLocalWorkerLocked(w, active) &&
		w.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE &&
		w.admission == pb.AdmissionState_ADMISSION_STATE_OPEN
}

// Reconcile runs at boot, before anything is served. Rows describing processes from a
// previous life are checked against their OS BIRTH identity: a matching birth is a real
// orphan and is killed (it holds a device grant and a socket this daemon no longer
// knows); a mismatch is a REUSED PID and is never signalled — only its row is closed.
func (c *Orchestrator) Reconcile() (killed, forgotten int, e *exit.Error) {
	unlock := inputasset.Guard()
	e = inputasset.Sweep(c.opt.Layout, c.opt.Store)
	unlock()
	if e != nil {
		return 0, 0, e
	}
	rows, e := c.opt.Store.LiveWorkers()
	if e != nil {
		return 0, 0, e
	}
	for _, row := range rows {
		if row.WorkerID == "local" && row.State == "spawned_without_birth" {
			return 0, 0, exit.Named(exit.Conflict, "worker_birth_identity_unresolved",
				"local worker %s may have started before its OS birth identity was journaled",
				row.InstanceID).
				WithRemedy("do not start another worker on devices [%s]; locate and stop the orphan, then explicitly close its retained worker row",
					strings.Join(row.Devices, ","))
		}
	}
	for _, row := range rows {
		if row.PID > 0 && birthOf(row.PID) == row.Birth && row.Birth != "" {
			_ = killGroup(row.PID, syscall.SIGKILL)
			killed++
			c.logf("orphan worker %s (pid %d, birth %s) killed on reconcile",
				row.InstanceID, row.PID, row.Birth)
		} else {
			forgotten++
			c.logf("worker row %s forgotten: pid %d is gone or reused (birth %q != %q)",
				row.InstanceID, row.PID, birthOf(row.PID), row.Birth)
		}
		if e := c.opt.Store.CloseWorker(row.InstanceID); e != nil {
			return killed, forgotten, e
		}
	}
	// THE DISPATCH QUEUE IS MEMORY, AND THE AUTHORITY IS NOT. A request recorded as owed
	// work before the crash has no attempt and no queue entry after it — it simply stopped
	// existing as far as scheduling was concerned, while its row went on saying `queued`
	// forever. Found live by cl-004's crash arm: a job submitted moments before the
	// orchestrator was `kill -9`ed never ran again. Rebuilding the queue from the authority
	// is the only place the two can be made to agree, and it happens before anything is
	// served. Order is the authority's own (created_at), so FIFO survives a crash too.
	owed, e := c.opt.Store.Owed()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range owed {
		c.enqueue(req.ID)
		c.logf("%s was owed work before the restart; it is back on the dispatch queue", req.ID)
	}
	if len(owed) > 0 {
		go c.reviveQueue()
	}
	// AND THE ATTEMPTS THAT OWE A TERMINAL. Killing an orphan is only half of a restart:
	// the attempts it held are unsettled, and the ONE thing that can settle them is the
	// supervisor's own journal replayed by a worker in the SAME SLOT (02 §6.2). Nothing
	// else in this process will ask for that slot — the requests are not queued, they have
	// ordinals — so a job dispatched moments before a `kill -9` of the orchestrator hung
	// forever. Found live by cl-004's crash arm; cl-006's own crash section had been
	// POSTing /v1/local/workers by hand to work around it.
	unsettled, e := c.opt.Store.Unsettled()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range unsettled {
		c.logf("%s holds an attempt with no terminal; making its slot resident so the "+
			"supervisor journal replays", req.ID)
		c.selectOrStart(req)
	}
	ready, e := c.opt.Store.ReadyRequeues()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range ready {
		c.logf("%s committed and acknowledged a retry before restart; resuming its requeue", req.ID)
		c.Requeue(req.ID, "restart-after-terminal-ack")
	}
	return killed, forgotten, nil
}
