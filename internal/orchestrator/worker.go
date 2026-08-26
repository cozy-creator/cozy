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

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/media"
	"github.com/cozy-creator/cozy-creator-v2/internal/plan"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	pb "github.com/cozy-creator/cozy-creator-v2/protocol/cozy/worker/v1"
)

// Binding is cozy-creator's LOCAL PINNED-BINDING RECORD: the resolution of one
// `entrypoint_binding_plan_id` against this machine. The orchestrator names a plan by
// digest on the wire and stages the record the worker resolves that digest against.
//
// The record's key set is CLOSED at both ends — cozy-runtime refuses an unknown key —
// and th-004 owns the real EntrypointBindingPlan document. This is the named seam, not a
// pretend hub: when th-004 lands, the id keeps being the digest of a canonical document
// and only the document changes.
//
// `Record` is the WHOLE record; which half of it is IDENTITY and which half is this
// machine's RESOLUTION is declared once, in internal/plan (#506a), and both the id and
// the pod's re-derivation of it read that one declaration.
type Binding struct {
	Entrypoint string         `json:"entrypoint"`
	Record     map[string]any `json:"record"`
	planID     string
	// Outputs names this entrypoint's RESULT FIELD PATHS (`image`, `preview`,
	// `detail.thumb`). It is not part of the binding record's canonical bytes — it is
	// the LAUNCHER's knowledge of the entrypoint's declared result shape, which cl-006
	// needs so a client is not forced to restate it on every submission. cr-003's
	// descriptor is where it comes from once an installed generation exists.
	Outputs []string `json:"outputs,omitempty"`
}

// PlanID is the binding's identity: sha256 over the canonical bytes of the record's
// IDENTITY half, without its own id (a document never contains its own digest) and
// without the paths that say where this machine put things (#506a).
func (b *Binding) PlanID() (string, *exit.Error) {
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

// Staged is the record exactly as it lands on the worker's disk — identity, resolution
// and the id — and is what the remote delivery path ships to a pod's media server.
func (b *Binding) Staged() (string, []byte, *exit.Error) {
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

// THE THREE OBJECTS THAT USED TO BE ONE (#484). `EndpointSpec` conflated three
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
	Endpoint  string `json:"endpoint"`   // org/name — the slot this placement serves under
	ReleaseID string `json:"release_id"` // PROVENANCE: what the spec was resolved FROM
	// InstallID is the install this placement was resolved from ("" = an uninstalled dev
	// tree). Was `Generation`, which named a protocol word this side does not own (#484).
	InstallID string `json:"install_id"`
	// DescriptorDigest is the cr-003 descriptor's own identity, as the install VERIFIED
	// it in the generation's own venv. It rides the PlacementSpec because a placement is
	// named by the bytes it serves, and the descriptor is one of them.
	DescriptorDigest string     `json:"descriptor_digest"`
	Bindings         []*Binding `json:"bindings"`
	// Jobs is the JOB-mode declaration (cl-004). A worker is in exactly ONE mode — the
	// DesiredWorkerState's own oneof says which — so a placement carries bindings or jobs,
	// never both, and `IsJob` is read from it rather than re-derived from what happens to
	// be populated later.
	Jobs []*JobPlan `json:"jobs,omitempty"`
}

// WorkerConnection is the dial triple for a worker this service did not spawn (was
// RemoteSpec — it describes a CONNECTION, and "remote" was a claim about geography that a
// loopback pod falsifies). Token is a secret.Value rather than a string so that a spec
// which is logged, rendered or marshalled prints the credential's DIGEST — there is no
// formatting verb that leaks it, and its one raw read is the same Claim.proof carrier a
// spawned worker's bootstrap uses.
type WorkerConnection struct {
	Addr   string       `json:"addr"`
	Token  secret.Value `json:"token"`
	CACert string       `json:"ca_cert"` // path to the worker's pinned PEM
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

// WorkerLaunchSpec is everything this owner needs to make one worker exist and host one
// placement. The caller (cl-010's `start`, or cl-001's live driver) resolves it from the
// install; the orchestrator itself resolves nothing about Python.
type WorkerLaunchSpec struct {
	Python   string       `json:"python"` // the interpreter inside the endpoint's own venv
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
	// Connection attaches an ALREADY-RUNNING worker (a rented pod's TLS leg, cl-015/#445)
	// instead of spawning one: the owner dials Addr with the cert at CACert pinned and
	// presents Token as Claim.proof. Nil = the ordinary local spawn.
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

// pinnedEndpoint is the endpoint name a request PINNED to a rental resolves its slot
// under. One attached worker per rental id: a second request naming the same rental finds
// the worker already conversing instead of attaching a second control stream to it, and a
// request naming no rental never lands in a rented worker's slot.
func pinnedEndpoint(endpoint, rental string) string {
	if rental == "" {
		return endpoint
	}
	return endpoint + "@" + rental
}

// WorkerID is what the Claim names as the expected worker identity; empty skips the
// worker-side check (a spawned worker's identity is already ours by construction).
func (s WorkerLaunchSpec) WorkerID() string { return "" }

// IsJob answers the worker's mode.
func (p DesiredPlacement) IsJob() bool { return len(p.Jobs) > 0 }

// IsJob answers the worker's mode, from the one placement it hosts.
func (s WorkerLaunchSpec) IsJob() bool { return s.Placement.IsJob() }

// OutputsFor names one entrypoint's declared result field paths.
func (p DesiredPlacement) OutputsFor(entrypoint string) []string {
	for _, b := range p.Bindings {
		if b.Entrypoint == entrypoint {
			return b.Outputs
		}
	}
	return nil
}

// InstanceID is the endpoint's local worker SLOT identity, and it is deliberately STABLE
// across supervisor restarts: `worker_instance_id` names one provisioned instance
// lifetime, and outcome replay across a restart is authorized by that identity (02 §2). A
// slot keeps its journal root, which is what lets a restarted worker report its held
// attempts at all. A NEW install is a genuinely new instance and gets a new id.
func (p DesiredPlacement) InstanceID() string {
	slot := "slot/" + p.Endpoint + "/" + p.InstallID
	if p.IsJob() {
		// A JOB worker is its own slot: one worker per (endpoint, install, job function),
		// because a JobDirective names ONE job and a worker is in one mode. It is also what
		// lets a serving worker and a job worker of the same endpoint coexist instead of
		// fighting over one instance id.
		slot += "/job/" + p.Jobs[0].Function
	}
	sum := sha256.Sum256([]byte(slot))
	return "ins-" + hex.EncodeToString(sum[:12])
}

func (s WorkerLaunchSpec) InstanceID() string { return s.Placement.InstanceID() }

// PlacementID is the RecordOwner-minted routing + journal key for the one placement this
// slot hosts (#481; renamed from deployment_id — tensorhub's "deployment" is an id-less
// semantic tuple and the collision was real). Routing, NEVER identity: an InvocationSpec
// digest excludes it, so the same invocation is the same work wherever it routes.
func (p DesiredPlacement) PlacementID() string {
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
	// slots is the owner's working copy of available_attempt_slots — the worker's free
	// seats, corrected by every report and RESERVED at dispatch so one drain pass cannot
	// hand two attempts to a one-seat worker on one reading.
	slots int
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
	// jobsAvail is the worker's own last `jobs_available`. It is RESERVED at dispatch
	// and corrected by the next Report: without the reservation one drain pass would
	// hand two queued jobs to the same one-attempt worker on one stale reading, and
	// relying on the worker to refuse the second is not a design.
	jobsAvail int
	// revision is the desired-state revision THIS owner last issued, with the placement
	// set it issued. The set travels as bytes, so the owner keeps the bytes it authored:
	// a worker's accepted digest is compared against these, never re-canonicalized.
	revision  uint64
	setDigest []byte
	setBytes  []byte
	// The worker's own last reason for not serving, and when it first said so. A worker
	// whose placement latches a fault has not answered "not yet" — it has answered "I
	// cannot", and a client waiting on it needs that answer rather than a longer wait.
	fault      string
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
	spawned time.Time
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

// EnsureWorker makes one worker exist hosting one placement, and SAYS WHICH OF THE THREE
// THINGS IT DID (#484). It is idempotent by construction rather than by a caller's check:
// a slot already hosting this placement is `none`, a slot that exists without it converges
// and is `placement_added`, and only an absent slot is spawned or connected.
//
// The three answers are not decoration. `cozy start` and POST /v1/local/workers both need
// to tell an idempotent no-op from a real convergence, and a boolean `resident` could only
// tell them "it was there", which is true of both.
func (c *Orchestrator) EnsureWorker(spec WorkerLaunchSpec) (string, WorkerChange, *exit.Error) {
	instanceID := spec.InstanceID()
	c.mu.Lock()
	live := c.workers[instanceID]
	if live != nil && live.exited {
		live = nil
	}
	c.mu.Unlock()
	if live != nil {
		// The worker is here. Does it already host what is wanted? The placement's
		// identity for this purpose is its plan set, which is what the desired set names.
		if hostsPlans(live, spec.Placement) {
			return instanceID, ChangeNone, nil
		}
		if e := c.ConvergePlacementSet(instanceID, []DesiredPlacement{spec.Placement}); e != nil {
			return instanceID, ChangeNone, e
		}
		return instanceID, ChangePlacementAdded, nil
	}
	if spec.Connection != nil {
		id, e := c.connectWorker(spec)
		return id, ChangeWorkerStarted, e
	}
	id, e := c.spawnWorker(spec)
	return id, ChangeWorkerStarted, e
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

// spawnWorker journals the device grant, stages the binding records, and spawns the
// worker. The grant is journaled BEFORE the process exists: a process that was never
// granted an envelope cannot appear, and two concurrent starts cannot both consume one.
func (c *Orchestrator) spawnWorker(spec WorkerLaunchSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	// The slot's root is REUSED on purpose: the supervisor journal under it is what a
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
		id, data, e := b.Staged()
		if e != nil {
			return "", e
		}
		if _, e := plan.Stage(filepath.Join(workerHome, "binding-plans"), b.Record); e != nil {
			return "", e
		}
		planIDs = append(planIDs, id)
		subjects = append(subjects, subjectOf(id, data))
	}
	sortStrings(planIDs) // the wire field is sorted lexicographic ascending
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
		InstanceID: instanceID,
		Endpoint:   spec.Placement.Endpoint,
		Generation: spec.Placement.InstallID,
		ReleaseID:  spec.Placement.ReleaseID,
		WorkerID:   "local",
		Devices:    spec.Devices,
	}); e != nil {
		logFile.Close()
		return "", e
	}

	// `cozy-runtime serve`'s CLOSED flag set — the launch facts nothing can discover for a
	// orchestrator. It is the runtime's PUBLIC launch grammar, so this is the only spelling
	// the orchestrator knows: `--socket`/`--out` are the path grants the verb translates
	// into the supervisor's own `--hub`/`--root`.
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
		"--release-id", spec.Placement.ReleaseID,
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
	// Registered BEFORE the process can dial: a worker that registers faster than its
	// launcher can record it would be refused as an instance nobody spawned.
	c.mu.Lock()
	c.workers[instanceID] = w
	c.mu.Unlock()

	if err := cmd.Start(); err != nil {
		c.mu.Lock()
		delete(c.workers, instanceID)
		c.mu.Unlock()
		_ = c.opt.Store.CloseWorker(instanceID)
		logFile.Close()
		return "", exit.Internalf("cannot start the endpoint worker: %s", err)
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
		return "", exit.Internalf("cannot contain the endpoint worker: %s", err)
	}
	c.mu.Lock()
	w.pid = cmd.Process.Pid
	c.mu.Unlock()
	if e := c.opt.Store.WorkerStarted(instanceID, cmd.Process.Pid, birthOf(cmd.Process.Pid)); e != nil {
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
		err := cmd.Wait()
		logFile.Close()
		// The process is reaped: retire its containment handle so a reused pid can
		// never meet a stale job (Windows; a no-op where the group is a kernel fact).
		releaseGroup(cmd.Process.Pid)
		c.mu.Lock()
		current, live := c.workers[instanceID]
		mine := live && current == w
		if mine {
			w.exitCode = cmd.ProcessState.ExitCode()
			// The entry STAYS, marked exited: a caller waiting on readiness needs to
			// learn the worker is gone and where its log is, and a row that vanishes
			// silently is the same lie as a row that outlives its process.
			w.exited = true
			if w.bootID != "" {
				delete(c.sessions, w.bootID)
			}
		}
		c.mu.Unlock()
		if mine {
			_ = c.opt.Store.CloseWorker(instanceID)
			c.logf("worker %s exited (%v); its device grant is released — log %s",
				instanceID, err, logPath)
			// The queue's answer to "does capacity exist" just changed, and so has the
			// fate of anything this worker was RUNNING.
			go c.recoverWorker(w.spec)
		}
	}()
	return instanceID, nil
}

// newWorker is the one place a worker's observed state starts, and it starts UNKNOWN:
// every enum at its UNSPECIFIED zero, every map empty, zero seats. A worker this owner has
// not heard from is not dispatchable, and the gates read that off these values rather than
// off a separate "have we heard from it" flag that could disagree with them.
func newWorker(instanceID string, spec WorkerLaunchSpec) *worker {
	return &worker{
		instanceID:     instanceID,
		spec:           spec,
		placementID:    spec.Placement.PlacementID(),
		dispatchable:   map[string]bool{},
		materializable: map[string]bool{},
		spawned:        time.Now(),
	}
}

// connectWorker registers an ALREADY-RUNNING worker (a rented pod's TLS leg, cl-015):
// no spawn, no device grant (the pod's card is the pod's), no birth identity — the
// conversation is the same claim the local path runs, dialed at the rental's address
// with the pinned cert and the owner token as proof (#445).
// It also DELIVERS this placement's binding-plan records to the pod. That is the half that
// was missing: a desired set names plan ids, and the worker resolves each one against a
// record on ITS OWN disk (`<worker home>/binding-plans/<id>.json`). The local path stages
// those records by writing files; the connected path had no channel at all, so a real pod
// would have been directed to serve plans it had never been given (#506a/#506b). The
// channel is the pod's media server, and the pod re-derives each id from the record's own
// identity before it keeps the bytes — so delivery either agrees or refuses typed.
func (c *Orchestrator) connectWorker(spec WorkerLaunchSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	if spec.Connection.Media == nil {
		return "", exit.Named(exit.Unavailable, "rental_no_media_plane",
			"rental %s pins a control address and no media plane, and a pod that cannot be "+
				"handed bytes cannot be served", spec.Placement.Endpoint).
			WithRemedy("a rented pod runs its worker and a co-resident media server " +
				"(cl-014); this host will not fall back to granting paths on its own disk, " +
				"because the pod cannot reach them").
			WithNext("cozy rent ls")
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
	var planIDs []string
	var subjects []*pb.ArtifactSubject
	for _, b := range spec.Placement.Bindings {
		id, data, e := b.Staged()
		if e != nil {
			return "", e
		}
		path, e := byteplane.PutPlan(id, data)
		if e != nil {
			return "", e
		}
		c.logf("binding plan %s delivered to %s at %s (%d B)",
			shortDigest(id), byteplane.Addr(), path, len(data))
		planIDs = append(planIDs, id)
		subjects = append(subjects, subjectOf(id, data))
	}
	sortStrings(planIDs)
	sortSubjects(subjects)
	// AttachWorker, not SpawnWorker: the device-envelope admission arbitrates THIS host's
	// cards, and the pod's card is the pod's. There is no grant to journal and none to
	// release, which is also why nothing here has a pid or a birth identity to record.
	if e := c.opt.Store.AttachWorker(records.WorkerProcess{
		InstanceID: instanceID,
		Endpoint:   spec.Placement.Endpoint,
		Generation: spec.Placement.InstallID,
		ReleaseID:  spec.Placement.ReleaseID,
		WorkerID:   "remote",
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

// ErrorGrace is how long a placement may hold a latched fault before the wait gives up on
// it. A shortfall on a shared card is often transient — a neighbouring process gives the
// device back and the next activation succeeds — so a fault is not instantly fatal. What
// is fatal is staying there: cl-003 watched a worker whose fill was refused
// `device_shortfall` report every two seconds for eleven minutes while a `cozy run`
// waited out the full thirty-minute readiness window with nothing on its event stream.
const ErrorGrace = 90 * time.Second

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
// its placement holds a fault past `ErrorGrace`, or it goes SILENT. A worker that is
// materializing is none of those, however long it takes.
func (c *Orchestrator) EnsurePlacementReady(instanceID, planID string) *exit.Error {
	silent := SilentReports * ReportCadence
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		ok := w != nil && !w.exited && w.dispatchableFor(planID)
		gone := w == nil || w.exited
		logPath, fault, stuck, code := "", "", time.Duration(0), 0
		quiet := time.Duration(0)
		var refused *exit.Error
		if w != nil {
			logPath, fault, code, refused = w.logPath, w.fault, w.exitCode, w.refusal
			if !w.errorSince.IsZero() {
				stuck = time.Since(w.errorSince)
			}
			if !w.lastReport.IsZero() {
				quiet = time.Since(w.lastReport)
			} else if !w.spawned.IsZero() {
				quiet = time.Since(w.spawned)
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
			return exit.New(exit.Failed, "the endpoint worker exited before reporting ready").
				WithRemedy("its log is %s", logPath)
		}
		if stuck > ErrorGrace {
			// The worker's OWN words, under the code its reason projects to. Waiting
			// longer on a worker that has been saying "I cannot" for a minute and a half
			// is not patience, it is a client with no answer.
			return workerError(fault, stuck).WithRemedy("its log is %s", logPath)
		}
		if quiet > silent {
			// STALLED, and said as an observation rather than as an elapsed time: this
			// worker owes a Report every `ReportCadence` and has missed `SilentReports`
			// of them. A slow load is not this — a loading worker keeps reporting.
			return exit.Named(exit.Failed, "worker_silent",
				"the endpoint worker has reported no observed state for %s, which is %d missed "+
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
func workerError(fault string, stuck time.Duration) *exit.Error {
	said := fault
	if said == "" {
		said = "no reason reported"
	}
	if strings.Contains(fault, "shortfall") || strings.Contains(fault, "capacity") {
		return exit.New(exit.Capacity,
			"the endpoint worker cannot make its binding resident on this device (%s, for %s)",
			said, stuck.Round(time.Second))
	}
	return exit.New(exit.Failed,
		"the endpoint worker's placement has held a fault for %s and never became "+
			"dispatchable: %s", stuck.Round(time.Second), said)
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
	InstanceID  string   `json:"instance_id"`
	Endpoint    string   `json:"endpoint"`
	ReleaseID   string   `json:"release_id"`
	BootID      string   `json:"worker_boot_id"`
	PlacementID string   `json:"placement_id"`
	PID         int      `json:"pid"`
	Generation  uint64   `json:"executor_generation"`
	Exited      bool     `json:"exited"`
	Devices     []string `json:"devices"`

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
	DesiredRevision   uint64 `json:"desired_state_revision"`
	AcceptedRevision  uint64 `json:"accepted_desired_state_revision"`
	ConvergedRevision uint64 `json:"converged_revision"`

	// The two OBSERVATIONS a waiter needs and could not see. `cozy warm` polls this
	// listing and used to give up on a 10-minute clock, because the facts that decide
	// — how long the worker has been silent, and how long it has been saying it cannot
	// serve — lived only inside the orchestrator. A slow load is neither of them.
	QuietMS    int64  `json:"quiet_ms"`
	ErrorForMS int64  `json:"error_for_ms"`
	Fault      string `json:"fault"`
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

// Workers is every worker this service currently owns — the LOCAL extension module's
// listing (cl-006) and `cozy status`'s source.
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
		InstanceID: w.instanceID, Endpoint: w.spec.Placement.Endpoint,
		ReleaseID: w.spec.Placement.ReleaseID, BootID: w.bootID,
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

		DesiredRevision:   w.revision,
		AcceptedRevision:  w.acceptedRevision,
		ConvergedRevision: w.convergedRevision,

		Fault: w.fault,
	}
	// THIS OWNER'S OWN VERDICT IS A FACT ABOUT THE WORKER, so it is reported as one. A
	// claim this side refused — a foreign instance, an unpinned release, a schema this
	// build does not speak — is the answer a poller of `/v1/local/workers` needs; without
	// it the only observable was a readiness wait that timed out, which reads as "slow"
	// for something that has already been decided.
	if w.refusal != nil {
		f.Fault = w.refusal.ErrName() + ": " + w.refusal.Message
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

// trimEnum renders a protocol enum by its own name, minus the type prefix proto3's
// package-level value scoping forces onto it. The NUMBERS are normative; this is for a
// person reading `cozy status`.
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
	sortStrings(out)
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

// RetirePlacement converges a live worker onto the desired set WITHOUT this placement —
// the honest first half of taking one out of service, and a verb of its own because it is
// not the same act as ending a process (#484). The worker drains what it holds; nothing
// here waits, because a drain's completion is an OBSERVED fact (converged_revision) and
// not something a caller can be told synchronously.
//
// Retiring the only placement leaves an empty set, which is a worker hosting nothing —
// legal, and what a slot being replaced should look like before its process is stopped.
func (c *Orchestrator) RetirePlacement(instanceID, placementID string) *exit.Error {
	c.mu.Lock()
	w := c.workers[instanceID]
	c.mu.Unlock()
	if w == nil {
		return exit.New(exit.NotFound, "no worker %s on this host", instanceID)
	}
	if w.placementID != placementID {
		return exit.New(exit.NotFound, "worker %s hosts placement %s, not %s",
			instanceID, w.placementID, placementID)
	}
	c.logf("retiring placement %s from %s: the desired set becomes empty and it drains",
		placementID, instanceID)
	return c.ConvergePlacementSet(instanceID, nil)
}

// ShutdownWorker drains and stops the WHOLE worker process group — the only yield
// mechanism there is. An attempt is never killed to improve queue latency, and no suspend
// path exists anywhere in this package. The wait for the process to go is `alive(pid)`,
// the kernel's own answer, polled until `StopGrace` is spent.
func (c *Orchestrator) ShutdownWorker(instanceID string, grace time.Duration) {
	c.mu.Lock()
	w := c.workers[instanceID]
	c.mu.Unlock()
	if w == nil {
		return
	}
	if w.cmd != nil && w.cmd.Process != nil && !w.exited {
		// The cooperative tier: SIGTERM to the group, CTRL_BREAK to the job's console
		// group on Windows. A failure here is loud but not an escalation by itself —
		// the bounded wait below is what separates asking from insisting.
		if err := killGroup(w.cmd.Process.Pid, syscall.SIGTERM); err != nil {
			c.logf("worker %s: the cooperative stop could not be delivered (%s); "+
				"the forced tier follows the grace window", instanceID, err)
		}
		deadline := time.Now().Add(grace)
		for time.Now().Before(deadline) {
			if !alive(w.cmd.Process.Pid) {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if alive(w.cmd.Process.Pid) {
			_ = killGroup(w.cmd.Process.Pid, syscall.SIGKILL)
		}
	}
	_ = c.opt.Store.CloseWorker(instanceID)
	c.mu.Lock()
	w.exited = true
	delete(c.workers, instanceID)
	if w.bootID != "" {
		delete(c.sessions, w.bootID)
	}
	c.mu.Unlock()
	c.logf("worker %s stopped; its device grant is released", instanceID)
	go c.reviveQueue()
}

// Reconcile runs at boot, before anything is served. Rows describing processes from a
// previous life are checked against their OS BIRTH identity: a matching birth is a real
// orphan and is killed (it holds a device grant and a socket this service no longer
// knows); a mismatch is a REUSED PID and is never signalled — only its row is closed.
func (c *Orchestrator) Reconcile() (killed, forgotten int, e *exit.Error) {
	rows, e := c.opt.Store.LiveWorkers()
	if e != nil {
		return 0, 0, e
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
	return killed, forgotten, nil
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
