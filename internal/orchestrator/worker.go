package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/processtree"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/units"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Entrypoint is the small dispatch projection of one entrypoint already bound inside
// the exact PlacementSet. Digest is the wire identity; Outputs comes from the exact
// PackageInterface and is presentation metadata, never a second binding document.
type Entrypoint struct {
	Name    string   `json:"name"`
	Digest  string   `json:"digest"`
	Outputs []string `json:"outputs,omitempty"`
}

// THE THREE OBJECTS THAT USED TO BE ONE (#484). `PackageSpec` conflated three
// independent facts — how a worker PROCESS comes to exist, WHAT that worker is asked to
// host, and HOW this owner reaches a worker it did not spawn — and the conflation is
// exactly what rev-2 makes unstateable: a placement set is a full-replace document a
// RecordOwner converges a LIVE worker onto, so "the placement" cannot be a field of "the
// launch". Splitting them is what lets `ConvergePlacementSet` exist at all.

// DesiredPlacement is ONE assignment this owner wants a worker to host — the local half of
// rev-2's Placement nested directly in PlacementSet (#481). It carries the identity
// facts, and nothing about how a process is started.
type DesiredPlacement struct {
	Package             string `json:"package"` // org/name — the slot this placement serves under
	Release             string `json:"release"`
	SourceDigest        string `json:"source_digest,omitempty"` // local-only DevelopmentPackage identity
	LocalRevisionDigest string `json:"local_revision_digest,omitempty"`
	// InstallID is the install this placement was resolved from ("" = an uninstalled dev
	// tree). Was `Generation`, which named a protocol word this side does not own (#484).
	InstallID string `json:"install_id"`
	// PlacementSetBytes are the exact Hub-selected PlacementSet/1 bytes. Creator
	// validates and relays them unchanged for both a local venv and a rented pod.
	// Entrypoints is a read-only projection used for request selection; it never
	// participates in identity.
	PlacementSetDigest string       `json:"placement_set_digest"`
	PlacementSetBytes  []byte       `json:"placement_set_bytes"`
	EnvironmentDigest  string       `json:"environment_digest"`
	Entrypoints        []Entrypoint `json:"entrypoints"`
	PlacementIDValue   string       `json:"placement_id,omitempty"`
	// Models is the exact selection this placement was resolved with (empty = the package's
	// own defaults). A projection like Entrypoints, never identity: the editable refresh
	// re-prepares a worker under the same selection a run gave it.
	Models []ModelRef `json:"models,omitempty"`
	// Hidden names the entrypoints this placement deliberately does NOT serve (#572d).
	// Recorded so an operator reading a placement can tell "no binding was staged" from
	// "a binding was staged and broke".
	// Jobs is the JOB-mode declaration (cl-004). A worker is in exactly ONE mode — the
	// DesiredWorkerState's own oneof says which — so a placement carries bindings or jobs,
	// never both, and `IsJob` is read from it rather than re-derived from what happens to
	// be populated later.
	Jobs []*JobPlan `json:"jobs,omitempty"`
}

// PlacementFromExact validates the one selected PlacementSet and builds only the
// small request-routing projection Creator needs. The exact bytes remain the sole
// desired-state authority.
func PlacementFromExact(pkg, installID, digest string, data []byte,
	outputs map[string][]string) (DesiredPlacement, *exit.Error) {
	declared, err := canonical.Raw(digest)
	if err != nil || !bytes.Equal(canonical.Digest(data), declared) {
		return DesiredPlacement{}, exit.Named(exit.Conflict, "placement_set_identity_mismatch",
			"PlacementSet bytes do not match %s", digest)
	}
	doc, err := canonical.Read(data, &pb.PlacementSet{})
	if err != nil {
		return DesiredPlacement{}, exit.Named(exit.Conflict, "placement_set_invalid",
			"PlacementSet is not its closed canonical /1 document: %s", err)
	}
	rows := doc.List("placements")
	if len(rows) != 1 {
		return DesiredPlacement{}, exit.Named(exit.Structural, "placement_set_cardinality",
			"Creator supports exactly one placement, got %d", len(rows))
	}
	row := rows[0]
	packageFact, development := row.Sub("package"), row.Sub("development")
	environment := row.Sub("environment")
	placement := DesiredPlacement{
		Package: pkg, InstallID: installID, PlacementIDValue: row.Str("placement_id"),
		Release:            packageFact.Str("release"),
		EnvironmentDigest:  row.Str("environment_digest"),
		PlacementSetDigest: digest, PlacementSetBytes: append([]byte(nil), data...),
	}
	if development.Str("source_digest") != "" {
		placement.Release = development.Str("release")
		placement.SourceDigest = development.Str("source_digest")
		placement.LocalRevisionDigest = development.Str("local_revision_digest")
		if development.Str("package") != pkg || placement.Release == "" ||
			placement.PlacementIDValue == "" || len(packageFact) != 0 || len(environment.Sub("locked_requirements")) != 0 {
			return DesiredPlacement{}, exit.Named(exit.Structural, "development_placement_incomplete",
				"development PlacementSet mixes local source with published selection facts")
		}
		project := development.Sub("project_wheel")
		if placement.LocalRevisionDigest == "" {
			if len(project) != 0 || placement.EnvironmentDigest != "" {
				return DesiredPlacement{}, exit.Named(exit.Structural, "development_placement_incomplete",
					"source-only development placement cannot carry an incomplete captured environment")
			}
		} else if project.Sub("ref").Str("digest") == "" || placement.EnvironmentDigest == "" {
			return DesiredPlacement{}, exit.Named(exit.Structural, "development_placement_incomplete",
				"captured development placement requires its project wheel and environment identity")
		}
	} else if packageFact.Str("package") != pkg || placement.Release == "" ||
		placement.EnvironmentDigest == "" ||
		placement.PlacementIDValue == "" {
		return DesiredPlacement{}, exit.Named(exit.Structural, "placement_set_incomplete",
			"PlacementSet omits or mismatches its placement, package selection, or environment identity")
	}
	if placement.EnvironmentDigest != "" {
		// Wire 30: a published Environment is its locked-requirements ref; supplied
		// wheels belong to editable revisions only.
		environmentIdentity := map[string]canonical.Value{
			"format":       "cozy.worker.v1.Environment/1",
			"local_wheels": arrayOrEmpty(environment["local_wheels"]),
		}
		if locked, ok := environment["locked_requirements"]; ok {
			environmentIdentity["locked_requirements"] = locked
		}
		if !digestMatches(environmentIdentity, placement.EnvironmentDigest) {
			return DesiredPlacement{}, exit.Named(exit.Conflict, "environment_identity_mismatch",
				"environment_digest does not hash the exact nested Environment")
		}
	}
	for _, entrypoint := range row.List("entrypoints") {
		name, binding := entrypoint.Str("name"), entrypoint.Str("entrypoint_binding_digest")
		identity := map[string]canonical.Value{
			"name": entrypoint["name"], "slots": arrayOrEmpty(entrypoint["slots"]),
		}
		if name == "" || binding == "" || !digestMatches(identity, binding) {
			return DesiredPlacement{}, exit.Named(exit.Conflict, "entrypoint_binding_identity_mismatch",
				"entrypoint %q has an invalid binding digest", name)
		}
		placement.Entrypoints = append(placement.Entrypoints, Entrypoint{
			Name: name, Digest: binding, Outputs: append([]string(nil), outputs[name]...),
		})
	}
	bindings := map[string]canonical.Value{
		"entrypoints": arrayOrEmpty(row["entrypoints"]), "models": arrayOrEmpty(row["models"]),
	}
	if !digestMatches(bindings, row.Str("bindings_digest")) {
		return DesiredPlacement{}, exit.Named(exit.Conflict, "bindings_identity_mismatch",
			"bindings_digest does not hash the exact selected entrypoints and models")
	}
	// Every placement carries the exact model selection its set binds — a development
	// placement's frozen at install from the local store (cozy-runtime
	// `_resolve_development_models`, cl-101), a published placement's selected by
	// install or per-request selection (cl-114). The projection is what the matching
	// layer holds a request's own selection against: the plan id hashes the entrypoint's
	// interface, not its weights, so without it two selections of one package are
	// indistinguishable warm capacity.
	placement.Models = placementModels(pkg, row)
	return placement, nil
}

// placementModels reads the exact model rows a PlacementSet binds and names each by the
// descriptor slot path its entrypoint binds it to — the spelling every download
// download set, private placement, and request selection is addressed by.
func placementModels(pkg string, row canonical.Doc) []ModelRef {
	byID := map[string]canonical.Doc{}
	for _, model := range row.List("models") {
		byID[model.Str("id")] = model
	}
	var out []ModelRef
	seen := map[string]bool{}
	for _, entrypoint := range row.List("entrypoints") {
		for _, slot := range entrypoint.List("slots") {
			model, ok := byID[slot.Str("reference_model_id")]
			path := entrypoint.Str("name") + ".models." + slot.Str("slot")
			if !ok || seen[path] {
				continue
			}
			seen[path] = true
			manifest := model.Sub("manifest")
			out = append(out, ModelRef{Package: pkg, Slot: path,
				Model: model.Str("repo"), Release: model.Str("version"), Lane: model.Str("lane"),
				Manifest: manifest.Str("digest"), ManifestLength: manifest.Int("length")})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Slot < out[j].Slot })
	return out
}

func arrayOrEmpty(value canonical.Value) canonical.Value {
	if value == nil {
		return []canonical.Value{}
	}
	return value
}

func digestMatches(value canonical.Value, spelled string) bool {
	data, err := canonical.Write(value)
	if err != nil {
		return false
	}
	want, err := canonical.Raw(spelled)
	return err == nil && bytes.Equal(canonical.Digest(data), want)
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

// RemoteTarget is the dial identity for generic rented capacity, and the WIDTH that
// identity was paid for. Desired package/model state is a later Creator-to-worker command.
type RemoteTarget struct {
	Connection *WorkerConnection
	// Devices is the pod's device envelope: RentalDeviceEnvelope over the rental's paid
	// accelerator count, empty for a CPU pod. It travels with the dial identity because a
	// rental's width is a property of the machine, settled when it was bought, and every
	// later reader — the lane breach check, the decision log's device names, the device
	// pin this owner authors — is reading one fact and must read the same one.
	Devices []string
}

// WorkerLaunchSpec is everything this owner needs to make one worker exist and host one
// placement. The caller (cl-010's `start`, or cl-001's live driver) resolves it from the
// install; the orchestrator itself resolves nothing about Python.
type WorkerLaunchSpec struct {
	Python  string   `json:"python"` // package-independent control Runtime
	Args    []string `json:"args"`
	Dir     string   `json:"dir"`
	Imposed []string `json:"imposed"` // exact env values the launcher imposes, never inherited
	// Devices is the device envelope this worker holds: for a spawned worker the names on
	// its own `--devices` line, which this daemon GRANTS and arbitrates; for an attached
	// rental the pod's paid width, which this daemon only reads. Lane ordinals are
	// positions in it, and its length is the placement's device-group degree.
	Devices           []string `json:"devices"`
	GraceSec          float64  `json:"grace_sec"`
	ArtifactCache     string   `json:"artifact_cache,omitempty"`
	InstallRoot       string   `json:"install_root,omitempty"`
	EnvironmentPython string   `json:"environment_python,omitempty"` // preinstalled package venv
	TensorFSRoot      string   `json:"tensorfs_root,omitempty"`
	// Placement is what this worker is launched to host. LAUNCH CLAMPS THE SET TO ONE
	// (worker-protocol header): a longer set is a typed refusal at the worker, so this
	// side names one placement rather than pretending to a generality it cannot deliver.
	Placement DesiredPlacement `json:"placement"`
	// Connection attaches an ALREADY-RUNNING worker instead of spawning one: the owner
	// pins CACert and signs Claim with its per-rental key. Nil = the ordinary local spawn.
	Connection  *WorkerConnection        `json:"connection,omitempty"`
	Preparation *LocalServingPreparation `json:"preparation,omitempty"`
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

func (p DesiredPlacement) EntrypointDigest(name string) (string, *exit.Error) {
	for _, entrypoint := range p.Entrypoints {
		if entrypoint.Name == name {
			return entrypoint.Digest, nil
		}
	}
	return "", exit.Named(exit.NotFound, "entrypoint_not_selected",
		"the selected PlacementSet has no entrypoint %q", name)
}

// IsJob answers the worker's mode, from the one placement it hosts.
func (s WorkerLaunchSpec) IsJob() bool { return s.Placement.IsJob() }

// InstanceID is the package's stable local worker SLOT identity. Creator's records store
// owns attempts across child restarts; Runtime itself retains no recovery state. A NEW
// install is a genuinely new instance and gets a new id.
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
		return rentalInstanceID(s.Connection.RentalID)
	}
	return s.Placement.InstanceID()
}

func rentalInstanceID(id string) string {
	sum := sha256.Sum256([]byte("rental/" + id))
	return "ins-" + hex.EncodeToString(sum[:12])
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

// supportsCurrentProtocol is shared by placement reuse and dispatch. A peer that
// lacks a compatible protocol range cannot be claimed or take new directives.
func (w *worker) supportsCurrentProtocol() bool {
	if w.refusal != nil && w.refusal.ErrName() == "worker.protocol_incompatible" {
		return false
	}
	return w.declaredInstance == "" || w.wireMinor >= pb.MinCompatibleWireMinor
}

type worker struct {
	instanceID string
	spec       WorkerLaunchSpec
	cmd        *exec.Cmd
	logPath    string
	home       string
	planIDs    []string
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

	// declaredInstance is the worker's self-minted process-lifetime identity. Creator
	// names the stable slot; every local or attached worker names its own incarnation and
	// must keep that identity stable across a control-stream reconnect.
	declaredInstance string
	wireMinor        uint32 // negotiated in ClaimAck, not inferred from the image profile
	remoteWorkerID   string
	// desiredPackages/models are Creator's logical private-rental intent. They survive a
	// control-stream reconnect so the new authenticated stream does not reset a loaded
	// worker to the empty package_set. They are refs only, never download locations or a
	// locally reconstructed placement.
	desiredPackages []*pb.DownloadPackageRef
	desiredModels   []*pb.DownloadModelRef
	// desiredDownloadSets hold, PER PACKAGE, the exact canonical download-set bytes the
	// current package_set is prepared under. Prepare is a per-package operation (the
	// Runtime's package_prepare takes exactly one package), and the Runtime seeds the
	// package's placement_id from these bytes. They are a pure function of content, so
	// an unchanged selection re-authors identical bytes and adding package B cannot
	// retire package A's serving placement. A package-interface-falsifying refusal is
	// relayed to the hub (cl-078/th-106); defectReported latches per revision so one
	// falsification files one report.
	desiredDownloadSets    map[string][]byte
	defectReportedRevision uint64
	desiredMu              sync.Mutex
	// A rental is one machine and may host several package environments. Keep the
	// worker-reported placement for every binding instead of overwriting package A when
	// package B joins the same desired set.
	remotePlacements map[string]DesiredPlacement
	observedRemote   map[string]remotePlacementObservation
	// desiredLocal is the exact command-scoped local wheel inventory. It survives
	// control reconnect so a prepared pod can replay its ledgered PlacementSet directly.
	desiredLocal *pb.DesiredLocalPackageSet
	// desiredPrivatePlacement is the model-only join for the already-prepared private
	// revision. It survives a control reconnect so pod-supervisor can replay its exact journal.
	desiredPrivatePlacement *pb.DesiredPrivatePlacementSet
	// desiredEpoch is the control-stream epoch the local desire above was issued on. A
	// desire issued on the live session and not refused is in flight or done; the same one
	// asked again waits on the pod's report rather than asking the pod to prepare twice.
	desiredEpoch uint64
	// localMu serializes ConvergeLocalPackage on this worker. It is never held by the
	// control stream's receive loop, whose reports the holder waits on.
	localMu sync.Mutex
	// hostPrepareSeq numbers the logical desires issued through PodHost (proto-025); a
	// prepare that completes for an older number sends nothing.
	hostPrepareSeq uint64

	// what the worker itself reported; the orchestrator echoes, never invents
	exited bool
	// refusal is a claim-time verdict this owner reached about the thing at the other end —
	// a pod serving a different release than the one this host pinned, say. It is kept so a
	// waiter gets the ANSWER instead of waiting out the silence window for a worker this
	// owner has already decided not to talk to (#505's carried-not-verified gap).
	refusal *exit.Error
	// desiredRefusal is the worker's permanent verdict on the exact desired revision this
	// owner issued. It is distinct from a claim refusal and from capacity: a config or
	// placement-set refusal can never become dispatchable by waiting longer.
	desiredRefusal *exit.Error
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
	phase             pb.WorkerPhase          // machine lifecycle, out of the placement enum
	materialization   pb.MaterializationState // axis 1: what is on disk
	serving           pb.ServingState         // axis 2: what it will take
	executorEpoch     uint64                  // THIS placement's executor epoch
	dispatchable      map[string]bool         // dispatchable_plan_ids
	jobReady          map[string]bool         // prepared job executors, including occupied slots
	materializable    map[string]bool         // DISJOINT from dispatchable
	heldSetDigest     []byte                  // parent set the placement actually holds
	fallbackSetDigest []byte                  // predecessor set kept for restore; empty = replacement PAUSED

	// THE ADMISSION FENCE (#472e/#482/#486c), ONE COUNTER PER SERIALIZED RESOURCE. Per-
	// placement credits are DELETED: N counters over ONE serialized device advertise N x
	// the real capacity. With proto-024 the serialized resource is a device LANE and a
	// worker may have several, so `seats` is the worker-level window (the reported sum)
	// and `lanes` holds one window per reported lane; an offer for a placement draws from
	// its lane's and the worker's. Dispatchability stays a PLACEMENT property.
	admission      pb.AdmissionState
	admissionEpoch uint64 // echoed on every offer; a stale echo refuses deterministically
	seats          seatLedger
	lanes          laneTable
	// held is how many attempts the worker last reported holding, in every state from
	// admission to ack; the lanes carry their own share (route.go reads both).
	held int
	// heldManifests is the worker's last word on which TensorFS manifests its verified
	// store holds complete (`held_manifests`, proto-026), placed or not — the disk-tier
	// fact the capacity decision prefers a rental by (residency-aware-routing.md §3.2).
	heldManifests map[string]bool
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
	acceptedRevision     uint64
	convergedRevision    uint64
	acceptedSetDigest    []byte
	snapshotAcknowledged bool
	environmentDigest    string
	// The job lane uses the same reported-versus-pre-offer split as serving. jobsAvail is
	// the effective number dispatch reads.
	reportedJobs int
	reservedJobs int
	jobsAvail    int
	// revision is the desired-state revision THIS owner last issued, with the placement
	// set it issued. The set travels as bytes, so the owner keeps the bytes it authored:
	// a worker's accepted digest is compared against these, never re-canonicalized.
	revision    uint64
	acquisition PlacementAcquisitionFacts
	setDigest   []byte
	setBytes    []byte
	// The worker's last diagnostic fault and the first report that carried it. The FAILED
	// axes, not this text, decide terminality: BINDING_DEGRADED may coexist with service.
	fault      string
	faulted    bool
	errorSince time.Time
	// The fault the worker keeps against the desired revision it accepted, and how many
	// consecutive reports have carried it unchanged (observeLatchedFault).
	latchedFault         string
	latchedFaultRevision uint64
	latchedFaultReports  int
	// refused is, per attempt this worker holds, the outcome this owner could not honour
	// and how many consecutive worker reports have restated it unchanged
	// (observeRefusedOutcome).
	refused map[string]*refusedOutcome
	// lastReport is telemetry only. Elapsed time since it never settles or retires work.
	lastReport time.Time
	// activitySeq is the last activity entry logged from this worker's reports; a report
	// whose newest entry is below it is a restarted worker whose lane starts over.
	activitySeq uint64
	// loggedFaults is every fault line this owner has logged from the worker's reports
	// and not yet seen absent, so a fault repeating on the report cadence logs once.
	loggedFaults map[string]bool
	// LRU is dispatch-based, not report-based: reports say the worker lives, while an
	// accepted attempt says a user actually used it. Never-used workers fall back to
	// residentRevision so two cold holders still have a deterministic oldest member.
	residentRevision uint64
	lastUseRevision  uint64
}

type remotePlacementObservation struct {
	placementID         string
	placementSetDigest  string
	environmentDigest   string
	materialization     pb.MaterializationState
	serving             pb.ServingState
	dispatchablePlanIDs map[string]bool
	knownPlanIDs        map[string]bool
}

// dispatchableFor is the ROUTING GATE, and it is two questions with two owners (#482).
// DISPATCHABILITY is a PLACEMENT property: the serving axis says DISPATCHABLE and the
// placement advertises this plan. ADMISSION is a WORKER property: the fence is OPEN and a
// seat is free. A placement that is STAGED but OFFLINE is not capacity, however much of it
// is on disk.
func (w *worker) dispatchableFor(planID string) bool {
	if w.spec.IsJob() {
		// Job capacity is the readiness axis. A serving convergence or serving
		// status does not exist, but credits must describe the current directive.
		return w.acceptedRevision >= w.revision && w.dispatchable[planID]
	}
	if w.spec.Connection != nil {
		for key, placement := range w.remotePlacements {
			if strings.HasSuffix(key, "\x00"+planID) && w.remoteDispatchable(placement, planID) {
				return true
			}
		}
		return false
	}
	// THE ISSUED REVISION GATES DISPATCH (cl-114 follow-up). Between this owner issuing a
	// new desired set and the worker converging to it, every observed fact — serving
	// DISPATCHABLE, the plan advertised — describes the placement being REPLACED. Run 168
	// proved the window live: the fp8 re-stage was accepted, the runtime kept the bf16
	// placement serving through the handoff, and an offer dispatched on the pre-converge
	// observation executed on the vacating selection. The remote ready path already
	// requires accepted and converged to reach the issued revision; the local one now
	// does too, so an offer is only made against the placement the owner last asked for.
	return w.acceptedRevision >= w.revision && w.convergedRevision >= w.revision &&
		w.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE && w.dispatchable[planID]
}

func remotePlanKey(packageName, planID string) string { return packageName + "\x00" + planID }

// placementFor names the placement an offer for this plan routes to: the one placement a
// local slot hosts, or the rental placement advertising the plan under the pinned package.
func (w *worker) placementFor(slot, planID string) string {
	if w.spec.Connection != nil {
		if placement, ok := w.remotePlacements[remotePlanKey(slot, planID)]; ok {
			return placement.PlacementIDValue
		}
	}
	return w.placementID
}

func (w *worker) remoteDispatchable(placement DesiredPlacement, planID string) bool {
	observed := w.observedRemote[placement.PlacementIDValue]
	return placement.PlacementIDValue != "" &&
		observed.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE &&
		observed.dispatchablePlanIDs[planID]
}

func (w *worker) remoteStaged(packageName, planID, release, localRevision string,
	models []ModelRef) bool {
	placement, ok := w.remotePlacements[remotePlanKey(packageName, planID)]
	if !ok || placement.Release != release ||
		(localRevision != "" && placement.LocalRevisionDigest != localRevision) ||
		!selectionServes(models, placement.Models) {
		return false
	}
	observed, ok := w.observedRemote[placement.PlacementIDValue]
	return ok && observed.knownPlanIDs[planID]
}

func preparedRemotePlacement(w *worker, pkg, release string) (DesiredPlacement, bool, *exit.Error) {
	if len(w.setBytes) == 0 || len(w.setDigest) == 0 {
		return DesiredPlacement{}, false, nil
	}
	doc, err := canonical.Read(w.setBytes, &pb.PlacementSet{})
	if err != nil {
		return DesiredPlacement{}, false, exit.Named(exit.Structural,
			"rental.placement_set_invalid", "prepared PlacementSet is not canonical: %s", err)
	}
	digest := spellOf(w.setDigest)
	for _, row := range doc.List("placements") {
		selected := row.Sub("package")
		development := row.Sub("development")
		if development.Str("package") != "" {
			selected = development
		}
		if selected.Str("package") != pkg || selected.Str("release") != release {
			continue
		}
		placement := DesiredPlacement{Package: pkg, Release: release,
			PlacementIDValue: row.Str("placement_id"), PlacementSetDigest: digest,
			PlacementSetBytes: append([]byte(nil), w.setBytes...),
			EnvironmentDigest: row.Str("environment_digest")}
		if development.Str("package") != "" {
			placement.SourceDigest = development.Str("source_digest")
			placement.LocalRevisionDigest = development.Str("local_revision_digest")
		}
		if placement.PlacementIDValue == "" ||
			(placement.SourceDigest == "" && placement.EnvironmentDigest == "") {
			return DesiredPlacement{}, false, exit.Named(exit.Structural,
				"rental.placement_incomplete", "prepared placement for %s@%s is incomplete", pkg, release)
		}
		seen := make(map[string]bool)
		for _, entrypoint := range row.List("entrypoints") {
			name, binding := entrypoint.Str("name"), entrypoint.Str("entrypoint_binding_digest")
			if name == "" || seen[name] || !validDigest(binding) {
				return DesiredPlacement{}, false, exit.Named(exit.Structural,
					"rental.entrypoint_binding_invalid", "prepared placement has an invalid or repeated callable binding")
			}
			seen[name] = true
			placement.Entrypoints = append(placement.Entrypoints, Entrypoint{Name: name, Digest: binding})
		}
		return placement, true, nil
	}
	return DesiredPlacement{}, false, nil
}

// preparedPlacementServes answers whether the set this rental last converged to already
// binds the request's function to the request's exact models (h3a-018): the placement for
// the package release names the function, and every slot the pod bound for it references
// the manifest the request pins for that slot. A request that is served by what the pod
// holds issues no package prepare — the prepare it would issue authors the same bytes and
// costs the pod a full materialization (28 s measured, for zero new bytes) — and reaches
// dispatch through the ordinary ready wait. A refused or failed desire is never "held".
func preparedPlacementServes(w *worker, logical LogicalPackage) bool {
	if w == nil || w.refusal != nil || w.desiredRefusal != nil || len(logical.Models) == 0 {
		return false
	}
	desired, found, problem := preparedRemotePlacement(w, logical.Package, logical.Release)
	if problem != nil || !found {
		return false
	}
	if observed, ok := w.observedRemote[desired.PlacementIDValue]; ok &&
		observed.materialization == pb.MaterializationState_MATERIALIZATION_STATE_FAILED {
		return false
	}
	doc, err := canonical.Read(w.setBytes, &pb.PlacementSet{})
	if err != nil {
		return false
	}
	pinned := make(map[string]string, len(logical.Models))
	for _, model := range logical.Models {
		if !model.Pinned() {
			return false
		}
		pinned[model.Slot] = model.Manifest
	}
	for _, row := range doc.List("placements") {
		if row.Str("placement_id") != desired.PlacementIDValue {
			continue
		}
		manifests := map[string]string{}
		for _, model := range row.List("models") {
			manifests[model.Str("id")] = model.Sub("manifest").Str("digest")
		}
		for _, entrypoint := range row.List("entrypoints") {
			if entrypoint.Str("name") != logical.Function {
				continue
			}
			slots := entrypoint.List("slots")
			if len(slots) != len(pinned) {
				return false
			}
			for _, slot := range slots {
				path := logical.Function + ".models." + slot.Str("slot")
				held := manifests[slot.Str("reference_model_id")]
				if held == "" || pinned[path] != held {
					return false
				}
			}
			return true
		}
	}
	return false
}

// admissible answers the worker-level half. CLOSED is STRUCTURAL (pre-snapshot-barrier,
// draining, mid-cutover); OPEN with zero seats is TRANSIENT saturation — both refuse under
// CAUSE_CODE_NO_CAPACITY at the worker, and an owner backs off differently for each
// (#486c), which is why they are two fields here and not one. The per-placement half —
// the placement's own lane — is roomFor (route.go).
func (w *worker) admissible() bool {
	return w.admission == pb.AdmissionState_ADMISSION_STATE_OPEN && w.seats.slots > 0
}

func (w *worker) observeSlots(n int) { w.seats.observe(n) }

// observeHeld records the worker's held attempts, worker-wide and per lane, from the
// placement each names.
func (w *worker) observeHeld(placementIDs []string) {
	w.held = len(placementIDs)
	w.lanes.observeHeld(placementIDs)
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
// The three answers are not decoration. `cozy run` and POST /v1/local/workers both need
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
		if spec.Preparation != nil {
			return instanceID, ChangeNone, nil
		}
		// A concurrent empty-rental attach may have waited for this connection.
		// It requests the existing claim, never an empty replacement placement.
		if spec.Connection != nil && spec.Placement.Package == "" {
			return instanceID, ChangeNone, nil
		}
		// The worker is here. Does it already host what is wanted? The placement's
		// identity for this purpose is its plan set, which is what the desired set names.
		c.mu.Lock()
		hosts := c.workers[instanceID] == live && hostsPlans(live, spec.Placement)
		c.mu.Unlock()
		if hosts {
			return instanceID, ChangeNone, nil
		}
		if live.spec.Connection != nil {
			if e := c.ConvergePlacementSet(instanceID, []DesiredPlacement{spec.Placement}); e != nil {
				return instanceID, ChangeNone, e
			}
			c.mu.Lock()
			live.spec.Placement = spec.Placement
			live.planIDs = live.planIDs[:0]
			for _, entrypoint := range spec.Placement.Entrypoints {
				live.planIDs = append(live.planIDs, entrypoint.Digest)
			}
			sort.Strings(live.planIDs)
			c.mu.Unlock()
			return instanceID, ChangePlacementAdded, nil
		}
		if e := c.ConvergePlacementSet(instanceID, []DesiredPlacement{spec.Placement}); e != nil {
			return instanceID, ChangeNone, e
		}
		// The launcher's record is what the matching layer reads. A local re-stage keeps
		// the process but replaces its placement (cl-114: the same plans under a new
		// model selection), so the spec must say what the worker was just asked to host —
		// or the next request would be matched against the selection that was vacated.
		c.mu.Lock()
		if c.workers[instanceID] == live {
			live.spec.Placement = spec.Placement
			live.planIDs = live.planIDs[:0]
			for _, entrypoint := range spec.Placement.Entrypoints {
				live.planIDs = append(live.planIDs, entrypoint.Digest)
			}
			for _, job := range spec.Placement.Jobs {
				live.planIDs = append(live.planIDs, job.DescriptorID)
			}
			sort.Strings(live.planIDs)
		}
		c.mu.Unlock()
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

// EnsureRental attaches one generic empty worker without selecting a package. Later
// desired state is Creator-owned and travels directly on this control stream.
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
	instance := rentalInstanceID(id)
	c.mu.Lock()
	live := c.workers[instance]
	already := live != nil && !live.exited && !live.stopping && live.spec.Connection != nil &&
		live.spec.Connection.RentalID == id
	c.mu.Unlock()
	if already {
		return instance, "", ChangeNone, c.ensureWorkerClaimed(instance)
	}
	spec := WorkerLaunchSpec{Connection: target.Connection, Devices: target.Devices}
	instance, change, problem := c.EnsureWorker(spec)
	if problem == nil {
		problem = c.ensureWorkerClaimed(instance)
	}
	return instance, "", change, problem
}

func (c *Orchestrator) ensureWorkerClaimed(instanceID string) *exit.Error {
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		claimed := w != nil && !w.exited && w.bootID != "" && !w.lastReport.IsZero() &&
			w.snapshotAcknowledged
		gone := w == nil || w.exited
		var refused *exit.Error
		if w != nil {
			refused = w.refusal
		}
		c.mu.Unlock()
		switch {
		case claimed:
			return nil
		case refused != nil:
			return refused
		case gone:
			return exit.New(exit.Failed, "the rented worker exited before accepting this Creator claim")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (c *Orchestrator) ensureLogicalPackageReady(instanceID, rentalID string,
	logical LogicalPackage) (WorkerLaunchSpec, string, *exit.Error) {
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		gone := w == nil || w.exited
		var refused, desiredRefusal *exit.Error
		placementFailed := false
		if w != nil {
			refused, desiredRefusal = w.refusal, w.desiredRefusal
			desired, found, placementProblem := preparedRemotePlacement(w, logical.Package, logical.Release)
			if placementProblem != nil {
				c.mu.Unlock()
				return WorkerLaunchSpec{}, "", placementProblem
			}
			observed := w.observedRemote[desired.PlacementIDValue]
			placementFailed = found && w.acceptedRevision == w.revision &&
				observed.placementSetDigest == desired.PlacementSetDigest &&
				observed.materialization == pb.MaterializationState_MATERIALIZATION_STATE_FAILED
			planID := logical.PlanID
			if planID == "" {
				// Runtime authored each binding beside its callable name. Other
				// dispatchable entrypoints cannot identify this request's function.
				for _, entrypoint := range desired.Entrypoints {
					if entrypoint.Name == logical.Function {
						planID = entrypoint.Digest
						break
					}
				}
			}
			ready := found && planID != "" && observed.dispatchablePlanIDs[planID] &&
				observed.placementID != "" &&
				observed.placementSetDigest == desired.PlacementSetDigest &&
				observed.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE &&
				w.acceptedRevision >= w.revision && w.convergedRevision >= w.revision
			if ready && desired.SourceDigest == "" &&
				(!validDigest(observed.environmentDigest) ||
					observed.environmentDigest != desired.EnvironmentDigest) {
				c.mu.Unlock()
				return WorkerLaunchSpec{}, "", exit.Named(exit.Structural,
					"rental.invocation_identity_invalid",
					"worker resolved package %s without complete invocation identity", logical.Package)
			}
			if ready {
				placement := desired
				placement.Package = pinnedPackage(logical.Package, rentalID)
				placement.Entrypoints = []Entrypoint{{Name: logical.Function, Digest: planID,
					Outputs: append([]string(nil), logical.Outputs...)}}
				placement.Models = append([]ModelRef(nil), logical.Models...)
				w.remotePlacements[remotePlanKey(placement.Package, planID)] = placement
				spec := w.spec
				spec.Placement = placement
				c.mu.Unlock()
				return spec, planID, nil
			}
		}
		c.mu.Unlock()
		switch {
		case refused != nil:
			return WorkerLaunchSpec{}, "", refused
		case desiredRefusal != nil:
			return WorkerLaunchSpec{}, "", desiredRefusal
		case placementFailed:
			return WorkerLaunchSpec{}, "", exit.Named(exit.Failed,
				"rental.package_materialization_failed",
				"the rented worker could not materialize package %s", logical.Package)
		case gone:
			return WorkerLaunchSpec{}, "", exit.New(exit.Failed,
				"the rented worker exited before making package %s ready", logical.Package)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (c *Orchestrator) waitPackageStaged(instanceID string) *exit.Error {
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		ready := w != nil && !w.exited &&
			w.acceptedRevision >= w.revision &&
			w.materialization == pb.MaterializationState_MATERIALIZATION_STATE_STAGED
		gone := w == nil || w.exited
		var refused, desiredRefusal *exit.Error
		faulted := false
		if w != nil {
			refused, desiredRefusal, faulted = w.refusal, w.desiredRefusal, w.faulted
		}
		c.mu.Unlock()
		switch {
		case ready:
			return nil
		case refused != nil:
			return refused
		case desiredRefusal != nil:
			return desiredRefusal
		case faulted:
			return exit.New(exit.Failed, "the rented worker refused package preparation")
		case gone:
			return exit.New(exit.Failed, "the rented worker exited while preparing the package")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func validDigest(value string) bool {
	_, err := canonical.Raw(value)
	return err == nil
}

// DetachRental stops this daemon's control loop for one rented worker. It waits for an
// in-flight attach to finish choosing the slot, then waits for the exact worker epoch
// to quiesce. The caller may delete the rental's pinned certificate only after this returns.
func (c *Orchestrator) DetachRental(id string) bool {
	instanceID := rentalInstanceID(id)
	for {
		c.mu.Lock()
		if inFlight := c.ensuring[instanceID]; inFlight != nil {
			c.mu.Unlock()
			<-inFlight
			continue
		}
		w := c.workers[instanceID]
		if w == nil || w.spec.Connection == nil || w.spec.Connection.RentalID != id {
			c.mu.Unlock()
			return false
		}
		c.mu.Unlock()
		c.shutdownWorker(w, StopGrace)
		return true
	}
}

// hostsPlans answers whether a live worker was launched for exactly the selected
// entrypoint bindings. The exact PlacementSet remains the authority: a local worker
// holding the same plans under a DIFFERENT set — the same interface bound to a different
// model selection (cl-114) — does not host this placement, and EnsureWorker re-stages it.
func hostsPlans(w *worker, p DesiredPlacement) bool {
	want := map[string]bool{}
	for _, entrypoint := range p.Entrypoints {
		want[entrypoint.Digest] = true
	}
	for _, j := range p.Jobs {
		want[j.DescriptorID] = true
		if j.OrchestrationParent != nil {
			want[j.OrchestrationParent.DescriptorID] = true
		}
	}
	if w.spec.Connection != nil && !w.spec.IsJob() {
		for id := range want {
			if _, ok := w.remotePlacements[remotePlanKey(p.Package, id)]; !ok {
				return false
			}
		}
		return len(want) > 0
	}
	if w.spec.Placement.PlacementSetDigest != p.PlacementSetDigest {
		return false
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

// spawnWorker journals the device grant and spawns the worker for one exact
// PlacementSet. The grant is journaled BEFORE the process exists: a process that was never
// granted an envelope cannot appear, and two concurrent starts cannot both consume one.
func (c *Orchestrator) spawnWorker(spec WorkerLaunchSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	// The slot root is reused for logs and local paths only. Attempt authority stays in
	// Creator's records store; Runtime owns no durable journal under this directory.
	root := c.opt.Layout.WorkerDir(instanceID)
	workerHome := filepath.Join(root, "home")
	if err := os.MkdirAll(workerHome, 0o755); err != nil {
		return "", exit.Internalf("cannot create the worker home %s: %s", workerHome, err)
	}

	var planIDs []string
	for _, entrypoint := range spec.Placement.Entrypoints {
		planIDs = append(planIDs, entrypoint.Digest)
	}
	sort.Strings(planIDs)
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
		InstanceID: instanceID, Package: spec.Placement.Package,
		InstallID: spec.Placement.InstallID, WorkerID: "local", Devices: spec.Devices,
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
		"--release-id", spec.Placement.Release,
		"--devices", strings.Join(spec.Devices, ","),
		"--grace", strconv.FormatFloat(graceOr(spec.GraceSec), 'f', -1, 64),
	)
	for _, option := range []struct{ flag, value string }{
		{"--artifact-cache", spec.ArtifactCache},
		{"--install-root", spec.InstallRoot},
		{"--environment-python", spec.EnvironmentPython},
		{"--tensorfs-root", spec.TensorFSRoot},
	} {
		if option.value != "" {
			args = append(args, option.flag, option.value)
		}
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
	processtree.Prepare(cmd)

	w := newWorker(instanceID, spec)
	w.cmd, w.logPath, w.home = cmd, logPath, workerHome
	w.planIDs, w.bootstrap = planIDs, bootstrap
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
	if err := processtree.Adopt(cmd); err != nil {
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
		_ = processtree.Kill(cmd.Process.Pid, syscall.SIGKILL)
		_ = cmd.Wait()
		logFile.Close()
		processtree.Release(cmd.Process.Pid)
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
		processtree.Release(cmd.Process.Pid)
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
			// Reclaimed BEFORE the row closes: the row holds the slot's device grant, so
			// no replacement for this instance id can be spawned into the same root
			// until it does.
			c.reclaimWorker(instanceID)
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
		instanceID:       instanceID,
		spec:             spec,
		placementID:      spec.Placement.PlacementID(),
		dispatchable:     map[string]bool{},
		materializable:   map[string]bool{},
		remotePlacements: map[string]DesiredPlacement{},
		observedRemote:   map[string]remotePlacementObservation{},
		refused:          map[string]*refusedOutcome{},
		stopped:          make(chan struct{}),
		attachDone:       make(chan struct{}),
	}
	return w
}

// connectWorker registers an ALREADY-RUNNING worker (a rented pod's TLS leg, cl-015):
// no spawn, no device grant (the pod's card is the pod's), no birth identity — the
// conversation is the same claim the local path runs, dialed at the rental's address
// with the pinned server cert and signed Creator ClaimProof (#445/proto-013).
// The media plane remains the invocation byte path, but it is NOT a package distribution
// path. Binding plans are ordinary artifact-grant subjects now: Tensorhub supplies their
// locations and the worker verifies their digests while materializing PlacementSet/1.
func (c *Orchestrator) connectWorker(spec WorkerLaunchSpec) (string, *exit.Error) {
	instanceID := spec.InstanceID()
	if spec.Connection.Media == nil {
		return "", exit.Named(exit.Unavailable, "rental_no_media_plane",
			"rental %s pins a control address and no media plane, and a pod that cannot be "+
				"handed bytes cannot be served", spec.Placement.Package).
			WithRemedy("a rented pod runs its worker and a co-resident media server " +
				"(cl-014); this host will not fall back to granting paths on its own disk, " +
				"because the pod cannot reach them").
			WithNext("cozy rental")
	}
	// This is a per-I/O byte-stall bound, not a worker lifecycle deadline.
	byteplane, e := media.Dial(*spec.Connection.Media, mediaIOStallBudget,
		int64(c.maxOutputBytes()))
	if e != nil {
		return "", e
	}
	if e := byteplane.Health(); e != nil {
		return "", e
	}
	// Health can block while the retained rental is released. Revalidate its
	// existing authority without replacing this connection's pinned target.
	if c.opt.Rentals != nil {
		if _, problem := c.opt.Rentals(spec.Connection.RentalID); problem != nil {
			return "", problem
		}
	}
	planIDs := make([]string, 0, len(spec.Placement.Entrypoints))
	for _, entrypoint := range spec.Placement.Entrypoints {
		planIDs = append(planIDs, entrypoint.Digest)
	}
	sort.Strings(planIDs)
	// AttachWorker, not SpawnWorker: the device-envelope admission arbitrates THIS host's
	// cards, and the pod's card is the pod's. There is no grant to journal and none to
	// release, which is also why nothing here has a pid or a birth identity to record.
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return "", exit.Unavailablef("the daemon is closing; no rental worker was registered")
	}
	if e := c.opt.Store.AttachWorker(records.WorkerProcess{
		InstanceID: instanceID, Package: spec.Placement.Package,
		InstallID: spec.Placement.InstallID, WorkerID: "remote",
	}); e != nil {
		c.mu.Unlock()
		return "", e
	}
	w := newWorker(instanceID, spec)
	w.logPath = "(connected worker: its log lives on the pod)"
	w.planIDs, w.media = planIDs, byteplane
	c.workers[instanceID] = w
	c.mu.Unlock()
	c.logf("worker %s CONNECTED at %s (media %s) plans=%d",
		instanceID, spec.Connection.Addr, byteplane.Addr(), len(planIDs))
	go c.attach(w)
	return instanceID, nil
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

// StillFactor is the fleet's still-factor (xs-007 row 29): how many consecutive
// observations may find a meter where they left it before the thing is called stopped
// rather than slow. It is the count `transfer.stillSamples` spends before calling a
// transfer stalled, and every silence budget this package derives comes from it, so one
// slow read, a GC pause, or a retry inside a transport can never be mistaken for a stall —
// only silence that repeats can.
const StillFactor = 8

// mediaIOStallBudget bounds one socket operation that moves no bytes. It does not settle,
// retire, or reclaim a worker and resets whenever bytes move.
//
// The figure is DERIVED, not chosen: a live pod proves it is there every ReportCadence on
// the control stream, so StillFactor consecutive report periods with not one byte moving
// on the byte plane is silence by the pod's own published cadence.
const mediaIOStallBudget = StillFactor * ReportCadence

// observeLatchedFault reads the fault a worker keeps against the desired revision it has
// accepted but not converged. The Runtime LATCHES a materialization refusal for a desired
// revision — in its own words, "retry is a RecordOwner act: a NEW revision, or a grant
// refresh" — and reports it unchanged on every ReportCadence until one arrives. Neither
// is this owner's to give from here: a rental's local wheels travelled as capabilities
// the pod has already spent, and a local plan has nothing to refresh. So the same fault
// repeating across StillFactor consecutive reports, with no revision issued in between
// and no placement progressing, is the worker's final word on that revision, and the
// request waiting on it fails typed instead of sitting queued behind a report loop.
//
// COUNTED, NOT TIMED. A materialization that is landing bytes reports MATERIALIZING and
// never trips this; a fault whose text changes starts the count over; a new desired
// revision clears it. A fault against a placement that still serves (fallback retention,
// a degraded binding) is an explanation, not a stall, and is never counted.
func (w *worker) observeLatchedFault(r *pb.ObservedWorkerState) *exit.Error {
	reset := func() *exit.Error {
		w.latchedFault, w.latchedFaultRevision, w.latchedFaultReports = "", 0, 0
		return nil
	}
	if w.revision == 0 || r.AcceptedDesiredStateRevision != w.revision ||
		r.ConvergedRevision >= w.revision || len(w.setDigest) == 0 ||
		!bytes.Equal(r.AcceptedPlacementSetDigest, w.setDigest) {
		return reset()
	}
	// Only the exact desired set can identify the placement this request is
	// waiting for. The worker may still report its outgoing fallback beside it.
	desired, err := canonical.Read(w.setBytes, &pb.PlacementSet{})
	if err != nil {
		return reset()
	}
	current := make(map[string]bool)
	for _, p := range desired.List("placements") {
		current[p.Str("placement_id")] = true
	}
	reported := make(map[string]*pb.PlacementStatus, len(r.Placements))
	seen := make(map[string]bool, len(r.Placements))
	for _, p := range r.Placements {
		if p != nil {
			seen[p.PlacementId] = true
		}
		if p == nil || !current[p.PlacementId] || !bytes.Equal(p.PlacementSetDigest, w.setDigest) {
			continue
		}
		// Historical diagnostics can survive a cutover. Downloading, activating
		// or draining is a current transition, never proof of a latched failure.
		if p.Materialization == pb.MaterializationState_MATERIALIZATION_STATE_MATERIALIZING ||
			p.Serving == pb.ServingState_SERVING_STATE_ACTIVATING ||
			p.Serving == pb.ServingState_SERVING_STATE_DRAINING {
			return reset()
		}
		reported[p.PlacementId] = p
	}
	offline := func(p *pb.PlacementStatus) bool {
		return p != nil && p.Serving == pb.ServingState_SERVING_STATE_OFFLINE
	}
	var fault *pb.Fault
	var placement *pb.PlacementStatus
	for _, p := range r.Placements {
		if p != nil && reported[p.PlacementId] == p && offline(p) {
			for _, f := range p.Faults {
				if f != nil {
					fault, placement = f, p
					break
				}
			}
		}
		if fault != nil {
			break
		}
	}
	for _, f := range r.Faults {
		if fault != nil || f == nil || !current[f.Subject] {
			continue
		}
		p := reported[f.Subject]
		// A pending replacement can fail before it has a PlacementStatus row,
		// while the predecessor still serves. Only its exact incoming ID from
		// the owned set can associate that global failure. A present row from a
		// different set cannot use this absence case.
		if offline(p) || p == nil && !seen[f.Subject] {
			fault, placement = f, p
		}
	}
	if fault == nil {
		return reset()
	}
	placementID, executorEpoch := fault.Subject, uint64(0)
	if placement != nil {
		placementID, executorEpoch = placement.PlacementId, placement.ExecutorEpoch
	}
	key := fmt.Sprintf("%x\x00%s\x00%d\x00%d\x00%s\x00%s\x00%s",
		w.setDigest, placementID, executorEpoch, fault.Kind, fault.Subject, fault.Reason, fault.Detail)
	if key != w.latchedFault || w.latchedFaultRevision != w.revision {
		w.latchedFault, w.latchedFaultRevision, w.latchedFaultReports = key, w.revision, 0
	}
	w.latchedFaultReports++
	if w.latchedFaultReports < StillFactor {
		return nil
	}
	return exit.Named(exit.Failed, "worker.placement_refused", "%s: %s",
		fault.Reason, brief(fault.Detail, 1024))
}

// Jobs have no placement set, so placement-fault reconciliation cannot settle
// their preparation. This explicit safety refusal means no executor was created.
// A stale revision or any usable/live job slot is never preparation-failure proof.
func (w *worker) jobExecutorRefusal(r *pb.ObservedWorkerState) *exit.Error {
	capacity := r.GetJobCapacity()
	if !w.spec.IsJob() || w.revision == 0 || r.AcceptedDesiredStateRevision != w.revision ||
		capacity.GetJobsAvailable() != 0 || capacity.GetJobsInFlight() != 0 ||
		capacity.GetOrchestrationAvailable() != 0 || capacity.GetOrchestrationInFlight() != 0 {
		return nil
	}
	for _, fault := range r.Faults {
		if fault != nil && fault.Kind == pb.FaultKind_FAULT_KIND_LOCAL_SAFETY_REFUSAL && fault.Reason == "job_executor_absent" {
			return exit.Named(exit.Structural, "job_executor_absent",
				"the worker cannot create its contained job executor; run it within a delegated cgroup or use a configured private rental")
		}
	}
	return nil
}

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
// reports a FAILED axis, or its process/stream exits. A worker that is materializing is none
// of those, however long it takes.
//
// `requestID` is the request this wait exists FOR, and the wait ends when it does (cl-186).
// That is not a timeout — it is the observation that the only reason to keep waiting has
// gone. Without it a wedged preparation held `starting[slot]` forever: the request could be
// cancelled and the loop went on spinning, every later request for the slot returned early
// from selectOrStart and parked on "no attached rental has a DISPATCHABLE placement", and
// the only recovery anyone found was `cozy rental end`. Two H100s were surrendered that way
// on 2026-09-08. The guard two paragraphs down already refuses one particular version of
// this wedge (`plan_not_staged`) and says so in those words; this is the general case.
//
// An empty `requestID` is a wait no request owns — `cozy install --sync`'s foreground
// relaunch, which the operator interrupts directly.
func (c *Orchestrator) EnsurePlacementReady(instanceID, planID, requestID string) *exit.Error {
	settledCheck := time.Time{}
	for {
		c.mu.Lock()
		w := c.workers[instanceID]
		ok := w != nil && !w.exited && w.dispatchableFor(planID)
		if w != nil && w.spec.IsJob() {
			// Jobs have no serving axis or placement convergence. An occupied
			// executor is ready too: its accepted attempt must not be killed by
			// the preparation waiter racing with dispatch.
			ok = !w.exited && w.acceptedRevision >= w.revision && w.jobReady[planID]
		}
		gone := w == nil || w.exited
		logPath, fault, code := "", "", 0
		workerFaulted := false
		unstaged, holds := false, ""
		var refused, desiredRefusal *exit.Error
		if w != nil {
			logPath, fault, code, refused = w.logPath, w.fault, w.exitCode, w.refusal
			desiredRefusal = w.desiredRefusal
			workerFaulted = w.faulted
			if !w.exited && !staged(w, planID) {
				unstaged, holds = true, strings.Join(w.planIDs, ", ")
			}
		}
		c.mu.Unlock()
		if ok {
			return nil
		}
		// THIS OWNER'S OWN VERDICT COMES FIRST. A worker whose claim was refused here has
		// already received an identity verdict from this side.
		if refused != nil {
			return refused
		}
		if desiredRefusal != nil {
			return desiredRefusal
		}
		// THE WAIT ENDS WHEN ITS REQUEST DOES (cl-186), and it is asked here — after
		// readiness and after this owner's own verdicts, so a placement that became
		// dispatchable still wins and a real refusal is still the answer. The durable row
		// is read on entry and then no oftener than the fact can usefully change; the
		// loop's 20 ms cadence is for in-memory observations, not for SQLite.
		if requestID != "" && time.Since(settledCheck) >= time.Second {
			settledCheck = time.Now()
			if row, problem := c.opt.Store.RequestRow(requestID); problem == nil &&
				(row == nil || records.Settled(row.State)) {
				return exit.Named(exit.Canceled, "request.settled_while_preparing",
					"request %s settled while worker %s was still preparing %s; the wait it "+
						"started is over", requestID, instanceID, planID)
			}
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
			// The runtime's own typed refusal is already on stderr, which is this log.
			// Saying only "read the log" turned a launch-grammar mismatch (cl-120's
			// `--artifact-store`) into a mute death; the tool's last words are the answer.
			return exit.New(exit.Failed,
				"the package worker exited (%d) before reporting ready: %s",
				code, lastWords(logPath)).
				WithRemedy("its log is %s", logPath)
		}
		if workerFaulted {
			// FAILED/fault is the worker's settled typed answer, not a timer start.
			// Transient work remains MATERIALIZING/ACTIVATING and keeps reporting progress.
			return workerError(fault).WithRemedy("its log is %s", logPath)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// lastWords is the tail of a dead worker's log, condensed to one line. A worker that
// refuses its own launch says why on stderr and then exits; this is that sentence, read
// at the only moment anyone needs it.
func lastWords(path string) string {
	if path == "" {
		return "it wrote no log"
	}
	f, err := os.Open(path)
	if err != nil {
		return "its log is unreadable: " + err.Error()
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "its log is unreadable: " + err.Error()
	}
	const window = 4 << 10
	at := info.Size() - window
	if at < 0 {
		at = 0
	}
	buf := make([]byte, info.Size()-at)
	if _, err := f.ReadAt(buf, at); err != nil && err != io.EOF {
		return "its log is unreadable: " + err.Error()
	}
	var said []string
	for _, line := range strings.Split(string(buf), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			said = append(said, line)
		}
	}
	if len(said) == 0 {
		return "it said nothing"
	}
	if len(said) > 3 {
		said = said[len(said)-3:]
	}
	tail := strings.Join(strings.Fields(strings.Join(said, " · ")), " ")
	if len(tail) > 400 {
		return "…" + tail[len(tail)-400:]
	}
	return tail
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
	InstanceID                string   `json:"instance_id"`
	RentalID                  string   `json:"rental_id,omitempty"`
	Package                   string   `json:"package"`
	Release                   string   `json:"release"`
	BootID                    string   `json:"worker_boot_id"`
	PlacementID               string   `json:"placement_id"`
	PlacementSetDigest        string   `json:"placement_set_digest"`
	RetainedFallbackSetDigest string   `json:"retained_fallback_placement_set_digest"`
	PID                       int      `json:"pid"`
	ExecutorEpoch             uint64   `json:"executor_epoch"`
	Exited                    bool     `json:"exited"`
	Devices                   []string `json:"devices"`

	Phase           string `json:"worker_phase"`
	Materialization string `json:"materialization"`
	Serving         string `json:"serving"`
	// Dispatchable is the placement's own `dispatchable_plan_ids`; Materializable is the
	// DISJOINT set it could take but has not activated.
	Dispatchable   []string `json:"dispatchable_plan_ids"`
	Materializable []string `json:"materializable_plan_ids"`

	Admission      string `json:"admission_state"`
	AdmissionEpoch uint64 `json:"admission_epoch"`
	AvailableSlots int    `json:"available_attempt_slots"`
	// HeldAttempts is what the worker last reported holding, admission to ack.
	HeldAttempts int `json:"held_attempts"`
	// Lanes is the worker's own account of its serialized resources (proto-024): one seat
	// each, ordinals into the granted Devices. Empty for a worker that reports none.
	// PlacementLane is the lane this slot's placement is on, "" until the worker says.
	Lanes         []LaneFacts `json:"lanes,omitempty"`
	PlacementLane string      `json:"placement_lane,omitempty"`
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

func factsOf(w *worker) WorkerFacts {
	f := WorkerFacts{
		InstanceID: w.instanceID, Package: w.spec.Placement.Package,
		Release: w.spec.Placement.Release, BootID: w.bootID,
		PlacementID: w.placementID, ExecutorEpoch: w.executorEpoch,
		Exited: w.exited, Devices: w.spec.Devices,

		Phase:           trimEnum(pb.WorkerPhase_name[int32(w.phase)], "WORKER_PHASE_"),
		Materialization: trimEnum(pb.MaterializationState_name[int32(w.materialization)], "MATERIALIZATION_STATE_"),
		Serving:         trimEnum(pb.ServingState_name[int32(w.serving)], "SERVING_STATE_"),
		Dispatchable:    keysOf(w.dispatchable),
		Materializable:  keysOf(w.materializable),

		Admission:       trimEnum(pb.AdmissionState_name[int32(w.admission)], "ADMISSION_STATE_"),
		AdmissionEpoch:  w.admissionEpoch,
		AvailableSlots:  w.seats.slots,
		HeldAttempts:    w.held,
		UnackedOutcomes: w.unacked,
		Lanes:           laneFactsOf(w),
		PlacementLane:   w.laneOf(w.placementID),

		DesiredRevision:   w.revision,
		AcceptedRevision:  w.acceptedRevision,
		ConvergedRevision: w.convergedRevision,
		Acquisition:       w.acquisition,

		Fault: w.fault,
	}
	if w.spec.Connection != nil {
		f.RentalID = w.spec.Connection.RentalID
	}
	f.PlacementSetDigest, _ = canonical.Spell(w.heldSetDigest)
	f.RetainedFallbackSetDigest, _ = canonical.Spell(w.fallbackSetDigest)
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
// person reading `cozy run list`.
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

// retireWorker is the epoch-fenced retirement used by the stall path. A recovered
// worker intentionally reuses instance and placement ids; only the exact object whose
// observations established the retirement ground may receive the empty desired set.
func (c *Orchestrator) retireWorker(w *worker) *exit.Error {
	c.mu.Lock()
	if c.workers[w.instanceID] != w || w.exited || w.stopping {
		c.mu.Unlock()
		return exit.New(exit.NotFound, "worker %s is no longer the observed epoch", w.instanceID)
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
// path exists anywhere in this package. The wait for the process to go uses its process handle,
// the kernel's own answer, polled until `StopGrace` is spent.
func (c *Orchestrator) ShutdownWorker(instanceID string, grace time.Duration) {
	c.mu.Lock()
	w := c.workers[instanceID]
	c.mu.Unlock()
	c.shutdownWorker(w, grace)
}

// shutdownWorker stops exactly the worker object the caller observed. Instance ids are
// deterministic and reused across child restarts, so an id-only teardown can otherwise close
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
		if err := processtree.Kill(pid, syscall.SIGTERM); err != nil {
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
			_ = processtree.Kill(pid, syscall.SIGKILL)
			<-processDone // process reaping is an observed fact, not another timeout
		}
	} else if processDone != nil {
		<-processDone
	}
	if attachDone != nil {
		<-attachDone
	}
	if w.media == nil {
		c.reclaimWorker(w.instanceID) // before the row closes; see the exit path
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

// reclaimWorker removes an exited local worker's root — its runtime home, scratch and
// model-ingest copies — once its process is reaped and before its row closes. The log
// stays until the next daemon start for the person the exit message pointed at it.
func (c *Orchestrator) reclaimWorker(instanceID string) {
	freed, problem := reclaim.Worker(c.opt.Layout, instanceID, true)
	if problem != nil {
		c.logf("worker %s root reclaim deferred: %s", instanceID, problem.Message)
		return
	}
	if freed > 0 {
		c.logf("worker %s root reclaimed: %s", instanceID, units.Bytes(freed))
	}
}

// UnloadIdleLocalWorkers stops every definitely-idle local serving worker. Remote
// workers are paid resources owned by the rental lifecycle, job workers are run-once,
// and any active request, outstanding offer, reservation, or unacked terminal keeps a
// worker alive. Process exit is the reliable release of its GPU-resident model.
func (c *Orchestrator) UnloadIdleLocalWorkers() ([]WorkerFacts, *exit.Error) {
	return c.unloadIdleLocalWorkers("", "")
}

// UnloadIdleLocalPackage stops only stale idle workers for one refreshed package.
// The current install stays warm; other warm packages and active work also stay alive.
func (c *Orchestrator) UnloadIdleLocalPackage(pkg, keepInstallID string) ([]WorkerFacts, *exit.Error) {
	return c.unloadIdleLocalWorkers(pkg, keepInstallID)
}

func (c *Orchestrator) unloadIdleLocalWorkers(pkg, keepInstallID string) ([]WorkerFacts, *exit.Error) {
	active, e := c.opt.Store.ActiveRequests()
	if e != nil {
		return nil, e
	}

	c.mu.Lock()
	candidates := make([]*worker, 0, len(c.workers))
	for _, w := range c.workers {
		if w.spec.Connection == nil && !w.spec.IsJob() &&
			(pkg == "" || w.spec.Placement.Package == pkg) &&
			(keepInstallID == "" || w.spec.Placement.InstallID != keepInstallID) {
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
		// Capacity changed, so the queue must be re-asked — but that wake is not part
		// of reclamation. Its head may synchronously acquire a managed rental. Joining
		// that unrelated network operation here kept `cozy unload` waiting after every
		// selected process was reaped and its device grant was already released.
		go c.reviveQueue()
	}
	return stopped, nil
}

func (c *Orchestrator) idleLocalWorkerLocked(w *worker, active []records.Request) bool {
	if w == nil || c.workers[w.instanceID] != w || w.exited || w.stopping ||
		w.spec.Connection != nil || w.spec.IsJob() || w.seats.reserved != 0 || w.unacked != 0 {
		return false
	}
	for _, reservation := range c.offers {
		if reservation.worker == w {
			return false
		}
	}
	for _, req := range active {
		// Submitted/queued requests own a FIFO position, not this worker. Counting a
		// same-package request behind a different-package head as active would protect
		// the current holder forever while FIFO prevents that later request dispatching.
		if req.State != "dispatching" && req.State != "requeue_pending" {
			continue
		}
		// Binding-plan digests are content identities, not package identities. Two
		// packages can legitimately expose byte-identical bindings; treating the plan
		// digest alone as ownership made a queued package B protect package A's idle
		// worker from eviction. Match the local package slot and immutable install too.
		if req.Worker == "" && req.Package == w.spec.Placement.Package &&
			(req.InstallID == "" || req.InstallID == w.spec.Placement.InstallID) &&
			staged(w, req.PlanID) {
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
	unlockLocal := localpackage.Guard()
	e = localpackage.Sweep(c.opt.Layout, c.opt.Store)
	unlockLocal()
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
			_ = processtree.Kill(row.PID, syscall.SIGKILL)
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
	// AND THE ATTEMPTS THAT OWE A TERMINAL. Remote rentals reconnect to pod-supervisor,
	// whose worker-local ledger replays their exact state. Local Runtime is stateless, so
	// Creator settles its own persisted assignment as ABANDONED and requeues it; restarting
	// an execution child to ask it what happened would recreate the duplicate journal this
	// boundary removes.
	unsettled, e := c.opt.Store.Unsettled()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range unsettled {
		if req.Worker != "" {
			// RECONNECTING ONLY WORKS IF THERE IS SOMETHING TO RECONNECT TO. pod-supervisor's
			// ledger is what replays a remote attempt, and it lives on the pod: once the
			// rental is gone or terminally failed, that ledger was destroyed with it and no
			// amount of reconnecting will ever produce the terminal this attempt owes.
			// Before this check the daemon said "reconnecting to its supervisor ledger" at
			// every boot for a pod that had not existed for hours, and the request behind it
			// could never settle (observed live: req-b2df33d17e663a8a1e047246, 2590 s and
			// climbing on a destroyed container).
			if !c.rentalCanServe(req.Worker) {
				c.recoverPinned(req, req.Worker, c.lostRentalCause(req.Worker))
				continue
			}
			c.logf("%s holds a remote attempt with no terminal; reconnecting to its supervisor ledger",
				req.ID)
			c.selectOrStart(req)
			continue
		}
		attempts, problem := c.opt.Store.Attempts(req.ID)
		if problem != nil {
			return killed, forgotten, problem
		}
		for _, attempt := range attempts {
			switch attempt.State {
			case "preparing", "offered", "accepted", "recovered_open", "terminal":
				c.settleLocalProcessDeath(attempt)
			}
		}
	}
	ready, e := c.opt.Store.ReadyRequeues()
	if e != nil {
		return killed, forgotten, e
	}
	for _, req := range ready {
		c.logf("%s committed and acknowledged a retry before restart; resuming its requeue", req.ID)
		c.Requeue(req.ID, "restart-after-terminal-ack")
	}
	if problem := c.ResumeOutputExports(); problem != nil {
		return killed, forgotten, problem
	}
	if problem := c.ResumeNativeEffects(); problem != nil {
		c.logf("native effect recovery pending: %s", problem.ErrName())
	}
	if problem := c.ResumeModelTransfers(); problem != nil {
		return killed, forgotten, problem
	}
	if problem := c.ResumeQueuedRequests(); problem != nil {
		return killed, forgotten, problem
	}
	if problem := c.resumeManualRentals(); problem != nil {
		return killed, forgotten, problem
	}
	if problem := c.restoreRetainedWork(); problem != nil {
		return killed, forgotten, problem
	}
	return killed, forgotten, nil
}

// ResumeQueuedRequests re-enters the one placement path for every request a
// prior daemon accepted but never made resident. A submission's selectOrStart
// dies with its daemon, and a queued row with no attempt then waits forever —
// the "worst of both" selectOrStart's own comment warns about (live: three
// four-lane submissions orphaned by daemon restarts, 2026-09-02). A rental
// request whose prior daemon had already acquired may briefly double-rent;
// the idle-release timer reaps the orphan.
func (c *Orchestrator) ResumeQueuedRequests() *exit.Error {
	queued, problem := c.opt.Store.Requests("queued", "", 512)
	if problem != nil {
		return problem
	}
	for _, req := range queued {
		c.logf("%s was queued when a prior daemon exited; resuming its placement", req.ID)
		c.selectOrStart(req)
	}
	return nil
}

// freshActivity is the report's activity entries this owner has not logged yet, in
// order. Callers hold c.mu.
func (w *worker) freshActivity(entries []*pb.ActivityEvent) []*pb.ActivityEvent {
	var newest uint64
	for _, a := range entries {
		if a != nil && a.Seq > newest {
			newest = a.Seq
		}
	}
	if newest < w.activitySeq {
		w.activitySeq = 0
	}
	var out []*pb.ActivityEvent
	for _, a := range entries {
		if a == nil || a.Seq <= w.activitySeq {
			continue
		}
		w.activitySeq = a.Seq
		out = append(out, a)
	}
	return out
}

// freshFaults is the report's worker and placement faults this owner has not logged
// since they last appeared. A fault absent from a report is forgotten, so its return is
// news again. Callers hold c.mu.
func (w *worker) freshFaults(worker, placement []*pb.Fault) (fresh, freshPlacement []*pb.Fault) {
	seen := map[string]bool{}
	pick := func(scope string, faults []*pb.Fault) []*pb.Fault {
		var out []*pb.Fault
		for _, f := range faults {
			if f == nil {
				continue
			}
			key := fmt.Sprintf("%s\x00%d\x00%s\x00%s\x00%s", scope, f.Kind, f.Subject, f.Reason, f.Detail)
			seen[key] = true
			if !w.loggedFaults[key] {
				out = append(out, f)
			}
		}
		return out
	}
	fresh = pick("worker", worker)
	freshPlacement = pick("placement", placement)
	w.loggedFaults = seen
	return fresh, freshPlacement
}
