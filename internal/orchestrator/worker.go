package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"github.com/cozy-creator/cozy/internal/archive"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/records"
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
	Package        string `json:"package"` // org/name — the slot this placement serves under
	Release        string `json:"release"`
	InstallationID string `json:"installation_id"` // worker environment lifetime; never package content
	// InstallID is the install this placement was resolved from ("" = an uninstalled dev
	// tree). Was `Generation`, which named a protocol word this side does not own (#484).
	InstallID string `json:"install_id"`
	// PlacementSetBytes are the exact Hub-selected PlacementSet/1 bytes. Creator
	// validates and relays them unchanged for both a local venv and a rented pod.
	// Entrypoints is a read-only projection used for request selection; it never
	// participates in identity.
	PlacementSetDigest string       `json:"placement_set_digest"`
	PlacementSetBytes  []byte       `json:"placement_set_bytes"`
	BindingsDigest     string       `json:"bindings_digest"`
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
	doc, err := archive.Read(data, archive.Placement)
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
	selected := packageFact
	if development.Str("package") != "" {
		if len(packageFact) != 0 || development.Str("installation_id") != row.Str("installation_id") {
			return DesiredPlacement{}, exit.Named(exit.Structural, "development_placement_incomplete",
				"development placement mixes package selections or installation handles")
		}
		selected = development
	}
	placement := DesiredPlacement{
		Package: pkg, InstallID: installID, PlacementIDValue: row.Str("placement_id"),
		Release: selected.Str("release"), InstallationID: row.Str("installation_id"),
		PlacementSetDigest: digest, PlacementSetBytes: append([]byte(nil), data...),
	}
	if selected.Str("package") != pkg || placement.Release == "" ||
		placement.InstallationID == "" || placement.PlacementIDValue == "" ||
		row.Str("package_interface") == "" {
		return DesiredPlacement{}, exit.Named(exit.Structural, "placement_set_incomplete",
			"PlacementSet requires its selected package, installation and callable metadata")
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
	placement.BindingsDigest = row.Str("bindings_digest")
	placement.Models = placementModels(pkg, row)
	return placement, nil
}

// placementModels reads the exact model rows a PlacementSet binds and names each by the
// descriptor slot path its entrypoint binds it to — the spelling every download
// download set, unpublished package placement, and request selection is addressed by.
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
				Manifest: manifest.Str("digest"), ManifestLength: manifest.Int("length"), Adapters: placementAdapters(slot, byID)})
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
	// accelerator count, empty for a CPU pod, settled when it was bought.
	Devices []string
}

// WorkerLaunchSpec names one attached rental worker and the placement it hosts. The
// orchestrator starts no process: this computer runs work through its machine's Host.
type WorkerLaunchSpec struct {
	// Devices is the rental's paid width, which this daemon only reads. Lane ordinals are
	// positions in it, and its length is the placement's device-group degree.
	Devices []string `json:"devices"`
	// Placement is what this worker is launched to host. LAUNCH CLAMPS THE SET TO ONE
	// (worker-protocol header): a longer set is a typed refusal at the worker, so this
	// side names one placement rather than pretending to a generality it cannot deliver.
	Placement DesiredPlacement `json:"placement"`
	// Connection attaches the rental's ALREADY-RUNNING worker: the owner pins CACert and
	// signs Claim with its per-rental key.
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

// RecycleExit is the run-once COMPLETION disposition (cr-009, v1 rc 75). A job worker
// exits with it after its terminal is acknowledged: the process ending is the successful
// end of the work, and it is deliberately distinct from any death.
const RecycleExit = 75

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
	// Lanes is the worker's own account of its serialized resources (proto-024), each with
	// the seats it still admits, ordinals into the granted Devices. Empty for a worker that reports none.
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

// ClassicRetired names work accepted for the retired classic worker session, local or
// rented. Every machine now runs work as a machine execution; such work is never revived.
func ClassicRetired() *exit.Error {
	return exit.Named(exit.Conflict, records.ClassicRetiredCode,
		"this work was accepted for the retired classic worker; machines now run work as machine executions").
		WithRemedy("submit it again with the current cozy")
}

// StopGrace is how long a worker has between SIGTERM and SIGKILL. It is a CANCELLATION
// BUDGET, not a wait for work to finish: the supervisor's own cooperative-cancel budget
// plus the terminal transaction that follows it, which is what a worker still needs to do
// after it is told to stop. It is deliberately ONE number — this operation used to be
// spelled 20 s in the stall watchdog, 20 s for a stale worker, 20 s for one that failed to
// become ready, and 30 s on the API route, with nothing anywhere saying why the same
// SIGTERM deserved three budgets.
const StopGrace = 30 * time.Second

// Reconcile rebuilds the daemon's work from its records before anything is served.
func (c *Orchestrator) Reconcile() *exit.Error {
	writer, problem := home.LockWriter(c.opt.Layout)
	if problem != nil {
		// A CLI may be starting this daemon while handing off a captured run.
		// That writer owns its not-yet-submitted revision; defer optional GC.
		c.logf("local package sweep deferred: %s", problem.Message)
	} else {
		unlockLocal := localpackage.Guard()
		e := localpackage.Sweep(c.opt.Layout, c.opt.Store)
		unlockLocal()
		writer.Unlock()
		if e != nil {
			return e
		}
	}
	// Work accepted for the retired classic worker session ends here, named, and custody no
	// store can release any more (classic work, ended rentals) is forgotten.
	retired, e := c.opt.Store.RetireClassicWork()
	if e != nil {
		return e
	}
	for _, id := range retired {
		c.logf("%s was accepted for the retired classic worker; ended as %s", id, records.ClassicRetiredCode)
	}
	if e := c.opt.Store.ForgetEndedRentalCustody(); e != nil {
		return e
	}
	// THE QUEUE IS MEMORY, AND THE AUTHORITY IS NOT. A request recorded as owed work before
	// the crash has no queue entry after it; rebuilding the queue from the authority, in
	// its own created_at order, happens before anything is served.
	owed, e := c.opt.Store.Owed()
	if e != nil {
		return e
	}
	for _, req := range owed {
		c.enqueue(req.ID)
		c.logf("%s was owed work before the restart; it is back on the queue", req.ID)
	}
	if len(owed) > 0 {
		go c.reviveQueue()
	}
	if problem := c.ResumeOutputExports(); problem != nil {
		return problem
	}
	if problem := c.ResumeModelTransfers(); problem != nil {
		return problem
	}
	return c.ResumeQueuedRequests()
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
		c.start(req)
	}
	return nil
}
