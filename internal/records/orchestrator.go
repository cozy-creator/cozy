package records

import (
	"bytes"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The orchestrator's half of the ONE lifecycle authority (cl-001). Worker processes,
// their sessions, requests, attempts and outputs are rows in the SAME database as the
// package installs and pins — one authority, one transaction boundary, no second
// lifecycle store anywhere (the fence's `store` family proves the absence).
//
// Two properties this file exists for:
//
//   1. "terminal accepted + output visible" is ONE transaction, and it commits BEFORE
//      TerminalAck. A crash between them exposes no output and cannot double-apply.
//   2. A device envelope cannot be consumed twice. The grant is an ATTRIBUTE of the
//      worker process row (1:1, never a separate identity), and the admission is a
//      single INSERT...SELECT...WHERE NOT EXISTS statement — atomic by construction,
//      so two concurrent starts cannot both win.

const requestsDDL = `
CREATE TABLE IF NOT EXISTS requests (
  id           TEXT PRIMARY KEY,
  idem_key     TEXT    NOT NULL UNIQUE,
  body_digest  TEXT    NOT NULL,
  package     TEXT    NOT NULL,
  entrypoint   TEXT    NOT NULL,
  plan_id      TEXT    NOT NULL,
  package_release TEXT NOT NULL DEFAULT '',
  local_installation_id TEXT NOT NULL DEFAULT '',
  local_package_uploaded_boot_id TEXT NOT NULL DEFAULT '',
  installation_id TEXT NOT NULL DEFAULT '',
  payload      BLOB    NOT NULL,
  outputs      TEXT    NOT NULL DEFAULT '',
  state        TEXT    NOT NULL,
  ordinal      INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT    NOT NULL,
  kind         TEXT    NOT NULL DEFAULT 'serving',
  needs_accelerator INTEGER NOT NULL DEFAULT 0,
  org          TEXT    NOT NULL DEFAULT '',
  trees        TEXT    NOT NULL DEFAULT '',
  worker       TEXT    NOT NULL DEFAULT '',
  machine      TEXT    NOT NULL DEFAULT '',
  rental       INTEGER NOT NULL DEFAULT 0,
  rental_required INTEGER NOT NULL DEFAULT 0,
  rent_new INTEGER NOT NULL DEFAULT 0 CHECK(rent_new IN (0,1)),
  hub TEXT NOT NULL DEFAULT '',
  install_id   TEXT    REFERENCES installs(id),
  assets       TEXT    NOT NULL DEFAULT '[]',
  capture      TEXT    NOT NULL DEFAULT '',
  attention_kernel TEXT NOT NULL DEFAULT '',
  models       TEXT    NOT NULL DEFAULT '[]',
  weights_outputs TEXT NOT NULL DEFAULT '[]',
  retain_work INTEGER NOT NULL DEFAULT 0 CHECK(retain_work IN (0,1)),
  retry_of TEXT NOT NULL DEFAULT '',
  reuse_scope TEXT NOT NULL DEFAULT '',
  control_revision INTEGER NOT NULL DEFAULT 0 CHECK(control_revision>=0),
  parent_request_id TEXT NOT NULL DEFAULT '',
  parent_call_index INTEGER NOT NULL DEFAULT -1 CHECK(parent_call_index>=-1 AND parent_call_index<4294967296),
  child_intent_digest TEXT NOT NULL DEFAULT '',
  child_target_digest TEXT NOT NULL DEFAULT '',
  child_reusable INTEGER NOT NULL DEFAULT 0 CHECK(child_reusable IN (0,1)),
  reused_from TEXT NOT NULL DEFAULT '',
  orchestration_directive BLOB NOT NULL DEFAULT x'',
  child_artifacts INTEGER NOT NULL DEFAULT 0 CHECK(child_artifacts IN (0,1)),
  requested_rental TEXT NOT NULL DEFAULT ''
)`

const workerProcessesDDL = `
CREATE TABLE IF NOT EXISTS worker_processes (
  instance_id     TEXT PRIMARY KEY,
  package        TEXT    NOT NULL,
  install_id      TEXT    REFERENCES installs(id),
  worker_id       TEXT    NOT NULL,
  devices         TEXT    NOT NULL,
	pid             INTEGER NOT NULL,
	birth           TEXT    NOT NULL,
	session_id      TEXT,
	state           TEXT    NOT NULL,
  opened_at       TEXT    NOT NULL,
  closed_at       TEXT    NOT NULL DEFAULT ''
)`

const workerSessionIndex = `
CREATE UNIQUE INDEX IF NOT EXISTS worker_session ON worker_processes(session_id)
  WHERE session_id IS NOT NULL`

const attemptsDDL = `
CREATE TABLE IF NOT EXISTS attempts (
  request_id       TEXT    NOT NULL REFERENCES requests(id),
  attempt          INTEGER NOT NULL,
  attempt_key      TEXT    NOT NULL UNIQUE,
  instance_id      TEXT    NOT NULL REFERENCES worker_processes(instance_id),
  session_id       TEXT    NOT NULL,
  invocation_digest TEXT    NOT NULL,
  invocation        BLOB    NOT NULL,
  weights_outputs  TEXT    NOT NULL DEFAULT '[]',
  state            TEXT    NOT NULL,
  plan_digest      TEXT    NOT NULL DEFAULT '',
  construction     TEXT    NOT NULL DEFAULT '',
  plan_summary     TEXT    NOT NULL DEFAULT '',
  terminal_id      TEXT    NOT NULL DEFAULT '',
  terminal_digest  TEXT    NOT NULL DEFAULT '',
  terminal_status  TEXT    NOT NULL DEFAULT '',
  terminal_cause   TEXT    NOT NULL DEFAULT '',
  safe_message     TEXT    NOT NULL DEFAULT '',
  triage_subject   TEXT    NOT NULL DEFAULT '',
  triage_digest    TEXT    NOT NULL DEFAULT '',
  triage_length    INTEGER NOT NULL DEFAULT 0,
  triage_bundle    BLOB    NOT NULL DEFAULT x'',
  terminal_body    BLOB,
  dispatched_at    TEXT    NOT NULL,
  accepted_at      TEXT    NOT NULL DEFAULT '',
  closed_at        TEXT    NOT NULL DEFAULT '',
  media_cleaned    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (request_id, attempt)
)`

var orchestratorSchema = []string{workerProcessesDDL, workerSessionIndex, requestsDDL, childRequestIndex, `
-- The PUBLICATION (cl-004). One row per job request, written INSIDE the terminal
-- transaction: a publication that a terminal did not commit does not exist, which is
-- what "killing before commit exposes no partial bundle" means as a schema property
-- rather than as a check. The status column is the terminal's own verdict STAMPED as
-- metadata: a failed run's landed writes still land (jobs.md), and nothing here gates it.
CREATE TABLE IF NOT EXISTS publications (
  request_id   TEXT PRIMARY KEY REFERENCES requests(id),
  attempt      INTEGER NOT NULL,
  repo         TEXT    NOT NULL,
  root         TEXT    NOT NULL,
  status       TEXT    NOT NULL,
  cause        TEXT    NOT NULL DEFAULT '',
  entries      INTEGER NOT NULL DEFAULT 0,
  bytes        INTEGER NOT NULL DEFAULT 0,
  committed_at TEXT    NOT NULL
)`, `
-- The DURABLE checkpoint exchange's orchestrator half (cr-009 §Checkpoints). The worker
-- has already made the save durable in its OWN journal; this is a SECOND observation,
-- at-least-once like every other durable message (law 9). The identity is closed:
-- repeating it replays the receipt, and the same key with different bytes CONFLICTS —
-- a orchestrator conflict is a journaled fault and never un-writes the worker's fact.
--
-- THE ATTEMPT IS PART OF THE IDENTITY (#553a). It was stored and then left out of the key,
-- so two attempts of one request shared one checkpoint identity: attempt 2 saving the same
-- logical key with different bytes read as a CONFLICT with attempt 1 instead of as its own
-- checkpoint, and identical bytes read as attempt 1's receipt REPLAYED. The foreign key is
-- the other half — the same shape the outputs table already carries — so a checkpoint row
-- cannot name an attempt that does not exist.
CREATE TABLE IF NOT EXISTS job_checkpoints (
  request_id    TEXT    NOT NULL,
  attempt       INTEGER NOT NULL,
  operation_key TEXT    NOT NULL,
  logical_key   TEXT    NOT NULL,
  content_digest TEXT   NOT NULL,
  receipt_id    TEXT    NOT NULL,
  outcome       TEXT    NOT NULL,
  recorded_at   TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, operation_key, logical_key),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`, attemptsDDL, `
CREATE TABLE IF NOT EXISTS outputs (
  request_id TEXT    NOT NULL,
  attempt    INTEGER NOT NULL,
  output_id  TEXT    NOT NULL,
  media_id   TEXT    NOT NULL UNIQUE,
  path       TEXT    NOT NULL,
  digest     TEXT    NOT NULL,
  length     INTEGER NOT NULL,
  mime_type  TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, output_id),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`, outputExportSchema, `
-- Cozy's first-wins typed weights disposition and Runtime completion.
CREATE TABLE IF NOT EXISTS weights_finalizations (
  request_id         TEXT    NOT NULL,
  attempt            INTEGER NOT NULL,
  instance_id        TEXT    NOT NULL,
  owner_scope        TEXT    NOT NULL,
  invocation_digest  TEXT    NOT NULL,
  output_slot        TEXT    NOT NULL,
  disposition        TEXT    NOT NULL,
  receipt_digest     TEXT    NOT NULL DEFAULT '',
  scratch_root_id    TEXT    NOT NULL DEFAULT '',
  result_outcome     TEXT    NOT NULL DEFAULT '',
  result_receipt_digest TEXT NOT NULL DEFAULT '',
  result_receipt_bytes  BLOB NOT NULL DEFAULT x'',
  recorded_at        TEXT    NOT NULL,
  completed_at       TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (request_id, invocation_digest, output_slot),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`}

// The ONE spelling of "still owes work or a terminal". Every lifecycle fence — the down
// refusal, the idle exit — reads these; a request in any other
// state is settled and an attempt in any other state is closed.
const (
	activeRequestStates = `'submitted','queued','dispatching','requeue_pending','finalizing','pausing','paused','blocked','canceling','releasing'`
	openAttemptStates   = `'preparing','offered','accepted','recovered_open','terminal'`
	// settledRequestStates is the SQL spelling of settledRequestState: a row here owes
	// nothing and no later observation may contradict it.
	settledRequestStates = `'succeeded','failed','canceled','refused','abandoned'`
)

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// NewID mints an opaque local id. Attempt keys are opaque on purpose: a triage bundle is
// served by attempt key, never by a path a client can shape.
func NewID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + "-" + hex.EncodeToString(b[:])
}

// --------------------------------------------------------------------------- workers

// WorkerProcess is one package worker: its OS process-birth identity, the protocol
// identities it reported, and the install-scoped device grant it holds. The grant is
// this row's `Devices` field — one process, one visible device set, one install.
type WorkerProcess struct {
	InstanceID string
	Package    string
	InstallID  string
	WorkerID   string
	Devices    []string
	PID        int
	Birth      string // the OS process-birth identity: /proc starttime, never the pid alone
	SessionID  string
	State      string // spawned_without_birth | spawned | registered | closed
	OpenedAt   string
}

func deviceMark(d string) string { return "|" + d + "|" }

func deviceList(devices []string) string {
	var sb strings.Builder
	for _, d := range devices {
		sb.WriteString(deviceMark(d))
	}
	return sb.String()
}

// SpawnWorker journals a worker process AND its device grant in one atomic statement.
// The WHERE NOT EXISTS clause is the whole arbitration: no live process may already hold
// any device in this envelope. Two concurrent starts race one statement, and exactly one
// row appears.
func (s *Store) SpawnWorker(w WorkerProcess) *exit.Error {
	clauses := make([]string, 0, len(w.Devices))
	args := []any{
		w.InstanceID, w.Package, nullable(w.InstallID), w.WorkerID,
		deviceList(w.Devices), w.PID, w.Birth, "spawned_without_birth", now(),
	}
	for _, d := range w.Devices {
		clauses = append(clauses, "w.devices LIKE ?")
		args = append(args, "%"+deviceMark(d)+"%")
	}
	if len(clauses) == 0 {
		// A CPU-only process owns no accelerator. Keeping a failed CPU job's
		// state must not block another revision on a fabricated device grant.
		clauses = append(clauses, "0")
	}
	// THE ADMISSION ASKS ABOUT ANOTHER PROCESS, which is why the slot's own row is
	// excluded. A slot restarting itself — the recovered-attempts path, where a dead
	// worker's journal must be replayed by a worker in the SAME slot — is not a second
	// consumer of the envelope, and refusing it deadlocked exactly the recovery it was
	// meant to protect. Concurrency within one slot is serialized by `selectOrStart`;
	// this statement is the fence against a DIFFERENT slot.
	args = append(args, w.InstanceID)
	res, err := s.db.Exec(`
		INSERT INTO worker_processes(instance_id,package,install_id,worker_id,
		  devices,pid,birth,state,opened_at)
		SELECT ?,?,?,?,?,?,?,?,?
		WHERE NOT EXISTS (
		  SELECT 1 FROM worker_processes w WHERE w.state != 'closed' AND (`+
		strings.Join(clauses, " OR ")+`) AND w.instance_id != ?)
		ON CONFLICT(instance_id) DO UPDATE SET
		  pid=excluded.pid, birth=excluded.birth, devices=excluded.devices,
		  install_id=excluded.install_id,
		  session_id=NULL,
		  state='spawned_without_birth', opened_at=excluded.opened_at, closed_at=''`, args...)
	if err != nil {
		return exit.Internalf("cannot journal the worker process %s: %s", w.InstanceID, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		held, _ := s.DeviceHolders(w.Devices)
		return exit.Named(exit.Conflict, "device_envelope_held",
			"the device envelope [%s] is already granted to %s",
			strings.Join(w.Devices, ","), strings.Join(held, ",")).
			WithRemedy("one process per device: stop the holding worker first").
			WithNext("cozy run list")
	}
	return nil
}

// AttachWorker journals a worker this host did not spawn (a rented pod's, cl-015). It is
// SpawnWorker minus the device arbitration, and that subtraction is the whole reason it
// exists: the envelope fence arbitrates a scarce LOCAL resource, and a process on someone
// else's machine holds none of it. Routing an attach through the fence refused it with
// `a worker process needs a device envelope` — a local rule applied to a non-local fact.
//
// The identity fence is still here: the instance id is the primary key, so re-attaching
// one rental lands on its own row rather than accumulating one per request.
func (s *Store) AttachWorker(w WorkerProcess) *exit.Error {
	if _, err := s.db.Exec(`
		INSERT INTO worker_processes(instance_id,package,install_id,worker_id,
		  devices,pid,birth,state,opened_at)
		VALUES(?,?,?,?,'',0,'','spawned',?)
		ON CONFLICT(instance_id) DO UPDATE SET
		  install_id=excluded.install_id,
		  session_id=NULL,
		  state='spawned', opened_at=excluded.opened_at, closed_at=''`,
		w.InstanceID, w.Package, nullable(w.InstallID), w.WorkerID,
		now()); err != nil {
		return exit.Internalf("cannot journal the attached worker %s: %s", w.InstanceID, err)
	}
	return nil
}

func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// DeviceHolders names the live processes holding any of these devices.
func (s *Store) DeviceHolders(devices []string) ([]string, *exit.Error) {
	if len(devices) == 0 {
		return nil, nil
	}
	clauses := make([]string, 0, len(devices))
	args := make([]any, 0, len(devices))
	for _, d := range devices {
		clauses = append(clauses, "devices LIKE ?")
		args = append(args, "%"+deviceMark(d)+"%")
	}
	rows, err := s.db.Query(`SELECT instance_id FROM worker_processes
		WHERE state != 'closed' AND (`+strings.Join(clauses, " OR ")+`)`, args...)
	if err != nil {
		return nil, exit.Internalf("cannot read the device ledger: %s", err)
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot read a device ledger row: %s", err)
		}
		out = append(out, id)
	}
	return out, nil
}

// BindSession records what Register reported. session_id is the WORKER's to mint; the
// orchestrator only binds it, and the unique index refuses two live workers sharing one.
func (s *Store) BindSession(instanceID, sessionID string) *exit.Error {
	res, err := s.db.Exec(`UPDATE worker_processes
		SET session_id=?, state='registered'
		WHERE instance_id=? AND state != 'closed'`, sessionID, instanceID)
	if err != nil {
		return exit.New(exit.Conflict, "cannot bind session %s to %s: %s", sessionID, instanceID, err).
			WithRemedy("a session_id is bound to at most one live worker")
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "no live worker process %s to bind session %s to",
			instanceID, sessionID)
	}
	return nil
}

// WorkerStarted attaches the OS process-birth identity to a row whose device grant was
// already journaled. The grant comes FIRST and the process second: a process that was
// never admitted cannot exist, and the identity that proves it is the kernel's.
func (s *Store) WorkerStarted(instanceID string, pid int, birth string) *exit.Error {
	if pid <= 0 || birth == "" {
		return exit.Named(exit.Conflict, "worker_birth_identity_missing",
			"worker %s started without a complete OS process-birth identity", instanceID).
			WithRemedy("stop the child before releasing its device grant")
	}
	result, err := s.db.Exec(`UPDATE worker_processes SET pid=?, birth=?, state='spawned'
		WHERE instance_id=? AND worker_id='local' AND state='spawned_without_birth'`,
		pid, birth, instanceID)
	if err != nil {
		return exit.Internalf("cannot record the birth identity of %s: %s", instanceID, err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return exit.Named(exit.Conflict, "worker_birth_identity_not_pending",
			"worker %s has no pending local birth identity", instanceID).
			WithRemedy("only the launcher that reserved this worker may attach its OS identity")
	}
	return nil
}

// CloseWorker releases the process row AND, with it, its device grant.
func (s *Store) CloseWorker(instanceID string) *exit.Error {
	_, err := s.db.Exec(`UPDATE worker_processes SET state='closed', closed_at=?, session_id=NULL
		WHERE instance_id=?`, now(), instanceID)
	if err != nil {
		return exit.Internalf("cannot close the worker process %s: %s", instanceID, err)
	}
	return nil
}

// LiveWorkers is every process row this root still believes in. Restart reconciliation
// reads it and checks each against its OS process-birth identity before adopting.
func (s *Store) LiveWorkers() ([]WorkerProcess, *exit.Error) {
	rows, err := s.db.Query(`SELECT instance_id,package,COALESCE(install_id,''),
		worker_id,devices,pid,birth,COALESCE(session_id,''),state,opened_at FROM worker_processes WHERE state != 'closed'
		ORDER BY opened_at`)
	if err != nil {
		return nil, exit.Internalf("cannot list worker processes: %s", err)
	}
	defer rows.Close()
	var out []WorkerProcess
	for rows.Next() {
		var w WorkerProcess
		var devices string
		if err := rows.Scan(&w.InstanceID, &w.Package, &w.InstallID,
			&w.WorkerID, &devices, &w.PID, &w.Birth, &w.SessionID, &w.State, &w.OpenedAt); err != nil {
			return nil, exit.Internalf("cannot read a worker process row: %s", err)
		}
		for _, d := range strings.Split(strings.Trim(devices, "|"), "||") {
			if d != "" {
				w.Devices = append(w.Devices, d)
			}
		}
		out = append(out, w)
	}
	return out, nil
}

// --------------------------------------------------------------------------- requests

// Warning is one fact about a run that did not fail it.
type Warning struct {
	Code    string   `json:"code,omitempty"`
	Message string   `json:"message"`
	Fields  []string `json:"fields,omitempty"`
}

type Request struct {
	// Hub is the Tensorhub origin this request belongs to. Every hub operation for it
	// (resolution, rental acquisition, transfer, publication) addresses this origin.
	Hub string
	// RequestedRental is immutable caller affinity; Worker is the current assignment.
	RequestedRental string
	Capture         string
	// AttentionKernel is the optional execution-path pin carried into InvocationSpec.
	AttentionKernel string
	// Number is this host's short user-facing request reference. The globally unique ID
	// remains the durable internal/Hub identity; Number is derived from the retained local
	// request chronology and is never sent across the worker protocol.
	Number     int64
	ID         string
	IdemKey    string
	BodyDigest string
	Package    string
	Entrypoint string
	PlanID     string
	Release    string
	// LocalInstallationID names Creator's sealed carrier set. UploadedBootID binds the
	// completed transfer to the exact pod generation that acknowledged every file.
	LocalInstallationID        string
	LocalPackageUploadedBootID string
	InstallationID             string
	Payload                    []byte
	// Outputs names one destination per RESULT FIELD PATH. It lives on the request
	// because a REQUEUE re-derives the same grant shape without a client saying so again.
	Outputs   string
	State     string
	Ordinal   int64
	CreatedAt string
	// Kind is the ATTEMPT CLASS: `serving` or `job`. It is the one discriminator the
	// whole job branch hangs off, and it lives on the request because a requeue must
	// re-derive the same class without a client saying so again (cr-009: a job is an
	// attempt class on the one machinery, not a second runtime).
	Kind string
	// RetainWork preserves a unpublished package job's artifacts and capacity until the owner
	// resumes or permanently cancels it, including after an attempt fails.
	RetainWork bool
	// ReleaseImplicitWork is derived from a new captured result schema.
	// Its durable authority is successful_work_releases, not this admission-only field.
	ReleaseImplicitWork bool `json:"-"`
	// MachineExecutionObserver is admission-only. The marker is committed with
	// this request, before any scheduler can create a local attempt.
	MachineExecutionObserver bool   `json:"-"`
	DeadlineUnixMS           uint64 `json:"-"` // frozen submission event is its durable source
	// Warnings are admission-only: each becomes one request.warning event with the row.
	Warnings []Warning `json:"-"`
	// RetryOf names immutable predecessor history; ReuseScope identifies the
	// retained operation namespace shared by explicitly related revisions.
	RetryOf                string
	ReuseScope             string
	ControlRevision        uint64
	ParentRequestID        string
	ParentCallIndex        int64
	ChildIntentDigest      string
	ChildTargetDigest      string
	ChildReusable          bool
	ChildArtifacts         bool
	ReusedFrom             string
	OrchestrationDirective []byte
	// NeedsAccelerator is derived once from the selected package's immutable
	// dependency facts. It is not an author-supplied resource request.
	NeedsAccelerator bool
	// Org is the publishing org a job's scratch repo is named under. Empty for serving.
	Org string
	// Trees are the job's typed input TREES, `ref=dir` joined by commas. Each becomes
	// one grant input `tree:<ref>`; a field naming a ref the grant does not cover never
	// reaches a filesystem.
	Trees string
	// Worker names an ATTACHED remote worker (a rental id) this request must run on.
	// It lives on the request because a requeue must re-derive the same placement
	// without a client saying so again. Empty = any local worker.
	Worker string
	// Machine is the human machine word of the rental this request was bound to,
	// recorded when the word is known (acquisition start, claim, or pinned
	// submission) and kept as history after the rental row is gone (cl-107).
	// Empty = never bound to a rental.
	Machine string
	// Rental authorizes placement on Creator-managed rented capacity.
	Rental bool
	// RentalRequired forbids local placement for the hidden development/E2E override.
	RentalRequired bool
	// RentNew preserves fresh acquisition intent across queueing and restart.
	RentNew bool
	// InstallID pins the exact immutable local install resolved before
	// submission. The initial remote lane reads only its published release and PackageInterface.
	InstallID string
	// Assets are durable identity claims and borrowed source paths. Requeues reverify
	// those files; callers must keep their bytes available and unchanged.
	Assets []AssetBinding
	// Models are exact package-slot-to-model bindings resolved before rental creation.
	// They are local request state and are disclosed only after worker attachment.
	Models []ModelRef
	// WeightsOutputs is the immutable WeightsSink output subset projected beside the
	// InvocationSpec. Rev5 OutputBinding has no kind, so this may never be inferred from
	// ordinary asset outputs or from whichever receipts happen to arrive.
	WeightsOutputs string
	// OutputExport is a CLI-authenticated, PackageInterface-derived local publication intent.
	// It is recorded in its own durable row in the same transaction as this request.
	OutputExport *OutputExportIntent
	// ModelTransfer is a source materializer/output finalizer attached to this
	// ordinary request. It owns no lifecycle, placement, attempt, or event identity.
	ModelTransfer *ModelTransferIntent
	// PlannedSourceBytes is what a script ingest declares it will pull, so a rental
	// bought for it is sized to the ingest. Recorded at submission; read back through
	// Store.PlannedSourceBytes.
	PlannedSourceBytes int64 `json:"-"`
}

// AssetBinding ties one payload asset field to the exact local bytes the request owns.
// FieldPath is also the worker-protocol input_id (`first_frame`,
// `references.0.image`); position is identity for mixed reference lists, never a file
// name inferred later. LocalPath is resolution only and is excluded from submission
// identity; Digest, Length and MediaType are the claims inside InvocationSpec.
type ByteInputSnapshot struct {
	Body         []byte            `json:"body"`
	Reference    string            `json:"reference,omitempty"`
	Manifest     ArtifactObjectRef `json:"manifest"`
	ContentBytes int64             `json:"content_bytes"`
	Path         string            `json:"path"`
}

type AssetBinding struct {
	Snapshot  *ByteInputSnapshot `json:"snapshot,omitempty"`
	Native    *ByteAssetBinding  `json:"native,omitempty"`
	FieldPath string             `json:"field_path"`
	LocalPath string             `json:"local_path"`
	Digest    string             `json:"digest"`
	Length    int64              `json:"length"`
	MediaType string             `json:"media_type,omitempty"`
	Order     uint32             `json:"order"`
	MaxBytes  int64              `json:"max_bytes,omitempty"`
	// ModTime is the file's modification time, in Unix nanoseconds, when Digest was
	// computed. While the file keeps this time and Length, Digest still names its bytes.
	ModTime int64 `json:"mtime_ns,omitempty"`
}

// ModelRef is one exact user-selected model binding. Creator resolves the human
// spelling before it rents anything, records this row with the request, and sends it
// only to the attached worker in the desired download set.
type ModelRef struct {
	// Choice is the caller's own `model.<param>=` selection, left for the machine that runs
	// the call to resolve (worker-protocol ModelChoice); no ladder or rung is read for it.
	Choice bool `json:"choice,omitempty"`
	// Source is a Choice naming a provider checkpoint (hf://org/repo@commit, civitai://id)
	// that the machine resolves, narrows to Profiles (or the one it selects) and converts.
	Source   string            `json:"source,omitempty"`
	Profiles []string          `json:"profiles,omitempty"`
	Callable string            `json:"callable,omitempty"` // independently scheduled captured callable, qualified by package
	GPUs     int               `json:"gpus,omitempty"`     // exact selected execution group, zero is unspecified
	Adapters []ModelAdapterRef `json:"adapters,omitempty"`
	Package  string            `json:"package"`
	Slot     string            `json:"slot"`
	// BindingPath is the exact interface Model path used for package preparation.
	// Jobs keep Slot as the bare invocation parameter; serving already uses a path.
	BindingPath string `json:"binding_path,omitempty"`
	Model       string `json:"model"`
	// CatalogRepository is populated only by native local/Hub model resolution.
	// Model also carries synthetic producer/slot names for private artifacts;
	// those never gain repository authority from their spelling.
	CatalogRepository string `json:"catalog_repository,omitempty"`
	Release           string `json:"release"`
	Lane              string `json:"lane,omitempty"`
	Manifest          string `json:"manifest"`
	// HubCheckpoint records that Hub resolved an exact retained checkpoint without
	// a release. It routes downloads; the Hub still checks repository custody.
	HubCheckpoint bool `json:"hub_checkpoint,omitempty"`
	// ManifestLength is the manifest's exact byte length. Only a JOB declares its models
	// as invocation inputs (orchestrator.jobModels); a serving request's models reach
	// the worker through the placement's package-set lane, never as inputs.
	ManifestLength int64 `json:"manifest_length,omitempty"`
	// Bytes is the model tree's size as Tensorhub publishes it on the release card — the
	// hub fact the capacity decision ranks a rental's missing download by (§3.2). Zero
	// when the resolver did not carry it.
	Bytes int64 `json:"bytes,omitempty"`
	// ComponentBytes is the pinned lane's per-component stored bytes as the card publishes
	// them (th-181), and ComponentUse the slot's method → components map from the package
	// interface. Together they size what one entrypoint holds resident (cl-168).
	ComponentBytes map[string]int64    `json:"component_bytes,omitempty"`
	ComponentUse   map[string][]string `json:"component_use,omitempty"`
	// Ladder is the hub binding's FIT MAP (cl-166): which lane belongs on which GPU
	// class, in the owner's order, each rung resolved to the exact manifest the model
	// card publishes for that lane. A ref carrying a Ladder and no Manifest is UNPINNED:
	// the machine decision picks the rung whose gpu pattern matches the winning
	// accelerator and pins Lane/Manifest/Bytes from it. A pinned ref keeps the ladder as
	// the owner's word on where its lane fits (cl-170): an explicit override is not a rung.
	Ladder []ModelRung `json:"ladder,omitempty"`
	// SharedSlots are the OTHER interface slots this selection also binds, under the same
	// bytes (h3a-018): sibling entrypoints whose slot declares the same model class and
	// whose hub default names the same model release and ladder. The download set selects
	// the model under every one of them, so the pod prepares ONE placement carrying every
	// entrypoint of the construction and a switch between them is a dispatch, not a
	// prepare. Sizing reads Slot alone: the siblings share the construction, they do not
	// add to it.
	SharedSlots []string `json:"shared_slots,omitempty"`
}

// ModelAdapterRef is an ordered, exact adapter checkpoint for one base component.
// Scales are canonical decimal strings, so zero is distinct from omission.
type ModelAdapterRef struct {
	Component       string `json:"component"`
	Model           string `json:"model"`
	Release         string `json:"release,omitempty"`
	Lane            string `json:"lane,omitempty"`
	Manifest        string `json:"manifest"`
	ManifestLength  int64  `json:"manifest_length,omitempty"`
	SourceComponent string `json:"source_component"`
	Scale           string `json:"scale"`
	Bytes           int64  `json:"bytes,omitempty"`
}

// SameAdapters compares execution/custody identity, excluding sizing observations.
func SameAdapters(a, b []ModelAdapterRef) bool {
	if len(a) != len(b) {
		return false
	}
	for i, left := range a {
		right := b[i]
		if left.Component != right.Component || left.Model != right.Model || left.Release != right.Release ||
			left.Lane != right.Lane || left.Manifest != right.Manifest || left.SourceComponent != right.SourceComponent || left.Scale != right.Scale {
			return false
		}
	}
	return true
}

// BindingSlot is the slot's one identity: its package interface model path. A job keeps
// Slot as the bare invocation parameter and carries the path beside it.
func (m ModelRef) BindingSlot() string {
	if m.BindingPath != "" {
		return m.BindingPath
	}
	return m.Slot
}

// OneSelectionPerSlot joins selections in precedence order and keeps the first for each
// package slot, counting the slots a selection shares. The request's own selection, made
// once as an explicit override or its resolved default, therefore supersedes a callee
// default for the same slot, and a repeated contribution adds nothing.
func OneSelectionPerSlot(groups ...[]ModelRef) []ModelRef {
	var out []ModelRef
	held := map[string]bool{}
	for _, models := range groups {
		for _, model := range models {
			if held[model.Package+"\x00"+model.BindingSlot()] {
				continue
			}
			for _, slot := range append([]string{model.BindingSlot()}, model.SharedSlots...) {
				held[model.Package+"\x00"+slot] = true
			}
			out = append(out, model)
		}
	}
	return out
}

// ModelRung is one GPU class, execution group and lane resolved against the model card.
type ModelRung struct {
	GPUs           int              `json:"gpus,omitempty"`
	GPU            string           `json:"gpu"`
	Lane           string           `json:"lane"`
	Manifest       string           `json:"manifest"`
	Bytes          int64            `json:"bytes"`
	ComponentBytes map[string]int64 `json:"component_bytes,omitempty"`
}

func (r ModelRung) String() string {
	if r.GPUs > 0 {
		return fmt.Sprintf("%dx%s=%s", r.GPUs, r.GPU, r.Lane)
	}
	return r.GPU + "=" + r.Lane
}

// Pinned says the ref names one exact manifest; an unpinned ref still carries its ladder.
func (m ModelRef) Pinned() bool { return m.Manifest != "" }

// RungAt is the ref's rung, and its index, in a group `width` cards wide on `accelerator`:
// its first rung of exactly that width, else its first uncounted rung, which serves any
// width. A pinned ref is its own rung (index 0): at any width when uncounted, else only at
// its exact group, which its ladder, if any, must author on this accelerator. An empty
// accelerator — a host without an NVIDIA device — matches only a "*" rung.
func (m ModelRef) RungAt(accelerator string, width int) (ModelRung, int, bool) {
	if m.Source != "" {
		return m.sourceRung()
	}
	if m.Pinned() {
		authored := m.GPUs == 0 || len(m.Ladder) == 0
		for _, rung := range m.Ladder {
			authored = authored || rung.GPUs == m.GPUs && rung.Lane == m.Lane && RungMatches(rung.GPU, accelerator)
		}
		return ModelRung{GPU: "*", GPUs: m.GPUs, Lane: m.Lane, Manifest: m.Manifest, Bytes: m.Bytes,
			ComponentBytes: m.ComponentBytes}, 0, authored && (m.GPUs == 0 || m.GPUs == width)
	}
	uncounted := -1
	for i, rung := range m.Ladder {
		switch {
		case !RungMatches(rung.GPU, accelerator):
		case width > 0 && rung.GPUs == width:
			return rung, i, true
		case rung.GPUs == 0 && uncounted < 0:
			uncounted = i
		}
	}
	if uncounted < 0 {
		return ModelRung{}, -1, false
	}
	return m.Ladder[uncounted], uncounted, true
}

// Width is the device group a pinned selection takes on a machine of `machine` cards: its
// widest exact group, or the whole machine when no ref is counted.
func Width(models []ModelRef, machine int) int {
	width := 0
	for _, model := range models {
		width = max(width, model.GPUs)
	}
	if width == 0 {
		return machine
	}
	return width
}

// PurchaseRung only buys a wider machine when that exact group is authored.
func (m ModelRef) PurchaseRung(accelerator string, count int) (ModelRung, int, bool) {
	if m.Source != "" {
		return m.sourceRung()
	}
	if m.Pinned() {
		rung, index, ok := m.RungAt(accelerator, m.GPUs)
		return rung, index, ok && m.GPUs <= count
	}
	for i, rung := range m.Ladder {
		if rung.GPUs == count && RungMatches(rung.GPU, accelerator) {
			return rung, i, true
		}
	}
	for i, rung := range m.Ladder {
		if rung.GPUs <= 1 && RungMatches(rung.GPU, accelerator) {
			return rung, i, true
		}
	}
	return ModelRung{}, -1, false
}

// sourceRung is a provider source's only rung: it has no ladder, and the machine that makes
// it sizes it, so it fits any accelerator at its authored width.
func (m ModelRef) sourceRung() (ModelRung, int, bool) {
	return ModelRung{GPU: "*", GPUs: m.GPUs}, 0, true
}

// Pin returns the ref bound to one rung, its ladder kept.
func (m ModelRef) Pin(rung ModelRung) ModelRef {
	m.GPUs = rung.GPUs
	m.Lane, m.Manifest, m.Bytes, m.ComponentBytes = rung.Lane, rung.Manifest, rung.Bytes, rung.ComponentBytes
	return m
}

// How a machine-class decision sized the device against one slot (cl-168, cl-170).
const (
	// FitComponents: the card published every component's bytes, so the need is the
	// entrypoint's largest resident group and the device is held to it.
	FitComponents = "components"
	// FitRungAsserted: no component bytes, but a rung of the owner's ladder names this
	// accelerator AND this lane. That rung asserts the lane fits; no figure is compared.
	FitRungAsserted = "rung_asserted"
	// FitLaneBytes: no component bytes and no rung asserting this lane on this
	// accelerator — an explicit override off the ladder, or bound to none — so the whole
	// lane's bytes are the need, conservatively.
	FitLaneBytes = "lane_bytes"
	// FitDeriveOnly: the request is a JOB, whose Model inputs cozy-runtime never
	// constructs — they are derive-only Manifest capabilities with no load, no component
	// scope and no residency (cozy-runtime `internal/worker/plan.py` JobBinding) — so no
	// component of one is ever on the device and no figure is compared.
	FitDeriveOnly = "derive_only"
)

// Residency is what a pinned selection must hold on one device at once, and the rule that
// sized it — per slot, distinct rules joined by "+".
type Residency struct {
	Bytes int64
	Fit   string
	// Need names each slot's figure: `condition_text: text_encoder`, the largest single
	// component when the slot declares no component_use, or `lane <lane>` for whole-lane bytes.
	Need string
	// Weights is the resident figure alone. Bytes either adds the legacy Working
	// peak or takes the maximum with an exactly matched ObservedTotal. Separate run
	// counts keep an unknown measurement distinct from measured zero.
	Weights           int64
	Working           int64
	WorkingRuns       int
	ObservedTotal     int64
	ObservedTotalRuns int
}

// Resident sizes the pinned selection against ONE accelerator, at every rental width
// (cl-179): nothing here is tensor- or pipeline-parallel, so under a sequence-parallel
// group every GPU holds the whole selection and a K-card pod holds exactly what one of
// its cards holds. A width is latency and activation headroom, never capacity.
//
// It sizes that one accelerator the way cozy-runtime loads it: per slot, the largest sum
// over the slot's component_use groups (a method stages only
// the components it names), or the largest single component when the slot declares none;
// with no component bytes, the owner's rung for this accelerator and lane asserts the fit
// with no figure, and failing that the lane's whole bytes stand in. Summed over slots,
// because one callable holds its slots at once. Independently scheduled captured
// callables use their maximum resident group, not the sum across the workflow.
//
// `job` sizes the SAME selection as a job's inputs instead, and there the answer is
// nothing: cozy-runtime hands a job a derive-only view of the Manifest and refuses load
// and component access on it, so a job's model never reaches the device however large its
// closure is. Sizing one by the components a SERVING construction would stage is a figure
// about a different run (cl-180).
func Resident(models []ModelRef, accelerator string, job bool) Residency {
	groups := map[string][]ModelRef{}
	for _, model := range models {
		groups[model.Callable] = append(groups[model.Callable], model)
	}
	if len(groups) > 1 || (len(groups) == 1 && len(groups[""]) == 0) {
		var peak Residency
		// Callable groups execute independently; slots inside each group are simultaneous.
		for _, key := range slices.Sorted(maps.Keys(groups)) {
			current := residentTogether(groups[key], accelerator, job && key == "")
			if current.Bytes > peak.Bytes || peak.Fit == "" {
				peak = current
			}
		}
		return peak
	}
	return residentTogether(models, accelerator, job)
}

func residentTogether(models []ModelRef, accelerator string, job bool) Residency {
	var out Residency
	var fits, needs []string
	for _, model := range models {
		bytes, need, fit := model.resident(accelerator, job)
		if !job {
			for _, adapter := range model.Adapters {
				bytes = addResidencyBytes(bytes, adapter.Bytes)
			}
		}
		out.Bytes = addResidencyBytes(out.Bytes, bytes)
		if need != "" {
			needs = append(needs, need)
		}
		if !slices.Contains(fits, fit) {
			fits = append(fits, fit)
		}
	}
	out.Fit, out.Need = strings.Join(fits, "+"), strings.Join(needs, "; ")
	return out
}

func addResidencyBytes(current, additional int64) int64 {
	if current < 0 || additional < 0 || additional > math.MaxInt64-current {
		return math.MaxInt64
	}
	return current + additional
}

func (m ModelRef) resident(accelerator string, job bool) (int64, string, string) {
	if job {
		return 0, "", FitDeriveOnly
	}
	if len(m.ComponentBytes) > 0 {
		bytes, need := m.largestGroup()
		return bytes, need, FitComponents
	}
	for _, rung := range m.Ladder {
		if rung.Lane == m.Lane && RungMatches(rung.GPU, accelerator) {
			return 0, "", FitRungAsserted
		}
	}
	return m.Bytes, "lane " + m.Lane, FitLaneBytes
}

func (m ModelRef) largestGroup() (int64, string) {
	var best int64
	need := ""
	for _, method := range slices.Sorted(maps.Keys(m.ComponentUse)) {
		components := m.ComponentUse[method]
		if len(components) == 0 {
			continue
		}
		var sum int64
		for _, component := range components {
			sum += m.ComponentBytes[component]
		}
		if need == "" || sum > best {
			best, need = sum, method+": "+strings.Join(components, "+")
		}
	}
	if need != "" {
		return best, need
	}
	for _, component := range slices.Sorted(maps.Keys(m.ComponentBytes)) {
		if bytes := m.ComponentBytes[component]; need == "" || bytes > best {
			best, need = bytes, component
		}
	}
	return best, need
}

// Lanes renders the pinned selection for a log line or audit row: the lane alone for one
// slot, `slot=lane` pairs for several.
func Lanes(models []ModelRef) string {
	if len(models) == 1 {
		return models[0].Lane
	}
	parts := make([]string, 0, len(models))
	for _, model := range models {
		parts = append(parts, model.BindingSlot()+"="+model.Lane)
	}
	return strings.Join(parts, ",")
}

const requestCols = `id,idem_key,body_digest,package,entrypoint,plan_id,package_release,
	local_installation_id,local_package_uploaded_boot_id,
	installation_id,payload,outputs,
	state,ordinal,created_at,kind,needs_accelerator,org,trees,worker,machine,rental,rental_required,
	COALESCE(install_id,''),assets,capture,attention_kernel,models,weights_outputs,retain_work,retry_of,reuse_scope,control_revision,
	parent_request_id,parent_call_index,child_intent_digest,child_target_digest,child_reusable,reused_from,orchestration_directive,child_artifacts,requested_rental,rent_new,hub`

func requestScanTargets(r *Request, assets, models *string) []any {
	return []any{&r.ID, &r.IdemKey, &r.BodyDigest, &r.Package, &r.Entrypoint, &r.PlanID,
		&r.Release, &r.LocalInstallationID,
		&r.LocalPackageUploadedBootID, &r.InstallationID, &r.Payload, &r.Outputs,
		&r.State, &r.Ordinal, &r.CreatedAt,
		&r.Kind, &r.NeedsAccelerator, &r.Org, &r.Trees, &r.Worker, &r.Machine, &r.Rental, &r.RentalRequired,
		&r.InstallID, assets, &r.Capture, &r.AttentionKernel, models, &r.WeightsOutputs, &r.RetainWork, &r.RetryOf, &r.ReuseScope, &r.ControlRevision,
		&r.ParentRequestID, &r.ParentCallIndex, &r.ChildIntentDigest, &r.ChildTargetDigest, &r.ChildReusable, &r.ReusedFrom, &r.OrchestrationDirective, &r.ChildArtifacts, &r.RequestedRental, &r.RentNew, &r.Hub}
}

func finishRequestScan(r Request, assets, models string, err error) (Request, error) {
	if err == nil && assets != "" {
		err = json.Unmarshal([]byte(assets), &r.Assets)
	}
	if err == nil && models != "" {
		err = json.Unmarshal([]byte(models), &r.Models)
	}
	return r, err
}

func scanRequest(row interface{ Scan(...any) error }) (Request, error) {
	var r Request
	var assets, models string
	err := row.Scan(requestScanTargets(&r, &assets, &models)...)
	return finishRequestScan(r, assets, models, err)
}

func scanNumberedRequest(row interface{ Scan(...any) error }) (Request, error) {
	var r Request
	var assets, models string
	targets := append([]any{&r.Number}, requestScanTargets(&r, &assets, &models)...)
	err := row.Scan(targets...)
	return finishRequestScan(r, assets, models, err)
}

// IsJob answers the attempt class. The default spelling is `serving` so a row written
// before the column existed reads as what it was.
func (r Request) IsJob() bool { return r.Kind == "job" }

// OwnModels are this caller's model inputs. Captured defaults can belong to callees,
// imported or of the same package; they remain on the request for child resolution but
// are not its inputs. A selection binds the caller when its package is the caller's and
// one of its slots is under the caller's entrypoint. Missing package or slot attribution
// is not proof that an input belongs elsewhere.
func (r Request) OwnModels() []ModelRef {
	var models []ModelRef
	for _, model := range r.Models {
		if (model.Package == "" || model.Package == r.Package) && r.bindsOwnSlot(model) {
			models = append(models, model)
		}
	}
	return models
}

func (r Request) bindsOwnSlot(model ModelRef) bool {
	for _, slot := range append([]string{model.BindingSlot()}, model.SharedSlots...) {
		entrypoint, _, attributed := strings.Cut(slot, ".models.")
		if !attributed || r.Entrypoint == "" || entrypoint == r.Entrypoint {
			return true
		}
	}
	return false
}

// SizedByOwnModels says the request's own execution holds its model slots on a device, so
// its rental is chosen for them and never for its callees' defaults: a serving callable
// constructs its slots, and a job's Model inputs are derive-only unless its package itself
// needs an accelerator.
func (r Request) SizedByOwnModels() bool {
	return len(r.OwnModels()) > 0 && (r.NeedsAccelerator || !r.IsJob())
}

// ComposesChildren says the request's own execution may host managed children that need
// capacity of their own. A request sized by its own slots holds them itself, and a
// weights-producing job runs on Runtime's ordinary job slot, beneath which Runtime refuses
// every model-bearing child ("managed child cannot coexist with an ancestor on the device
// lane", worker/machine_calls.py `_check_ancestors`).
func (r Request) ComposesChildren() bool {
	return !r.SizedByOwnModels() && !(r.IsJob() && r.WeightsOutputs != "" && r.WeightsOutputs != "[]")
}

// RequestRow reads one request back. Requeue re-derives its dispatch from this row and
// from nothing a caller has to repeat.
func (s *Store) RequestRow(id string) (*Request, *exit.Error) {
	r, err := scanRequest(s.db.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read request %s: %s", id, err)
	}
	transfer, problem := s.ModelTransferOf(id)
	if problem != nil {
		return nil, problem
	}
	if transfer != nil {
		r.ModelTransfer = &transfer.ModelTransferIntent
		if transfer.HasAcquisition() {
			r.Models = append([]ModelRef(nil), transfer.Models...)
		}
	}
	return &r, nil
}

// RequestByReference resolves the LOCAL short number or the globally unique durable id.
// Internal lifecycle code continues to use RequestRow so a number can never cross into the
// worker/Hub identity plane by accident.
func (s *Store) RequestByReference(reference string) (*Request, *exit.Error) {
	reference = strings.TrimSpace(reference)
	if number, err := strconv.ParseInt(reference, 10, 64); err == nil && number > 0 &&
		strconv.FormatInt(number, 10) == reference {
		var id string
		err := s.db.QueryRow(`SELECT id FROM requests ORDER BY created_at,id LIMIT 1 OFFSET ?`,
			number-1).Scan(&id)
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		if err != nil {
			return nil, exit.Internalf("cannot resolve local request number %d: %s", number, err)
		}
		row, problem := s.RequestRow(id)
		if row != nil {
			row.Number = number
		}
		return row, problem
	}
	row, problem := s.RequestRow(reference)
	if problem != nil || row == nil {
		return row, problem
	}
	number, err := requestNumber(s.db, *row)
	if err != nil {
		return nil, exit.Internalf("cannot number request %s: %s", row.ID, err)
	}
	row.Number = number
	return row, nil
}

func requestNumber(q interface{ QueryRow(string, ...any) *sql.Row }, row Request) (int64, error) {
	var number int64
	err := q.QueryRow(`SELECT COUNT(*) FROM requests
		WHERE created_at < ? OR (created_at = ? AND id <= ?)`,
		row.CreatedAt, row.CreatedAt, row.ID).Scan(&number)
	return number, err
}

// BindRequestPlan records the binding a rental request routes on once the rented worker
// has resolved it. A published request that names models freezes no plan at submit; the
// drain re-reads the row before every route, so a plan learned only in memory never
// reaches the route and the worker's DISPATCHABLE placement stays invisible to its own
// request (found live: the queue re-prepared the same package on a busy pod).
// Until its first attempt, a queued request may learn a newer binding for its
// frozen logical selection. Once dispatch has recorded any attempt, even an
// aborted or refused one, the binding remains immutable across retries.
func (s *Store) BindRequestPlan(id, planID string) *exit.Error {
	if id == "" || planID == "" {
		return exit.Internalf("cannot bind an empty request plan")
	}
	result, err := s.db.Exec(`UPDATE requests SET plan_id=? WHERE id=? AND (
		plan_id=? OR (state IN ('submitted','queued') AND ordinal=0 AND
		NOT EXISTS (SELECT 1 FROM attempts WHERE request_id=requests.id)))`,
		planID, id, planID)
	if err != nil {
		return exit.Internalf("cannot bind request %s plan: %s", id, err)
	}
	if changed, err := result.RowsAffected(); err == nil && changed == 1 {
		return nil
	}
	var held string
	if err := s.db.QueryRow(`SELECT plan_id FROM requests WHERE id=?`, id).Scan(&held); err != nil {
		return exit.Internalf("cannot read request %s plan: %s", id, err)
	}
	if held != planID {
		return exit.Named(exit.Conflict, "request_invocation_identity_changed",
			"request %s already binds plan %s, not %s", id, held, planID)
	}
	return nil
}

// BindRemoteInvocation records the exact invocation identity learned from the worker
// after logical package_set resolution. The client never supplies these values.
func (s *Store) BindRemoteInvocation(id, planID, installation string) *exit.Error {
	if id == "" || planID == "" || installation == "" {
		return exit.Internalf("cannot bind an incomplete remote invocation identity")
	}
	result, err := s.db.Exec(`UPDATE requests SET plan_id=?,installation_id=?
		WHERE id=? AND (plan_id='' OR plan_id=?) AND installation_id=''`,
		planID, installation, id, planID)
	if err != nil {
		return exit.Internalf("cannot bind request %s remote invocation: %s", id, err)
	}
	changed, err := result.RowsAffected()
	if err != nil {
		return exit.Internalf("cannot read request %s remote invocation binding result: %s", id, err)
	}
	if changed == 1 {
		return nil
	}
	var heldPlan, heldInstallation string
	if err := s.db.QueryRow(`SELECT plan_id,installation_id FROM requests WHERE id=?`, id).Scan(
		&heldPlan, &heldInstallation); err != nil {
		return exit.Internalf("cannot read request %s remote invocation binding: %s", id, err)
	}
	if heldPlan != planID || heldInstallation != installation {
		return exit.Named(exit.Conflict, "request_invocation_identity_changed",
			"request %s already binds a different worker-derived invocation identity", id)
	}
	return nil
}

// MarkLocalPackageUploaded crosses the durable boundary between verified carrier
// acknowledgements and DesiredLocalPackageSet. The caller proves every acknowledgement belongs
// to this exact boot/session before moving the marker; a replacement boot therefore overwrites an
// old marker only after it has independently re-received and verified the whole revision.
func (s *Store) MarkLocalPackageUploaded(id, digest, bootID string) *exit.Error {
	if id == "" || digest == "" || bootID == "" {
		return exit.Internalf("cannot record an incomplete local package upload")
	}
	result, err := s.db.Exec(`UPDATE requests SET local_package_uploaded_boot_id=?
		WHERE id=? AND local_installation_id=?`, bootID, id, digest)
	if err != nil {
		return exit.Internalf("cannot record request %s local package upload: %s", id, err)
	}
	changed, err := result.RowsAffected()
	if err != nil {
		return exit.Internalf("cannot read request %s local package upload result: %s", id, err)
	}
	if changed == 1 {
		return nil
	}
	var heldDigest string
	if err := s.db.QueryRow(`SELECT local_installation_id FROM requests WHERE id=?`, id).
		Scan(&heldDigest); err != nil {
		return exit.Internalf("cannot read request %s local package upload: %s", id, err)
	}
	if heldDigest != digest {
		return exit.Named(exit.Conflict, "local_package_revision_changed",
			"request %s already names another local package revision", id)
	}
	return exit.Named(exit.Conflict, "local_package_request_changed",
		"request %s cannot record its verified local package boot", id)
}

// PinRental pins one still-queued --rental request to the rental routing chose for it:
// the argmin lane at dispatch, or the rental the capacity decision stages its placement
// on (cl-092 step 4). A cancellation that wins first leaves worker empty, which tells a
// caller that bought the rental to release it. The pin also records the rental's machine
// word on the request (cl-107): the word is history the run keeps after the rental row
// is gone, never a read-time join.
func (s *Store) PinRental(id, rentalID string, models []ModelRef) (bool, *exit.Error) {
	// The pin and the lane are ONE decision (cl-166): the machine decides the rung, so
	// the pinned selection lands in the same statement as the worker. Nil models keep
	// the row's selection (a request that was already exact).
	set := ""
	args := []any{rentalID, rentalID}
	if models != nil {
		encoded, err := json.Marshal(models)
		if err != nil {
			return false, exit.Internalf("cannot encode request %s models: %s", id, err)
		}
		set, args = ", models=?", append(args, string(encoded))
	}
	result, err := s.db.Exec(`UPDATE requests SET worker=?,
		machine=COALESCE((SELECT machine_name FROM rentals WHERE id=?),machine)`+set+`
		WHERE id=? AND rental=1 AND worker='' AND (requested_rental='' OR requested_rental=?) AND
		state IN ('submitted','queued','requeue_pending') AND NOT EXISTS(SELECT 1 FROM rentals WHERE id=? AND state IN ('release_requested','released','failed'))`, append(args, id, rentalID, rentalID)...)
	if err != nil {
		return false, exit.Internalf("cannot assign request %s to rental %s: %s", id, rentalID, err)
	}
	changed, err := result.RowsAffected()
	if err != nil {
		return false, exit.Internalf("cannot read rental assignment for request %s: %s", id, err)
	}
	if changed == 1 {
		return true, nil
	}
	if problem := refuseReleasedRental(s.db, rentalID); problem != nil {
		return false, problem
	}
	row, problem := s.RequestRow(id)
	if problem != nil {
		return false, problem
	}
	return row != nil && row.Rental && row.Worker == rentalID &&
		!settledRequestState(row.State), nil
}

// PinRequestModels writes an unassigned --rental request's selection before the paid ask,
// so the rental POST declares the lane the machine decision chose (th-155). A later
// rung — the hub refused the first for inventory — overwrites it.
func (s *Store) PinRequestModels(id string, models []ModelRef) *exit.Error {
	encoded, err := json.Marshal(models)
	if err != nil {
		return exit.Internalf("cannot encode request %s models: %s", id, err)
	}
	if _, err := s.db.Exec(`UPDATE requests SET models=? WHERE id=? AND rental=1 AND worker=''
		AND state IN ('submitted','queued','requeue_pending')`, string(encoded), id); err != nil {
		return exit.Internalf("cannot record request %s models: %s", id, err)
	}
	return nil
}

// PinMachineModels records the rungs the executing machine's measured devices fixed, before
// its submission is frozen.
func (s *Store) PinMachineModels(id string, models []ModelRef) *exit.Error {
	encoded, err := json.Marshal(models)
	if err != nil {
		return exit.Internalf("cannot encode request %s models: %s", id, err)
	}
	if _, err := s.db.Exec(`UPDATE requests SET models=? WHERE id=?
		AND state IN ('submitted','queued','requeue_pending')`, string(encoded), id); err != nil {
		return exit.Internalf("cannot record request %s models: %s", id, err)
	}
	return nil
}

func settledRequestState(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// ReleaseCanceledIdempotencyKey frees a key whose run ended without an
// outcome worth replaying: an explicit cancellation withdrew the intent, and a
// failure already delivered its refusal once — pinning either to the key
// forever leaves an identical resubmission no honest path after the cause is
// fixed. Succeeded runs keep replaying; that is what idempotency is for. The
// row keeps its history under a derived key that can never collide with a
// caller key (callers never contain "\x00").
func (s *Store) ReleaseCanceledIdempotencyKey(key string) (bool, *exit.Error) {
	result, err := s.db.Exec(
		`UPDATE requests SET idem_key = idem_key || char(0) || id WHERE idem_key=? AND state IN ('canceled','failed','refused')`, key)
	if err != nil {
		return false, exit.Internalf("cannot release canceled idempotency key: %s", err)
	}
	n, err := result.RowsAffected()
	if err != nil {
		return false, exit.Internalf("cannot release canceled idempotency key: %s", err)
	}
	return n == 1, nil
}

// RequestByIdempotencyKey resolves the durable identity before a retry touches any
// caller-owned resources. In particular, a settled request's original and staged asset
// files may both be gone; its recorded semantic body is still the answer for that key.
func (s *Store) RequestByIdempotencyKey(key string) (*Request, *exit.Error) {
	r, err := scanRequest(s.db.QueryRow(`SELECT `+requestCols+` FROM requests WHERE idem_key=?`, key))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read request for idempotency key %s: %s", key, err)
	}
	transfer, problem := s.ModelTransferOf(r.ID)
	if problem != nil {
		return nil, problem
	}
	if transfer != nil {
		r.ModelTransfer = &transfer.ModelTransferIntent
		if transfer.HasAcquisition() {
			r.Models = append([]ModelRef(nil), transfer.Models...)
		}
	}
	return &r, nil
}

// Requests lists rows newest-first, optionally filtered by state and package. Empty
// filters select every value; the client contract's listing route reads exactly this.
func (s *Store) Requests(state, packageName string, limit int) ([]Request, *exit.Error) {
	return s.RequestsOfKind("", state, packageName, limit)
}

// RequestsOfKind narrows the same listing to one ATTEMPT CLASS. `cozy run list` reads jobs
// and the request listing reads serving rows — one table, one reader, two questions.
func (s *Store) RequestsOfKind(kind, state, packageName string, limit int) ([]Request, *exit.Error) {
	return s.requestsBefore(kind, state, packageName, "", "", limit, 0, false)
}

// RequestsBefore reads a bounded history page, excluding the supplied run number.
func (s *Store) RequestsBefore(state, packageName string, limit int, before int64) ([]Request, *exit.Error) {
	return s.requestsBefore("", state, packageName, "", "", limit, before, false)
}

// PublicRequestsBefore is one history page; a non-empty hub keeps only that hub's runs.
// A request recorded without a hub belongs to its rental's hub, else fallbackHub, so a
// run from before hubs were recorded never drops out of its listing. Run numbers stay
// host-wide either way.
func (s *Store) PublicRequestsBefore(state, packageName, hub, fallbackHub string, limit int, before int64) ([]Request, *exit.Error) {
	return s.requestsBefore("", state, packageName, hub, fallbackHub, limit, before, true)
}

// requestHubSQL is a request row's hub: its own, else its rental's, else the fallback
// bound as the statement's parameter.
const requestHubSQL = `COALESCE(NULLIF(requests.hub,''),(SELECT rentals.hub FROM rentals
  WHERE rentals.id IN (requests.worker,requests.requested_rental) AND rentals.hub<>'' LIMIT 1),?)`

// RequestHub is the hub one request belongs to, by the same rule as the listing.
func (s *Store) RequestHub(id, fallbackHub string) (string, *exit.Error) {
	var hub string
	err := s.db.QueryRow(`SELECT `+requestHubSQL+` FROM requests WHERE id=?`,
		strings.TrimRight(fallbackHub, "/"), id).Scan(&hub)
	if errors.Is(err, sql.ErrNoRows) {
		return strings.TrimRight(fallbackHub, "/"), nil
	}
	if err != nil {
		return "", exit.Internalf("cannot read the hub of request %s: %s", id, err)
	}
	return hub, nil
}

func (s *Store) requestsBefore(kind, state, packageName, hub, fallbackHub string, limit int, before int64, public bool) ([]Request, *exit.Error) {
	// Number only narrow index facts across history; load payloads and other
	// request documents only for the bounded selected page.
	selectedState := "state"
	if public && state != "" {
		selectedState = publicRunStatusSQL()
	}
	selectedHub := "hub"
	args := []any{}
	if hub != "" {
		selectedHub = requestHubSQL
		args = append(args, strings.TrimRight(fallbackHub, "/"))
	}
	query := `WITH numbered AS (SELECT ROW_NUMBER() OVER (ORDER BY created_at,id) AS number,
  id,created_at,kind,` + selectedState + ` AS state,package,` + selectedHub + ` AS hub FROM requests), page AS (SELECT number,id AS page_request_id FROM numbered`
	where := []string{}
	if before > 0 {
		where = append(where, `number<?`)
		args = append(args, before)
	}
	if kind != "" {
		where = append(where, `kind=?`)
		args = append(args, kind)
	}
	if state != "" {
		where = append(where, `state=?`)
		args = append(args, state)
	}
	if packageName != "" {
		where = append(where, `package=?`)
		args = append(args, packageName)
	}
	if hub != "" {
		where = append(where, `hub=?`)
		args = append(args, hub)
	}
	if len(where) > 0 {
		query += ` WHERE ` + strings.Join(where, " AND ")
	}
	query += ` ORDER BY created_at DESC, id DESC LIMIT ?) SELECT page.number, ` + requestCols +
		` FROM requests JOIN page ON requests.id=page.page_request_id ORDER BY page.number DESC`
	args = append(args, limit)
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, exit.Internalf("cannot list requests: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanNumberedRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a request row: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

// ActiveRequests is every invocation the daemon still owes work or a terminal.
// Lifecycle operations use the complete set rather than a presentation-limited request
// listing: omitting row 501 from a safety fence would make `exit` destructive by accident.
func (s *Store) ActiveRequests() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests
		WHERE state IN (` + activeRequestStates + `)
		ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot list active requests: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an active request: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

// Owed is every request this authority still owes work for and that has NO live attempt:
// the ones a restarted daemon must put back on its dispatch queue. A request WITH a live
// or recovered attempt is not owed capacity — it is owed a terminal, and the
// recovered-attempts law is what settles that.
func (s *Store) Owed() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests r
		WHERE r.state IN ('submitted','queued')
		  AND NOT (r.package='cozy/platform' AND r.entrypoint='model-pass-through' AND EXISTS
		      (SELECT 1 FROM request_model_transfers t WHERE t.request_id=r.id))
		  AND NOT EXISTS (SELECT 1 FROM attempts a WHERE a.request_id=r.id
		                  AND a.state IN (` + openAttemptStates + `))
		ORDER BY r.created_at, r.id`)
	if err != nil {
		return nil, exit.Internalf("cannot read the owed requests: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an owed request row: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

// Unsettled is every request that holds an OPEN attempt: one this authority assigned and
// has no terminal for. These are not owed capacity — they are owed a TERMINAL, and the
// only thing that can produce one is the supervisor's own journal replayed by a worker in
// the same slot (worker-protocol/02 §6.2).
func (s *Store) Unsettled() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests r
		WHERE EXISTS (SELECT 1 FROM attempts a WHERE a.request_id=r.id
		              AND a.state IN (` + openAttemptStates + `))
		ORDER BY r.created_at, r.id`)
	if err != nil {
		return nil, exit.Internalf("cannot read the unsettled requests: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an unsettled request row: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

// LocalInstallationInUse preserves inputs owned by accepted requests, bindings or pins.
func (s *Store) LocalInstallationInUse(id string) (bool, *exit.Error) {
	var used bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM private_child_bindings WHERE child_install_id=?)
		OR EXISTS(SELECT 1 FROM requests WHERE local_installation_id=? AND state IN (`+activeRequestStates+`))
		OR EXISTS(SELECT 1 FROM pins WHERE install_id=?)`, id, id, id).Scan(&used)
	if err != nil {
		return false, exit.Internalf("cannot read installation ownership: %s", err)
	}
	return used, nil
}

// CanceledLocalPackages are durable transfer tombstones owed to one attached worker. Replaying
// them on every claimed session is idempotent and finishes cleanup after a daemon/stream crash.
func (s *Store) CanceledLocalPackages(workerID string) ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+requestCols+` FROM requests
		WHERE state='canceled' AND worker=? AND local_installation_id<>''
		ORDER BY created_at,id`, workerID)
	if err != nil {
		return nil, exit.Internalf("cannot read canceled local package transfers: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		row, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot scan canceled local package transfer: %s", err)
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish canceled local package transfers: %s", err)
	}
	return out, nil
}

// Settled answers whether a request state is final: nothing will run for it again.
func Settled(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// SettleRequest records the request's final state. Only a terminal the orchestrator
// ACCEPTED can settle one.
func (s *Store) SettleRequest(id, state string) *exit.Error {
	if _, err := s.db.Exec(`UPDATE requests SET state=? WHERE id=?`, state, id); err != nil {
		return exit.Internalf("cannot settle request %s: %s", id, err)
	}
	return nil
}

// BeginRequeue crosses the post-ack boundary in one transition. Replays after it are
// harmless: only `requeue_pending` moves, so a duplicate terminal ack cannot mint two
// ordinals.
func (s *Store) BeginRequeue(id string) (bool, *exit.Error) {
	result, err := s.db.Exec(`UPDATE requests SET state='queued' WHERE id=? AND state='requeue_pending'`, id)
	if err != nil {
		return false, exit.Internalf("cannot requeue %s: %s", id, err)
	}
	n, _ := result.RowsAffected()
	return n == 1, nil
}

// Submit records one durable request under its idempotency key. The same key with the
// same body digest answers the SAME request; the same key with a different body is a
// conflict, never a second execution wearing one name.
func (s *Store) Submit(r Request) (Request, bool, *exit.Error) {
	return s.SubmitWithEvent(r, nil)
}

// SubmitWithEvent freezes intake-only facts with the request. Deadline and
// publication consent must survive a crash before any observer goroutine starts.
func (s *Store) SubmitWithEvent(r Request, event map[string]any) (Request, bool, *exit.Error) {
	if problem := NormalizeModelTransferIntent(r.ModelTransfer); problem != nil {
		return Request{}, false, problem
	}
	r, assets, models, exportOutputs, problem := prepareRequest(r)
	if problem != nil {
		return Request{}, false, problem
	}
	tx, err := s.db.Begin()
	if err != nil {
		return Request{}, false, exit.Internalf("cannot begin request submission: %s", err)
	}
	defer tx.Rollback()
	recorded, fresh, problem := submitRequestTx(tx, r, assets, models, exportOutputs)
	if problem != nil {
		return Request{}, false, problem
	}
	if fresh && event != nil {
		if err := appendEventTx(tx, recorded.ID, "run.created", 0, event); err != nil {
			return Request{}, false, exit.Internalf("cannot freeze request submission intent: %s", err)
		}
	}
	for _, warning := range r.Warnings {
		if !fresh {
			break
		}
		payload := map[string]any{"code": warning.Code, "message": warning.Message, "fields": warning.Fields}
		if err := appendEventTx(tx, recorded.ID, "request.warning", 0, payload); err != nil {
			return Request{}, false, exit.Internalf("cannot record a request warning: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return Request{}, false, exit.Internalf("cannot commit request %s: %s", r.ID, err)
	}
	return recorded, fresh, nil
}

func prepareRequest(r Request) (Request, string, string, string, *exit.Error) {
	r.CreatedAt = now()
	r.State = "submitted"
	r.OrchestrationDirective = nil
	if r.Kind == "" {
		r.Kind = "serving"
	}
	if r.ParentRequestID == "" {
		r.ParentCallIndex = -1
	}
	if r.RetainWork {
		r.ReuseScope = r.ID
	}
	assets := []byte("[]")
	if len(r.Assets) > 0 {
		var err error
		assets, err = json.Marshal(r.Assets)
		if err != nil {
			return Request{}, "", "", "", exit.Internalf("cannot record request %s assets: %s", r.ID, err)
		}
	}
	models := []byte("[]")
	if len(r.Models) > 0 {
		var err error
		models, err = json.Marshal(r.Models)
		if err != nil {
			return Request{}, "", "", "", exit.Internalf("cannot record request %s models: %s", r.ID, err)
		}
	}
	exportOutputs, problem := prepareOutputExport(r.OutputExport)
	if problem != nil {
		return Request{}, "", "", "", problem
	}
	return r, string(assets), string(models), exportOutputs, nil
}

func submitRequestTx(tx *sql.Tx, r Request, assets, models, exportOutputs string,
) (Request, bool, *exit.Error) {
	existing, err := scanRequest(tx.QueryRow(
		`SELECT `+requestCols+` FROM requests WHERE idem_key=?`, r.IdemKey))
	if err == nil {
		originalRetention, retentionErr := submittedRetainWorkTx(tx, existing)
		if retentionErr != nil {
			return Request{}, false, exit.Internalf("cannot read original retention intent: %s", retentionErr)
		}
		if existing.BodyDigest != r.BodyDigest || originalRetention != r.RetainWork || existing.RetryOf != r.RetryOf {
			return Request{}, false, exit.New(exit.Conflict,
				"idempotency key %s already names a request with a different body", r.IdemKey).
				WithRemedy("one key, one body: %s was recorded, %s was submitted",
					short(existing.BodyDigest), short(r.BodyDigest))
		}
		existing.Number, err = requestNumber(tx, existing)
		if err != nil {
			return Request{}, false, exit.Internalf("cannot number request %s: %s", existing.ID, err)
		}
		return existing, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return Request{}, false, exit.Internalf("cannot read request %s: %s", r.IdemKey, err)
	}
	if r.RetryOf != "" {
		if problem := retainRetryTx(tx, &r); problem != nil {
			return Request{}, false, problem
		}
	}
	if r.Rental && r.Worker != "" {
		if problem := refuseReleasedRental(tx, r.Worker); problem != nil {
			return Request{}, false, problem
		}
	}

	machineRental := r.Worker
	if machineRental == "" {
		machineRental = r.RequestedRental
	}
	if _, err := tx.Exec(`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,
		plan_id,package_release,local_installation_id,
		local_package_uploaded_boot_id,installation_id,
		payload,outputs,state,ordinal,created_at,kind,needs_accelerator,org,trees,worker,machine,rental,rental_required,install_id,assets,attention_kernel,models,
		weights_outputs,retain_work,retry_of,reuse_scope,control_revision,parent_request_id,parent_call_index,child_intent_digest,child_target_digest,child_reusable,reused_from,orchestration_directive,child_artifacts,requested_rental,rent_new,hub)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?, ?,0,?,?,?,?,?,?,
		COALESCE((SELECT machine_name FROM rentals WHERE id=?),''),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		r.ID, r.IdemKey, r.BodyDigest, r.Package, r.Entrypoint, r.PlanID,
		r.Release, r.LocalInstallationID,
		r.LocalPackageUploadedBootID, r.InstallationID, r.Payload,
		r.Outputs, r.State, r.CreatedAt, r.Kind, r.NeedsAccelerator, r.Org, r.Trees, r.Worker,
		machineRental, r.Rental,
		r.RentalRequired,
		nullable(r.InstallID),
		assets, r.AttentionKernel, models, r.WeightsOutputs, r.RetainWork, r.RetryOf, r.ReuseScope, r.ControlRevision,
		r.ParentRequestID, r.ParentCallIndex, r.ChildIntentDigest, r.ChildTargetDigest, r.ChildReusable, r.ReusedFrom, blobOrEmpty(r.OrchestrationDirective), r.ChildArtifacts, r.RequestedRental, r.RentNew, r.Hub); err != nil {
		return Request{}, false, exit.Internalf("cannot record request %s: %s", r.ID, err)
	}
	if r.MachineExecutionObserver {
		if r.ParentRequestID != "" {
			return Request{}, false, exit.New(exit.Validation, "a machine execution observer must be a root request")
		}
		if _, err := tx.Exec(`INSERT INTO machine_executions(request_id) VALUES(?)`, r.ID); err != nil {
			return Request{}, false, exit.Internalf("cannot mark machine execution observation: %s", err)
		}
	} else {
		if problem := armSuccessfulWorkTx(tx, r); problem != nil {
			return Request{}, false, problem
		}
	}
	if problem := recordOutputExportTx(tx, r.ID, r.OutputExport, exportOutputs); problem != nil {
		return Request{}, false, problem
	}
	if problem := recordModelTransferTx(tx, r.ID, r.ModelTransfer); problem != nil {
		return Request{}, false, problem
	}
	if problem := recordPlannedSourcesTx(tx, r.ID, r.PlannedSourceBytes); problem != nil {
		return Request{}, false, problem
	}
	r.Number, err = requestNumber(tx, r)
	if err != nil {
		return Request{}, false, exit.Internalf("cannot number request %s: %s", r.ID, err)
	}
	return r, true, nil
}

func short(s string) string {
	if len(s) > 19 {
		return s[:19] + "…"
	}
	return s
}

// --------------------------------------------------------------------------- attempts

// Attempt is one execution of one spec. `AttemptKey` is the opaque id triage is served
// by; nothing about it is a path.
type Attempt struct {
	RequestID           string
	Attempt             int64
	AttemptKey          string
	InstanceID          string
	SessionID           string
	InvocationDigest    string
	InvocationCanonical []byte
	ServingPlacementSet []byte // exact prepared PlacementSet, committed with the serving offer
	WeightsOutputs      string
	State               string // preparing | offered | dispatch_aborted | accepted | recovered_open | terminal | closed
	PlanDigest          string
	Construction        string
	PlanSummary         string
	TerminalID          string
	TerminalDigest      string
	TerminalStatus      string
	TerminalCause       string
	SafeMessage         string
	TriageSubject       string
	TriageDigest        string
	TriageLength        int64
	TriageKept          bool // the verified bundle bytes are in this row
	TerminalBody        []byte
	DispatchedAt        string
	AcceptedAt          string
	ClosedAt            string
}

// ordinalLaws is the ordinal-minting law, read INSIDE the caller's transaction. It refuses
// while the request holds an open recovered attempt (02 §6.2) or a live one: a new
// session_id never manufactures absence, and supersession is explicit.
func ordinalLaws(tx *sql.Tx, requestID string) (int64, *exit.Error) {
	var requestState string
	if err := tx.QueryRow(`SELECT state FROM requests WHERE id=?`, requestID).Scan(&requestState); err != nil {
		return 0, exit.Internalf("cannot read request %s before dispatch: %s", requestID, err)
	}
	if requestState != "submitted" && requestState != "queued" {
		return 0, exit.New(exit.Conflict,
			"request %s is %s and may not mint another attempt", requestID, requestState)
	}
	var open int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM attempts
		WHERE request_id=? AND state='recovered_open'`, requestID).Scan(&open); err != nil {
		return 0, exit.Internalf("cannot read the recovered attempts of %s: %s", requestID, err)
	}
	if open > 0 {
		return 0, exit.New(exit.Conflict,
			"%s has %d recovered attempt(s) still open: no next ordinal may be minted", requestID, open).
			WithRemedy("a recovered attempt is closed by its own journaled terminal, never by assumption")
	}
	var live int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM attempts
		WHERE request_id=? AND state IN ('preparing','offered','accepted','terminal')`, requestID).Scan(&live); err != nil {
		return 0, exit.Internalf("cannot read the live attempts of %s: %s", requestID, err)
	}
	if live > 0 {
		return 0, exit.New(exit.Conflict,
			"%s still holds a live attempt: supersession is explicit, never a new ordinal", requestID).
			WithRemedy("cancel the held attempt (SUPERSEDED) and wait for its journaled terminal")
	}
	var max sql.NullInt64
	if err := tx.QueryRow(`SELECT MAX(attempt) FROM attempts WHERE request_id=?`, requestID).
		Scan(&max); err != nil {
		return 0, exit.Internalf("cannot read the attempt ordinals of %s: %s", requestID, err)
	}
	return max.Int64 + 1, nil
}

// NextOrdinal answers what the next ordinal WOULD be, under the same laws. It is a
// question, not a claim: minting is `Dispatch`'s, in the transaction that writes the row.
func (s *Store) NextOrdinal(requestID string) (int64, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return 0, exit.Internalf("cannot begin the ordinal transaction: %s", err)
	}
	defer tx.Rollback()
	return ordinalLaws(tx, requestID)
}

// Dispatch MINTS the ordinal and journals the assignment in ONE transaction, before
// StartAttempt goes out: a terminal may only cross a restart when the persisted assignment
// authorizes it, and an ordinal may only exist when its row does.
//
// The two used to be separate calls (`NextOrdinal` then `Dispatch`), and cl-003's very
// first cold four-component run found what that costs. Two `drain()` goroutines — one from
// the worker's READY report, one from `selectOrStart` — raced across the THREE unrelated
// reads inside the old `NextOrdinal`: the second read (is an attempt live?) ran before the
// first goroutine's insert committed, and the third (what is the highest ordinal?) ran
// after it. So the law said "nothing is live" and the arithmetic said "one exists", and
// the orchestrator dispatched attempt 2 of a request whose attempt 1 was starting. The
// worker refused it — `live_attempt_supersession`, which is the runtime's own fence doing
// its job — and the request FAILED. The ordinal is now minted by the writer that owns the
// row, which is the only place it can be minted atomically.
func (s *Store) Dispatch(a Attempt) (int64, *exit.Error) {
	a.DispatchedAt = now()
	if a.AttemptKey == "" {
		a.AttemptKey = NewID("att")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return 0, exit.Internalf("cannot begin the dispatch transaction: %s", err)
	}
	defer tx.Rollback()
	ordinal, e := ordinalLaws(tx, a.RequestID)
	if e != nil {
		return 0, e
	}
	a.Attempt = ordinal
	var request Request
	if err := tx.QueryRow(`SELECT package,entrypoint,kind,worker,rental FROM requests WHERE id=?`, a.RequestID).Scan(&request.Package, &request.Entrypoint, &request.Kind, &request.Worker, &request.Rental); err != nil {
		return 0, exit.Internalf("cannot read serving dispatch subject: %s", err)
	}
	if request.Rental && request.Worker != "" {
		if problem := refuseReleasedRental(tx, request.Worker); problem != nil {
			return 0, problem
		}
	}
	if _, problem := BoundServingPlacement(request, a); problem != nil {
		return 0, problem
	}
	if _, err := tx.Exec(`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,
		session_id,invocation_digest,invocation,weights_outputs,state,dispatched_at)
		VALUES(?,?,?,?,?,?,?,?,'preparing',?)`,
		a.RequestID, a.Attempt, a.AttemptKey, a.InstanceID, a.SessionID,
		a.InvocationDigest, a.InvocationCanonical, a.WeightsOutputs, a.DispatchedAt); err != nil {
		return 0, exit.New(exit.Conflict, "cannot journal attempt %s#%d: %s", a.RequestID, a.Attempt, err).
			WithRemedy("an attempt ordinal is written once")
	}
	if len(a.ServingPlacementSet) > 0 {
		if _, err := tx.Exec(`INSERT INTO attempt_serving_placements(request_id,attempt,body) VALUES(?,?,?)`, a.RequestID, a.Attempt, a.ServingPlacementSet); err != nil {
			return 0, exit.Internalf("cannot retain prepared serving placement: %s", err)
		}
	}
	advanced, err := tx.Exec(`UPDATE requests SET state='dispatching', ordinal=?
		WHERE id=? AND state IN ('submitted','queued')`, a.Attempt, a.RequestID)
	if err != nil {
		return 0, exit.Internalf("cannot advance request %s: %s", a.RequestID, err)
	}
	if n, _ := advanced.RowsAffected(); n != 1 {
		return 0, exit.New(exit.Conflict,
			"request %s changed state before attempt %d could be assigned", a.RequestID, a.Attempt)
	}
	if err := tx.Commit(); err != nil {
		return 0, exit.Internalf("the dispatch transaction did not commit: %s", err)
	}
	return ordinal, nil
}

// OfferDispatch durably crosses the offer boundary before the frame is queued. If the
// process dies on either side, the next worker snapshot decides the only ambiguity:
// held means recover it; absent means abort it and return the request to the queue.
func (s *Store) OfferDispatch(requestID string, attempt int64, sessionID string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the offer transaction: %s", err)
	}
	defer tx.Rollback()
	res, err := tx.Exec(`UPDATE attempts SET state='offered'
		WHERE request_id=? AND attempt=? AND session_id=? AND state='preparing'`,
		requestID, attempt, sessionID)
	if err != nil {
		return exit.Internalf("cannot offer %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "%s#%d is not a prepared assignment of session %s",
			requestID, attempt, sessionID)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("the offer transaction did not commit for %s#%d: %s",
			requestID, attempt, err)
	}
	return nil
}

// AbortDispatch closes an assignment that never crossed the AttemptOffer boundary. It is
// not a terminal: no worker saw the attempt and therefore no worker owes an outcome. The
// row remains as durable history while the request becomes owed capacity again.
func (s *Store) AbortDispatch(requestID string, attempt int64, sessionID, reason string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the dispatch-abort transaction: %s", err)
	}
	defer tx.Rollback()
	res, err := tx.Exec(`UPDATE attempts SET state='dispatch_aborted', safe_message=?, closed_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('preparing','offered')`,
		reason, now(), requestID, attempt, sessionID)
	if err != nil {
		return exit.Internalf("cannot abort dispatch %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict,
			"cannot abort dispatch %s#%d: it is not an unoffered assignment of session %s",
			requestID, attempt, sessionID)
	}
	request, err := tx.Exec(`UPDATE requests SET state=CASE WHEN state='dispatching' THEN 'submitted' ELSE state END
		WHERE id=? AND ordinal=? AND state IN ('dispatching','pausing','canceling')`, requestID, attempt)
	if err != nil {
		return exit.Internalf("cannot return request %s to submitted: %s", requestID, err)
	}
	if n, _ := request.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict,
			"cannot abort dispatch %s#%d: its request no longer names that live assignment",
			requestID, attempt)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("the dispatch-abort transaction did not commit for %s#%d: %s",
			requestID, attempt, err)
	}
	return nil
}

// MediaCleanupOwed derives cleanup work from durable attempt state. Only an aborted
// pre-offer assignment or an acknowledged closed outcome is disposable; a terminal that
// has not been acked still needs its remote outputs for replay.
func (s *Store) MediaCleanupOwed(instanceID string) ([]Attempt, *exit.Error) {
	return s.attemptsWhere(`instance_id=? AND media_cleaned=0
		AND state IN ('dispatch_aborted','closed')`, instanceID)
}

func (s *Store) MarkMediaCleaned(requestID string, attempt int64) *exit.Error {
	if _, err := s.db.Exec(`UPDATE attempts SET media_cleaned=1 WHERE request_id=? AND attempt=?`,
		requestID, attempt); err != nil {
		return exit.Internalf("cannot record media cleanup for %s#%d: %s", requestID, attempt, err)
	}
	return nil
}

// Accepted records queue admission. The plan and construction digests bind at device entry,
// not here: they arrive on HeldAttempt from RUNNING on (proto-026).
func (s *Store) Accepted(requestID string, attempt int64, sessionID string) *exit.Error {
	res, err := s.db.Exec(`UPDATE attempts SET state='accepted', accepted_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('offered','recovered_open')`,
		now(), requestID, attempt, sessionID)
	if err != nil {
		return exit.Internalf("cannot record acceptance of %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict,
			"acceptance of %s#%d refused: it is not a dispatching attempt of session %s",
			requestID, attempt, sessionID).
			WithRemedy("only the session the attempt was assigned to may write its row")
	}
	return nil
}

// Output is one written asset, bound to the result FIELD PATH it came from.
//
// `MediaID` is the client's ONLY handle on the bytes (cl-006). It is minted here, inside
// the terminal transaction, and it is opaque: the client contract's media route takes a
// media id and nothing else, so there is no representation of "fetch this path" for a
// caller to shape. `Path` is the orchestrator's own knowledge of where the runtime was
// granted to write, and it never leaves this process.
type Output struct {
	OutputID string
	MediaID  string
	Path     string
	Digest   string
	Length   int64
	MimeType string
}

// WeightsReceipt is the validated Runtime-authored WeightsReceipt/1 carried by an outcome.
type WeightsReceipt struct {
	RequestID, OwnerScope, InvocationDigest, OutputSlot string
	Attempt                                             int64
	ReceiptDigest                                       string
	ReceiptBytes                                        []byte
}

// WeightsFinalization is Cozy's durable typed first-wins disposition and Runtime completion.
type WeightsFinalization struct {
	RequestID, InstanceID, OwnerScope, InvocationDigest, OutputSlot string
	Attempt                                                         int64
	Disposition, ReceiptDigest, ScratchRootID                       string
	ResultOutcome                                                   string
	ResultReceiptDigest                                             string
	ResultReceiptBytes                                              []byte
	RecordedAt, CompletedAt                                         string
}

// Terminal is what a worker journaled and the orchestrator is about to make authoritative.
type Terminal struct {
	RequestID        string
	Attempt          int64
	SessionID        string
	InvocationDigest string
	TerminalID       string
	TerminalDigest   string
	Status           string
	Cause            string
	SafeMessage      string
	TriageSubject    string
	// TriageDigest/TriageLength/TriageBundle are cl-006's PERSISTENCE of the worker's
	// bundle: the bounded bytes are copied out of the worker/media plane and verified
	// against the terminal's own TriageBundleRef before this transaction runs, then
	// live IN the attempt row (cl-116). A bundle belongs to exactly one attempt, so an
	// orphan bundle file is unrepresentable.
	TriageDigest         string
	TriageLength         int64
	TriageBundle         []byte
	Body                 []byte
	Outputs              []Output
	ByteOutputs          []ByteOutput
	WeightsFinalizations []WeightsFinalization
	// Event is the attempt-end lifecycle event, appended INSIDE this transaction so the
	// stream cannot disagree with the authority about whether the request ended.
	EventType    string
	EventPayload map[string]any
	// RequestState is what the REQUEST row becomes. It is not always the attempt's own
	// status: an attempt the orchestrator will requeue leaves the request QUEUED, and
	// writing `abandoned` there would make the status document say `failed` for a
	// request that is still going.
	RequestState string
	// ExpectedRequestState prevents a pause/cancel that committed during outcome
	// verification from being overwritten by the earlier lifecycle projection.
	ExpectedRequestState string
	// Publication is the job lane's durable publication (cl-004), written INSIDE this
	// transaction. `nil` for a serving attempt — and for a job attempt the orchestrator
	// will requeue, because a requeued attempt has not ended the request.
	Publication *Publication
}

// Publication is one job's DURABLE PUBLICATION: the scratch repo it landed under, the
// root that holds its bytes, and whatever it declared into the local CAS.
type Publication struct {
	RequestID   string
	Attempt     int64
	Repo        string // <org>/_job-<request-id>
	Root        string // the durable directory the grant named
	Status      string // the terminal's verdict, STAMPED — never a persistence gate
	Cause       string
	Entries     int64
	Bytes       int64
	CommittedAt string
}

// AcceptTerminal is THE transaction cl-001 exists for: the terminal becomes
// authoritative and its outputs become visible together, or neither does. It commits
// BEFORE TerminalAck is sent, so a crash in between replays (the worker never stops
// replaying an unacked terminal) and re-applies idempotently — never twice.
//
// `applied` is false for an exact replay of a terminal already closed: the caller still
// acks, because the ack is what lets the worker compact.
func (s *Store) AcceptTerminal(t Terminal) (applied bool, e *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin the terminal transaction: %s", err)
	}
	defer tx.Rollback()

	var state, digest, assignedSession, assignedSpec string
	var knownBody []byte
	err = tx.QueryRow(`SELECT state, terminal_digest, session_id, invocation_digest, COALESCE(terminal_body,x'') FROM attempts
		WHERE request_id=? AND attempt=?`, t.RequestID, t.Attempt).
		Scan(&state, &digest, &assignedSession, &assignedSpec, &knownBody)
	if errors.Is(err, sql.ErrNoRows) {
		return false, exit.New(exit.NotFound,
			"terminal for %s#%d refused: no such assigned attempt", t.RequestID, t.Attempt).
			WithRemedy("a terminal is authorized by the persisted assignment, never by its own claim")
	}
	if err != nil {
		return false, exit.Internalf("cannot read attempt %s#%d: %s", t.RequestID, t.Attempt, err)
	}
	if assignedSpec != t.InvocationDigest {
		return false, exit.New(exit.Validation,
			"terminal for %s#%d refused: it closes %s, the assignment is %s",
			t.RequestID, t.Attempt, short(t.InvocationDigest), short(assignedSpec))
	}
	if state == "terminal" || state == "closed" {
		if digest != t.TerminalDigest || !bytes.Equal(knownBody, t.Body) {
			return false, exit.New(exit.Conflict,
				"%s#%d already closed with terminal %s; %s is a different body",
				t.RequestID, t.Attempt, short(digest), short(t.TerminalDigest))
		}
		// A claimed peer can carry this already committed outcome after recovery.
		// ACK its exact bytes without rewriting who actually executed it.
		return false, nil
	}
	if assignedSession != t.SessionID {
		return false, exit.New(exit.Conflict,
			"terminal for %s#%d refused: session %s does not own that attempt row (%s does)",
			t.RequestID, t.Attempt, t.SessionID, assignedSession).
			WithRemedy("one writer per attempt row")
	}
	if t.ExpectedRequestState != "" {
		var current string
		if err := tx.QueryRow(`SELECT state FROM requests WHERE id=?`, t.RequestID).Scan(&current); err != nil {
			return false, exit.Internalf("cannot read request disposition: %s", err)
		}
		if current != t.ExpectedRequestState {
			return false, exit.Named(exit.Conflict, "request.lifecycle_changed",
				"request %s changed from %s to %s while accepting its outcome", t.RequestID, t.ExpectedRequestState, current)
		}
	}

	res, err := tx.Exec(`UPDATE attempts SET state='terminal', terminal_id=?, terminal_digest=?,
		terminal_status=?, terminal_cause=?, safe_message=?, triage_subject=?, triage_digest=?,
		triage_length=?, triage_bundle=?, terminal_body=?, closed_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('offered','accepted','recovered_open')`,
		t.TerminalID, t.TerminalDigest, t.Status, t.Cause, t.SafeMessage, t.TriageSubject,
		t.TriageDigest, t.TriageLength, blobOrEmpty(t.TriageBundle),
		t.Body, now(), t.RequestID, t.Attempt, t.SessionID)
	if err != nil {
		return false, exit.Internalf("cannot apply the terminal of %s#%d: %s", t.RequestID, t.Attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return false, exit.New(exit.Conflict,
			"terminal for %s#%d refused: the attempt row is in state %q", t.RequestID, t.Attempt, state)
	}
	for _, o := range t.Outputs {
		if _, err := tx.Exec(`INSERT INTO outputs(request_id,attempt,output_id,media_id,path,digest,
			length,mime_type) VALUES(?,?,?,?,?,?,?,?)`,
			t.RequestID, t.Attempt, o.OutputID, o.MediaID, o.Path, o.Digest, o.Length,
			o.MimeType); err != nil {
			return false, exit.Internalf("cannot publish output %s of %s#%d: %s",
				o.OutputID, t.RequestID, t.Attempt, err)
		}
	}
	if problem := recordByteOutputsTx(tx, t); problem != nil {
		return false, problem
	}
	if problem := recordDeviceMemoryTx(tx, t); problem != nil {
		return false, problem
	}
	visible := now()
	for _, finalization := range t.WeightsFinalizations {
		var held WeightsFinalization
		err := tx.QueryRow(`SELECT attempt,instance_id,owner_scope,disposition,
			receipt_digest,scratch_root_id FROM weights_finalizations
			WHERE request_id=? AND invocation_digest=? AND output_slot=?`, finalization.RequestID,
			finalization.InvocationDigest, finalization.OutputSlot).Scan(&held.Attempt,
			&held.InstanceID, &held.OwnerScope, &held.Disposition, &held.ReceiptDigest,
			&held.ScratchRootID)
		if err == nil {
			if held.Attempt != finalization.Attempt || held.InstanceID != finalization.InstanceID ||
				held.OwnerScope != finalization.OwnerScope ||
				held.Disposition != finalization.Disposition ||
				held.ReceiptDigest != finalization.ReceiptDigest ||
				held.ScratchRootID != finalization.ScratchRootID {
				return false, exit.New(exit.Conflict,
					"weights output %s already has a different typed final intent",
					finalization.OutputSlot)
			}
			continue
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return false, exit.Internalf("cannot read weights final intent %s of %s: %s",
				finalization.OutputSlot, t.RequestID, err)
		}
		if _, err := tx.Exec(`INSERT INTO weights_finalizations(request_id,attempt,instance_id,
			owner_scope,invocation_digest,output_slot,disposition,receipt_digest,scratch_root_id,
			recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?)`,
			finalization.RequestID, finalization.Attempt, finalization.InstanceID,
			finalization.OwnerScope, finalization.InvocationDigest, finalization.OutputSlot,
			finalization.Disposition, finalization.ReceiptDigest, finalization.ScratchRootID,
			visible); err != nil {
			return false, exit.Internalf("cannot record weights final intent %s of %s#%d: %s",
				finalization.OutputSlot, t.RequestID, t.Attempt, err)
		}
	}
	requestState := t.RequestState
	if requestState == "" {
		requestState = strings.ToLower(t.Status)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=? WHERE id=?`,
		requestState, t.RequestID); err != nil {
		return false, exit.Internalf("cannot settle request %s: %s", t.RequestID, err)
	}
	if requestState == "failed" || requestState == "canceled" || requestState == "refused" ||
		requestState == "abandoned" || (requestState == "finalizing" && t.Status != "SUCCEEDED") {
		transferState := "failed"
		if requestState == "canceled" || t.Status == "CANCELED" {
			transferState = "canceled"
		}
		if _, err := tx.Exec(`UPDATE request_model_transfers SET state=?,error_code=?,safe_error=?,
			updated_at=? WHERE request_id=? AND state NOT IN ('completed','failed','canceled')`,
			transferState, t.Cause, t.SafeMessage, now(), t.RequestID); err != nil {
			return false, exit.Internalf("cannot settle model transfer %s: %s", t.RequestID, err)
		}
	}
	// THE PUBLICATION rides the same commit as the terminal and the outputs (cl-004).
	// "The bundle is visible" and "the terminal was accepted" are therefore one fact:
	// a kill before this commit leaves the bytes on disk with nothing claiming them,
	// which is what "no partial bundle is visible" means. A LATER attempt of the same
	// request republishes into the same root, so the row is upserted rather than
	// duplicated — the publication belongs to the REQUEST, not to one ordinal.
	if p := t.Publication; p != nil {
		if _, err := tx.Exec(`INSERT INTO publications(request_id,attempt,repo,root,status,cause,
			entries,bytes,committed_at) VALUES(?,?,?,?,?,?,?,?,?)
			ON CONFLICT(request_id) DO UPDATE SET attempt=excluded.attempt,
			  status=excluded.status, cause=excluded.cause, entries=excluded.entries,
			  bytes=excluded.bytes, committed_at=excluded.committed_at`,
			p.RequestID, p.Attempt, p.Repo, p.Root, p.Status, p.Cause, p.Entries, p.Bytes,
			visible); err != nil {
			return false, exit.Internalf("cannot commit the publication of %s: %s", p.Repo, err)
		}
	}
	// The terminal EVENT rides the same transaction as the terminal fact and its outputs
	// (cl-006). A stream that announced a terminal the authority had not committed — or an
	// authority that committed one the stream never announced — is unrepresentable.
	if t.EventType != "" {
		if err := appendEventTx(tx, t.RequestID, t.EventType, t.Attempt, t.EventPayload); err != nil {
			return false, exit.Internalf("cannot append the terminal event of %s#%d: %s",
				t.RequestID, t.Attempt, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return false, exit.New(exit.Conflict,
			"the terminal transaction did not commit for %s#%d: %s", t.RequestID, t.Attempt, err).
			WithRemedy("no output is visible and the worker will replay: nothing was half-applied")
	}
	return true, nil
}

// Closed records the ack. Only a closed entry is compactable, and closure follows the
// commit — never precedes it.
func (s *Store) Closed(requestID string, attempt int64) *exit.Error {
	res, err := s.db.Exec(`UPDATE attempts SET state='closed' WHERE request_id=? AND attempt=?
		AND state='terminal'`, requestID, attempt)
	if err != nil {
		return exit.Internalf("cannot close %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n == 0 {
		var state string
		if err := s.db.QueryRow(`SELECT state FROM attempts WHERE request_id=? AND attempt=?`,
			requestID, attempt).Scan(&state); err != nil || state != "closed" {
			return exit.New(exit.Conflict, "cannot close %s#%d from state %q", requestID, attempt, state)
		}
	}
	return nil
}

func (s *Store) AttemptRow(requestID string, attempt int64) (*Attempt, *exit.Error) {
	rows, e := s.attemptsWhere(`request_id=? AND attempt=?`, requestID, attempt)
	if e != nil || len(rows) == 0 {
		return nil, e
	}
	return &rows[0], nil
}

func (s *Store) Attempts(requestID string) ([]Attempt, *exit.Error) {
	return s.attemptsWhere(`request_id=?`, requestID)
}

// OpenAttemptsOf is every attempt one worker INSTANCE still owes a terminal for. It is
// what a orchestrator asks when that worker's process dies: those attempts are not
// finished and not failed — they are unsettled, and the only thing that can settle one is
// the supervisor's own journal, replayed by a worker in the SAME slot.
func (s *Store) OpenAttemptsOf(instanceID string) ([]Attempt, *exit.Error) {
	return s.attemptsWhere(`instance_id=? AND state IN (`+openAttemptStates+`)`, instanceID)
}

// ReadyRequeues are committed requeue decisions whose old terminal has crossed the
// acknowledgement boundary. BeginRequeue changes them to queued.
func (s *Store) ReadyRequeues() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests
		WHERE state='requeue_pending' AND EXISTS (
		  SELECT 1 FROM attempts a WHERE a.request_id=requests.id
		  AND a.attempt=requests.ordinal AND a.state='closed')
		ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot read pending requeues: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a pending requeue: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

func (s *Store) attemptsWhere(where string, args ...any) ([]Attempt, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,attempt,attempt_key,instance_id,session_id,
		invocation_digest,invocation,weights_outputs,state,plan_digest,construction,plan_summary,terminal_id,
		terminal_digest,terminal_status,terminal_cause,safe_message,triage_subject,
		triage_digest,triage_length,length(triage_bundle)>0,
		COALESCE(terminal_body,x''),dispatched_at,accepted_at,closed_at,
		COALESCE((SELECT body FROM attempt_serving_placements p WHERE p.request_id=attempts.request_id AND p.attempt=attempts.attempt),x'')
		FROM attempts WHERE `+where+` ORDER BY request_id, attempt`, args...)
	if err != nil {
		return nil, exit.Internalf("cannot read attempts: %s", err)
	}
	defer rows.Close()
	var out []Attempt
	for rows.Next() {
		var a Attempt
		if err := rows.Scan(&a.RequestID, &a.Attempt, &a.AttemptKey, &a.InstanceID, &a.SessionID,
			&a.InvocationDigest, &a.InvocationCanonical, &a.WeightsOutputs, &a.State, &a.PlanDigest, &a.Construction,
			&a.PlanSummary, &a.TerminalID, &a.TerminalDigest, &a.TerminalStatus, &a.TerminalCause,
			&a.SafeMessage, &a.TriageSubject, &a.TriageDigest, &a.TriageLength, &a.TriageKept,
			&a.TerminalBody, &a.DispatchedAt, &a.AcceptedAt, &a.ClosedAt, &a.ServingPlacementSet); err != nil {
			return nil, exit.Internalf("cannot read an attempt row: %s", err)
		}
		out = append(out, a)
	}
	return out, nil
}

// TriageBundle serves one kept bundle by the attempt's OPAQUE key. Empty bytes mean the
// terminal named no bundle or verification refused it; the caller answers 404, not 500.
func (s *Store) TriageBundle(attemptKey string) (subject string, bundle []byte, problem *exit.Error) {
	row := s.db.QueryRow(`SELECT triage_subject, triage_bundle FROM attempts WHERE attempt_key=?`, attemptKey)
	if err := row.Scan(&subject, &bundle); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil, nil
		}
		return "", nil, exit.Internalf("cannot read the triage bundle of attempt %s: %s", attemptKey, err)
	}
	return subject, bundle, nil
}

// blobOrEmpty keeps a NOT NULL blob column honest: an absent bundle is zero bytes.
func blobOrEmpty(b []byte) []byte {
	if b == nil {
		return []byte{}
	}
	return b
}

const weightsFinalizationCols = `request_id,attempt,instance_id,owner_scope,invocation_digest,
	output_slot,disposition,receipt_digest,scratch_root_id,result_outcome,
	result_receipt_digest,result_receipt_bytes,
	recorded_at,completed_at`

func scanWeightsFinalization(row interface{ Scan(...any) error }) (WeightsFinalization, error) {
	var f WeightsFinalization
	err := row.Scan(&f.RequestID, &f.Attempt, &f.InstanceID, &f.OwnerScope,
		&f.InvocationDigest, &f.OutputSlot, &f.Disposition, &f.ReceiptDigest,
		&f.ScratchRootID, &f.ResultOutcome, &f.ResultReceiptDigest, &f.ResultReceiptBytes,
		&f.RecordedAt, &f.CompletedAt)
	return f, err
}

// WeightsFinalization reads the exact first-wins intent for one semantic output.
func (s *Store) WeightsFinalization(requestID, invocationDigest, outputSlot string) (*WeightsFinalization, *exit.Error) {
	f, err := scanWeightsFinalization(s.db.QueryRow(`SELECT `+weightsFinalizationCols+`
		FROM weights_finalizations WHERE request_id=? AND invocation_digest=? AND output_slot=?`,
		requestID, invocationDigest, outputSlot))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read weights finalization %s/%s: %s",
			requestID, outputSlot, err)
	}
	return &f, nil
}

// PendingWeightsFinalizations is the exact decision set that must be (re)sent before
// this attempt's outcome may be acknowledged. Ordering by slot makes replay deterministic.
func (s *Store) PendingWeightsFinalizations(requestID string, attempt int64) ([]WeightsFinalization, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+weightsFinalizationCols+` FROM weights_finalizations
		WHERE request_id=? AND attempt=? AND completed_at='' ORDER BY output_slot`, requestID, attempt)
	if err != nil {
		return nil, exit.Internalf("cannot read pending weights finalizations of %s#%d: %s",
			requestID, attempt, err)
	}
	defer rows.Close()
	out := []WeightsFinalization{}
	for rows.Next() {
		f, err := scanWeightsFinalization(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a weights finalization row: %s", err)
		}
		out = append(out, f)
	}
	return out, nil
}

func (s *Store) WeightsFinalizationsOf(requestID string) ([]WeightsFinalization, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+weightsFinalizationCols+` FROM weights_finalizations
		WHERE request_id=? ORDER BY attempt,output_slot`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read weights finalizations of %s: %s", requestID, err)
	}
	defer rows.Close()
	out := []WeightsFinalization{}
	for rows.Next() {
		f, err := scanWeightsFinalization(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a weights finalization row: %s", err)
		}
		out = append(out, f)
	}
	return out, nil
}

// RecordWeightsFinalizeResult journals Runtime's typed replayable result. The same result is
// idempotent; a changed result for the first-wins decision conflicts forever.
func (s *Store) RecordWeightsFinalizeResult(result WeightsFinalization) (applied bool, e *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin the weights finalize result transaction: %s", err)
	}
	defer tx.Rollback()
	held, err := scanWeightsFinalization(tx.QueryRow(`SELECT `+weightsFinalizationCols+`
		FROM weights_finalizations WHERE request_id=? AND invocation_digest=? AND output_slot=?`,
		result.RequestID, result.InvocationDigest, result.OutputSlot))
	if errors.Is(err, sql.ErrNoRows) {
		return false, exit.New(exit.NotFound, "no weights final intent for %s/%s",
			result.RequestID, result.OutputSlot)
	}
	if err != nil {
		return false, exit.Internalf("cannot read weights final intent %s/%s: %s",
			result.RequestID, result.OutputSlot, err)
	}
	if held.InstanceID != result.InstanceID {
		return false, exit.New(exit.Conflict,
			"weights finalize result for %s/%s came from worker %s; %s owns the transaction",
			result.RequestID, result.OutputSlot, result.InstanceID, held.InstanceID)
	}
	if held.CompletedAt != "" {
		if held.ResultOutcome != result.ResultOutcome ||
			held.ResultReceiptDigest != result.ResultReceiptDigest ||
			!bytes.Equal(held.ResultReceiptBytes, result.ResultReceiptBytes) {
			return false, exit.New(exit.Conflict,
				"weights finalization %s/%s already completed with a different typed result",
				result.RequestID, result.OutputSlot)
		}
		return false, nil
	}
	completed := now()
	updated, err := tx.Exec(`UPDATE weights_finalizations SET result_outcome=?,
		result_receipt_digest=?,result_receipt_bytes=?,completed_at=?
		WHERE request_id=? AND invocation_digest=? AND output_slot=? AND completed_at=''`,
		result.ResultOutcome, result.ResultReceiptDigest, blob(result.ResultReceiptBytes), completed,
		result.RequestID, result.InvocationDigest, result.OutputSlot)
	if err != nil {
		return false, exit.Internalf("cannot record weights finalize result %s/%s: %s",
			result.RequestID, result.OutputSlot, err)
	}
	if n, _ := updated.RowsAffected(); n != 1 {
		return false, exit.New(exit.Conflict, "weights finalization %s/%s moved while completing",
			result.RequestID, result.OutputSlot)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit weights finalize result %s/%s: %s",
			result.RequestID, result.OutputSlot, err)
	}
	return true, nil
}

// blob keeps a NOT NULL BLOB column NOT NULL. An ABANDON_UNCOMMITTED completion carries
// no receipt at all, and a nil slice is a NULL rather than the empty evidence it means.
func blob(b []byte) []byte {
	if b == nil {
		return []byte{}
	}
	return b
}

// VisibleOutputs answers ONLY for an attempt whose terminal the orchestrator accepted.
// The join is the invariant: an output row exists only inside the terminal transaction,
// so "visible without a terminal" has no representation.
func (s *Store) VisibleOutputs(requestID string) ([]Output, *exit.Error) {
	rows, err := s.db.Query(`SELECT o.output_id,o.media_id,o.path,o.digest,o.length,o.mime_type
		FROM outputs o JOIN attempts a ON a.request_id=o.request_id AND a.attempt=o.attempt
		WHERE o.request_id=? AND a.state IN ('terminal','closed')
		ORDER BY o.attempt, o.output_id`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read the outputs of %s: %s", requestID, err)
	}
	defer rows.Close()
	var out []Output
	for rows.Next() {
		var o Output
		if err := rows.Scan(&o.OutputID, &o.MediaID, &o.Path, &o.Digest, &o.Length, &o.MimeType); err != nil {
			return nil, exit.Internalf("cannot read an output row: %s", err)
		}
		out = append(out, o)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish output read: %s", err)
	}
	rows.Close()
	// Verified local copies survive loss of the remote collection ACK. The
	// complete result still requires its separate collection/custody gate.
	var readable bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e
 WHERE e.request_id=? AND (e.collected=1 OR `+machineExecutionLost+`))`, requestID).Scan(&readable); err != nil {
		return nil, exit.Internalf("cannot read machine output availability: %s", err)
	}
	if readable {
		products, problem := s.Products(requestID)
		if problem != nil {
			return nil, problem
		}
		for _, product := range Fold(products) {
			id := product.Output
			if product.Op == ProductAppend {
				id = fmt.Sprintf("%s.%d", product.Output, product.Index)
			}
			out = append(out, Output{OutputID: id, Path: product.Path, Digest: product.Digest, Length: product.Length, MimeType: product.MediaType})
		}
	}
	return out, nil
}

// Media is the OPAQUE-ID read (cl-006). The client contract's media route takes a media
// id and resolves it HERE, against a row that exists only because a terminal was accepted.
// Three properties fall out of that and none of them is a check the handler performs:
// a path cannot be asked for (there is no route shape that takes one), an output that no
// terminal published has no id at all, and an id is unguessable.
func (s *Store) Media(mediaID string) (*Output, string, int64, *exit.Error) {
	var o Output
	var requestID string
	var attempt int64
	err := s.db.QueryRow(`SELECT o.output_id,o.media_id,o.path,o.digest,o.length,o.mime_type,
		o.request_id,o.attempt
		FROM outputs o JOIN attempts a ON a.request_id=o.request_id AND a.attempt=o.attempt
		WHERE o.media_id=? AND a.state IN ('terminal','closed')`, mediaID).
		Scan(&o.OutputID, &o.MediaID, &o.Path, &o.Digest, &o.Length, &o.MimeType,
			&requestID, &attempt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, "", 0, nil
	}
	if err != nil {
		return nil, "", 0, exit.Internalf("cannot read media %s: %s", mediaID, err)
	}
	return &o, requestID, attempt, nil
}

// ---------------------------------------------------------------- publications (cl-004)

const publicationCols = `request_id,attempt,repo,root,status,cause,entries,bytes,committed_at`

func scanPublication(row interface{ Scan(...any) error }) (Publication, error) {
	var p Publication
	err := row.Scan(&p.RequestID, &p.Attempt, &p.Repo, &p.Root, &p.Status, &p.Cause,
		&p.Entries, &p.Bytes, &p.CommittedAt)
	return p, err
}

// PublicationOf answers for one request. A nil answer means no terminal ever committed
// one — which is exactly the question a crash arm asks.
func (s *Store) PublicationOf(requestID string) (*Publication, *exit.Error) {
	p, err := scanPublication(s.db.QueryRow(
		`SELECT `+publicationCols+` FROM publications WHERE request_id=?`, requestID))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the publication of %s: %s", requestID, err)
	}
	return &p, nil
}

// ---------------------------------------------------------------- job checkpoints

// Checkpoint is one durable checkpoint identity, as the worker presented it.
type Checkpoint struct {
	RequestID     string
	Attempt       int64
	OperationKey  string
	LogicalKey    string
	ContentDigest string
	ReceiptID     string
	Outcome       string
	RecordedAt    string
	// SessionID is the CLAIM the frame arrived under, never a stored column: the row is
	// admitted only if the attempt it names is open under this same session. It is an
	// input to the admission, not a fact about the checkpoint.
	SessionID string
}

// RecordCheckpoint journals the orchestrator's copy of one durable save and answers with
// the row plus what happened: RECORDED (new), REPLAYED (the same identity again) or
// CONFLICT (the same keys, different bytes). It never replaces a recorded digest — the
// worker's own journal already made that fact durable, and a orchestrator that overwrote
// it would be a second authority over one fact.
// A fourth answer joins the three: UNKNOWN_ATTEMPT, when the frame names an attempt that
// is not open under the sending session. Ownership is checked INSIDE the insert — one
// INSERT...SELECT...WHERE EXISTS, atomic by construction — so a terminal landing between a
// separate check and the write cannot slip a checkpoint into a closed attempt.
func (s *Store) RecordCheckpoint(c Checkpoint) (Checkpoint, string, *exit.Error) {
	// OWNERSHIP FIRST, and it decides before anything is read back. A sender that does not
	// own an open attempt learns nothing about what is journaled under it — not even that
	// a key is taken, which is what answering CONFLICT would have told it.
	var open int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM attempts
		WHERE request_id=? AND attempt=? AND session_id=?
		AND state IN ('offered','accepted','recovered_open')`,
		c.RequestID, c.Attempt, c.SessionID).Scan(&open); err != nil {
		return c, "", exit.Internalf("cannot read the attempt of a checkpoint: %s", err)
	}
	if open != 1 {
		return c, "UNKNOWN_ATTEMPT", unknownAttempt(c)
	}
	held, err := s.db.Query(`SELECT `+checkpointCols+` FROM job_checkpoints
		WHERE request_id=? AND attempt=? AND operation_key=? AND logical_key=?`,
		c.RequestID, c.Attempt, c.OperationKey, c.LogicalKey)
	if err != nil {
		return c, "", exit.Internalf("cannot read the checkpoint journal: %s", err)
	}
	defer held.Close()
	if held.Next() {
		var row Checkpoint
		if err := held.Scan(&row.RequestID, &row.Attempt, &row.OperationKey, &row.LogicalKey,
			&row.ContentDigest, &row.ReceiptID, &row.Outcome, &row.RecordedAt); err != nil {
			return c, "", exit.Internalf("cannot read a checkpoint row: %s", err)
		}
		if row.ContentDigest != c.ContentDigest {
			return row, "CONFLICT", nil
		}
		return row, "REPLAYED", nil
	}
	c.ReceiptID = NewID("crc")
	c.Outcome, c.RecordedAt = "RECORDED", now()
	res, err := s.db.Exec(`INSERT INTO job_checkpoints(request_id,attempt,operation_key,
		logical_key,content_digest,receipt_id,outcome,recorded_at)
		SELECT ?,?,?,?,?,?,?,?
		WHERE EXISTS (SELECT 1 FROM attempts WHERE request_id=? AND attempt=? AND session_id=?
		              AND state IN ('offered','accepted','recovered_open'))`,
		c.RequestID, c.Attempt, c.OperationKey, c.LogicalKey, c.ContentDigest,
		c.ReceiptID, c.Outcome, c.RecordedAt,
		c.RequestID, c.Attempt, c.SessionID)
	if err != nil {
		return c, "", exit.Internalf("cannot journal the checkpoint: %s", err)
	}
	// The pre-check above answers; THIS is the authority. A terminal landing between the
	// two closes the attempt and the insert writes nothing — one statement, no window.
	if n, _ := res.RowsAffected(); n != 1 {
		return c, "UNKNOWN_ATTEMPT", unknownAttempt(c)
	}
	return c, "RECORDED", nil
}

func unknownAttempt(c Checkpoint) *exit.Error {
	return exit.New(exit.NotFound,
		"checkpoint %s/%s refused: %s#%d is not an OPEN attempt of session %s",
		c.OperationKey, c.LogicalKey, c.RequestID, c.Attempt, c.SessionID).
		WithRemedy("a checkpoint belongs to the attempt that is running it, and only its " +
			"own session may declare one")
}

const checkpointCols = `request_id,attempt,operation_key,logical_key,content_digest,
	receipt_id,outcome,recorded_at`

// Checkpoints lists one request's journaled checkpoint identities, in arrival order.
func (s *Store) Checkpoints(requestID string) ([]Checkpoint, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+checkpointCols+` FROM job_checkpoints
		WHERE request_id=? ORDER BY attempt, recorded_at`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot list the checkpoints of %s: %s", requestID, err)
	}
	defer rows.Close()
	var out []Checkpoint
	for rows.Next() {
		var c Checkpoint
		if err := rows.Scan(&c.RequestID, &c.Attempt, &c.OperationKey, &c.LogicalKey,
			&c.ContentDigest, &c.ReceiptID, &c.Outcome, &c.RecordedAt); err != nil {
			return nil, exit.Internalf("cannot read a checkpoint row: %s", err)
		}
		out = append(out, c)
	}
	return out, nil
}
