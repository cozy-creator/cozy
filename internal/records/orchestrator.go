package records

import (
	"bytes"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

// The orchestrator's half of the ONE lifecycle authority (cl-001). Worker processes,
// their sessions, requests, attempts and outputs are rows in the SAME database as the
// install generations and pins — one authority, one transaction boundary, no second
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

var orchestratorSchema = []string{`
CREATE TABLE IF NOT EXISTS worker_processes (
  instance_id     TEXT PRIMARY KEY,
  endpoint        TEXT    NOT NULL,
  generation      TEXT    REFERENCES install_generations(id),
  release_id      TEXT    NOT NULL,
  worker_id       TEXT    NOT NULL,
  devices         TEXT    NOT NULL,
  pid             INTEGER NOT NULL,
  birth           TEXT    NOT NULL,
  session_id      TEXT,
  incarnation     INTEGER NOT NULL DEFAULT 0,
  readiness_epoch INTEGER NOT NULL DEFAULT 0,
  revision        INTEGER NOT NULL DEFAULT 0,
  intake          TEXT    NOT NULL DEFAULT '',
  state           TEXT    NOT NULL,
  opened_at       TEXT    NOT NULL,
  closed_at       TEXT    NOT NULL DEFAULT ''
)`, `
CREATE UNIQUE INDEX IF NOT EXISTS worker_session ON worker_processes(session_id)
  WHERE session_id IS NOT NULL`, `
CREATE TABLE IF NOT EXISTS placement_acquisition_observations (
  instance_id                 TEXT NOT NULL,
  worker_boot_id              TEXT NOT NULL,
  placement_id                TEXT NOT NULL,
  placement_spec_digest       TEXT NOT NULL,
  endpoint_started_ns         INTEGER NOT NULL,
  endpoint_ended_ns           INTEGER NOT NULL,
  endpoint_downloaded_bytes   INTEGER NOT NULL,
  endpoint_reused_bytes       INTEGER NOT NULL,
  model_started_ns            INTEGER NOT NULL,
  model_ended_ns              INTEGER NOT NULL,
  model_downloaded_bytes      INTEGER NOT NULL,
  model_reused_bytes          INTEGER NOT NULL,
  observed_at                 TEXT NOT NULL,
  PRIMARY KEY (instance_id, worker_boot_id, placement_id, placement_spec_digest)
)`, `
CREATE TABLE IF NOT EXISTS requests (
  id           TEXT PRIMARY KEY,
  idem_key     TEXT    NOT NULL UNIQUE,
  body_digest  TEXT    NOT NULL,
  endpoint     TEXT    NOT NULL,
  entrypoint   TEXT    NOT NULL,
  plan_id      TEXT    NOT NULL,
  payload      BLOB    NOT NULL,
  outputs      TEXT    NOT NULL DEFAULT '',
  state        TEXT    NOT NULL,
  ordinal      INTEGER NOT NULL DEFAULT 0,
  requeues     INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT    NOT NULL,
  kind         TEXT    NOT NULL DEFAULT 'serving',
  org          TEXT    NOT NULL DEFAULT '',
  trees        TEXT    NOT NULL DEFAULT '',
  worker       TEXT    NOT NULL DEFAULT '',
  install_id   TEXT    REFERENCES install_generations(id),
  assets       TEXT    NOT NULL DEFAULT '[]',
  artifact_outputs TEXT NOT NULL DEFAULT '[]'
)`, `
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
)`, `
CREATE TABLE IF NOT EXISTS attempts (
  request_id       TEXT    NOT NULL REFERENCES requests(id),
  attempt          INTEGER NOT NULL,
  attempt_key      TEXT    NOT NULL UNIQUE,
  instance_id      TEXT    NOT NULL REFERENCES worker_processes(instance_id),
  session_id       TEXT    NOT NULL,
  invocation_digest TEXT    NOT NULL,
  invocation        BLOB    NOT NULL,
  artifact_outputs  TEXT    NOT NULL DEFAULT '[]',
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
  triage_path      TEXT    NOT NULL DEFAULT '',
  terminal_body    BLOB,
  dispatched_at    TEXT    NOT NULL,
  accepted_at      TEXT    NOT NULL DEFAULT '',
  closed_at        TEXT    NOT NULL DEFAULT '',
  media_cleaned    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (request_id, attempt)
)`, `
CREATE TABLE IF NOT EXISTS outputs (
  request_id TEXT    NOT NULL,
  attempt    INTEGER NOT NULL,
  output_id  TEXT    NOT NULL,
  media_id   TEXT    NOT NULL UNIQUE,
  path       TEXT    NOT NULL,
  digest     TEXT    NOT NULL,
  length     INTEGER NOT NULL,
  mime_type  TEXT    NOT NULL,
  visible_at TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, output_id),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`, `
-- Exact Runtime-authored ArtifactReceipt/1 documents carried by AttemptOutcomeBody/3.
-- They commit in the same transaction as the outcome and therefore exist before Ack.
CREATE TABLE IF NOT EXISTS artifact_receipts (
  request_id         TEXT    NOT NULL,
  attempt            INTEGER NOT NULL,
  output_slot        TEXT    NOT NULL,
  owner_scope        TEXT    NOT NULL,
  invocation_digest  TEXT    NOT NULL,
  transaction_id     TEXT    NOT NULL,
  receipt_digest     TEXT    NOT NULL,
  receipt_bytes      BLOB    NOT NULL,
  recorded_at        TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, output_slot),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`, `
-- Creator's first-wins artifact disposition. The exact decision is journaled before send;
-- the exact Runtime result is journaled before the outcome is acknowledged.
CREATE TABLE IF NOT EXISTS artifact_finalizations (
  request_id         TEXT    NOT NULL,
  attempt            INTEGER NOT NULL,
  instance_id        TEXT    NOT NULL,
  owner_scope        TEXT    NOT NULL,
  invocation_digest  TEXT    NOT NULL,
  output_slot        TEXT    NOT NULL,
  disposition        TEXT    NOT NULL,
  receipt_digest     TEXT    NOT NULL DEFAULT '',
  scratch_root_id    TEXT    NOT NULL DEFAULT '',
  decision_digest    TEXT    NOT NULL,
  decision_bytes     BLOB    NOT NULL,
  result_outcome     TEXT    NOT NULL DEFAULT '',
  result_digest      TEXT    NOT NULL DEFAULT '',
  result_bytes       BLOB    NOT NULL DEFAULT x'',
  result_receipt_digest TEXT NOT NULL DEFAULT '',
  result_receipt_bytes  BLOB NOT NULL DEFAULT x'',
  recorded_at        TEXT    NOT NULL,
  completed_at       TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (request_id, invocation_digest, output_slot),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`}

// rebuild carries the tables whose IDENTITY changed — the one migration `ALTER TABLE`
// cannot express. `stale` is a fragment of the OLD table's own DDL, read back out of
// `sqlite_master`, so the rebuild is skipped on every root that already carries the new
// shape: idempotent by observation, not by a version counter nobody maintains.
type tableRebuild struct {
	table string
	stale string
	steps []string
}

var rebuild = []tableRebuild{{
	table: "job_checkpoints",
	stale: "PRIMARY KEY (request_id, operation_key, logical_key)",
	steps: []string{
		`ALTER TABLE job_checkpoints RENAME TO job_checkpoints_pre_attempt_key`,
		// Recreated by `schema` on the next Open? No — this runs inside the same Open, so
		// the new shape is created here, from the same text the schema declares.
		`CREATE TABLE job_checkpoints (
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
)`,
		// Rows whose attempt does not exist are dropped, and that is the correction, not a
		// loss: under the new identity a checkpoint of an attempt that was never dispatched
		// is exactly the row this key exists to make unrepresentable.
		`INSERT INTO job_checkpoints SELECT c.* FROM job_checkpoints_pre_attempt_key c
		   WHERE EXISTS (SELECT 1 FROM attempts a
		                 WHERE a.request_id=c.request_id AND a.attempt=c.attempt)`,
		`DROP TABLE job_checkpoints_pre_attempt_key`,
	},
}}

// widen carries the columns a table gained after some root already created it. Applied
// after `schema`, and a duplicate-column answer means it is already there.
var widen = []string{
	`ALTER TABLE install_generations ADD COLUMN runtime TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE install_generations ADD COLUMN project_dir TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE managed_profile_installs ADD COLUMN native_evidence_digest TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN kind TEXT NOT NULL DEFAULT 'serving'`,
	`ALTER TABLE requests ADD COLUMN worker TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN org TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN trees TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN assets TEXT NOT NULL DEFAULT '[]'`,
	`ALTER TABLE requests ADD COLUMN install_id TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN artifact_outputs TEXT NOT NULL DEFAULT '[]'`,
	`ALTER TABLE attempts ADD COLUMN media_cleaned INTEGER NOT NULL DEFAULT 0`,
	`ALTER TABLE attempts ADD COLUMN artifact_outputs TEXT NOT NULL DEFAULT '[]'`,
	`ALTER TABLE outputs ADD COLUMN reclaimed_at TEXT NOT NULL DEFAULT ''`,
}

// normalize hard-cuts pre-launch lifecycle spellings whose durable meaning was refined.
// Each statement is idempotent and runs after every open, so an interrupted upgrade is
// simply retried.
var normalize = []string{
	`UPDATE attempts SET state='offered' WHERE state='dispatching'`,
	`UPDATE requests SET state='requeue_pending' WHERE state='queued' AND EXISTS (
	  SELECT 1 FROM attempts a WHERE a.request_id=requests.id
	  AND a.attempt=requests.ordinal AND a.state IN ('terminal','closed'))`,
}

func now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// NewID mints an opaque local id. Attempt keys are opaque on purpose: a triage bundle is
// served by attempt key, never by a path a client can shape.
func NewID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + "-" + hex.EncodeToString(b[:])
}

// --------------------------------------------------------------------------- workers

// WorkerProcess is one endpoint worker: its OS process-birth identity, the protocol
// identities it reported, and the generation-scoped device grant it holds. The grant is
// this row's `Devices` field — one process, one visible device set, one generation.
type WorkerProcess struct {
	InstanceID                            string
	Endpoint                              string
	Generation                            string
	ReleaseID                             string
	WorkerID                              string
	Devices                               []string
	PID                                   int
	Birth                                 string // the OS process-birth identity: /proc starttime, never the pid alone
	SessionID                             string
	Incarnation, ReadinessEpoch, Revision int64
	Intake                                string
	State                                 string // spawned_without_birth | spawned | registered | closed
	OpenedAt                              string
}

type AcquisitionLeg struct {
	StartedNS, EndedNS           uint64
	DownloadedBytes, ReusedBytes uint64
}

type PlacementAcquisition struct {
	InstanceID, WorkerBootID, PlacementID, PlacementSpecDigest string
	Endpoint, Model                                            AcquisitionLeg
	ObservedAt                                                 string
}

const acquisitionColumns = `instance_id,worker_boot_id,placement_id,placement_spec_digest,
	endpoint_started_ns,endpoint_ended_ns,endpoint_downloaded_bytes,endpoint_reused_bytes,
	model_started_ns,model_ended_ns,model_downloaded_bytes,model_reused_bytes,observed_at`

func scanPlacementAcquisition(row interface{ Scan(...any) error }) (PlacementAcquisition, error) {
	var out PlacementAcquisition
	err := row.Scan(&out.InstanceID, &out.WorkerBootID, &out.PlacementID, &out.PlacementSpecDigest,
		&out.Endpoint.StartedNS, &out.Endpoint.EndedNS,
		&out.Endpoint.DownloadedBytes, &out.Endpoint.ReusedBytes,
		&out.Model.StartedNS, &out.Model.EndedNS,
		&out.Model.DownloadedBytes, &out.Model.ReusedBytes, &out.ObservedAt)
	return out, err
}

// ObservePlacementAcquisition persists the latest worker-authored observation for one
// exact placement spec. Counters and clock points may advance; they never move backward.
func (s *Store) ObservePlacementAcquisition(in PlacementAcquisition) *exit.Error {
	if in.InstanceID == "" || in.WorkerBootID == "" || in.PlacementID == "" ||
		in.PlacementSpecDigest == "" {
		return exit.Named(exit.Conflict, "placement.acquisition_identity_missing",
			"an acquisition observation is missing worker, boot, placement, or spec identity")
	}
	if err := validAcquisitionLeg("endpoint", in.Endpoint); err != nil {
		return err
	}
	if err := validAcquisitionLeg("model", in.Model); err != nil {
		return err
	}
	if in.ObservedAt == "" {
		in.ObservedAt = now()
	}
	result, err := s.db.Exec(`INSERT INTO placement_acquisition_observations(`+acquisitionColumns+`)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
		ON CONFLICT(instance_id,worker_boot_id,placement_id,placement_spec_digest)
		DO UPDATE SET endpoint_started_ns=excluded.endpoint_started_ns,
		 endpoint_ended_ns=excluded.endpoint_ended_ns,
		 endpoint_downloaded_bytes=excluded.endpoint_downloaded_bytes,
		 endpoint_reused_bytes=excluded.endpoint_reused_bytes,
		 model_started_ns=excluded.model_started_ns,model_ended_ns=excluded.model_ended_ns,
		 model_downloaded_bytes=excluded.model_downloaded_bytes,
		 model_reused_bytes=excluded.model_reused_bytes,observed_at=excluded.observed_at
		WHERE (placement_acquisition_observations.endpoint_started_ns=0 OR
		       (excluded.endpoint_started_ns=placement_acquisition_observations.endpoint_started_ns AND
		        excluded.endpoint_ended_ns>=placement_acquisition_observations.endpoint_ended_ns AND
		        (placement_acquisition_observations.endpoint_ended_ns=0 OR
		         excluded.endpoint_ended_ns=placement_acquisition_observations.endpoint_ended_ns) AND
		        excluded.endpoint_downloaded_bytes>=placement_acquisition_observations.endpoint_downloaded_bytes AND
		        excluded.endpoint_reused_bytes>=placement_acquisition_observations.endpoint_reused_bytes) OR
		       (placement_acquisition_observations.endpoint_ended_ns>0 AND
		        excluded.endpoint_started_ns>placement_acquisition_observations.endpoint_ended_ns))
		  AND (placement_acquisition_observations.model_started_ns=0 OR
		       (excluded.model_started_ns=placement_acquisition_observations.model_started_ns AND
		        excluded.model_ended_ns>=placement_acquisition_observations.model_ended_ns AND
		        (placement_acquisition_observations.model_ended_ns=0 OR
		         excluded.model_ended_ns=placement_acquisition_observations.model_ended_ns) AND
		        excluded.model_downloaded_bytes>=placement_acquisition_observations.model_downloaded_bytes AND
		        excluded.model_reused_bytes>=placement_acquisition_observations.model_reused_bytes) OR
		       (placement_acquisition_observations.model_ended_ns>0 AND
		        excluded.model_started_ns>placement_acquisition_observations.model_ended_ns))`,
		in.InstanceID, in.WorkerBootID, in.PlacementID, in.PlacementSpecDigest,
		in.Endpoint.StartedNS, in.Endpoint.EndedNS, in.Endpoint.DownloadedBytes, in.Endpoint.ReusedBytes,
		in.Model.StartedNS, in.Model.EndedNS, in.Model.DownloadedBytes, in.Model.ReusedBytes,
		in.ObservedAt)
	if err != nil {
		return exit.Internalf("cannot persist placement acquisition observation: %s", err)
	}
	changed, err := result.RowsAffected()
	if err != nil {
		return exit.Internalf("cannot read placement acquisition result: %s", err)
	}
	if changed != 1 {
		return exit.Named(exit.Conflict, "placement.acquisition_regressed",
			"worker %s acquisition counters or clock points moved backward", in.WorkerBootID)
	}
	return nil
}

func validAcquisitionLeg(name string, leg AcquisitionLeg) *exit.Error {
	if leg.StartedNS == 0 {
		if leg.EndedNS != 0 || leg.DownloadedBytes != 0 || leg.ReusedBytes != 0 {
			return exit.Named(exit.Conflict, "placement.acquisition_invalid",
				"%s acquisition has counters or an end without a start", name)
		}
		return nil
	}
	if leg.EndedNS != 0 && leg.EndedNS < leg.StartedNS {
		return exit.Named(exit.Conflict, "placement.acquisition_invalid",
			"%s acquisition ended before it started", name)
	}
	const maxSQLiteInteger = uint64(1<<63 - 1)
	if leg.StartedNS > maxSQLiteInteger || leg.EndedNS > maxSQLiteInteger ||
		leg.DownloadedBytes > maxSQLiteInteger || leg.ReusedBytes > maxSQLiteInteger {
		return exit.Named(exit.Conflict, "placement.acquisition_invalid",
			"%s acquisition exceeds the durable integer range", name)
	}
	return nil
}

func (s *Store) PlacementAcquisition(instanceID, bootID, placementID,
	specDigest string) (*PlacementAcquisition, *exit.Error) {
	out, err := scanPlacementAcquisition(s.db.QueryRow(`SELECT `+acquisitionColumns+`
		FROM placement_acquisition_observations
		WHERE instance_id=? AND worker_boot_id=? AND placement_id=? AND placement_spec_digest=?`,
		instanceID, bootID, placementID, specDigest))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read placement acquisition observation: %s", err)
	}
	return &out, nil
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
	if len(w.Devices) == 0 {
		return exit.New(exit.Validation, "a worker process needs a device envelope, even an empty-named one").
			WithRemedy("name the devices this endpoint process may see")
	}
	clauses := make([]string, 0, len(w.Devices))
	args := []any{
		w.InstanceID, w.Endpoint, nullable(w.Generation), w.ReleaseID, w.WorkerID,
		deviceList(w.Devices), w.PID, w.Birth, "spawned_without_birth", now(),
	}
	for _, d := range w.Devices {
		clauses = append(clauses, "w.devices LIKE ?")
		args = append(args, "%"+deviceMark(d)+"%")
	}
	// THE ADMISSION ASKS ABOUT ANOTHER PROCESS, which is why the slot's own row is
	// excluded. A slot restarting itself — the recovered-attempts path, where a dead
	// worker's journal must be replayed by a worker in the SAME slot — is not a second
	// consumer of the envelope, and refusing it deadlocked exactly the recovery it was
	// meant to protect. Concurrency within one slot is serialized by `selectOrStart`;
	// this statement is the fence against a DIFFERENT slot.
	args = append(args, w.InstanceID)
	res, err := s.db.Exec(`
		INSERT INTO worker_processes(instance_id,endpoint,generation,release_id,worker_id,
		  devices,pid,birth,state,opened_at)
		SELECT ?,?,?,?,?,?,?,?,?,?
		WHERE NOT EXISTS (
		  SELECT 1 FROM worker_processes w WHERE w.state != 'closed' AND (`+
		strings.Join(clauses, " OR ")+`) AND w.instance_id != ?)
		ON CONFLICT(instance_id) DO UPDATE SET
		  pid=excluded.pid, birth=excluded.birth, devices=excluded.devices,
		  generation=excluded.generation, release_id=excluded.release_id,
		  session_id=NULL, incarnation=0, readiness_epoch=0, revision=0, intake='',
		  state='spawned_without_birth', opened_at=excluded.opened_at, closed_at=''`, args...)
	if err != nil {
		return exit.Internalf("cannot journal the worker process %s: %s", w.InstanceID, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		held, _ := s.DeviceHolders(w.Devices)
		return exit.New(exit.Conflict,
			"the device envelope [%s] is already granted to %s",
			strings.Join(w.Devices, ","), strings.Join(held, ",")).
			WithRemedy("one process per device: stop the holding worker first").
			WithNext("cozy invoke list")
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
		INSERT INTO worker_processes(instance_id,endpoint,generation,release_id,worker_id,
		  devices,pid,birth,state,opened_at)
		VALUES(?,?,?,?,?,'',0,'','spawned',?)
		ON CONFLICT(instance_id) DO UPDATE SET
		  generation=excluded.generation, release_id=excluded.release_id,
		  session_id=NULL, incarnation=0, readiness_epoch=0, revision=0, intake='',
		  state='spawned', opened_at=excluded.opened_at, closed_at=''`,
		w.InstanceID, w.Endpoint, nullable(w.Generation), w.ReleaseID, w.WorkerID,
		now()); err != nil {
		return exit.Internalf("cannot journal the attached worker %s: %s", w.InstanceID, err)
	}
	return nil
}

func (s *Store) ReviseAttachedWorker(instanceID, endpoint, releaseID string) *exit.Error {
	result, err := s.db.Exec(`UPDATE worker_processes SET endpoint=?,release_id=?
		WHERE instance_id=? AND worker_id='remote' AND state!='closed'`,
		endpoint, releaseID, instanceID)
	if err != nil {
		return exit.Internalf("cannot revise attached worker %s: %s", instanceID, err)
	}
	changed, err := result.RowsAffected()
	if err != nil || changed != 1 {
		return exit.Named(exit.Conflict, "rental.worker_not_live",
			"attached worker %s is no longer one live remote worker", instanceID)
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
func (s *Store) BindSession(instanceID, sessionID string, incarnation int64) *exit.Error {
	res, err := s.db.Exec(`UPDATE worker_processes
		SET session_id=?, incarnation=?, state='registered'
		WHERE instance_id=? AND state != 'closed'`, sessionID, incarnation, instanceID)
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

// ReportWorker records the applied baseline a Report echoes.
func (s *Store) ReportWorker(sessionID, intake string, revision, epoch, incarnation int64) *exit.Error {
	_, err := s.db.Exec(`UPDATE worker_processes
		SET intake=?, revision=?, readiness_epoch=?, incarnation=?
		WHERE session_id=?`, intake, revision, epoch, incarnation, sessionID)
	if err != nil {
		return exit.Internalf("cannot record the worker report for %s: %s", sessionID, err)
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
	rows, err := s.db.Query(`SELECT instance_id,endpoint,COALESCE(generation,''),release_id,
		worker_id,devices,pid,birth,COALESCE(session_id,''),incarnation,readiness_epoch,
		revision,intake,state,opened_at FROM worker_processes WHERE state != 'closed'
		ORDER BY opened_at`)
	if err != nil {
		return nil, exit.Internalf("cannot list worker processes: %s", err)
	}
	defer rows.Close()
	var out []WorkerProcess
	for rows.Next() {
		var w WorkerProcess
		var devices string
		if err := rows.Scan(&w.InstanceID, &w.Endpoint, &w.Generation, &w.ReleaseID,
			&w.WorkerID, &devices, &w.PID, &w.Birth, &w.SessionID, &w.Incarnation,
			&w.ReadinessEpoch, &w.Revision, &w.Intake, &w.State, &w.OpenedAt); err != nil {
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

type Request struct {
	ID         string
	IdemKey    string
	BodyDigest string
	Endpoint   string
	Entrypoint string
	PlanID     string
	Payload    []byte
	// Outputs names one destination per RESULT FIELD PATH. It lives on the request
	// because a REQUEUE re-derives the same grant shape without a client saying so again.
	Outputs   string
	State     string
	Ordinal   int64
	Requeues  int64
	CreatedAt string
	// Kind is the ATTEMPT CLASS: `serving` or `job`. It is the one discriminator the
	// whole job branch hangs off, and it lives on the request because a requeue must
	// re-derive the same class without a client saying so again (cr-009: a job is an
	// attempt class on the one machinery, not a second runtime).
	Kind string
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
	// InstallID pins a durable local request to the exact immutable generation
	// resolved before submission. Remote requests leave it empty.
	InstallID string
	// Assets are the request's durable input-asset bindings. LocalPath points into the
	// authority-owned immutable input store, not at the caller's original file: a requeue
	// after a client exit or service restart therefore grants the same verified bytes.
	Assets []AssetBinding
	// ArtifactOutputs is the immutable ArtifactSink output subset projected beside the
	// InvocationSpec. Rev5 OutputBinding has no kind, so this may never be inferred from
	// ordinary asset outputs or from whichever receipts happen to arrive.
	ArtifactOutputs string
}

// AssetBinding ties one payload asset field to the exact local bytes the request owns.
// FieldPath is also the worker-protocol input_id (`first_frame`,
// `references.0.image`); position is identity for mixed reference lists, never a file
// name inferred later. LocalPath is resolution only and is excluded from submission
// identity; Digest, Length and MediaType are the claims inside InvocationSpec.
type AssetBinding struct {
	FieldPath string `json:"field_path"`
	LocalPath string `json:"local_path"`
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type,omitempty"`
	Order     uint32 `json:"order"`
	MaxBytes  int64  `json:"max_bytes,omitempty"`
}

const requestCols = `id,idem_key,body_digest,endpoint,entrypoint,plan_id,payload,outputs,
	state,ordinal,requeues,created_at,kind,org,trees,worker,COALESCE(install_id,''),assets,artifact_outputs`

func scanRequest(row interface{ Scan(...any) error }) (Request, error) {
	var r Request
	var assets string
	err := row.Scan(&r.ID, &r.IdemKey, &r.BodyDigest, &r.Endpoint, &r.Entrypoint, &r.PlanID,
		&r.Payload, &r.Outputs, &r.State, &r.Ordinal, &r.Requeues, &r.CreatedAt,
		&r.Kind, &r.Org, &r.Trees, &r.Worker, &r.InstallID, &assets, &r.ArtifactOutputs)
	if err == nil && assets != "" {
		err = json.Unmarshal([]byte(assets), &r.Assets)
	}
	return r, err
}

// IsJob answers the attempt class. The default spelling is `serving` so a row written
// before the column existed reads as what it was.
func (r Request) IsJob() bool { return r.Kind == "job" }

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
	return &r, nil
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
	return &r, nil
}

// Requests lists rows newest-first, optionally filtered by state. `state=""` is every
// request; the client contract's listing route reads exactly this.
func (s *Store) Requests(state string, limit int) ([]Request, *exit.Error) {
	return s.RequestsOfKind("", state, limit)
}

// RequestsOfKind narrows the same listing to one ATTEMPT CLASS. `cozy invoke list` reads jobs
// and the request listing reads serving rows — one table, one reader, two questions.
func (s *Store) RequestsOfKind(kind, state string, limit int) ([]Request, *exit.Error) {
	query := `SELECT ` + requestCols + ` FROM requests`
	where := []string{}
	args := []any{}
	if kind != "" {
		where = append(where, `kind=?`)
		args = append(args, kind)
	}
	if state != "" {
		where = append(where, `state=?`)
		args = append(args, state)
	}
	if len(where) > 0 {
		query += ` WHERE ` + strings.Join(where, " AND ")
	}
	query += ` ORDER BY created_at DESC, id DESC LIMIT ?`
	args = append(args, limit)
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, exit.Internalf("cannot list requests: %s", err)
	}
	defer rows.Close()
	var out []Request
	for rows.Next() {
		r, err := scanRequest(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a request row: %s", err)
		}
		out = append(out, r)
	}
	return out, nil
}

// ActiveRequests is every invocation the controller still owes work or a terminal.
// Lifecycle operations use the complete set rather than a presentation-limited request
// listing: omitting row 501 from a safety fence would make `exit` destructive by accident.
func (s *Store) ActiveRequests() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests
		WHERE state IN ('submitted','queued','dispatching','requeue_pending')
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
// the ones a restarted service must put back on its dispatch queue. A request WITH a live
// or recovered attempt is not owed capacity — it is owed a terminal, and the
// recovered-attempts law is what settles that.
func (s *Store) Owed() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests r
		WHERE r.state IN ('submitted','queued')
		  AND NOT EXISTS (SELECT 1 FROM attempts a WHERE a.request_id=r.id
		                  AND a.state IN ('preparing','offered','accepted','recovered_open','terminal'))
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
		              AND a.state IN ('preparing','offered','accepted','recovered_open','terminal'))
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

// AssetInUse answers whether an unsettled request still owns a staged content object.
// Settled request rows retain their identity claims for audit/idempotency, but their bytes
// are no longer execution inputs and must not pin the private input store forever.
func (s *Store) AssetInUse(digest string) (bool, *exit.Error) {
	rows, err := s.db.Query(`SELECT assets FROM requests
		WHERE state IN ('submitted','queued','dispatching','requeue_pending')
		  AND assets <> '[]'`)
	if err != nil {
		return false, exit.Internalf("cannot read live input asset ownership: %s", err)
	}
	defer rows.Close()
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return false, exit.Internalf("cannot read a live input asset binding: %s", err)
		}
		var assets []AssetBinding
		if err := json.Unmarshal([]byte(raw), &assets); err != nil {
			return false, exit.Internalf("cannot decode a live input asset binding: %s", err)
		}
		for _, asset := range assets {
			if asset.Digest == digest {
				return true, nil
			}
		}
	}
	return false, nil
}

// SettleRequest records the request's final state. Only a terminal the orchestrator
// ACCEPTED can settle one.
func (s *Store) SettleRequest(id, state string) *exit.Error {
	if _, err := s.db.Exec(`UPDATE requests SET state=? WHERE id=?`, state, id); err != nil {
		return exit.Internalf("cannot settle request %s: %s", id, err)
	}
	return nil
}

// BeginRequeue crosses the post-ack boundary and charges the durable retry budget in one
// transition. Replays after that transition are harmless: only `requeue_pending` can be
// charged, so a duplicate terminal ack cannot spend twice or mint two ordinals.
func (s *Store) BeginRequeue(id string, max int64) (count int64, started, canceled bool, e *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return 0, false, false, exit.Internalf("cannot begin the requeue transaction: %s", err)
	}
	defer tx.Rollback()
	var state string
	if err := tx.QueryRow(`SELECT state,requeues FROM requests WHERE id=?`, id).
		Scan(&state, &count); err != nil {
		return 0, false, false, exit.Internalf("cannot read requeue state for %s: %s", id, err)
	}
	if state != "requeue_pending" {
		return count, false, false, nil
	}
	if count >= max {
		return count, false, false, exit.New(exit.Failed, "%s exhausted its requeue budget of %d", id, max)
	}
	if _, err := tx.Exec(`UPDATE requests SET state='queued',requeues=requeues+1
		WHERE id=? AND state='requeue_pending'`, id); err != nil {
		return 0, false, false, exit.Internalf("cannot charge a requeue for %s: %s", id, err)
	}
	if err := tx.Commit(); err != nil {
		return 0, false, false, exit.Internalf("cannot commit requeue %s: %s", id, err)
	}
	return count + 1, true, false, nil
}

// Submit records one durable request under its idempotency key. The same key with the
// same body digest answers the SAME request; the same key with a different body is a
// conflict, never a second execution wearing one name.
func (s *Store) Submit(r Request) (Request, bool, *exit.Error) {
	r, assets, problem := prepareRequest(r)
	if problem != nil {
		return Request{}, false, problem
	}
	tx, err := s.db.Begin()
	if err != nil {
		return Request{}, false, exit.Internalf("cannot begin request submission: %s", err)
	}
	defer tx.Rollback()
	recorded, fresh, problem := submitRequestTx(tx, r, assets)
	if problem != nil {
		return Request{}, false, problem
	}
	if err := tx.Commit(); err != nil {
		return Request{}, false, exit.Internalf("cannot commit request %s: %s", r.ID, err)
	}
	return recorded, fresh, nil
}

func prepareRequest(r Request) (Request, string, *exit.Error) {
	r.CreatedAt = now()
	r.State = "submitted"
	if r.Kind == "" {
		r.Kind = "serving"
	}
	assets := []byte("[]")
	if len(r.Assets) > 0 {
		var err error
		assets, err = json.Marshal(r.Assets)
		if err != nil {
			return Request{}, "", exit.Internalf("cannot record request %s assets: %s", r.ID, err)
		}
	}
	return r, string(assets), nil
}

func submitRequestTx(tx *sql.Tx, r Request, assets string) (Request, bool, *exit.Error) {
	existing, err := scanRequest(tx.QueryRow(
		`SELECT `+requestCols+` FROM requests WHERE idem_key=?`, r.IdemKey))
	if err == nil {
		if existing.BodyDigest != r.BodyDigest {
			return Request{}, false, exit.New(exit.Conflict,
				"idempotency key %s already names a request with a different body", r.IdemKey).
				WithRemedy("one key, one body: %s was recorded, %s was submitted",
					short(existing.BodyDigest), short(r.BodyDigest))
		}
		return existing, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return Request{}, false, exit.Internalf("cannot read request %s: %s", r.IdemKey, err)
	}
	if _, err := tx.Exec(`INSERT INTO requests(id,idem_key,body_digest,endpoint,entrypoint,
		plan_id,payload,outputs,state,ordinal,requeues,created_at,kind,org,trees,worker,install_id,assets,
		artifact_outputs)
		VALUES(?,?,?,?,?,?,?,?,?,0,0,?,?,?,?,?,?,?,?)`,
		r.ID, r.IdemKey, r.BodyDigest, r.Endpoint, r.Entrypoint, r.PlanID, r.Payload,
		r.Outputs, r.State, r.CreatedAt, r.Kind, r.Org, r.Trees, r.Worker, nullable(r.InstallID),
		assets, r.ArtifactOutputs); err != nil {
		return Request{}, false, exit.Internalf("cannot record request %s: %s", r.ID, err)
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
	ArtifactOutputs     string
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
	TriagePath          string
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
	if _, err := tx.Exec(`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,
		session_id,invocation_digest,invocation,artifact_outputs,state,dispatched_at)
		VALUES(?,?,?,?,?,?,?,?,'preparing',?)`,
		a.RequestID, a.Attempt, a.AttemptKey, a.InstanceID, a.SessionID,
		a.InvocationDigest, a.InvocationCanonical, a.ArtifactOutputs, a.DispatchedAt); err != nil {
		return 0, exit.New(exit.Conflict, "cannot journal attempt %s#%d: %s", a.RequestID, a.Attempt, err).
			WithRemedy("an attempt ordinal is written once")
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
	request, err := tx.Exec(`UPDATE requests SET state='submitted'
		WHERE id=? AND ordinal=? AND state='dispatching'`, requestID, attempt)
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

// Accepted records AttemptAccepted's journaled digests. They never move afterwards.
func (s *Store) Accepted(requestID string, attempt int64, sessionID, planDigest, construction, summary string) *exit.Error {
	res, err := s.db.Exec(`UPDATE attempts SET state='accepted', plan_digest=?, construction=?,
		plan_summary=?, accepted_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('offered','recovered_open')`,
		planDigest, construction, summary, now(), requestID, attempt, sessionID)
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

// ArtifactReceipt is the exact Runtime-authored ArtifactReceipt/1 identity carried by
// AttemptOutcomeBody/3. ReceiptBytes are never parsed and reserialized for persistence.
type ArtifactReceipt struct {
	RequestID, OwnerScope, InvocationDigest, OutputSlot, TransactionID string
	Attempt                                                            int64
	ReceiptDigest                                                      string
	ReceiptBytes                                                       []byte
	RecordedAt                                                         string
}

// ArtifactFinalization is Creator's durable first-wins disposition and Runtime's exact
// replayable completion. DecisionDigest/Bytes are persisted before send; ResultDigest/Bytes
// are persisted before AttemptOutcomeAck.
type ArtifactFinalization struct {
	RequestID, InstanceID, OwnerScope, InvocationDigest, OutputSlot string
	Attempt                                                         int64
	Disposition, ReceiptDigest, ScratchRootID                       string
	DecisionDigest                                                  string
	DecisionBytes                                                   []byte
	ResultOutcome, ResultDigest                                     string
	ResultBytes                                                     []byte
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
	// TriageDigest/TriageLength/TriagePath are cl-006's PERSISTENCE of the worker's
	// bundle: the bytes are copied out of the worker root and verified against the
	// terminal's own TriageBundleRef before this transaction runs. A one-shot run
	// deletes its worker root (cr-011 §8), so a bundle worth keeping is the client's
	// to keep — and "the client" is this orchestrator.
	TriageDigest          string
	TriageLength          int64
	TriagePath            string
	Body                  []byte
	Outputs               []Output
	ArtifactReceipts      []ArtifactReceipt
	ArtifactFinalizations []ArtifactFinalization
	// Event is the attempt-end lifecycle event, appended INSIDE this transaction so the
	// stream cannot disagree with the authority about whether the request ended.
	EventType    string
	EventPayload map[string]any
	// RequestState is what the REQUEST row becomes. It is not always the attempt's own
	// status: an attempt the orchestrator will requeue leaves the request QUEUED, and
	// writing `abandoned` there would make the status document say `failed` for a
	// request that is still going.
	RequestState string
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
	err = tx.QueryRow(`SELECT state, terminal_digest, session_id, invocation_digest FROM attempts
		WHERE request_id=? AND attempt=?`, t.RequestID, t.Attempt).
		Scan(&state, &digest, &assignedSession, &assignedSpec)
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
	if assignedSession != t.SessionID {
		return false, exit.New(exit.Conflict,
			"terminal for %s#%d refused: session %s does not own that attempt row (%s does)",
			t.RequestID, t.Attempt, t.SessionID, assignedSession).
			WithRemedy("one writer per attempt row")
	}
	if state == "terminal" || state == "closed" {
		if digest != t.TerminalDigest {
			return false, exit.New(exit.Conflict,
				"%s#%d already closed with terminal %s; %s is a different body",
				t.RequestID, t.Attempt, short(digest), short(t.TerminalDigest))
		}
		return false, nil // exact replay: ack again, apply nothing
	}

	res, err := tx.Exec(`UPDATE attempts SET state='terminal', terminal_id=?, terminal_digest=?,
		terminal_status=?, terminal_cause=?, safe_message=?, triage_subject=?, triage_digest=?,
		triage_length=?, triage_path=?, terminal_body=?, closed_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('offered','accepted','recovered_open')`,
		t.TerminalID, t.TerminalDigest, t.Status, t.Cause, t.SafeMessage, t.TriageSubject,
		t.TriageDigest, t.TriageLength, t.TriagePath,
		t.Body, now(), t.RequestID, t.Attempt, t.SessionID)
	if err != nil {
		return false, exit.Internalf("cannot apply the terminal of %s#%d: %s", t.RequestID, t.Attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return false, exit.New(exit.Conflict,
			"terminal for %s#%d refused: the attempt row is in state %q", t.RequestID, t.Attempt, state)
	}
	visible := now()
	for _, o := range t.Outputs {
		if _, err := tx.Exec(`INSERT INTO outputs(request_id,attempt,output_id,media_id,path,digest,
			length,mime_type,visible_at) VALUES(?,?,?,?,?,?,?,?,?)`,
			t.RequestID, t.Attempt, o.OutputID, o.MediaID, o.Path, o.Digest, o.Length,
			o.MimeType, visible); err != nil {
			return false, exit.Internalf("cannot publish output %s of %s#%d: %s",
				o.OutputID, t.RequestID, t.Attempt, err)
		}
	}
	for _, receipt := range t.ArtifactReceipts {
		if _, err := tx.Exec(`INSERT INTO artifact_receipts(request_id,attempt,output_slot,
			owner_scope,invocation_digest,transaction_id,receipt_digest,receipt_bytes,recorded_at)
			VALUES(?,?,?,?,?,?,?,?,?)`, receipt.RequestID, receipt.Attempt, receipt.OutputSlot,
			receipt.OwnerScope, receipt.InvocationDigest, receipt.TransactionID,
			receipt.ReceiptDigest, receipt.ReceiptBytes, visible); err != nil {
			return false, exit.Internalf("cannot record artifact receipt %s of %s#%d: %s",
				receipt.OutputSlot, t.RequestID, t.Attempt, err)
		}
	}
	for _, finalization := range t.ArtifactFinalizations {
		var held string
		err := tx.QueryRow(`SELECT decision_digest FROM artifact_finalizations
			WHERE request_id=? AND invocation_digest=? AND output_slot=?`, finalization.RequestID,
			finalization.InvocationDigest, finalization.OutputSlot).Scan(&held)
		if err == nil {
			if held != finalization.DecisionDigest {
				return false, exit.New(exit.Conflict,
					"artifact output %s already has final intent %s, not %s",
					finalization.OutputSlot, short(held), short(finalization.DecisionDigest))
			}
			continue
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return false, exit.Internalf("cannot read artifact final intent %s of %s: %s",
				finalization.OutputSlot, t.RequestID, err)
		}
		if _, err := tx.Exec(`INSERT INTO artifact_finalizations(request_id,attempt,instance_id,
			owner_scope,invocation_digest,output_slot,disposition,receipt_digest,scratch_root_id,
			decision_digest,decision_bytes,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`,
			finalization.RequestID, finalization.Attempt, finalization.InstanceID,
			finalization.OwnerScope, finalization.InvocationDigest, finalization.OutputSlot,
			finalization.Disposition, finalization.ReceiptDigest, finalization.ScratchRootID,
			finalization.DecisionDigest, finalization.DecisionBytes, visible); err != nil {
			return false, exit.Internalf("cannot record artifact final intent %s of %s#%d: %s",
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

// Recover marks an attempt the worker reported on Register as an OPEN OBLIGATION. While
// it is open, NextOrdinal refuses for that request id — the whole point of the
// recovered-journal handshake.
func (s *Store) Recover(requestID string, attempt int64, sessionID string) *exit.Error {
	res, err := s.db.Exec(`UPDATE attempts SET
		state=CASE WHEN terminal_digest<>'' THEN 'terminal' ELSE 'recovered_open' END,
		session_id=? WHERE request_id=? AND attempt=?
		AND state IN ('preparing','offered','accepted','recovered_open','terminal','closed')`,
		sessionID, requestID, attempt)
	if err != nil {
		return exit.Internalf("cannot record the recovered attempt %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.NotFound,
			"the worker reported a recovered attempt %s#%d this orchestrator never assigned",
			requestID, attempt)
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

// AttemptByKey is the opaque-id read: triage is retrievable by attempt key and by
// nothing else. There is no path in the row and no way to name one.
func (s *Store) AttemptByKey(key string) (*Attempt, *exit.Error) {
	rows, e := s.attemptsWhere(`attempt_key=?`, key)
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
	return s.attemptsWhere(
		`instance_id=? AND state IN ('preparing','offered','accepted','recovered_open','terminal')`, instanceID)
}

// ReadyRequeues are committed retry decisions whose old terminal has crossed the
// acknowledgement boundary. BeginRequeue changes them to queued and charges the budget
// atomically.
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
		invocation_digest,invocation,artifact_outputs,state,plan_digest,construction,plan_summary,terminal_id,
		terminal_digest,terminal_status,terminal_cause,safe_message,triage_subject,
		triage_digest,triage_length,triage_path,
		COALESCE(terminal_body,x''),dispatched_at,accepted_at,closed_at
		FROM attempts WHERE `+where+` ORDER BY request_id, attempt`, args...)
	if err != nil {
		return nil, exit.Internalf("cannot read attempts: %s", err)
	}
	defer rows.Close()
	var out []Attempt
	for rows.Next() {
		var a Attempt
		if err := rows.Scan(&a.RequestID, &a.Attempt, &a.AttemptKey, &a.InstanceID, &a.SessionID,
			&a.InvocationDigest, &a.InvocationCanonical, &a.ArtifactOutputs, &a.State, &a.PlanDigest, &a.Construction,
			&a.PlanSummary, &a.TerminalID, &a.TerminalDigest, &a.TerminalStatus, &a.TerminalCause,
			&a.SafeMessage, &a.TriageSubject, &a.TriageDigest, &a.TriageLength, &a.TriagePath,
			&a.TerminalBody, &a.DispatchedAt, &a.AcceptedAt, &a.ClosedAt); err != nil {
			return nil, exit.Internalf("cannot read an attempt row: %s", err)
		}
		out = append(out, a)
	}
	return out, nil
}

const artifactFinalizationCols = `request_id,attempt,instance_id,owner_scope,invocation_digest,
	output_slot,disposition,receipt_digest,scratch_root_id,decision_digest,decision_bytes,
	result_outcome,result_digest,result_bytes,result_receipt_digest,result_receipt_bytes,
	recorded_at,completed_at`

func scanArtifactFinalization(row interface{ Scan(...any) error }) (ArtifactFinalization, error) {
	var f ArtifactFinalization
	err := row.Scan(&f.RequestID, &f.Attempt, &f.InstanceID, &f.OwnerScope,
		&f.InvocationDigest, &f.OutputSlot, &f.Disposition, &f.ReceiptDigest,
		&f.ScratchRootID, &f.DecisionDigest, &f.DecisionBytes, &f.ResultOutcome,
		&f.ResultDigest, &f.ResultBytes, &f.ResultReceiptDigest, &f.ResultReceiptBytes,
		&f.RecordedAt, &f.CompletedAt)
	return f, err
}

// ArtifactFinalization reads the exact first-wins intent for one semantic output.
func (s *Store) ArtifactFinalization(requestID, invocationDigest, outputSlot string) (*ArtifactFinalization, *exit.Error) {
	f, err := scanArtifactFinalization(s.db.QueryRow(`SELECT `+artifactFinalizationCols+`
		FROM artifact_finalizations WHERE request_id=? AND invocation_digest=? AND output_slot=?`,
		requestID, invocationDigest, outputSlot))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read artifact finalization %s/%s: %s",
			requestID, outputSlot, err)
	}
	return &f, nil
}

// PendingArtifactFinalizations is the exact decision set that must be (re)sent before
// this attempt's outcome may be acknowledged. Ordering by slot makes replay deterministic.
func (s *Store) PendingArtifactFinalizations(requestID string, attempt int64) ([]ArtifactFinalization, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+artifactFinalizationCols+` FROM artifact_finalizations
		WHERE request_id=? AND attempt=? AND result_digest='' ORDER BY output_slot`, requestID, attempt)
	if err != nil {
		return nil, exit.Internalf("cannot read pending artifact finalizations of %s#%d: %s",
			requestID, attempt, err)
	}
	defer rows.Close()
	out := []ArtifactFinalization{}
	for rows.Next() {
		f, err := scanArtifactFinalization(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an artifact finalization row: %s", err)
		}
		out = append(out, f)
	}
	return out, nil
}

func (s *Store) ArtifactFinalizationsOf(requestID string) ([]ArtifactFinalization, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+artifactFinalizationCols+` FROM artifact_finalizations
		WHERE request_id=? ORDER BY attempt,output_slot`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read artifact finalizations of %s: %s", requestID, err)
	}
	defer rows.Close()
	out := []ArtifactFinalization{}
	for rows.Next() {
		f, err := scanArtifactFinalization(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read an artifact finalization row: %s", err)
		}
		out = append(out, f)
	}
	return out, nil
}

// RecordArtifactFinalizeResult journals Runtime's exact replayable result. The same result
// is idempotent; any changed bytes for the first-wins decision conflict forever.
func (s *Store) RecordArtifactFinalizeResult(result ArtifactFinalization) (applied bool, e *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin the artifact finalize result transaction: %s", err)
	}
	defer tx.Rollback()
	held, err := scanArtifactFinalization(tx.QueryRow(`SELECT `+artifactFinalizationCols+`
		FROM artifact_finalizations WHERE request_id=? AND invocation_digest=? AND output_slot=?`,
		result.RequestID, result.InvocationDigest, result.OutputSlot))
	if errors.Is(err, sql.ErrNoRows) {
		return false, exit.New(exit.NotFound, "no artifact final intent for %s/%s",
			result.RequestID, result.OutputSlot)
	}
	if err != nil {
		return false, exit.Internalf("cannot read artifact final intent %s/%s: %s",
			result.RequestID, result.OutputSlot, err)
	}
	if held.InstanceID != result.InstanceID {
		return false, exit.New(exit.Conflict,
			"artifact finalize result for %s/%s came from worker %s; %s owns the transaction",
			result.RequestID, result.OutputSlot, result.InstanceID, held.InstanceID)
	}
	if held.ResultDigest != "" {
		if held.ResultDigest != result.ResultDigest || !bytes.Equal(held.ResultBytes, result.ResultBytes) {
			return false, exit.New(exit.Conflict,
				"artifact finalization %s/%s already completed with %s, not %s",
				result.RequestID, result.OutputSlot, short(held.ResultDigest), short(result.ResultDigest))
		}
		return false, nil
	}
	completed := now()
	updated, err := tx.Exec(`UPDATE artifact_finalizations SET result_outcome=?,result_digest=?,
		result_bytes=?,result_receipt_digest=?,result_receipt_bytes=?,completed_at=?
		WHERE request_id=? AND invocation_digest=? AND output_slot=? AND result_digest=''`,
		result.ResultOutcome, result.ResultDigest, blob(result.ResultBytes),
		result.ResultReceiptDigest, blob(result.ResultReceiptBytes), completed,
		result.RequestID, result.InvocationDigest, result.OutputSlot)
	if err != nil {
		return false, exit.Internalf("cannot record artifact finalize result %s/%s: %s",
			result.RequestID, result.OutputSlot, err)
	}
	if n, _ := updated.RowsAffected(); n != 1 {
		return false, exit.New(exit.Conflict, "artifact finalization %s/%s moved while completing",
			result.RequestID, result.OutputSlot)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit artifact finalize result %s/%s: %s",
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

// ArtifactReceiptsOf exposes the exact bytes the terminal transaction made durable.
func (s *Store) ArtifactReceiptsOf(requestID string, attempt int64) ([]ArtifactReceipt, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,attempt,owner_scope,invocation_digest,output_slot,
		transaction_id,receipt_digest,receipt_bytes,recorded_at FROM artifact_receipts
		WHERE request_id=? AND attempt=? ORDER BY output_slot`, requestID, attempt)
	if err != nil {
		return nil, exit.Internalf("cannot read artifact receipts of %s#%d: %s", requestID, attempt, err)
	}
	defer rows.Close()
	out := []ArtifactReceipt{}
	for rows.Next() {
		var receipt ArtifactReceipt
		if err := rows.Scan(&receipt.RequestID, &receipt.Attempt, &receipt.OwnerScope,
			&receipt.InvocationDigest, &receipt.OutputSlot, &receipt.TransactionID,
			&receipt.ReceiptDigest, &receipt.ReceiptBytes, &receipt.RecordedAt); err != nil {
			return nil, exit.Internalf("cannot read an artifact receipt row: %s", err)
		}
		out = append(out, receipt)
	}
	return out, nil
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
	return out, nil
}

// VisibleOutputsOf narrows visibility to ONE attempt — what a crash arm asks when it
// wants to know whether the bytes THAT attempt wrote ever became a result.
func (s *Store) VisibleOutputsOf(requestID string, attempt int64) ([]Output, *exit.Error) {
	rows, err := s.db.Query(`SELECT o.output_id,o.media_id,o.path,o.digest,o.length,o.mime_type
		FROM outputs o JOIN attempts a ON a.request_id=o.request_id AND a.attempt=o.attempt
		WHERE o.request_id=? AND o.attempt=? AND a.state IN ('terminal','closed')
		ORDER BY o.output_id`, requestID, attempt)
	if err != nil {
		return nil, exit.Internalf("cannot read the outputs of %s#%d: %s", requestID, attempt, err)
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

// Counts is the durable workload summary exposed by the local API beside its live
// endpoint and worker counts. Serving requests and jobs are separate products in the
// dashboard even though they share one lifecycle table and attempt authority.
func (s *Store) Counts() (map[string]int, *exit.Error) {
	out := map[string]int{}
	for name, query := range map[string]string{
		"requests": `SELECT COUNT(*) FROM requests WHERE kind='serving'`,
		"active_requests": `SELECT COUNT(*) FROM requests WHERE kind='serving'
			AND state IN ('submitted','queued','dispatching','requeue_pending')`,
		"jobs": `SELECT COUNT(*) FROM requests WHERE kind='job'`,
		"active_jobs": `SELECT COUNT(*) FROM requests WHERE kind='job'
			AND state IN ('submitted','queued','dispatching','requeue_pending')`,
		"attempts":           `SELECT COUNT(*) FROM attempts`,
		"active_attempts":    `SELECT COUNT(*) FROM attempts WHERE state IN ('preparing','offered','accepted','terminal')`,
		"recovered_attempts": `SELECT COUNT(*) FROM attempts WHERE state='recovered_open'`,
		"outputs":            `SELECT COUNT(*) FROM outputs`,
	} {
		var n int
		if err := s.db.QueryRow(query).Scan(&n); err != nil {
			return nil, exit.Internalf("cannot read the %s count: %s", name, err)
		}
		out[name] = n
	}
	return out, nil
}
