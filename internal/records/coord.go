package records

import (
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

// The coordinator's half of the ONE lifecycle authority (cl-001). Worker processes,
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

var coordSchema = []string{`
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
  worker       TEXT    NOT NULL DEFAULT ''
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
-- The DURABLE checkpoint exchange's coordinator half (cr-009 §Checkpoints). The worker
-- has already made the save durable in its OWN journal; this is a SECOND observation,
-- at-least-once like every other durable message (law 9). The identity is closed:
-- repeating it replays the receipt, and the same key with different bytes CONFLICTS —
-- a coordinator conflict is a journaled fault and never un-writes the worker's fact.
CREATE TABLE IF NOT EXISTS job_checkpoints (
  request_id    TEXT    NOT NULL,
  attempt       INTEGER NOT NULL,
  operation_key TEXT    NOT NULL,
  logical_key   TEXT    NOT NULL,
  content_digest TEXT   NOT NULL,
  receipt_id    TEXT    NOT NULL,
  outcome       TEXT    NOT NULL,
  recorded_at   TEXT    NOT NULL,
  PRIMARY KEY (request_id, operation_key, logical_key)
)`, `
CREATE TABLE IF NOT EXISTS attempts (
  request_id       TEXT    NOT NULL REFERENCES requests(id),
  attempt          INTEGER NOT NULL,
  attempt_key      TEXT    NOT NULL UNIQUE,
  instance_id      TEXT    NOT NULL REFERENCES worker_processes(instance_id),
  session_id       TEXT    NOT NULL,
  invocation_digest TEXT    NOT NULL,
  invocation        BLOB    NOT NULL,
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
)`}

// widen carries the columns a table gained after some root already created it. Applied
// after `schema`, and a duplicate-column answer means it is already there.
var widen = []string{
	`ALTER TABLE requests ADD COLUMN kind TEXT NOT NULL DEFAULT 'serving'`,
	`ALTER TABLE requests ADD COLUMN worker TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN org TEXT NOT NULL DEFAULT ''`,
	`ALTER TABLE requests ADD COLUMN trees TEXT NOT NULL DEFAULT ''`,
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
	State                                 string // spawned | registered | closed
	OpenedAt                              string
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
		deviceList(w.Devices), w.PID, w.Birth, "spawned", now(),
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
		  state='spawned', opened_at=excluded.opened_at, closed_at=''`, args...)
	if err != nil {
		return exit.Internalf("cannot journal the worker process %s: %s", w.InstanceID, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		held, _ := s.DeviceHolders(w.Devices)
		return exit.New(exit.Conflict,
			"the device envelope [%s] is already granted to %s",
			strings.Join(w.Devices, ","), strings.Join(held, ",")).
			WithRemedy("one process per device: stop the holding worker first").
			WithNext("cozy status")
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
// coordinator only binds it, and the unique index refuses two live workers sharing one.
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
	_, err := s.db.Exec(`UPDATE worker_processes SET pid=?, birth=? WHERE instance_id=?`,
		pid, birth, instanceID)
	if err != nil {
		return exit.Internalf("cannot record the birth identity of %s: %s", instanceID, err)
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
}

const requestCols = `id,idem_key,body_digest,endpoint,entrypoint,plan_id,payload,outputs,
	state,ordinal,requeues,created_at,kind,org,trees,worker`

func scanRequest(row interface{ Scan(...any) error }) (Request, error) {
	var r Request
	err := row.Scan(&r.ID, &r.IdemKey, &r.BodyDigest, &r.Endpoint, &r.Entrypoint, &r.PlanID,
		&r.Payload, &r.Outputs, &r.State, &r.Ordinal, &r.Requeues, &r.CreatedAt,
		&r.Kind, &r.Org, &r.Trees, &r.Worker)
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

// Requests lists rows newest-first, optionally filtered by state. `state=""` is every
// request; the client contract's listing route reads exactly this.
func (s *Store) Requests(state string, limit int) ([]Request, *exit.Error) {
	return s.RequestsOfKind("", state, limit)
}

// RequestsOfKind narrows the same listing to one ATTEMPT CLASS. `cozy job ls` reads jobs
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

// Owed is every request this authority still owes work for and that has NO live attempt:
// the ones a restarted service must put back on its dispatch queue. A request WITH a live
// or recovered attempt is not owed capacity — it is owed a terminal, and the
// recovered-attempts law is what settles that.
func (s *Store) Owed() ([]Request, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + requestCols + ` FROM requests r
		WHERE r.state IN ('submitted','queued')
		  AND NOT EXISTS (SELECT 1 FROM attempts a WHERE a.request_id=r.id
		                  AND a.state IN ('dispatching','accepted','recovered_open'))
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
		              AND a.state IN ('dispatching','accepted','recovered_open'))
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

// SettleRequest records the request's final state. Only a terminal the coordinator
// ACCEPTED can settle one.
func (s *Store) SettleRequest(id, state string) *exit.Error {
	if _, err := s.db.Exec(`UPDATE requests SET state=? WHERE id=?`, state, id); err != nil {
		return exit.Internalf("cannot settle request %s: %s", id, err)
	}
	return nil
}

// ChargeRequeue consumes one unit of the request's durable requeue budget. Retry is
// BOUNDED and the bound is durable: a worker that dies on every attempt exhausts it
// instead of running forever.
func (s *Store) ChargeRequeue(id string, max int64) (int64, *exit.Error) {
	res, err := s.db.Exec(`UPDATE requests SET requeues=requeues+1 WHERE id=? AND requeues<?`, id, max)
	if err != nil {
		return 0, exit.Internalf("cannot charge a requeue for %s: %s", id, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return max, exit.New(exit.Failed, "%s exhausted its requeue budget of %d", id, max)
	}
	r, e := s.RequestRow(id)
	if e != nil {
		return 0, e
	}
	return r.Requeues, nil
}

// Submit records one durable request under its idempotency key. The same key with the
// same body digest answers the SAME request; the same key with a different body is a
// conflict, never a second execution wearing one name.
func (s *Store) Submit(r Request) (Request, bool, *exit.Error) {
	existing, err := scanRequest(s.db.QueryRow(
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
	r.CreatedAt = now()
	r.State = "submitted"
	if r.Kind == "" {
		r.Kind = "serving"
	}
	if _, err := s.db.Exec(`INSERT INTO requests(id,idem_key,body_digest,endpoint,entrypoint,
		plan_id,payload,outputs,state,ordinal,requeues,created_at,kind,org,trees,worker)
		VALUES(?,?,?,?,?,?,?,?,?,0,0,?,?,?,?,?)`,
		r.ID, r.IdemKey, r.BodyDigest, r.Endpoint, r.Entrypoint, r.PlanID, r.Payload,
		r.Outputs, r.State, r.CreatedAt, r.Kind, r.Org, r.Trees, r.Worker); err != nil {
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
	State               string // dispatching | accepted | recovered_open | terminal | closed
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
		WHERE request_id=? AND state IN ('dispatching','accepted')`, requestID).Scan(&live); err != nil {
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
// the coordinator dispatched attempt 2 of a request whose attempt 1 was starting. The
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
		session_id,invocation_digest,invocation,state,dispatched_at)
		VALUES(?,?,?,?,?,?,?,'dispatching',?)`,
		a.RequestID, a.Attempt, a.AttemptKey, a.InstanceID, a.SessionID,
		a.InvocationDigest, a.InvocationCanonical, a.DispatchedAt); err != nil {
		return 0, exit.New(exit.Conflict, "cannot journal attempt %s#%d: %s", a.RequestID, a.Attempt, err).
			WithRemedy("an attempt ordinal is written once")
	}
	if _, err := tx.Exec(`UPDATE requests SET state='dispatching', ordinal=? WHERE id=?`,
		a.Attempt, a.RequestID); err != nil {
		return 0, exit.Internalf("cannot advance request %s: %s", a.RequestID, err)
	}
	if err := tx.Commit(); err != nil {
		return 0, exit.Internalf("the dispatch transaction did not commit: %s", err)
	}
	return ordinal, nil
}

// Accepted records AttemptAccepted's journaled digests. They never move afterwards.
func (s *Store) Accepted(requestID string, attempt int64, sessionID, planDigest, construction, summary string) *exit.Error {
	res, err := s.db.Exec(`UPDATE attempts SET state='accepted', plan_digest=?, construction=?,
		plan_summary=?, accepted_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state='dispatching'`,
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
// caller to shape. `Path` is the coordinator's own knowledge of where the runtime was
// granted to write, and it never leaves this process.
type Output struct {
	OutputID string
	MediaID  string
	Path     string
	Digest   string
	Length   int64
	MimeType string
}

// Terminal is what a worker journaled and the coordinator is about to make authoritative.
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
	// to keep — and "the client" is this coordinator.
	TriageDigest string
	TriageLength int64
	TriagePath   string
	Body         []byte
	Outputs      []Output
	// Event is the attempt-end lifecycle event, appended INSIDE this transaction so the
	// stream cannot disagree with the authority about whether the request ended.
	EventType    string
	EventPayload map[string]any
	// RequestState is what the REQUEST row becomes. It is not always the attempt's own
	// status: an attempt the coordinator will requeue leaves the request QUEUED, and
	// writing `abandoned` there would make the status document say `failed` for a
	// request that is still going.
	RequestState string
	// Publication is the job lane's durable publication (cl-004), written INSIDE this
	// transaction. `nil` for a serving attempt — and for a job attempt the coordinator
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
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('dispatching','accepted','recovered_open')`,
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
	for i, o := range t.Outputs {
		if o.MediaID == "" {
			o.MediaID = NewID("med")
			t.Outputs[i].MediaID = o.MediaID
		}
		if _, err := tx.Exec(`INSERT INTO outputs(request_id,attempt,output_id,media_id,path,digest,
			length,mime_type,visible_at) VALUES(?,?,?,?,?,?,?,?,?)`,
			t.RequestID, t.Attempt, o.OutputID, o.MediaID, o.Path, o.Digest, o.Length,
			o.MimeType, visible); err != nil {
			return false, exit.Internalf("cannot publish output %s of %s#%d: %s",
				o.OutputID, t.RequestID, t.Attempt, err)
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
	_, err := s.db.Exec(`UPDATE attempts SET state='closed' WHERE request_id=? AND attempt=?
		AND state='terminal'`, requestID, attempt)
	if err != nil {
		return exit.Internalf("cannot close %s#%d: %s", requestID, attempt, err)
	}
	return nil
}

// Recover marks an attempt the worker reported on Register as an OPEN OBLIGATION. While
// it is open, NextOrdinal refuses for that request id — the whole point of the
// recovered-journal handshake.
func (s *Store) Recover(requestID string, attempt int64, sessionID string) *exit.Error {
	res, err := s.db.Exec(`UPDATE attempts SET state='recovered_open', session_id=?
		WHERE request_id=? AND attempt=? AND state IN ('dispatching','accepted')`,
		sessionID, requestID, attempt)
	if err != nil {
		return exit.Internalf("cannot record the recovered attempt %s#%d: %s", requestID, attempt, err)
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return exit.New(exit.NotFound,
			"the worker reported a recovered attempt %s#%d this coordinator never assigned",
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
// what a coordinator asks when that worker's process dies: those attempts are not
// finished and not failed — they are unsettled, and the only thing that can settle one is
// the supervisor's own journal, replayed by a worker in the SAME slot.
func (s *Store) OpenAttemptsOf(instanceID string) ([]Attempt, *exit.Error) {
	return s.attemptsWhere(
		`instance_id=? AND state IN ('dispatching','accepted','recovered_open')`, instanceID)
}

func (s *Store) attemptsWhere(where string, args ...any) ([]Attempt, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,attempt,attempt_key,instance_id,session_id,
		invocation_digest,invocation,state,plan_digest,construction,plan_summary,terminal_id,
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
			&a.InvocationDigest, &a.InvocationCanonical, &a.State, &a.PlanDigest, &a.Construction,
			&a.PlanSummary, &a.TerminalID, &a.TerminalDigest, &a.TerminalStatus, &a.TerminalCause,
			&a.SafeMessage, &a.TriageSubject, &a.TriageDigest, &a.TriageLength, &a.TriagePath,
			&a.TerminalBody, &a.DispatchedAt, &a.AcceptedAt, &a.ClosedAt); err != nil {
			return nil, exit.Internalf("cannot read an attempt row: %s", err)
		}
		out = append(out, a)
	}
	return out, nil
}

// VisibleOutputs answers ONLY for an attempt whose terminal the coordinator accepted.
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

// Publications lists every committed publication, newest first.
func (s *Store) Publications(limit int) ([]Publication, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+publicationCols+` FROM publications
		ORDER BY committed_at DESC LIMIT ?`, limit)
	if err != nil {
		return nil, exit.Internalf("cannot list publications: %s", err)
	}
	defer rows.Close()
	var out []Publication
	for rows.Next() {
		p, err := scanPublication(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read a publication row: %s", err)
		}
		out = append(out, p)
	}
	return out, nil
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
}

// RecordCheckpoint journals the coordinator's copy of one durable save and answers with
// the row plus what happened: RECORDED (new), REPLAYED (the same identity again) or
// CONFLICT (the same keys, different bytes). It never replaces a recorded digest — the
// worker's own journal already made that fact durable, and a coordinator that overwrote
// it would be a second authority over one fact.
func (s *Store) RecordCheckpoint(c Checkpoint) (Checkpoint, string, *exit.Error) {
	held, err := s.db.Query(`SELECT `+checkpointCols+` FROM job_checkpoints
		WHERE request_id=? AND operation_key=? AND logical_key=?`,
		c.RequestID, c.OperationKey, c.LogicalKey)
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
	if _, err := s.db.Exec(`INSERT INTO job_checkpoints(request_id,attempt,operation_key,
		logical_key,content_digest,receipt_id,outcome,recorded_at) VALUES(?,?,?,?,?,?,?,?)`,
		c.RequestID, c.Attempt, c.OperationKey, c.LogicalKey, c.ContentDigest,
		c.ReceiptID, c.Outcome, c.RecordedAt); err != nil {
		return c, "", exit.Internalf("cannot journal the checkpoint: %s", err)
	}
	return c, "RECORDED", nil
}

const checkpointCols = `request_id,attempt,operation_key,logical_key,content_digest,
	receipt_id,outcome,recorded_at`

// Checkpoints lists one request's journaled checkpoint identities, in arrival order.
func (s *Store) Checkpoints(requestID string) ([]Checkpoint, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+checkpointCols+` FROM job_checkpoints
		WHERE request_id=? ORDER BY recorded_at`, requestID)
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

// Counts is what bare `cozy` renders: one line of live facts, read from the authority.
func (s *Store) Counts() (map[string]int, *exit.Error) {
	out := map[string]int{}
	for name, query := range map[string]string{
		"workers":   `SELECT COUNT(*) FROM worker_processes WHERE state != 'closed'`,
		"requests":  `SELECT COUNT(*) FROM requests`,
		"attempts":  `SELECT COUNT(*) FROM attempts`,
		"live":      `SELECT COUNT(*) FROM attempts WHERE state IN ('dispatching','accepted')`,
		"recovered": `SELECT COUNT(*) FROM attempts WHERE state='recovered_open'`,
		"outputs":   `SELECT COUNT(*) FROM outputs`,
	} {
		var n int
		if err := s.db.QueryRow(query).Scan(&n); err != nil {
			return nil, exit.Internalf("cannot read the %s count: %s", name, err)
		}
		out[name] = n
	}
	return out, nil
}
