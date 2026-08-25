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
  created_at   TEXT    NOT NULL
)`, `
CREATE TABLE IF NOT EXISTS attempts (
  request_id       TEXT    NOT NULL REFERENCES requests(id),
  attempt          INTEGER NOT NULL,
  attempt_key      TEXT    NOT NULL UNIQUE,
  instance_id      TEXT    NOT NULL REFERENCES worker_processes(instance_id),
  session_id       TEXT    NOT NULL,
  exec_spec_digest TEXT    NOT NULL,
  exec_spec        BLOB    NOT NULL,
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
  path       TEXT    NOT NULL,
  digest     TEXT    NOT NULL,
  length     INTEGER NOT NULL,
  mime_type  TEXT    NOT NULL,
  visible_at TEXT    NOT NULL,
  PRIMARY KEY (request_id, attempt, output_id),
  FOREIGN KEY (request_id, attempt) REFERENCES attempts(request_id, attempt)
)`}

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
	res, err := s.db.Exec(`
		INSERT INTO worker_processes(instance_id,endpoint,generation,release_id,worker_id,
		  devices,pid,birth,state,opened_at)
		SELECT ?,?,?,?,?,?,?,?,?,?
		WHERE NOT EXISTS (
		  SELECT 1 FROM worker_processes w WHERE w.state != 'closed' AND (`+
		strings.Join(clauses, " OR ")+`))
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
}

const requestCols = `id,idem_key,body_digest,endpoint,entrypoint,plan_id,payload,outputs,
	state,ordinal,requeues,created_at`

func scanRequest(row interface{ Scan(...any) error }) (Request, error) {
	var r Request
	err := row.Scan(&r.ID, &r.IdemKey, &r.BodyDigest, &r.Endpoint, &r.Entrypoint, &r.PlanID,
		&r.Payload, &r.Outputs, &r.State, &r.Ordinal, &r.Requeues, &r.CreatedAt)
	return r, err
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
	return &r, nil
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
	if _, err := s.db.Exec(`INSERT INTO requests(id,idem_key,body_digest,endpoint,entrypoint,
		plan_id,payload,outputs,state,ordinal,requeues,created_at) VALUES(?,?,?,?,?,?,?,?,?,0,0,?)`,
		r.ID, r.IdemKey, r.BodyDigest, r.Endpoint, r.Entrypoint, r.PlanID, r.Payload,
		r.Outputs, r.State, r.CreatedAt); err != nil {
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
	RequestID      string
	Attempt        int64
	AttemptKey     string
	InstanceID     string
	SessionID      string
	ExecSpecDigest string
	ExecSpec       []byte
	State          string // dispatching | accepted | recovered_open | terminal | closed
	PlanDigest     string
	Construction   string
	PlanSummary    string
	TerminalID     string
	TerminalDigest string
	TerminalStatus string
	TerminalCause  string
	SafeMessage    string
	TriageSubject  string
	TerminalBody   []byte
	DispatchedAt   string
	AcceptedAt     string
	ClosedAt       string
}

// NextOrdinal mints the next attempt ordinal for a request — and REFUSES while that
// request still has an open recovered attempt (02 §6.2). A new session_id never
// manufactures absence: the coordinator closes what Register reported before it invents
// the next ordinal.
func (s *Store) NextOrdinal(requestID string) (int64, *exit.Error) {
	var open int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM attempts
		WHERE request_id=? AND state='recovered_open'`, requestID).Scan(&open); err != nil {
		return 0, exit.Internalf("cannot read the recovered attempts of %s: %s", requestID, err)
	}
	if open > 0 {
		return 0, exit.New(exit.Conflict,
			"%s has %d recovered attempt(s) still open: no next ordinal may be minted", requestID, open).
			WithRemedy("a recovered attempt is closed by its own journaled terminal, never by assumption")
	}
	var live int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM attempts
		WHERE request_id=? AND state IN ('dispatching','accepted')`, requestID).Scan(&live); err != nil {
		return 0, exit.Internalf("cannot read the live attempts of %s: %s", requestID, err)
	}
	if live > 0 {
		return 0, exit.New(exit.Conflict,
			"%s still holds a live attempt: supersession is explicit, never a new ordinal", requestID).
			WithRemedy("cancel the held attempt (SUPERSEDED) and wait for its journaled terminal")
	}
	var max sql.NullInt64
	if err := s.db.QueryRow(`SELECT MAX(attempt) FROM attempts WHERE request_id=?`, requestID).
		Scan(&max); err != nil {
		return 0, exit.Internalf("cannot read the attempt ordinals of %s: %s", requestID, err)
	}
	return max.Int64 + 1, nil
}

// Dispatch journals the assignment BEFORE StartAttempt goes out: a terminal may only
// cross a restart when the persisted assignment authorizes it.
func (s *Store) Dispatch(a Attempt) *exit.Error {
	a.DispatchedAt = now()
	if a.AttemptKey == "" {
		a.AttemptKey = NewID("att")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the dispatch transaction: %s", err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,
		session_id,exec_spec_digest,exec_spec,state,dispatched_at)
		VALUES(?,?,?,?,?,?,?,'dispatching',?)`,
		a.RequestID, a.Attempt, a.AttemptKey, a.InstanceID, a.SessionID,
		a.ExecSpecDigest, a.ExecSpec, a.DispatchedAt); err != nil {
		return exit.New(exit.Conflict, "cannot journal attempt %s#%d: %s", a.RequestID, a.Attempt, err).
			WithRemedy("an attempt ordinal is written once")
	}
	if _, err := tx.Exec(`UPDATE requests SET state='dispatching', ordinal=? WHERE id=?`,
		a.Attempt, a.RequestID); err != nil {
		return exit.Internalf("cannot advance request %s: %s", a.RequestID, err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("the dispatch transaction did not commit: %s", err)
	}
	return nil
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
type Output struct {
	OutputID string
	Path     string
	Digest   string
	Length   int64
	MimeType string
}

// Terminal is what a worker journaled and the coordinator is about to make authoritative.
type Terminal struct {
	RequestID      string
	Attempt        int64
	SessionID      string
	ExecSpecDigest string
	TerminalID     string
	TerminalDigest string
	Status         string
	Cause          string
	SafeMessage    string
	TriageSubject  string
	Body           []byte
	Outputs        []Output
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
	err = tx.QueryRow(`SELECT state, terminal_digest, session_id, exec_spec_digest FROM attempts
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
	if assignedSpec != t.ExecSpecDigest {
		return false, exit.New(exit.Validation,
			"terminal for %s#%d refused: it closes %s, the assignment is %s",
			t.RequestID, t.Attempt, short(t.ExecSpecDigest), short(assignedSpec))
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
		terminal_status=?, terminal_cause=?, safe_message=?, triage_subject=?, terminal_body=?,
		closed_at=?
		WHERE request_id=? AND attempt=? AND session_id=? AND state IN ('dispatching','accepted','recovered_open')`,
		t.TerminalID, t.TerminalDigest, t.Status, t.Cause, t.SafeMessage, t.TriageSubject,
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
		if _, err := tx.Exec(`INSERT INTO outputs(request_id,attempt,output_id,path,digest,
			length,mime_type,visible_at) VALUES(?,?,?,?,?,?,?,?)`,
			t.RequestID, t.Attempt, o.OutputID, o.Path, o.Digest, o.Length, o.MimeType, visible); err != nil {
			return false, exit.Internalf("cannot publish output %s of %s#%d: %s",
				o.OutputID, t.RequestID, t.Attempt, err)
		}
	}
	if _, err := tx.Exec(`UPDATE requests SET state=? WHERE id=?`,
		strings.ToLower(t.Status), t.RequestID); err != nil {
		return false, exit.Internalf("cannot settle request %s: %s", t.RequestID, err)
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

// OpenRecovered is every unclosed recovered obligation — what `cozy status` shows and
// what dispatch waits on.
func (s *Store) OpenRecovered() ([]Attempt, *exit.Error) {
	return s.attemptsWhere(`state='recovered_open'`)
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

func (s *Store) attemptsWhere(where string, args ...any) ([]Attempt, *exit.Error) {
	rows, err := s.db.Query(`SELECT request_id,attempt,attempt_key,instance_id,session_id,
		exec_spec_digest,exec_spec,state,plan_digest,construction,plan_summary,terminal_id,
		terminal_digest,terminal_status,terminal_cause,safe_message,triage_subject,
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
			&a.ExecSpecDigest, &a.ExecSpec, &a.State, &a.PlanDigest, &a.Construction,
			&a.PlanSummary, &a.TerminalID, &a.TerminalDigest, &a.TerminalStatus, &a.TerminalCause,
			&a.SafeMessage, &a.TriageSubject, &a.TerminalBody, &a.DispatchedAt, &a.AcceptedAt,
			&a.ClosedAt); err != nil {
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
	rows, err := s.db.Query(`SELECT o.output_id,o.path,o.digest,o.length,o.mime_type
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
		if err := rows.Scan(&o.OutputID, &o.Path, &o.Digest, &o.Length, &o.MimeType); err != nil {
			return nil, exit.Internalf("cannot read an output row: %s", err)
		}
		out = append(out, o)
	}
	return out, nil
}

// VisibleOutputsOf narrows visibility to ONE attempt — what a crash arm asks when it
// wants to know whether the bytes THAT attempt wrote ever became a result.
func (s *Store) VisibleOutputsOf(requestID string, attempt int64) ([]Output, *exit.Error) {
	rows, err := s.db.Query(`SELECT o.output_id,o.path,o.digest,o.length,o.mime_type
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
		if err := rows.Scan(&o.OutputID, &o.Path, &o.Digest, &o.Length, &o.MimeType); err != nil {
			return nil, exit.Internalf("cannot read an output row: %s", err)
		}
		out = append(out, o)
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
