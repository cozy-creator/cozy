package records

import (
	"database/sql"
	"encoding/json"
	"math"
	"sort"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

// The DURABLE half of the client contract's event stream (cl-006). Lifecycle events are
// rows in the same authority as the requests they describe, which buys the three
// properties the contract names and nothing more:
//
//   - MONOTONIC and TOTALLY ORDERED: `seq` is one AUTOINCREMENT counter across every
//     request, so the multiplexed stream and the per-request stream read the same order
//     and one cursor works for both.
//   - REPLAYABLE: a client resumes from the last id it saw. A stream close is never a
//     terminal verdict — reconnecting from the cursor is always the right move.
//   - AS DURABLE AS THE FACT: the terminal event is appended INSIDE the terminal
//     transaction (records/orchestrator.go), so "the output became visible" and "the stream says
//     so" commit together. A crash between them is unrepresentable.
//
// The LOSSY half — progress ticks, logs, per-stage marks — is never written here. It is
// live-only by construction (orchestrator/dispatch.go's fanout), because a durable row per
// denoising step would make the authority a log sink. The attempt-end event may
// retain its final measured overall fraction in the same terminal transaction.
var eventSchema = []string{`
CREATE TABLE IF NOT EXISTS request_events (
  seq        INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT    NOT NULL REFERENCES requests(id),
  type       TEXT    NOT NULL,
  attempt    INTEGER NOT NULL DEFAULT 0,
  payload    TEXT    NOT NULL,
  at         TEXT    NOT NULL
)`, `
CREATE INDEX IF NOT EXISTS request_events_by_request ON request_events(request_id, seq)`}

// Event is one durable lifecycle row. `Payload` is a JSON object; the contract's event
// envelope is {type, request_id, attempt, event_id, at, payload}.
type Event struct {
	Seq       int64
	RequestID string
	Type      string
	Attempt   int64
	Payload   map[string]any
	// Raw is the payload as recorded, for a reader that decodes it into its own type.
	Raw json.RawMessage
	At  string
}

// Terminal event types. A client stops on these and never reconnects (the absorbing
// terminal rule the reference SSE client already implements).
var terminalTypes = map[string]bool{
	"request.completed": true,
	"request.failed":    true,
	"request.canceled":  true,
}

// TerminalEvent reports whether this event type ends the stream.
func TerminalEvent(eventType string) bool { return terminalTypes[eventType] }

func encodePayload(payload map[string]any) (string, *exit.Error) {
	if payload == nil {
		payload = map[string]any{}
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return "", exit.Internalf("cannot render an event payload: %s", err)
	}
	return string(data), nil
}

// AppendEvent records one lifecycle event outside any transaction. Used for the states
// that are not themselves a durable commit (submitted, dispatched, accepted, requeued).
func (s *Store) AppendEvent(requestID, eventType string, attempt int64, payload map[string]any) *exit.Error {
	body, e := encodePayload(payload)
	if e != nil {
		return e
	}
	if _, err := s.db.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at)
		VALUES(?,?,?,?,?)`, requestID, eventType, attempt, body, now()); err != nil {
		return exit.Internalf("cannot append the %s event for %s: %s", eventType, requestID, err)
	}
	return nil
}

func appendEventTx(tx *sql.Tx, requestID, eventType string, attempt int64, payload map[string]any) error {
	body, e := encodePayload(payload)
	if e != nil {
		return e
	}
	_, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at)
		VALUES(?,?,?,?,?)`, requestID, eventType, attempt, body, now())
	return err
}

// CancelQueuedRequest settles a request with no attempt and appends its terminal event in
// one transaction. A crash can therefore expose neither half by itself.
func (s *Store) CancelQueuedRequest(requestID string, payload map[string]any) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin queued cancellation: %s", err)
	}
	defer tx.Rollback()
	var state string
	var openAttempts int
	if err := tx.QueryRow(`SELECT state,(SELECT COUNT(*) FROM attempts WHERE request_id=r.id
		AND state IN ('preparing','offered','accepted','recovered_open','terminal'))
		FROM requests r WHERE id=?`, requestID).Scan(&state, &openAttempts); err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, exit.Internalf("cannot read queued request %s: %s", requestID, err)
	}
	if state == "canceled" || state == "succeeded" || state == "failed" ||
		state == "refused" || state == "abandoned" {
		return false, nil
	}
	if openAttempts != 0 {
		return false, exit.New(exit.Conflict,
			"request %s has an attempt and is not a queued cancellation", requestID)
	}
	if _, err := tx.Exec(`UPDATE requests SET state='canceled' WHERE id=?`, requestID); err != nil {
		return false, exit.Internalf("cannot settle queued request %s: %s", requestID, err)
	}
	if _, err := tx.Exec(`UPDATE request_model_transfers SET state='canceled',updated_at=?
		WHERE request_id=? AND state NOT IN ('completed','failed','canceled')`, now(), requestID); err != nil {
		return false, exit.Internalf("cannot cancel model transfer %s: %s", requestID, err)
	}
	if err := appendEventTx(tx, requestID, "request.canceled", 0, payload); err != nil {
		return false, exit.Internalf("cannot append queued cancellation for %s: %s", requestID, err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit queued cancellation for %s: %s", requestID, err)
	}
	return true, nil
}

// QueuedFailure is the terminal payload of a failure before any attempt: the error's
// name, its message and, when another component originated it, that component's code.
func QueuedFailure(cause *exit.Error) map[string]any {
	payload := map[string]any{"status": "FAILED", "cause": cause.ErrName(),
		"error_type": cause.ErrName(), "error": cause.Message,
		"outputs": []any{}, "requeuing": false}
	if cause.Cause != "" {
		payload["error_code"] = cause.Cause
	}
	return payload
}

// FailQueuedRequest atomically records the typed pre-attempt failure and its terminal
// event. After a 202 response, status and watch must never disagree about why activation
// failed merely because one of two separate writes was lost.
func (s *Store) FailQueuedRequest(requestID string, payload map[string]any) (bool, *exit.Error) {
	return s.failQueuedRequest(requestID, nil, payload)
}

// FailQueuedPreparation settles only the selection that produced the failure.
// Rental recovery or an explicit resume can replace it while preparation runs.
func (s *Store) FailQueuedPreparation(expected Request, payload map[string]any) (bool, *exit.Error) {
	return s.failQueuedRequest(expected.ID, &expected, payload)
}

func (s *Store) failQueuedRequest(requestID string, expected *Request, payload map[string]any) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin queued failure: %s", err)
	}
	defer tx.Rollback()
	var state, worker string
	var ordinal int64
	var revision uint64
	var retain bool
	var openAttempts int
	if err := tx.QueryRow(`SELECT state,retain_work,worker,ordinal,control_revision,(SELECT COUNT(*) FROM attempts WHERE request_id=r.id
		AND state IN ('preparing','offered','accepted','recovered_open','terminal'))
		FROM requests r WHERE id=?`, requestID).Scan(&state, &retain, &worker, &ordinal, &revision, &openAttempts); err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, exit.Internalf("cannot read queued request %s: %s", requestID, err)
	}
	if state != "submitted" && state != "queued" {
		return false, nil
	}
	if expected != nil && (worker != expected.Worker || ordinal != expected.Ordinal || revision != expected.ControlRevision) {
		return false, nil
	}
	if openAttempts != 0 {
		return false, exit.New(exit.Conflict,
			"request %s has an attempt and is not a queued failure", requestID)
	}
	if retain {
		if _, err := tx.Exec(`UPDATE requests SET state='blocked' WHERE id=?`, requestID); err != nil {
			return false, exit.Internalf("cannot retain queued failure: %s", err)
		}
		blocked := make(map[string]any, len(payload))
		for key, value := range payload {
			blocked[key] = value
		}
		blocked["status"] = "blocked"
		if err := appendEventTx(tx, requestID, "request.blocked", 0, blocked); err != nil {
			return false, exit.Internalf("cannot journal retained queued failure: %s", err)
		}
		if err := tx.Commit(); err != nil {
			return false, exit.Internalf("cannot commit retained queued failure: %s", err)
		}
		return true, nil
	}
	if _, err := tx.Exec(`UPDATE requests SET state='failed' WHERE id=?`, requestID); err != nil {
		return false, exit.Internalf("cannot settle queued request %s: %s", requestID, err)
	}
	code, _ := payload["error_type"].(string)
	detail, _ := payload["error"].(string)
	if _, err := tx.Exec(`UPDATE request_model_transfers SET state='failed',error_code=?,
		safe_error=?,updated_at=? WHERE request_id=? AND state NOT IN ('completed','canceling','canceled')`,
		code, detail, now(), requestID); err != nil {
		return false, exit.Internalf("cannot fail queued model transfer %s: %s", requestID, err)
	}
	if err := appendEventTx(tx, requestID, "request.failed", 0, payload); err != nil {
		return false, exit.Internalf("cannot append queued failure for %s: %s", requestID, err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit queued failure for %s: %s", requestID, err)
	}
	return true, nil
}

// EventsAfter reads durable events strictly after `cursor`. An empty requestID reads the
// MULTIPLEXED stream — every request, one order, one cursor. The browser opens exactly
// one of these instead of one connection per request (the connection-cap reason the
// local host multiplexes at all).
func (s *Store) EventsAfter(requestID string, cursor int64, limit int) ([]Event, *exit.Error) {
	return s.events(requestID, cursor, limit)
}

// maxEvidenceStreams bounds the progress streams one evidence read carries.
const maxEvidenceStreams = 256

// progressRow is a sample of a narrated position: Runtime's lossy progress lane, and a log
// carrying a `position` (a prefetch's download). progressStream is the stream it samples.
const (
	progressRow    = `(type='machine.progress' OR (type='request.log' AND json_extract(payload,'$.fields.position') IS NOT NULL))`
	progressStream = `COALESCE(json_extract(payload,'$.payload.stage'),json_extract(payload,'$.name')||' '||COALESCE(json_extract(payload,'$.fields.stage'),''))`
)

// EvidenceEvents is one request's durable record: its first `limit` lifecycle events in
// order, beside the latest sample of each progress stream. Progress reads under its own
// bound, coalesced per stream, so no rate of samples displaces a grant, a phase boundary
// or a terminal (run 1510: 4,065 prefetch positions filled its 4,096 rows).
func (s *Store) EvidenceEvents(requestID string, limit int) ([]Event, *exit.Error) {
	lifecycle, e := s.scanEvents(`SELECT seq,request_id,type,attempt,payload,at FROM request_events
		WHERE request_id=? AND type!='machine.triage' AND NOT `+progressRow+` ORDER BY seq LIMIT ?`, requestID, limit)
	if e != nil {
		return nil, e
	}
	latest, e := s.scanEvents(`SELECT seq,request_id,type,attempt,payload,at FROM request_events WHERE seq IN (
		SELECT MAX(seq) FROM request_events WHERE request_id=? AND `+progressRow+`
		GROUP BY type,`+progressStream+` ORDER BY MAX(seq) DESC LIMIT ?)`, requestID, maxEvidenceStreams)
	if e != nil {
		return nil, e
	}
	out := append(lifecycle, latest...)
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out, nil
}

func (s *Store) events(requestID string, cursor int64, limit int) ([]Event, *exit.Error) {
	// A kept triage bundle rides beside the events as a document, never as one of them.
	query := `SELECT seq,request_id,type,attempt,payload,at FROM request_events
		WHERE seq>? AND type!='machine.triage'`
	args := []any{cursor}
	if requestID != "" {
		query += ` AND request_id=?`
		args = append(args, requestID)
	}
	return s.scanEvents(query+` ORDER BY seq LIMIT ?`, append(args, limit)...)
}

func (s *Store) scanEvents(query string, args ...any) ([]Event, *exit.Error) {
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, exit.Internalf("cannot read the event stream: %s", err)
	}
	defer rows.Close()
	var out []Event
	for rows.Next() {
		var e Event
		var body string
		if err := rows.Scan(&e.Seq, &e.RequestID, &e.Type, &e.Attempt, &body, &e.At); err != nil {
			return nil, exit.Internalf("cannot read an event row: %s", err)
		}
		// One payload a stream consumer cannot read as an object is carried, not allowed
		// to end every stream reading past it.
		e.Raw = json.RawMessage(body)
		if err := json.Unmarshal([]byte(body), &e.Payload); err != nil {
			var value any
			if json.Unmarshal([]byte(body), &value) == nil {
				e.Payload = map[string]any{"value": value}
			} else {
				e.Payload = map[string]any{"unreadable": body}
			}
			e.Raw, _ = json.Marshal(e.Payload)
		}
		out = append(out, e)
	}
	return out, nil
}

// TerminalEventAt is the request's durable settlement clock. Pre-attempt failures and
// cancellations have no attempt close timestamp, but their terminal event is committed in
// the same transaction as the settled request state.
func (s *Store) TerminalEventAt(requestID string) (string, *exit.Error) {
	var at string
	err := s.db.QueryRow(`SELECT at FROM request_events
		WHERE request_id=? AND type IN ('request.completed','request.failed','request.canceled')
		ORDER BY seq DESC LIMIT 1`, requestID).Scan(&at)
	if err == sql.ErrNoRows {
		return "", nil
	}
	if err != nil {
		return "", exit.Internalf("cannot read terminal event time for %s: %s", requestID, err)
	}
	return at, nil
}

// TerminalOverallFraction reads only the current attempt's persisted outcome
// summary. A prior attempt or a stage-local denominator cannot supply this value.
func (s *Store) TerminalOverallFraction(requestID string, attempt int64) (*float64, *exit.Error) {
	var body string
	err := s.db.QueryRow(`SELECT payload FROM request_events
		WHERE request_id=? AND attempt=? AND type IN (
		'request.completed','request.failed','request.canceled','request.attempt_failed',
		'request.pausing','request.blocked','request.canceling','request.finalizing')
		AND json_extract(payload,'$.overall_fraction') IS NOT NULL
		ORDER BY seq DESC LIMIT 1`, requestID, attempt).Scan(&body)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read terminal progress for %s: %s", requestID, err)
	}
	var summary struct {
		OverallFraction *float64 `json:"overall_fraction"`
	}
	if err := json.Unmarshal([]byte(body), &summary); err != nil {
		return nil, exit.Internalf("cannot read terminal progress payload for %s: %s", requestID, err)
	}
	value := summary.OverallFraction
	if value == nil || math.IsNaN(*value) || math.IsInf(*value, 0) || *value < 0 || *value > 1 {
		return nil, nil
	}
	return value, nil
}

// CancelAttribution is who canceled this request. Every cancellation path records its
// actor durably — in the queued terminal's own payload, or in the request.cancel_requested
// event that precedes a live attempt's cancel frame — so a canceled run can always say
// WHO, not merely that it ended.
func (s *Store) CancelAttribution(requestID string) (actor, errType, errText string, problem *exit.Error) {
	rows, err := s.db.Query(`SELECT type, payload FROM request_events
		WHERE request_id=? AND type IN ('request.cancel_requested','request.canceled')
		ORDER BY seq DESC`, requestID)
	if err != nil {
		return "", "", "", exit.Internalf("cannot read cancellation events for %s: %s", requestID, err)
	}
	defer rows.Close()
	for rows.Next() {
		var eventType, body string
		if err := rows.Scan(&eventType, &body); err != nil {
			return "", "", "", exit.Internalf("cannot read a cancellation event row: %s", err)
		}
		var payload map[string]any
		if json.Unmarshal([]byte(body), &payload) != nil {
			continue
		}
		if actor == "" {
			actor, _ = payload["actor"].(string)
		}
		if eventType == "request.canceled" && errText == "" {
			errType, _ = payload["error_type"].(string)
			errText, _ = payload["error"].(string)
		}
	}
	return actor, errType, errText, nil
}

// SettledFailure is the typed cause journaled with a request's failed or blocked settlement.
// A failure before any attempt (a refused desired state, an unplaceable pin) has no attempt
// row to carry it; this event is its only durable record.
func (s *Store) SettledFailure(requestID string) (errType, errCode, errText string, problem *exit.Error) {
	var body string
	err := s.db.QueryRow(`SELECT payload FROM request_events
		WHERE request_id=? AND type IN ('request.failed','request.blocked')
		ORDER BY seq DESC LIMIT 1`, requestID).Scan(&body)
	if err == sql.ErrNoRows {
		return "", "", "", nil
	}
	if err != nil {
		return "", "", "", exit.Internalf("cannot read the settled failure of %s: %s", requestID, err)
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		return "", "", "", exit.Internalf("the settled failure of %s is unreadable: %s", requestID, err)
	}
	errType, _ = payload["error_type"].(string)
	errCode, _ = payload["error_code"].(string)
	errText, _ = payload["error"].(string)
	return errType, errCode, errText, nil
}

// LastEventSeq is the current head of the stream. A client that wants only what happens
// NEXT opens at the head instead of replaying history.
func (s *Store) LastEventSeq() (int64, *exit.Error) {
	var seq sql.NullInt64
	if err := s.db.QueryRow(`SELECT MAX(seq) FROM request_events`).Scan(&seq); err != nil {
		return 0, exit.Internalf("cannot read the event stream head: %s", err)
	}
	return seq.Int64, nil
}

// LatestMachineProgress reads the current attempt's already-imported Runtime
// event page. It does not add another telemetry writer or new progress journal.
func (s *Store) LatestMachineProgress(requestID string, attempt int64) (map[string]any, *exit.Error) {
	var body string
	err := s.db.QueryRow(`SELECT json_extract(payload,'$.payload') FROM request_events
		WHERE request_id=? AND attempt=? AND type='machine.progress'
		AND json_extract(payload,'$.type')='progress'
		AND json_type(payload,'$.payload')='object'
		ORDER BY seq DESC LIMIT 1`, requestID, attempt).Scan(&body)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read machine progress for %s: %s", requestID, err)
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(body), &value); err != nil {
		return nil, exit.Internalf("cannot decode machine progress for %s: %s", requestID, err)
	}
	return value, nil
}

// MachineProgressEstimate is how long an execution's attempt still needs, measured from its
// own progress samples: the Runtime-stamped time between the first and latest whole-job
// fraction, scaled to the remaining fraction. False without two advancing samples.
func (s *Store) MachineProgressEstimate(requestID string, attempt int64) (int64, bool) {
	sample := func(order string) (float64, time.Time, bool) {
		var fraction sql.NullFloat64
		var at string
		err := s.db.QueryRow(`SELECT json_extract(payload,'$.payload.overall_fraction'),at FROM request_events
			WHERE request_id=? AND attempt=? AND type='machine.progress'
			AND json_type(payload,'$.payload.overall_fraction') IN ('real','integer')
			ORDER BY seq `+order+` LIMIT 1`, requestID, attempt).Scan(&fraction, &at)
		when, parseErr := time.Parse(time.RFC3339Nano, at)
		return fraction.Float64, when, err == nil && fraction.Valid && parseErr == nil
	}
	firstFraction, firstAt, ok := sample("ASC")
	if !ok {
		return 0, false
	}
	lastFraction, lastAt, ok := sample("DESC")
	if !ok || lastFraction <= firstFraction || !lastAt.After(firstAt) || lastFraction > 1 {
		return 0, false
	}
	perFraction := float64(lastAt.Sub(firstAt).Milliseconds()) / (lastFraction - firstFraction)
	return int64((1 - lastFraction) * perFraction), true
}

// AwaitingPublication is a sent checkpoint publication its machine can no longer settle:
// the machine's publication authority stopped answering, so only the owner's Hub read can.
type AwaitingPublication struct {
	CallIndex   uint32 `json:"call_index"`
	Publication string `json:"publication"`
	Destination string `json:"destination"`
	Code        string `json:"code"`
}

// MachinePublicationsAwaitingOwner are the execution's unresolved publications Runtime has
// reported with no later settlement, oldest first.
func (s *Store) MachinePublicationsAwaitingOwner(requestID string) ([]AwaitingPublication, *exit.Error) {
	rows, err := s.db.Query(`SELECT payload FROM request_events u WHERE u.request_id=? AND u.type='machine.publication_unresolved'
		AND NOT EXISTS(SELECT 1 FROM request_events d WHERE d.request_id=u.request_id AND d.seq>u.seq
		AND d.type='machine.publication_settled' AND json_extract(d.payload,'$.publication')=json_extract(u.payload,'$.publication'))
		ORDER BY u.seq`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read unresolved publications for %s: %s", requestID, err)
	}
	defer rows.Close()
	var out []AwaitingPublication
	seen := map[string]bool{}
	for rows.Next() {
		var body string
		var awaiting AwaitingPublication
		if err := rows.Scan(&body); err != nil || json.Unmarshal([]byte(body), &awaiting) != nil || awaiting.Publication == "" || awaiting.Destination == "" {
			return nil, exit.Internalf("cannot decode an unresolved publication for %s", requestID)
		}
		if !seen[awaiting.Publication] {
			seen[awaiting.Publication] = true
			out = append(out, awaiting)
		}
	}
	return out, nil
}

// MachineGPUWait is the newest gpu.wait Runtime reported for this execution whose call has
// not since been granted or released devices, or nil when nothing waits for a GPU.
func (s *Store) MachineGPUWait(requestID string) (*MachineGPUWaiting, *exit.Error) {
	var body string
	err := s.db.QueryRow(`SELECT payload FROM request_events w WHERE w.request_id=? AND w.type='machine.gpu.wait'
		AND NOT EXISTS(SELECT 1 FROM request_events g WHERE g.request_id=w.request_id AND g.seq>w.seq
		AND g.type IN ('machine.gpu.grant','machine.gpu.release') AND json_extract(g.payload,'$.key')=json_extract(w.payload,'$.key'))
		ORDER BY w.seq DESC LIMIT 1`, requestID).Scan(&body)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read GPU wait for %s: %s", requestID, err)
	}
	var wait MachineGPUWaiting
	if err := json.Unmarshal([]byte(body), &wait); err != nil {
		return nil, exit.Internalf("cannot decode GPU wait for %s: %s", requestID, err)
	}
	return &wait, nil
}

// MachineGPUsHeld counts the devices this execution's calls hold: granted, not yet released.
func (s *Store) MachineGPUsHeld(requestID string) (int, *exit.Error) {
	var held int
	err := s.db.QueryRow(`SELECT COUNT(DISTINCT o.value) FROM request_events g, json_each(g.payload,'$.ordinals') o
		WHERE g.request_id=? AND g.type='machine.gpu.grant' AND NOT EXISTS(SELECT 1 FROM request_events r
		WHERE r.request_id=g.request_id AND r.seq>g.seq AND r.type='machine.gpu.release'
		AND json_extract(r.payload,'$.key')=json_extract(g.payload,'$.key'))`, requestID).Scan(&held)
	if err != nil {
		return 0, exit.Internalf("cannot count GPUs held by %s: %s", requestID, err)
	}
	return held, nil
}

// MachineGPUWaiting is Runtime's gpu.wait body: the call, the device count it needs and the
// roots holding the devices ahead of it.
type MachineGPUWaiting struct {
	Key       string   `json:"key"`
	Width     int      `json:"width"`
	BlockedBy []string `json:"blocked_by"`
}

// RecordMachineTriage keeps a Runtime execution's triage bundle beside its events, once
// per attempt; MachineTriage reads the newest back.
func (s *Store) RecordMachineTriage(requestID string, attempt int64, bundle []byte) *exit.Error {
	if !json.Valid(bundle) {
		return exit.New(exit.Conflict, "triage bundle is not a JSON document")
	}
	if _, err := s.db.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at)
		SELECT ?,'machine.triage',?,?,? WHERE NOT EXISTS(SELECT 1 FROM request_events
		WHERE request_id=? AND type='machine.triage' AND attempt=?)`,
		requestID, attempt, string(bundle), now(), requestID, attempt); err != nil {
		return exit.Internalf("cannot keep the triage bundle of %s: %s", requestID, err)
	}
	return nil
}

func (s *Store) MachineTriage(requestID string) (json.RawMessage, *exit.Error) {
	var bundle string
	err := s.db.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='machine.triage'
		ORDER BY seq DESC LIMIT 1`, requestID).Scan(&bundle)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the triage bundle of %s: %s", requestID, err)
	}
	return json.RawMessage(bundle), nil
}

// Wait is why a queued request last said it waits: a park or queue event's reason, the
// wait cause, and the machine it waits on.
type Wait struct {
	Reason    string `json:"reason"`
	Cause     string `json:"wait"`
	WaitingOn string `json:"waiting_on"`
}

// CurrentWait is the request's newest park or queue event when nothing has moved it on
// since (a preparation stage, an acceptance or a terminal); nil otherwise.
func (s *Store) CurrentWait(requestID string) (*Wait, *exit.Error) {
	var kind, body string
	err := s.db.QueryRow(`SELECT type,payload FROM request_events WHERE request_id=?
		AND type IN ('request.parked','request.queued','request.preparing','request.accepted',
		  'request.completed','request.failed','request.canceled')
		ORDER BY seq DESC LIMIT 1`, requestID).Scan(&kind, &body)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read the wait of %s: %s", requestID, err)
	}
	if kind != "request.parked" && kind != "request.queued" {
		return nil, nil
	}
	var wait Wait
	if err := json.Unmarshal([]byte(body), &wait); err != nil || wait.Reason == "" {
		return nil, nil
	}
	return &wait, nil
}
