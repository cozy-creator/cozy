package records

import (
	"database/sql"
	"encoding/json"
	"math"

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
	At        string
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
	// A kept triage bundle rides beside the events as a document, never as one of them.
	query := `SELECT seq,request_id,type,attempt,payload,at FROM request_events
		WHERE seq>? AND type!='machine.triage'`
	args := []any{cursor}
	if requestID != "" {
		query += ` AND request_id=?`
		args = append(args, requestID)
	}
	query += ` ORDER BY seq LIMIT ?`
	args = append(args, limit)

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
		if err := json.Unmarshal([]byte(body), &e.Payload); err != nil {
			var value any
			if json.Unmarshal([]byte(body), &value) == nil {
				e.Payload = map[string]any{"value": value}
			} else {
				e.Payload = map[string]any{"unreadable": body}
			}
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
