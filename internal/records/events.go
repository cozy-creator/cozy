package records

import (
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
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
// denoising step would make the authority a log sink.
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
	if err := appendEventTx(tx, requestID, "request.canceled", 0, payload); err != nil {
		return false, exit.Internalf("cannot append queued cancellation for %s: %s", requestID, err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit queued cancellation for %s: %s", requestID, err)
	}
	return true, nil
}

// EventsAfter reads durable events strictly after `cursor`. An empty requestID reads the
// MULTIPLEXED stream — every request, one order, one cursor. The browser opens exactly
// one of these instead of one connection per request (the connection-cap reason the
// local host multiplexes at all).
func (s *Store) EventsAfter(requestID string, cursor int64, limit int) ([]Event, *exit.Error) {
	query := `SELECT seq,request_id,type,attempt,payload,at FROM request_events
		WHERE seq>?`
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
		if err := json.Unmarshal([]byte(body), &e.Payload); err != nil {
			return nil, exit.Internalf("event %d has an unreadable payload: %s", e.Seq, err)
		}
		out = append(out, e)
	}
	return out, nil
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
