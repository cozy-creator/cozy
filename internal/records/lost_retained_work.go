package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

const LostRetainedWorkMessage = "The rented machine was lost and its unfinished work is gone. This run cannot resume; start a new run from the original inputs."

// FailLostRetainedWork is called only after confirmed loss. Unlike a temporary
// preparation failure, missing execution state cannot be recovered by --retry.
// The selected machine and any still-open attempt fence a stale observation.
func (s *Store) FailLostRetainedWork(id, machine, reason string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin retained-work loss: %s", err)
	}
	defer tx.Rollback()
	failed, problem := failLostRetainedWorkTx(tx, id, machine, reason)
	if problem != nil || !failed {
		return false, problem
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit retained-work loss: %s", err)
	}
	return true, nil
}

func failLostRetainedWorkTx(tx *sql.Tx, id, machine, reason string) (bool, *exit.Error) {
	var state, selected string
	var ordinal int64
	var retained bool
	var open int
	if err := tx.QueryRow(`SELECT state,worker,ordinal,retain_work,
 (SELECT count(*) FROM attempts WHERE request_id=r.id AND state IN (`+openAttemptStates+`))
 FROM requests r WHERE id=?`, id).Scan(&state, &selected, &ordinal, &retained, &open); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return false, nil
		}
		return false, exit.Internalf("cannot read retained-work loss: %s", err)
	}
	if !retained || machine == "" || selected != machine || open != 0 {
		return false, nil
	}
	switch state {
	case "submitted", "queued", "dispatching", "requeue_pending", "finalizing", "pausing", "paused", "blocked":
	default:
		return false, nil
	}
	at := now()
	payload := map[string]any{
		"status": "FAILED", "cause": "RENTAL_LOST", "error_type": "request.state_lost",
		"error": LostRetainedWorkMessage, "machine_id": machine, "requeuing": false,
	}
	if reason != "" {
		payload["reason"] = reason
	}
	if state == "blocked" {
		var sequence int64
		var priorAt, body string
		err := tx.QueryRow(`SELECT seq,at,payload FROM request_events WHERE request_id=?
 AND type='request.blocked' ORDER BY seq DESC LIMIT 1`, id).Scan(&sequence, &priorAt, &body)
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return false, exit.Internalf("cannot read original retained-work loss: %s", err)
		}
		var prior map[string]any
		if err == nil && json.Unmarshal([]byte(body), &prior) == nil && prior["error_type"] == "request.state_lost" {
			// Correct the projection, not the execution clock. Keep the old event
			// byte-for-byte and explicitly record when its classification changed.
			payload["reclassified_at"] = at
			payload["original_event_id"] = sequence
			payload["original_error"] = prior["error"]
			at = priorAt
		}
	}
	var linkedMachine string
	var accepted, outcome bool
	err := tx.QueryRow(`SELECT machine_id,length(receipt)>0,length(outcome)>0
 FROM machine_executions WHERE request_id=?`, id).Scan(&linkedMachine, &accepted, &outcome)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return false, exit.Internalf("cannot read lost execution custody: %s", err)
	}
	if err == nil {
		if linkedMachine != machine {
			return false, nil
		}
		lost, err := machineExecutionLostIn(tx, id)
		if err != nil {
			return false, exit.Internalf("cannot read execution loss observation: %s", err)
		}
		if !lost {
			payload["had_acceptance_receipt"], payload["had_recorded_outcome"] = accepted, outcome
			if accepted && ordinal == 0 {
				ordinal = 1
			}
			if err := appendLossEventTx(tx, id, "client.machine_lost", ordinal, payload, at); err != nil {
				return false, exit.Internalf("cannot record retained execution loss: %s", err)
			}
		}
	}
	if _, err := tx.Exec(`UPDATE requests SET state='failed',retain_work=0 WHERE id=?`, id); err != nil {
		return false, exit.Internalf("cannot settle retained-work loss: %s", err)
	}
	if err := appendLossEventTx(tx, id, "run.failed", ordinal, payload, at); err != nil {
		return false, exit.Internalf("cannot record retained-work failure: %s", err)
	}
	return true, nil
}

func appendLossEventTx(tx *sql.Tx, id, kind string, ordinal int64, payload map[string]any, at string) error {
	body, problem := encodePayload(payload)
	if problem != nil {
		return problem
	}
	_, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`, id, kind, ordinal, body, at)
	return err
}

// ReconcileLostRetainedWork repairs the prior blocked projection using its own
// durable explicit loss observation. No absent account listing or elapsed time
// is treated as evidence. The daemon owns this data correction on startup.
func (s *Store) ReconcileLostRetainedWork() *exit.Error {
	rows, err := s.db.Query(`SELECT r.id,r.worker FROM requests r JOIN request_events e ON e.seq=(
 SELECT max(last.seq) FROM request_events last WHERE last.request_id=r.id AND last.type='request.blocked')
 WHERE r.state='blocked' AND r.retain_work=1 AND r.rental=1 AND r.worker<>''
 AND json_extract(e.payload,'$.error_type')='request.state_lost'`)
	if err != nil {
		return exit.Internalf("cannot read legacy retained-work losses: %s", err)
	}
	type loss struct{ id, machine string }
	var pending []loss
	for rows.Next() {
		var item loss
		if err := rows.Scan(&item.id, &item.machine); err != nil {
			rows.Close()
			return exit.Internalf("cannot read retained-work loss identity: %s", err)
		}
		pending = append(pending, item)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return exit.Internalf("cannot finish retained-work loss identities: %s", err)
	}
	for _, item := range pending {
		if _, problem := s.FailLostRetainedWork(item.id, item.machine, ""); problem != nil {
			return problem
		}
	}
	return nil
}
