package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

const LostRetainedWorkMessage = "The rented machine was lost and its unfinished work is gone. This run cannot resume; start a new run from the original inputs."

// failLostRetainedWorkTx settles retained work after its machine's confirmed loss. Missing
// execution state cannot be recovered by --retry. The selected machine and any still-open
// attempt fence a stale observation.
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
	case "submitted", "queued", "dispatching", "requeue_pending", "finalizing", "pausing", "paused":
	default:
		return false, nil
	}
	payload := map[string]any{
		"status": "FAILED", "cause": "RENTAL_LOST", "error_type": "request.state_lost",
		"error": LostRetainedWorkMessage, "machine_id": machine, "requeuing": false,
	}
	if reason != "" {
		payload["reason"] = reason
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
			if err := appendEventTx(tx, id, "client.machine_lost", ordinal, payload); err != nil {
				return false, exit.Internalf("cannot record retained execution loss: %s", err)
			}
		}
	}
	if _, err := tx.Exec(`UPDATE requests SET state='failed',retain_work=0 WHERE id=?`, id); err != nil {
		return false, exit.Internalf("cannot settle retained-work loss: %s", err)
	}
	if err := appendEventTx(tx, id, "run.failed", ordinal, payload); err != nil {
		return false, exit.Internalf("cannot record retained-work failure: %s", err)
	}
	return true, nil
}
