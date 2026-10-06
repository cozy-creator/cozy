package records

import (
	"database/sql"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Destruction ends remote obligations, not by pretending that a native release
// or collection happened. Keep the exact acceptance, control and outcome history.
const machineExecutionLost = `EXISTS(SELECT 1 FROM request_events loss
 WHERE loss.request_id=e.request_id AND loss.type='client.machine_lost'
 AND json_extract(loss.payload,'$.machine_id')=e.machine_id)`

func machineExecutionLostIn(q interface {
	QueryRow(string, ...any) *sql.Row
}, id string) (bool, error) {
	var lost bool
	err := q.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e
 WHERE e.request_id=? AND `+machineExecutionLost+`)`, id).Scan(&lost)
	return lost, err
}

func (s *Store) MachineExecutionLost(id string) (bool, *exit.Error) {
	lost, err := machineExecutionLostIn(s.db, id)
	if err != nil {
		return false, exit.Internalf("cannot read machine loss: %s", err)
	}
	return lost, nil
}

// executionRental is the machine an execution ran on: its recorded machine, or for an explicit
// endpoint (a foreground --rental run of a daemon older than cozy.machine.v1) the rental its
// retained selector names as its workspace.
const executionRental = `CASE WHEN e.machine_id LIKE 'endpoint-%' THEN (SELECT json_extract(v.payload,'$.machine_endpoint.execution_workspace_id')
 FROM request_events v WHERE v.request_id=e.request_id AND v.type IN ('run.created','request.submitted') ORDER BY v.seq DESC LIMIT 1)
 ELSE e.machine_id END`

// ReconcileEndedMachineExecutions settles every run on a rental this owner has proof is
// gone: its confirmed-release ledger, or a Hub state committed only after provider
// absence. An absent row, timeout or empty account listing is not that proof.
func (s *Store) ReconcileEndedMachineExecutions() *exit.Error { return s.reconcileEnded("") }

// ReconcileEndedMachineExecution settles one run whose rental is proven gone, so a reader
// answers from the records instead of waiting on a machine that no longer exists.
func (s *Store) ReconcileEndedMachineExecution(id string) *exit.Error { return s.reconcileEnded(id) }

func (s *Store) reconcileEnded(request string) *exit.Error {
	// Read first: the writer lock is taken only when there is something to settle.
	ended, err := endedRentals(s.db, request)
	if err != nil {
		return exit.Internalf("cannot read ended rentals: %s", err)
	}
	if len(ended) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin ended-machine reconciliation: %s", err)
	}
	defer tx.Rollback()
	ended, err = endedRentals(tx, request)
	if err != nil {
		return exit.Internalf("cannot read ended rentals: %s", err)
	}
	for _, rental := range ended {
		if problem := settleLostMachine(tx, rental); problem != nil {
			return problem
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit ended-machine reconciliation: %s", err)
	}
	return nil
}

// endedRentals are the rentals proven gone that runs still owe work on: those runs or, given
// request, that one run.
func endedRentals(q interface {
	Query(string, ...any) (*sql.Rows, error)
}, request string) ([]string, error) {
	filter, args := "", []any{}
	if request != "" {
		filter, args = " AND e.request_id=?", []any{request}
	}
	rows, err := q.Query(`WITH ended(rental) AS (SELECT rental_id FROM rental_operations WHERE state='released' AND rental_id<>''
 UNION SELECT id FROM rentals WHERE state IN (`+absentRentalStates+`)),
 reached(request_id, rental) AS (SELECT e.request_id, `+executionRental+` FROM machine_executions e WHERE e.machine_id<>''`+filter+`)
 SELECT DISTINCT reached.rental FROM reached JOIN ended ON ended.rental=reached.rental
 JOIN machine_executions e ON e.request_id=reached.request_id JOIN requests r ON r.id=e.request_id WHERE `+machineExecutionOwed, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ended []string
	for rows.Next() {
		var rental string
		if err := rows.Scan(&rental); err != nil {
			return nil, err
		}
		ended = append(ended, rental)
	}
	return ended, rows.Err()
}

// LoseMachineExecution settles one accepted execution its machine proved it no longer
// holds (the worker restarted without its execution workspace). The machine and its
// other work continue.
func (s *Store) LoseMachineExecution(id, machine, message string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin lost execution: %s", err)
	}
	defer tx.Rollback()
	if problem := settleLost(tx, machine, id, message); problem != nil {
		return problem
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit lost execution: %s", err)
	}
	return nil
}

// LoseMachine settles everything a machine replaced for good still owed. Each run keeps the
// state it ended in and records, in message, that the machine's own record of it is gone.
func (s *Store) LoseMachine(machine, message string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin the replaced machine's settlement: %s", err)
	}
	defer tx.Rollback()
	if problem := settleLost(tx, machine, "", message); problem != nil {
		return problem
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit the replaced machine's settlement: %s", err)
	}
	return nil
}

// settleLostMachine ends every obligation on a rental proven gone, its runs and those that
// reached it as an explicit endpoint. A run its machine never confirmed stays in the outbox:
// released to be placed again under the same identity, charging nothing; the gone machine
// cannot run it, and a late acceptance from it is refused (AcceptRunV1). A confirmed run may
// have executed: it ends FAILED with its machine's loss, and its owner resubmits.
func settleLostMachine(tx *sql.Tx, machine string) *exit.Error {
	return settleLost(tx, machine, "", "")
}

func settleLost(tx *sql.Tx, machine, request, lost string) *exit.Error {
	var cause string
	if err := tx.QueryRow(`SELECT failure_code FROM rentals WHERE id=?`, machine).Scan(&cause); err != nil && err != sql.ErrNoRows {
		return exit.Internalf("cannot read lost machine cause: %s", err)
	}
	rows, err := tx.Query(`SELECT e.machine_id,r.id,r.state,r.retain_work,r.rental=1 AND r.requested_rental='',
 e.cancel_requested,length(e.submission)>0 OR `+runV1SentHere+`,length(e.receipt)>0,length(e.outcome)>0
 FROM machine_executions e JOIN requests r ON r.id=e.request_id
 WHERE e.machine_id<>'' AND (e.machine_id=?1 OR `+executionRental+`=?1) AND (?2='' OR e.request_id=?2) AND `+machineExecutionOwed, machine, request)
	if err != nil {
		return exit.Internalf("cannot read destroyed machine observers: %s", err)
	}
	type observation struct {
		machine, id, state                                          string
		retained, placeable, cancel, sent, accepted, hasResult bool
	}
	var observations []observation
	for rows.Next() {
		var value observation
		if err := rows.Scan(&value.machine, &value.id, &value.state, &value.retained, &value.placeable,
			&value.cancel, &value.sent, &value.accepted, &value.hasResult); err != nil {
			rows.Close()
			return exit.Internalf("cannot read destroyed machine observer: %s", err)
		}
		observations = append(observations, value)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return exit.Internalf("cannot finish destroyed machine observers: %s", err)
	}
	for _, value := range observations {
		if !value.accepted && value.placeable && !value.retained && (value.state == "submitted" || value.state == "queued") {
			if problem := releaseUnconfirmedTx(tx, value.id, value.machine, cause); problem != nil {
				return problem
			}
			continue
		}
		if value.retained {
			failed, problem := failLostRetainedWorkTx(tx, value.id, value.machine, cause)
			if problem != nil {
				return problem
			}
			if failed {
				continue
			}
		}
		message := "its rental ended before it finished"
		if settledRequestState(value.state) {
			message = "its rental ended; anything not yet collected from it is gone"
		} else if !value.sent {
			message = "its rental ended before the run was sent to it"
		} else if !value.accepted {
			message = "its rental ended before it confirmed the run"
		}
		if lost != "" {
			message = lost
		}
		detail := map[string]any{
			"machine_id": value.machine, "error_type": "machine_execution.state_lost", "error": message,
			"had_acceptance_receipt": value.accepted, "had_recorded_outcome": value.hasResult,
		}
		if cause != "" {
			detail["cause"] = cause
		}
		if err := appendEventTx(tx, value.id, "client.machine_lost", 0, detail); err != nil {
			return exit.Internalf("cannot record destroyed machine: %s", err)
		}
		// Nothing more is collected from a machine that is gone.
		if problem := skipOutputExport(tx, value.id, message); problem != nil {
			return problem
		}
		next := value.state
		if !settledRequestState(value.state) {
			next = "failed"
			if value.cancel || value.state == "canceling" {
				next = "canceled"
			}
		}
		if _, err := tx.Exec(`UPDATE requests SET state=?,retain_work=0 WHERE id=?`, next, value.id); err != nil {
			return exit.Internalf("cannot project destroyed execution: %s", err)
		}
		if next != value.state {
			if err := appendEventTx(tx, value.id, StateEvent(next), 0, detail); err != nil {
				return exit.Internalf("cannot record destroyed execution projection: %s", err)
			}
		}
	}
	return nil
}

// releaseUnconfirmedTx puts the run back in the outbox: unlinked, unpinned and unsent, with
// the same identity. Input bytes the lost machine held are staged again wherever the run is
// placed; `machine` stays as the run's history.
func releaseUnconfirmedTx(tx *sql.Tx, id, machine, cause string) *exit.Error {
	if _, err := tx.Exec(`UPDATE machine_executions SET machine_id='',remote_cursor=0 WHERE request_id=?`, id); err != nil {
		return exit.Internalf("cannot release unsent execution: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET worker='' WHERE id=?`, id); err != nil {
		return exit.Internalf("cannot unpin unsent execution: %s", err)
	}
	inputs, problem := machineInputsIn(tx, id)
	if problem != nil {
		return problem
	}
	for _, input := range inputs {
		if input.State != "held" {
			continue
		}
		if err := appendEventTx(tx, id, "machine.input", 0, map[string]any{"input_id": input.InputID,
			"manifest": input.Manifest, "content_bytes": input.ContentBytes, "state": "pending"}); err != nil {
			return exit.Internalf("cannot restage input from lost machine: %s", err)
		}
	}
	reason := "rented machine " + machine + " was lost before it confirmed this run"
	if cause != "" {
		reason += " (" + cause + ")"
	}
	if err := appendEventTx(tx, id, "request.queued", 0, map[string]any{"reason": reason, "machine_id": machine}); err != nil {
		return exit.Internalf("cannot record released execution: %s", err)
	}
	return nil
}
