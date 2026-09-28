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

// ReconcileEndedMachineExecutions settles every machine this owner has proof is
// gone: its confirmed-release ledger, or a Hub state committed only after provider
// absence. An absent row, timeout or empty account listing is not that proof.
func (s *Store) ReconcileEndedMachineExecutions() *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin ended-machine reconciliation: %s", err)
	}
	defer tx.Rollback()
	rows, err := tx.Query(`SELECT DISTINCT e.machine_id FROM machine_executions e WHERE e.machine_id<>'' AND (
 EXISTS(SELECT 1 FROM rental_operations o WHERE o.rental_id=e.machine_id AND o.state='released') OR
 EXISTS(SELECT 1 FROM rentals WHERE id=e.machine_id AND state IN (` + absentRentalStates + `)))`)
	if err != nil {
		return exit.Internalf("cannot read confirmed ended machines: %s", err)
	}
	var machines []string
	for rows.Next() {
		var machine string
		if err := rows.Scan(&machine); err != nil {
			rows.Close()
			return exit.Internalf("cannot read ended machine identity: %s", err)
		}
		machines = append(machines, machine)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return exit.Internalf("cannot finish ended machine identities: %s", err)
	}
	for _, machine := range machines {
		if problem := settleLostMachine(tx, machine); problem != nil {
			return problem
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit ended-machine reconciliation: %s", err)
	}
	return nil
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

// settleLostMachine ends every obligation on a machine proven gone. A run whose offer
// never left this host is released to be placed again, charging nothing. A sent offer
// may have executed, and that execution and its bytes died with the machine.
func settleLostMachine(tx *sql.Tx, machine string) *exit.Error {
	return settleLost(tx, machine, "", "")
}

func settleLost(tx *sql.Tx, machine, request, lost string) *exit.Error {
	var cause string
	if err := tx.QueryRow(`SELECT failure_code FROM rentals WHERE id=?`, machine).Scan(&cause); err != nil && err != sql.ErrNoRows {
		return exit.Internalf("cannot read lost machine cause: %s", err)
	}
	rows, err := tx.Query(`SELECT r.id,r.state,r.retain_work,r.rental=1 AND r.requested_rental='',
 e.cancel_requested,length(e.submission)>0,length(e.receipt)>0,length(e.outcome)>0
 FROM machine_executions e JOIN requests r ON r.id=e.request_id
 WHERE e.machine_id=? AND (?='' OR e.request_id=?) AND `+machineExecutionOwed, machine, request, request)
	if err != nil {
		return exit.Internalf("cannot read destroyed machine observers: %s", err)
	}
	type observation struct {
		id, state                                                   string
		retained, placeable, cancel, submitted, accepted, hasResult bool
	}
	var observations []observation
	for rows.Next() {
		var value observation
		if err := rows.Scan(&value.id, &value.state, &value.retained, &value.placeable,
			&value.cancel, &value.submitted, &value.accepted, &value.hasResult); err != nil {
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
		if !value.submitted && value.placeable && !value.retained && (value.state == "submitted" || value.state == "queued") {
			if problem := releaseUnsentTx(tx, value.id, machine, cause); problem != nil {
				return problem
			}
			continue
		}
		if value.retained {
			failed, problem := failLostRetainedWorkTx(tx, value.id, machine, cause)
			if problem != nil {
				return problem
			}
			if failed {
				continue
			}
		}
		message := "rented machine was confirmed destroyed; its execution and retained bytes can no longer be observed"
		if !value.submitted {
			message = "rented machine was confirmed destroyed before this run was submitted to it"
		}
		if lost != "" {
			message = lost
		}
		detail := map[string]any{
			"machine_id": machine, "error_type": "machine_execution.state_lost", "error": message,
			"had_acceptance_receipt": value.accepted, "had_recorded_outcome": value.hasResult,
		}
		if cause != "" {
			detail["cause"] = cause
		}
		if err := appendEventTx(tx, value.id, "client.machine_lost", 0, detail); err != nil {
			return exit.Internalf("cannot record destroyed machine: %s", err)
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

// releaseUnsentTx unlinks and unpins the run. Input bytes the lost machine held are
// staged again wherever the run is placed; `machine` stays as the run's history.
func releaseUnsentTx(tx *sql.Tx, id, machine, cause string) *exit.Error {
	if _, err := tx.Exec(`UPDATE machine_executions SET machine_id='' WHERE request_id=?`, id); err != nil {
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
	reason := "rented machine " + machine + " was lost before this run was submitted"
	if cause != "" {
		reason += " (" + cause + ")"
	}
	if err := appendEventTx(tx, id, "request.queued", 0, map[string]any{"reason": reason, "machine_id": machine}); err != nil {
		return exit.Internalf("cannot record released execution: %s", err)
	}
	return nil
}
