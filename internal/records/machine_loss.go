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

// ReconcileEndedMachineExecutions repairs observers predating loss projection.
// An absent row, timeout, failed acquisition or empty account listing is not
// enough: only this owner's durable confirmed-release ledger admits recovery.
func (s *Store) ReconcileEndedMachineExecutions() *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin ended-machine reconciliation: %s", err)
	}
	defer tx.Rollback()
	rows, err := tx.Query(`SELECT DISTINCT e.machine_id FROM machine_executions e
 JOIN rental_operations o ON o.rental_id=e.machine_id AND o.state='released'`)
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
		if problem := loseMachineExecutions(tx, machine); problem != nil {
			return problem
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit ended-machine reconciliation: %s", err)
	}
	return nil
}

func loseMachineExecutions(tx *sql.Tx, machine string) *exit.Error {
	rows, err := tx.Query(`SELECT r.id,r.state,e.cancel_requested,length(e.receipt)>0,length(e.outcome)>0
 FROM machine_executions e JOIN requests r ON r.id=e.request_id
 WHERE e.machine_id=? AND `+machineExecutionOwed, machine)
	if err != nil {
		return exit.Internalf("cannot read destroyed machine observers: %s", err)
	}
	type observation struct {
		id, state                   string
		cancel, accepted, hasResult bool
	}
	var observations []observation
	for rows.Next() {
		var value observation
		if err := rows.Scan(&value.id, &value.state, &value.cancel, &value.accepted, &value.hasResult); err != nil {
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
		detail := map[string]any{
			"machine_id": machine, "error_type": "machine_execution.state_lost",
			"error":                  "rented machine was confirmed destroyed; its execution and retained bytes can no longer be observed",
			"had_acceptance_receipt": value.accepted, "had_recorded_outcome": value.hasResult,
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
			if err := appendEventTx(tx, value.id, "request."+next, 0, detail); err != nil {
				return exit.Internalf("cannot record destroyed execution projection: %s", err)
			}
		}
	}
	return nil
}
