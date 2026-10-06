package records

import (
	"cmp"
	"database/sql"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Abandonment is owner intent, not proof of remote loss, stop, closure or release.
const machineExecutionAbandoned = `EXISTS(SELECT 1 FROM request_events abandoned
 WHERE abandoned.request_id=e.request_id AND abandoned.type='client.machine_abandoned')`
const machineExecutionAdmissionOpen = `NOT EXISTS(SELECT 1 FROM request_events abandoned
 WHERE abandoned.request_id=machine_executions.request_id AND abandoned.type='client.machine_abandoned')`

// AbandonMachineExecution closes this client's execution intent. Every frozen
// submission, receipt, outcome and pending-control byte remains available as evidence.
func (s *Store) AbandonMachineExecution(id, actor, why string) (bool, *exit.Error) {
	actor = strings.TrimSpace(actor)
	if actor == "" {
		return false, exit.New(exit.Validation, "local abandonment requires an explicit actor")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin local abandonment: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return false, exit.Internalf("cannot read abandonment evidence: %s", err)
	}
	if link == nil {
		return false, exit.Named(exit.Conflict, "request.abandon_unsupported", "only a Runtime-owned run can be abandoned locally")
	}
	if link.Abandoned {
		return false, nil
	}
	var state string
	var nativeSent bool
	if err := tx.QueryRow(`SELECT state,
 EXISTS(SELECT 1 FROM request_events WHERE request_id=requests.id AND type=?)
 FROM requests WHERE id=?`, RunV1Sent, id).Scan(&state, &nativeSent); err != nil {
		return false, exit.Internalf("cannot read abandoned run: %s", err)
	}
	facts := map[string]any{"actor": actor, "machine_id": link.MachineID, "scope": "local_abandonment", "machine_execution": true,
		"had_acceptance_receipt": len(link.Receipt) > 0, "acceptance_unknown": (nativeSent || len(link.Submission) > 0) && len(link.Receipt) == 0 && !link.SubmissionClosed,
		"remote_stop_confirmed": false, "error_type": "request.abandoned", "error": cmp.Or(why, "the owner abandoned local tracking; remote stop and rental release are not confirmed")}
	if err := appendEventTx(tx, id, "client.machine_abandoned", 0, facts); err != nil {
		return false, exit.Internalf("cannot retain abandonment intent: %s", err)
	}
	next := state
	if !settledRequestState(state) {
		next = "abandoned"
	}
	if _, err := tx.Exec(`UPDATE requests SET state=?,retain_work=0 WHERE id=?`, next, id); err != nil {
		return false, exit.Internalf("cannot close abandoned local intent: %s", err)
	}
	if next != state {
		// This is the local run's terminal, never a fabricated Runtime outcome.
		facts["status"] = "ABANDONED"
		if err := appendEventTx(tx, id, "run.failed", 0, facts); err != nil {
			return false, exit.Internalf("cannot record abandoned local terminal: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit local abandonment: %s", err)
	}
	return true, nil
}

// Retain late positive facts without turning an abandoned user's run back into
// remote execution intent. The first real outcome remains its canonical evidence.
func recordAbandonedOutcome(tx *sql.Tx, id string, attempt uint64, digest string, raw []byte) *exit.Error {
	var exists bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type='client.abandoned_machine_outcome' AND json_extract(payload,'$.outcome_digest')=?)`, id, digest).Scan(&exists); err != nil {
		return exit.Internalf("cannot read late outcome evidence: %s", err)
	}
	if !exists {
		if err := appendEventTx(tx, id, "client.abandoned_machine_outcome", int64(attempt), map[string]any{"outcome_digest": digest, "outcome": raw}); err != nil {
			return exit.Internalf("cannot retain late outcome evidence: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit late outcome evidence: %s", err)
	}
	return nil
}
