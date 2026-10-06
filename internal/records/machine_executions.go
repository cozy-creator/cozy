package records

// These rows are client submission and observation records. Runtime owns the
// attempts, children, retry policy, and output custody on the selected machine.

import (
	"database/sql"
	"errors"
	"github.com/cozy-creator/cozy/internal/exit"
)

var machineExecutionSchema = []string{
	`CREATE TABLE machine_executions (
 request_id TEXT PRIMARY KEY REFERENCES requests(id),
 machine_id TEXT NOT NULL DEFAULT '',
 submission BLOB NOT NULL DEFAULT x'',
 receipt BLOB NOT NULL DEFAULT x'',
 observed_state BLOB NOT NULL DEFAULT x'',
 remote_cursor INTEGER NOT NULL DEFAULT 0 CHECK(remote_cursor>=0),
 outcome BLOB NOT NULL DEFAULT x'',
 pending_control BLOB NOT NULL DEFAULT x'',
 cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
 collected INTEGER NOT NULL DEFAULT 0 CHECK(collected IN (0,1))
 ) STRICT`,
	`CREATE TRIGGER machine_execution_no_local_attempt BEFORE INSERT ON attempts
 WHEN EXISTS(SELECT 1 FROM machine_executions WHERE request_id=NEW.request_id)
 BEGIN SELECT RAISE(ABORT,'machine execution observation cannot own a local attempt'); END`,
}

type MachineExecution struct {
	RequestID, MachineID string
	Submission           []byte
	Receipt              []byte
	ObservedState        []byte
	RemoteCursor         int64
	Outcome              []byte
	PendingControl       []byte
	CancelRequested      bool
	Collected            bool
	Abandoned            bool
	SubmissionClosed     bool
	Lost                 bool // its machine is gone: nothing more of the run is there to observe
}

const machineExecutionColumns = `request_id,machine_id,submission,receipt,observed_state,remote_cursor,outcome,pending_control,cancel_requested,collected,
 EXISTS(SELECT 1 FROM request_events closed WHERE closed.request_id=machine_executions.request_id AND closed.type='machine.submission_closed'),
 NOT (` + machineExecutionAdmissionOpen + `),
 EXISTS(SELECT 1 FROM request_events loss WHERE loss.request_id=machine_executions.request_id AND loss.type='client.machine_lost'
 AND json_extract(loss.payload,'$.machine_id')=machine_executions.machine_id)`

type MachinePackageTransferProgress struct {
	Operation string
	Uploaded  bool
	Completed int
}

// MachinePackageTransfer observes only this request's transfer progress. Recording
// its start before the first byte lets reconnect resume an incomplete initial
// upload without asking the installer to consume unverified wheel carriers.
func (s *Store) MachinePackageTransfer(request, boot, revision string) (MachinePackageTransferProgress, *exit.Error) {
	var progress MachinePackageTransferProgress
	err := s.db.QueryRow(`SELECT COALESCE(json_extract(payload,'$.operation_id'),''),
 type='machine.package_uploaded',sum(CASE WHEN type='machine.package_uploaded' THEN 1 ELSE 0 END) OVER ()
 FROM request_events WHERE request_id=? AND type IN ('machine.package_upload_started','machine.package_uploaded')
 AND json_extract(payload,'$.worker_boot_id')=? AND json_extract(payload,'$.revision')=?
 ORDER BY seq DESC LIMIT 1`, request, boot, revision).Scan(&progress.Operation, &progress.Uploaded, &progress.Completed)
	if err == sql.ErrNoRows {
		return progress, nil
	}
	if err != nil {
		return progress, exit.Internalf("cannot read machine package transfer operation: %s", err)
	}
	return progress, nil
}

// e/r are the observer and request aliases. Explicit Runtime release is stronger
// than a status projection. A failed or paused root remains live after its small
// error result is collected, and cancellation alone never proves native cleanup.
// Local abandonment stops observation; it cannot waive remote lifecycle custody.
const machineExecutionLive = `(NOT ` + machineExecutionLost + ` AND NOT ` + machineRetentionReleased + ` AND (
 (length(e.receipt)=0 AND (e.cancel_requested=1 OR ` + machineExecutionAbandoned + `) AND length(e.submission)>0 AND NOT EXISTS
 (SELECT 1 FROM request_events closed WHERE closed.request_id=r.id AND closed.type='machine.submission_closed')) OR
 (length(e.receipt)=0 AND r.state NOT IN ('refused','failed','succeeded','abandoned','pausing','paused','blocked','canceled')) OR
 (length(e.receipt)>0 AND (r.state!='succeeded' OR e.collected=0 OR e.cancel_requested=1 OR length(e.pending_control)>0))))`

// Recipient custody (staged inputs, collected models) outlives the execution. It is owed to
// the machine, but it is bytes on its disk, not a worker state to preserve. A finished run's
// output log is owed until this client holds its products and acknowledges the terminal.
const machineExecutionOwed = `((NOT ` + machineExecutionLost + ` AND (` + machineInputOwed + ` OR ` + machineModelRetentionOwed + ` OR ` + machineLogOwed + `)) OR ` + machineExecutionLive + `)`

// Runtime's explicit release waives its log too: nothing of the execution is owed after it.
const machineLogOwed = `(length(e.receipt)>0 AND e.collected=0 AND r.state IN ('succeeded','failed','canceled') AND NOT ` + machineRetentionReleased + `)`

const machineRetentionReleased = `EXISTS(SELECT 1 FROM request_events released
 WHERE released.request_id=r.id AND released.type='machine.retention_released')`

func (s *Store) MachineExecutionOwesWork(id string) (bool, *exit.Error) {
	var owed bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE r.id=? AND NOT `+machineExecutionAbandoned+` AND `+machineExecutionOwed+`)`, id).Scan(&owed)
	if err != nil {
		return false, exit.Internalf("cannot inspect Runtime execution obligations: %s", err)
	}
	return owed, nil
}

// The events that settle a finished machine execution's result custody after its terminal:
// collected, refused until the owner acts, kept on the machine, or lost with it.
const (
	MachineResultCollected    = "client.machine_result_collected"
	MachineCollectionRefused  = "client.machine_collection_refused"
	MachineResultRetainedType = "machine.result_retained"
)

// CustodyEvent reports whether an event settles, for now, where a finished result is.
func CustodyEvent(eventType string) bool {
	switch eventType {
	case MachineResultCollected, MachineCollectionRefused, MachineResultRetainedType, "client.machine_lost", "client.machine_abandoned":
		return true
	}
	return false
}

// RefuseMachineCollection records why a finished execution's result cannot be collected
// until its owner acts. Nothing is recorded before the outcome or after collection, and a
// repeated refusal is recorded once.
func (s *Store) MachineCollectionRefusal(id string) (string, string, *exit.Error) {
	return machineCollectionRefusal(s.db, id)
}

func machineCollectionRefusal(q interface {
	QueryRow(string, ...any) *sql.Row
}, id string) (string, string, *exit.Error) {
	var code, message string
	err := q.QueryRow(`SELECT COALESCE(json_extract(payload,'$.error_code'),''),COALESCE(json_extract(payload,'$.error'),'')
 FROM request_events WHERE request_id=? AND type=? ORDER BY seq DESC LIMIT 1`, id, MachineCollectionRefused).Scan(&code, &message)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", nil
	}
	if err != nil {
		return "", "", exit.Internalf("cannot read collection refusal: %s", err)
	}
	return code, message, nil
}

// MachineResultRetained is why an execution's finished result stays with its machine
// because this host has no recipient for it, or "" when none was recorded.
func (s *Store) MachineResultRetained(id string) (string, *exit.Error) {
	var reason string
	err := s.db.QueryRow(`SELECT COALESCE(json_extract(payload,'$.reason'),'') FROM request_events
 WHERE request_id=? AND type=? ORDER BY seq DESC LIMIT 1`, id, MachineResultRetainedType).Scan(&reason)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", exit.Internalf("cannot read retained machine result: %s", err)
	}
	return reason, nil
}

// RentalHasLiveMachineExecutions is whether an execution on the rental still needs its
// worker's current state, which work Creator drives would replace.
func (s *Store) RentalHasLiveMachineExecutions(id string) (bool, *exit.Error) {
	var live bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE (e.machine_id=? OR r.worker=?) AND `+machineExecutionLive+`)`, id, id).Scan(&live)
	if err != nil {
		return false, exit.Internalf("cannot inspect live rented Runtime executions: %s", err)
	}
	return live, nil
}

// RentalRetainedModelBytes is the disk the rental's held machine models take: what each
// output's write added, as its native receipt reported it. A hold recorded before that
// was counted keeps the estimate that an ingest's output is at least its planned source.
func (s *Store) RentalRetainedModelBytes(id string) (int64, *exit.Error) {
	var total int64
	err := s.db.QueryRow(`SELECT COALESCE(SUM(CASE WHEN written>0 THEN written ELSE estimate END),0) FROM (
 SELECT (SELECT COALESCE(SUM(json_extract(hold.payload,'$.bytes')),0) FROM request_events hold
   WHERE hold.request_id=r.id AND hold.type='machine.model_retention' AND json_extract(hold.payload,'$.state')!='released'
   AND NOT EXISTS(SELECT 1 FROM request_events newer WHERE newer.request_id=hold.request_id AND newer.type=hold.type
   AND newer.seq>hold.seq AND json_extract(newer.payload,'$.retention_id')=json_extract(hold.payload,'$.retention_id'))) AS written,
 COALESCE((SELECT SUM(length) FROM request_model_transfer_files f WHERE f.request_id=r.id),0)
 + COALESCE((SELECT source_bytes FROM request_planned_sources p WHERE p.request_id=r.id),0) AS estimate
 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE (e.machine_id=? OR r.worker=?) AND `+machineModelRetentionOwed+`)`, id, id).Scan(&total)
	if err != nil {
		return 0, exit.Internalf("cannot total retained rented model bytes: %s", err)
	}
	return total, nil
}

func (s *Store) RentalHasMachineObligations(id string) (bool, *exit.Error) {
	var owed bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE (e.machine_id=? OR r.worker=?) AND `+machineExecutionOwed+`)`, id, id).Scan(&owed)
	if err != nil {
		return false, exit.Internalf("cannot inspect rented Runtime execution obligations: %s", err)
	}
	return owed, nil
}

func (s *Store) RequestExecutionTiming(id string) (int64, uint64, *exit.Error) {
	var duration int64
	var deadline uint64
	err := s.db.QueryRow(`SELECT COALESCE(json_extract(payload,'$.timeout_ms'),0),COALESCE(json_extract(payload,'$.deadline_unix_ms'),0) FROM request_events WHERE request_id=? AND type='run.created' ORDER BY seq LIMIT 1`, id).Scan(&duration, &deadline)
	if err == sql.ErrNoRows {
		return 0, 0, nil
	}
	if err != nil {
		return 0, 0, exit.Internalf("cannot read frozen execution deadline: %s", err)
	}
	return duration, deadline, nil
}

func scanMachineExecution(row interface{ Scan(...any) error }) (*MachineExecution, error) {
	var value MachineExecution
	err := row.Scan(&value.RequestID, &value.MachineID, &value.Submission, &value.Receipt,
		&value.ObservedState, &value.RemoteCursor, &value.Outcome, &value.PendingControl, &value.CancelRequested, &value.Collected, &value.SubmissionClosed, &value.Abandoned, &value.Lost)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return &value, err
}

func (s *Store) MachineExecution(id string) (*MachineExecution, *exit.Error) {
	value, err := scanMachineExecution(s.db.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return nil, exit.Internalf("cannot read machine execution observation: %s", err)
	}
	return value, nil
}

func (s *Store) MachineExecutions() ([]MachineExecution, *exit.Error) {
	rows, err := s.db.Query(`SELECT ` + machineExecutionColumns + ` FROM machine_executions ORDER BY rowid`)
	if err != nil {
		return nil, exit.Internalf("cannot list machine execution observations: %s", err)
	}
	defer rows.Close()
	var result []MachineExecution
	for rows.Next() {
		value, err := scanMachineExecution(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read machine execution observation: %s", err)
		}
		result = append(result, *value)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish machine execution observations: %s", err)
	}
	return result, nil
}

// LinkMachineExecution pins the selected machine before preparation or submission.
// Reconnect cannot silently move an accepted or ambiguous execution to another pod.
func (s *Store) LinkMachineExecution(id, machine string) *exit.Error {
	if machine == "" || len(machine) > 256 {
		return exit.New(exit.Validation, "machine execution requires one bounded machine identity")
	}
	result, err := s.db.Exec(`UPDATE machine_executions SET machine_id=? WHERE request_id=? AND (machine_id='' OR machine_id=?) AND NOT EXISTS(SELECT 1 FROM rentals WHERE id=? AND state IN ('release_requested','released','failed')) AND `+machineExecutionAdmissionOpen, machine, id, machine, machine)
	if err != nil {
		return exit.Internalf("cannot record machine execution destination: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "machine execution already names another machine")
	}
	return nil
}

// RequestMachineCancellation records intent and its actor before any remote I/O.
// Unsent work is canceled here. Sent or accepted work remains canceling until its
// machine proves the outcome. The return value reports recorded acceptance.
// An empty actor is used only when reconciling an intent that is already durable.
func (s *Store) RequestMachineCancellation(id, actor string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot record machine cancellation intent: %s", err)
	}
	defer tx.Rollback()
	lost, err := machineExecutionLostIn(tx, id)
	if err != nil {
		return false, exit.Internalf("cannot inspect canceled machine loss: %s", err)
	}
	if lost {
		return false, exit.Named(exit.Conflict, "machine_execution.state_lost", "execution's rented machine was confirmed destroyed")
	}
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return false, exit.Internalf("cannot read machine cancellation intent: %s", err)
	}
	if link != nil && link.Abandoned {
		return false, nil
	}
	if link == nil {
		return false, nil
	}
	if _, err := tx.Exec(`UPDATE machine_executions SET cancel_requested=1 WHERE request_id=?`, id); err != nil {
		return false, exit.Internalf("cannot retain machine cancellation intent: %s", err)
	}
	if len(link.Receipt) == 0 {
		state, scope := "canceled", "before_machine_submission"
		var sent bool
		if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type=?)`, id, RunV1Sent).Scan(&sent); err != nil {
			return false, exit.Internalf("cannot inspect machine dispatch before cancellation: %s", err)
		}
		if sent || len(link.Submission) > 0 && !link.SubmissionClosed {
			state, scope = "canceling", "machine_acceptance_unknown"
		}
		if err := projectCancellationTx(tx, id, state, scope); err != nil {
			return false, exit.Internalf("cannot project pending machine cancellation: %s", err)
		}
	} else if _, err := tx.Exec(`UPDATE requests SET state='canceling' WHERE id=? AND state NOT IN (`+settledRequestStates+`)`, id); err != nil {
		return false, exit.Internalf("cannot project requested machine cancellation: %s", err)
	}
	if actor != "" {
		var recorded bool
		if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type='request.cancel_requested')`, id).Scan(&recorded); err != nil {
			return false, exit.Internalf("cannot inspect cancellation attribution: %s", err)
		}
		if !recorded {
			var ordinal int64
			if err := tx.QueryRow(`SELECT ordinal FROM requests WHERE id=?`, id).Scan(&ordinal); err != nil {
				return false, exit.Internalf("cannot read canceled attempt: %s", err)
			}
			if err := appendEventTx(tx, id, "request.cancel_requested", ordinal, map[string]any{"actor": actor, "source": "machine_control"}); err != nil {
				return false, exit.Internalf("cannot record cancellation attribution: %s", err)
			}
		}
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit machine cancellation intent: %s", err)
	}
	return len(link.Receipt) > 0, nil
}

// projectCancellationTx moves an unfinished run to `state` with its event. A finished run
// keeps its terminal: what settles after it is a note, never a second status.
func projectCancellationTx(tx *sql.Tx, id, state, scope string) error {
	projected, err := tx.Exec(`UPDATE requests SET state=?,retain_work=0 WHERE id=? AND state NOT IN (`+settledRequestStates+`)`, state, id)
	if err != nil {
		return err
	}
	if n, _ := projected.RowsAffected(); n != 1 {
		return nil
	}
	// The terminal carries the run's current attempt: a follower skips an older attempt's.
	var ordinal int64
	if err := tx.QueryRow(`SELECT ordinal FROM requests WHERE id=?`, id).Scan(&ordinal); err != nil {
		return err
	}
	return appendEventTx(tx, id, StateEvent(state), ordinal, map[string]any{"scope": scope})
}
