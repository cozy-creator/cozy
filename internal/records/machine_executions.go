package records

// These rows are client submission and observation records. Runtime owns the
// attempts, children, retry policy, and output custody on the selected machine.

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"math"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
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
}

const machineExecutionColumns = `request_id,machine_id,submission,receipt,observed_state,remote_cursor,outcome,pending_control,cancel_requested,collected`

func (s *Store) MachinePackageUpload(boot, revision string) (string, *exit.Error) {
	var request string
	err := s.db.QueryRow(`SELECT request_id FROM request_events WHERE type='machine.package_uploaded' AND json_extract(payload,'$.worker_boot_id')=? AND json_extract(payload,'$.revision')=? ORDER BY seq DESC LIMIT 1`, boot, revision).Scan(&request)
	if err == sql.ErrNoRows {
		return "", nil
	}
	if err != nil {
		return "", exit.Internalf("cannot read machine package transfer progress: %s", err)
	}
	return request, nil
}

// e/r are the observer and request aliases. Explicit Runtime release is stronger
// than a status projection. A failed or paused root remains owed after its small
// error result is collected, and cancellation alone never proves native cleanup.
const machineExecutionOwed = `NOT EXISTS(SELECT 1 FROM request_events released
 WHERE released.request_id=r.id AND released.type='machine.retention_released') AND (
 (length(e.receipt)=0 AND r.state!='refused' AND (length(e.submission)>0 OR r.state!='canceled')) OR
 (length(e.receipt)>0 AND (r.state!='succeeded' OR e.collected=0 OR e.cancel_requested=1 OR length(e.pending_control)>0)))`

func (s *Store) MachineExecutionOwesWork(id string) (bool, *exit.Error) {
	var owed bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE r.id=? AND `+machineExecutionOwed+`)`, id).Scan(&owed)
	if err != nil {
		return false, exit.Internalf("cannot inspect Runtime execution obligations: %s", err)
	}
	return owed, nil
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
	err := s.db.QueryRow(`SELECT COALESCE(json_extract(payload,'$.timeout_ms'),0),COALESCE(json_extract(payload,'$.deadline_unix_ms'),0) FROM request_events WHERE request_id=? AND type='request.submitted' ORDER BY seq LIMIT 1`, id).Scan(&duration, &deadline)
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
		&value.ObservedState, &value.RemoteCursor, &value.Outcome, &value.PendingControl, &value.CancelRequested, &value.Collected)
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
	result, err := s.db.Exec(`UPDATE machine_executions SET machine_id=? WHERE request_id=? AND (machine_id='' OR machine_id=?)`, machine, id, machine)
	if err != nil {
		return exit.Internalf("cannot record machine execution destination: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "machine execution already names another machine")
	}
	return nil
}

// RecordMachineSubmission freezes the exact offer before its first transmission.
// A connection error thereafter is ambiguous acceptance, never permission to mint
// another submission identity or start a local attempt.
func (s *Store) RecordMachineSubmission(id string, submission *pb.MachineExecutionSubmit) *exit.Error {
	if submission == nil || submission.Offer == nil || submission.Offer.RequestId != id ||
		submission.SubmissionId == "" || len(submission.SubmissionId) > 256 ||
		len(submission.CaptureDigest) != 32 || !bytes.Equal(canonical.Digest(submission.CaptureCanonicalBytes), submission.CaptureDigest) ||
		len(submission.Offer.InvocationSpecDigest) != 32 || !bytes.Equal(canonical.Digest(submission.Offer.InvocationSpecCanonicalBytes), submission.Offer.InvocationSpecDigest) {
		return exit.New(exit.Validation, "machine submission identity is incomplete")
	}
	value := proto.Clone(submission).(*pb.MachineExecutionSubmit)
	value.Claim = nil // transport authentication is refreshed, not capture authority
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(value)
	if err != nil || len(raw) > 8<<20 {
		return exit.New(exit.Validation, "machine submission exceeds its bounded envelope")
	}
	result, err := s.db.Exec(`UPDATE machine_executions SET submission=? WHERE request_id=? AND machine_id!='' AND (length(submission)=0 OR submission=?) AND EXISTS(SELECT 1 FROM requests WHERE id=? AND state!='canceled')`, raw, id, raw, id)
	if err != nil {
		return exit.Internalf("cannot retain machine submission: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "machine submission differs from the already recorded offer")
	}
	return nil
}

func (s *Store) AcceptMachineExecution(id string, receipt *pb.MachineExecutionReceipt) *exit.Error {
	if receipt == nil || receipt.RequestId != id || receipt.AcceptedAtMs == 0 || receipt.AcceptedAtMs > math.MaxInt64 ||
		receipt.WorkerId == "" || receipt.WorkerBootId == "" || receipt.ExecutionWorkspaceId == "" || len(receipt.ExecutionWorkspaceId) > 256 {
		return exit.New(exit.Conflict, "machine returned an incomplete acceptance receipt")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot record machine acceptance: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil || link == nil {
		return exit.New(exit.Conflict, "machine acceptance has no client submission")
	}
	var submission pb.MachineExecutionSubmit
	if proto.Unmarshal(link.Submission, &submission) != nil || submission.Offer == nil ||
		receipt.SubmissionId != submission.SubmissionId || !bytes.Equal(receipt.CaptureDigest, submission.CaptureDigest) ||
		!bytes.Equal(receipt.InvocationSpecDigest, submission.Offer.InvocationSpecDigest) ||
		receipt.PublicationAuthorizationId != submission.PublicationAuthorizationId {
		return exit.New(exit.Conflict, "machine accepted a different capture or invocation")
	}
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(receipt)
	if err != nil || len(raw) > 4096 {
		return exit.New(exit.Conflict, "machine acceptance receipt is not bounded")
	}
	if len(link.Receipt) > 0 {
		if !bytes.Equal(link.Receipt, raw) {
			return exit.New(exit.Conflict, "machine acceptance differs from the retained receipt")
		}
		return nil
	}
	if _, err := tx.Exec(`UPDATE machine_executions SET receipt=? WHERE request_id=?`, raw, id); err != nil {
		return exit.Internalf("cannot retain machine receipt: %s", err)
	}
	payload, _ := json.Marshal(map[string]any{"execution_workspace_id": receipt.ExecutionWorkspaceId, "worker_id": receipt.WorkerId, "worker_boot_id": receipt.WorkerBootId, "durable": true})
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'request.machine_accepted',0,?,?)`, id, payload, time.UnixMilli(int64(receipt.AcceptedAtMs)).UTC().Format(time.RFC3339Nano)); err != nil {
		return exit.Internalf("cannot record machine acceptance event: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit machine acceptance: %s", err)
	}
	return nil
}

func machineObservationIdentity(link *MachineExecution, state *pb.MachineExecutionState) *exit.Error {
	var receipt pb.MachineExecutionReceipt
	if link == nil || proto.Unmarshal(link.Receipt, &receipt) != nil || receipt.RequestId == "" || state == nil ||
		state.RequestId != link.RequestID || state.ExecutionWorkspaceId != receipt.ExecutionWorkspaceId || state.WorkerId != receipt.WorkerId ||
		state.WorkerBootId == "" || state.Generation == 0 || state.Generation > math.MaxInt64 || state.Sequence > math.MaxInt64 || state.AttemptOrdinal > math.MaxInt64 {
		return exit.New(exit.Conflict, "machine observation differs from its accepted workspace")
	}
	return nil
}

func observedMachineRequestState(value string) string {
	switch value {
	case "queued", "retrying":
		return "queued"
	case "running":
		return "dispatching"
	case "pausing", "canceling", "paused", "succeeded", "failed", "canceled", "blocked":
		return value
	default:
		return ""
	}
}

// ObserveMachineExecution commits one authenticated page and its cursor together.
// Request rows and event ordinals are projections; the attempts table stays empty.
func (s *Store) ObserveMachineExecution(id string, state *pb.MachineExecutionState, page *pb.MachineExecutionEventPage) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin machine observation: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return exit.Internalf("cannot read machine observation: %s", err)
	}
	if problem := machineObservationIdentity(link, state); problem != nil {
		return problem
	}
	nextState := observedMachineRequestState(state.State)
	if nextState == "" || page == nil || page.NextAfter > page.HeadSequence || page.HeadSequence > math.MaxInt64 || page.CompactedThrough > page.HeadSequence {
		return exit.New(exit.Conflict, "machine observation has an invalid state or event cursor")
	}
	var previous pb.MachineExecutionState
	if len(link.ObservedState) > 0 {
		if proto.Unmarshal(link.ObservedState, &previous) != nil {
			return exit.Internalf("recorded machine state is unreadable")
		}
		if state.Sequence < previous.Sequence || state.Generation < previous.Generation {
			return nil
		}
		if state.Generation == previous.Generation && state.AttemptOrdinal < previous.AttemptOrdinal {
			return exit.New(exit.Conflict, "machine observation regressed its attempt ordinal")
		}
	}
	cursor := uint64(link.RemoteCursor)
	retentionReleased := false
	// A replayed page cannot regress the durable cursor. Its state snapshot may
	// still be newer, so project it after ignoring already recorded event bytes.
	if page.NextAfter < cursor {
		page = &pb.MachineExecutionEventPage{NextAfter: cursor, HeadSequence: max(cursor, page.HeadSequence)}
	}
	if page.CompactedThrough > cursor {
		cursor = page.CompactedThrough
	}
	for _, event := range page.Events {
		if event == nil || event.Sequence <= cursor {
			continue
		}
		if event.Sequence > page.NextAfter || event.AtMs > math.MaxInt64 || event.AttemptOrdinal > math.MaxInt64 ||
			event.Kind == "" || len(event.Kind) > 128 || len(event.BodyCanonicalBytes) > 64<<10 || !json.Valid(event.BodyCanonicalBytes) {
			return exit.New(exit.Conflict, "machine returned an invalid execution event")
		}
		kind, payload := "machine."+event.Kind, event.BodyCanonicalBytes
		if event.Kind == "retention_released" {
			retentionReleased = true
		}
		if event.Kind == "running" {
			kind = "request.accepted"
		}
		if event.Kind == "progress" || event.Kind == "log" {
			var progress struct {
				Type    string          `json:"type"`
				Payload json.RawMessage `json:"payload"`
			}
			if json.Unmarshal(payload, &progress) == nil && progress.Type == "log" && json.Valid(progress.Payload) {
				kind, payload = "request.log", progress.Payload
			}
		}
		if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`,
			id, kind, event.AttemptOrdinal, string(payload), time.UnixMilli(int64(event.AtMs)).UTC().Format(time.RFC3339Nano)); err != nil {
			return exit.Internalf("cannot cache machine event: %s", err)
		}
		cursor = event.Sequence
	}
	if page.NextAfter < cursor || page.NextAfter > cursor && page.NextAfter > page.CompactedThrough {
		return exit.New(exit.Conflict, "machine event page skips unaccounted history")
	}
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(state)
	if err != nil {
		return exit.Internalf("cannot retain machine state: %s", err)
	}
	resetOutcome := previous.AttemptOrdinal != 0 && previous.AttemptOrdinal != state.AttemptOrdinal
	if _, err := tx.Exec(`UPDATE machine_executions SET observed_state=?,remote_cursor=?,collected=?,outcome=CASE WHEN ? THEN x'' ELSE outcome END WHERE request_id=?`, raw, cursor, state.Collected, resetOutcome, id); err != nil {
		return exit.Internalf("cannot update machine observation cursor: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=?,ordinal=?,retain_work=CASE WHEN (? AND ?='succeeded') OR ?='canceled' THEN 0 ELSE retain_work END WHERE id=?`, nextState, state.AttemptOrdinal, state.Collected, nextState, nextState, id); err != nil {
		return exit.Internalf("cannot project machine execution status: %s", err)
	}
	if retentionReleased {
		if _, err := tx.Exec(`UPDATE machine_executions SET pending_control=x'',cancel_requested=0 WHERE request_id=?`, id); err != nil {
			return exit.Internalf("cannot record Runtime retention release: %s", err)
		}
		if _, err := tx.Exec(`UPDATE requests SET retain_work=0 WHERE id=?`, id); err != nil {
			return exit.Internalf("cannot project Runtime retention release: %s", err)
		}
	}
	if previous.State != state.State && (state.State == "paused" || state.State == "canceled") {
		if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`, id, "request."+state.State, state.AttemptOrdinal, `{"machine_execution":true}`, now()); err != nil {
			return exit.Internalf("cannot record observed machine control completion: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit machine observation: %s", err)
	}
	return nil
}

// RecordMachineOutcome retains the exact Runtime terminal before result readback
// or acknowledgement. It does not claim custody of referenced model/file bytes.
func (s *Store) RecordMachineOutcome(id string, outcome *pb.AttemptOutcome) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin machine outcome observation: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return exit.Internalf("cannot read machine outcome observation: %s", err)
	}
	var state pb.MachineExecutionState
	var body pb.AttemptOutcomeBody
	if link == nil || proto.Unmarshal(link.ObservedState, &state) != nil || outcome == nil ||
		outcome.RequestId != id || outcome.AttemptOrdinal != state.AttemptOrdinal ||
		!bytes.Equal(canonical.Digest(outcome.OutcomeCanonicalBytes), outcome.OutcomeDigest) ||
		canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body) != nil ||
		body.RequestId != id || body.AttemptOrdinal != outcome.AttemptOrdinal ||
		body.InvocationSpecDigest != spellMachineDigest(outcome.InvocationSpecDigest) {
		return exit.New(exit.Conflict, "machine outcome differs from its current execution")
	}
	var receipt pb.MachineExecutionReceipt
	if proto.Unmarshal(link.Receipt, &receipt) != nil || !bytes.Equal(receipt.InvocationSpecDigest, outcome.InvocationSpecDigest) {
		return exit.New(exit.Conflict, "machine outcome names another invocation")
	}
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(outcome)
	if err != nil || len(raw) > 8<<20 {
		return exit.New(exit.Conflict, "machine outcome is not bounded")
	}
	if len(link.Outcome) > 0 {
		if !bytes.Equal(link.Outcome, raw) {
			return exit.New(exit.Conflict, "machine returned a different terminal for the observed attempt")
		}
		return nil
	}
	result, err := tx.Exec(`UPDATE machine_executions SET outcome=? WHERE request_id=? AND observed_state=?`, raw, id, link.ObservedState)
	if err != nil {
		return exit.Internalf("cannot retain machine outcome: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "machine execution changed during result collection")
	}
	kind := "request.failed"
	if body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		kind = "request.completed"
	} else if body.Status == pb.OutcomeStatus_OUTCOME_STATUS_CANCELED {
		kind = "request.canceled"
	}
	facts := map[string]any{"machine_execution": true, "status": strings.TrimPrefix(body.Status.String(), "OUTCOME_STATUS_"), "outputs": []any{}}
	if body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		facts["error"] = body.SafeMessage
	}
	payload, _ := json.Marshal(facts)
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`, id, kind, outcome.AttemptOrdinal, string(payload), now()); err != nil {
		return exit.Internalf("cannot record machine terminal observation: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit machine terminal observation: %s", err)
	}
	return nil
}

func spellMachineDigest(raw []byte) string {
	value, _ := canonical.Spell(raw)
	return value
}

// RecordMachineControl preserves the generation and command identity across a
// lost reply. A caller must reconcile this command before issuing another one.
func (s *Store) RecordMachineControl(id string, command *pb.MachineExecutionControl) *exit.Error {
	link, problem := s.MachineExecution(id)
	if problem != nil {
		return problem
	}
	var receipt pb.MachineExecutionReceipt
	if link == nil || proto.Unmarshal(link.Receipt, &receipt) != nil || command == nil || command.Execution == nil ||
		command.Execution.RequestId != id || command.Execution.ExpectedExecutionWorkspaceId != receipt.ExecutionWorkspaceId ||
		command.CommandId == "" || len(command.CommandId) > 256 || command.ExpectedGeneration == 0 {
		return exit.New(exit.Conflict, "machine control does not name its accepted execution")
	}
	value := proto.Clone(command).(*pb.MachineExecutionControl)
	value.Execution.Claim = nil
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(value)
	if err != nil {
		return exit.Internalf("cannot encode machine control: %s", err)
	}
	result, err := s.db.Exec(`UPDATE machine_executions SET pending_control=?,cancel_requested=CASE WHEN ? THEN 1 ELSE cancel_requested END WHERE request_id=? AND (length(pending_control)=0 OR pending_control=?)`, raw, command.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL, id, raw)
	if err != nil {
		return exit.Internalf("cannot retain machine control: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "another machine control is awaiting acknowledgement")
	}
	return nil
}

func (s *Store) CompleteMachineControl(id string, command []byte) *exit.Error {
	var decoded pb.MachineExecutionControl
	if proto.Unmarshal(command, &decoded) != nil {
		return exit.New(exit.Conflict, "machine control acknowledgement has an invalid command")
	}
	if decoded.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL {
		var released bool
		if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type='machine.retention_released')`, id).Scan(&released); err != nil {
			return exit.Internalf("cannot inspect cancellation cleanup: %s", err)
		}
		if !released {
			return nil // acknowledgement is not proof that native retention was released
		}
	}
	result, err := s.db.Exec(`UPDATE machine_executions SET pending_control=x'',cancel_requested=CASE WHEN ? THEN 0 ELSE cancel_requested END WHERE request_id=? AND pending_control=?`, decoded.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL, id, command)
	if err != nil {
		return exit.Internalf("cannot acknowledge machine control: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		current, problem := s.MachineExecution(id)
		if problem == nil && current != nil && len(current.PendingControl) == 0 {
			return nil
		}
		return exit.New(exit.Conflict, "machine control acknowledgement changed its command")
	}
	return nil
}

// CancelMachineBeforeAcceptance distinguishes an unsent local intent from a
// submission whose reply may have been lost. The latter must reconcile its exact
// receipt and cancel Runtime; the client cannot claim it never executed.
func (s *Store) CancelMachineBeforeAcceptance(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot record machine cancellation intent: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return false, exit.Internalf("cannot read machine cancellation intent: %s", err)
	}
	if link == nil || len(link.Receipt) > 0 {
		return false, nil
	}
	state := "canceling"
	if len(link.Submission) == 0 {
		state = "canceled"
	}
	if _, err := tx.Exec(`UPDATE machine_executions SET cancel_requested=1 WHERE request_id=?`, id); err != nil {
		return false, exit.Internalf("cannot retain machine cancellation intent: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=?,retain_work=CASE WHEN ?='canceled' THEN 0 ELSE retain_work END WHERE id=?`, state, state, id); err != nil {
		return false, exit.Internalf("cannot project pending machine cancellation: %s", err)
	}
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,0,?,?)`, id, "request."+state, `{"scope":"before_machine_acceptance"}`, now()); err != nil {
		return false, exit.Internalf("cannot record pending machine cancellation event: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit machine cancellation intent: %s", err)
	}
	return true, nil
}

// RefuseMachineSubmission is called only for Runtime's explicit guarantee that
// this request was not accepted. An arbitrary RPC failure is never that proof.
func (s *Store) RefuseMachineSubmission(id, code, detail string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot record machine submission refusal: %s", err)
	}
	defer tx.Rollback()
	var receipt []byte
	if err := tx.QueryRow(`SELECT receipt FROM machine_executions WHERE request_id=?`, id).Scan(&receipt); err != nil || len(receipt) != 0 {
		return exit.New(exit.Conflict, "machine refusal contradicts accepted execution")
	}
	if _, err := tx.Exec(`UPDATE requests SET state='refused',retain_work=0 WHERE id=?`, id); err != nil {
		return exit.Internalf("cannot retain machine refusal state: %s", err)
	}
	if err := appendEventTx(tx, id, "request.failed", 0, map[string]any{"error_type": code, "error": detail, "machine_accepted": false}); err != nil {
		return exit.Internalf("cannot retain machine refusal event: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit machine refusal: %s", err)
	}
	return nil
}

func (s *Store) RejectMachineControl(id string, command []byte) *exit.Error {
	// Unlike an acknowledgement, a rejected cancellation must retain its desire
	// so acceptance-reconciliation can retry it against the current generation.
	_, err := s.db.Exec(`UPDATE machine_executions SET pending_control=x'' WHERE request_id=? AND pending_control=?`, id, command)
	if err != nil {
		return exit.Internalf("cannot record rejected machine control: %s", err)
	}
	return nil
}
