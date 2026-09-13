package records

// These rows are client submission and observation records. Runtime owns the
// attempts, children, retry policy, and output custody on the selected machine.

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"math"
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
	Collected            bool
}

const machineExecutionColumns = `request_id,machine_id,submission,receipt,observed_state,remote_cursor,outcome,pending_control,collected`

func scanMachineExecution(row interface{ Scan(...any) error }) (*MachineExecution, error) {
	var value MachineExecution
	err := row.Scan(&value.RequestID, &value.MachineID, &value.Submission, &value.Receipt,
		&value.ObservedState, &value.RemoteCursor, &value.Outcome, &value.PendingControl, &value.Collected)
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
	result, err := s.db.Exec(`UPDATE machine_executions SET submission=? WHERE request_id=? AND machine_id!='' AND (length(submission)=0 OR submission=?)`, raw, id, raw)
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
		!bytes.Equal(receipt.InvocationSpecDigest, submission.Offer.InvocationSpecDigest) {
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
	case "queued":
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
		if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`,
			id, "machine."+event.Kind, event.AttemptOrdinal, event.BodyCanonicalBytes, time.UnixMilli(int64(event.AtMs)).UTC().Format(time.RFC3339Nano)); err != nil {
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
	if _, err := tx.Exec(`UPDATE machine_executions SET observed_state=?,remote_cursor=?,collected=? WHERE request_id=?`, raw, cursor, state.Collected, id); err != nil {
		return exit.Internalf("cannot update machine observation cursor: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=?,ordinal=? WHERE id=?`, nextState, state.AttemptOrdinal, id); err != nil {
		return exit.Internalf("cannot project machine execution status: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit machine observation: %s", err)
	}
	return nil
}

// RecordMachineOutcome retains the exact Runtime terminal before result readback
// or acknowledgement. It does not claim custody of referenced model/file bytes.
func (s *Store) RecordMachineOutcome(id string, outcome *pb.AttemptOutcome) *exit.Error {
	link, problem := s.MachineExecution(id)
	if problem != nil {
		return problem
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
	result, err := s.db.Exec(`UPDATE machine_executions SET outcome=? WHERE request_id=? AND observed_state=? AND (length(outcome)=0 OR outcome=?)`, raw, id, link.ObservedState, raw)
	if err != nil {
		return exit.Internalf("cannot retain machine outcome: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "machine execution changed during result collection")
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
	result, err := s.db.Exec(`UPDATE machine_executions SET pending_control=? WHERE request_id=? AND (length(pending_control)=0 OR pending_control=?)`, raw, id, raw)
	if err != nil {
		return exit.Internalf("cannot retain machine control: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "another machine control is awaiting acknowledgement")
	}
	return nil
}

func (s *Store) CompleteMachineControl(id string, command []byte) *exit.Error {
	result, err := s.db.Exec(`UPDATE machine_executions SET pending_control=x'' WHERE request_id=? AND pending_control=?`, id, command)
	if err != nil {
		return exit.Internalf("cannot acknowledge machine control: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.New(exit.Conflict, "machine control acknowledgement changed its command")
	}
	return nil
}
