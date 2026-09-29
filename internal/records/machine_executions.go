package records

// These rows are client submission and observation records. Runtime owns the
// attempts, children, retry policy, and output custody on the selected machine.

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
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
	Abandoned            bool
	SubmissionClosed     bool
}

const machineExecutionColumns = `request_id,machine_id,submission,receipt,observed_state,remote_cursor,outcome,pending_control,cancel_requested,collected,
 EXISTS(SELECT 1 FROM request_events closed WHERE closed.request_id=machine_executions.request_id AND closed.type='machine.submission_closed'),
 NOT (` + machineExecutionAdmissionOpen + `)`

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
func (s *Store) RefuseMachineCollection(id string, refusal *exit.Error) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin collection refusal: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return exit.Internalf("cannot read machine collection: %s", err)
	}
	var outcome pb.AttemptOutcome
	if link == nil || link.Collected || len(link.Outcome) == 0 || proto.Unmarshal(link.Outcome, &outcome) != nil {
		return nil
	}
	code, message, problem := machineCollectionRefusal(tx, id)
	if problem != nil || code == refusal.ErrName() && message == refusal.Message {
		return problem
	}
	if err := appendEventTx(tx, id, MachineCollectionRefused, int64(outcome.AttemptOrdinal),
		map[string]any{"machine_execution": true, "error_code": refusal.ErrName(), "error": refusal.Message}); err != nil {
		return exit.Internalf("cannot record collection refusal: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit collection refusal: %s", err)
	}
	return nil
}

// MachineCollectionRefusal is the latest recorded reason a result could not be collected.
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
		&value.ObservedState, &value.RemoteCursor, &value.Outcome, &value.PendingControl, &value.CancelRequested, &value.Collected, &value.SubmissionClosed, &value.Abandoned)
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

// machineSubmissionGrammar is Runtime's execution identity grammar
// (workspace_executions.py `_ID`); a submission outside it is refused on the machine.
var machineSubmissionGrammar = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$`)

// MachineSubmissionID is the machine identity of a request's idempotency key: the key
// itself when Runtime accepts it, otherwise a stable digest of it, so a resume key such
// as `<base>/retry-of/<job>` still names one submission.
func MachineSubmissionID(idemKey string) string {
	if machineSubmissionGrammar.MatchString(idemKey) {
		return idemKey
	}
	digest := sha256.Sum256([]byte(idemKey))
	return "idem-" + hex.EncodeToString(digest[:])
}

// RecordMachineSubmission freezes the exact offer before its first transmission.
// A connection error thereafter is ambiguous acceptance, never permission to mint
// another submission identity or start a local attempt.
func (s *Store) RecordMachineSubmission(id string, submission *pb.MachineExecutionSubmit) *exit.Error {
	// A root named by its release carries no capture or invocation: the machine mints them.
	minted := submission != nil && submission.ReleaseRoot != nil && len(submission.CaptureCanonicalBytes) == 0 &&
		submission.Offer != nil && len(submission.Offer.InvocationSpecCanonicalBytes) == 0
	if submission == nil || submission.Offer == nil || submission.Offer.RequestId != id ||
		!machineSubmissionGrammar.MatchString(submission.SubmissionId) ||
		submission.ExpectedExecutionWorkspaceId == "" || len(submission.ExpectedExecutionWorkspaceId) > 256 ||
		!minted && (len(submission.CaptureDigest) != 32 || !bytes.Equal(canonical.Digest(submission.CaptureCanonicalBytes), submission.CaptureDigest) ||
			len(submission.Offer.InvocationSpecDigest) != 32 || !bytes.Equal(canonical.Digest(submission.Offer.InvocationSpecCanonicalBytes), submission.Offer.InvocationSpecDigest)) {
		return exit.New(exit.Validation, "machine submission identity is incomplete")
	}
	value := proto.Clone(submission).(*pb.MachineExecutionSubmit)
	value.Claim = nil // transport authentication is refreshed, not capture authority
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(value)
	if err != nil || len(raw) > 8<<20 {
		return exit.New(exit.Validation, "machine submission exceeds its bounded envelope")
	}
	result, err := s.db.Exec(`UPDATE machine_executions SET submission=? WHERE request_id=? AND machine_id!='' AND (length(submission)=0 OR submission=?) AND EXISTS(SELECT 1 FROM requests WHERE id=? AND state!='canceled') AND NOT EXISTS(SELECT 1 FROM rentals WHERE id=machine_executions.machine_id AND state IN ('release_requested','released','failed')) AND `+machineExecutionAdmissionOpen, raw, id, raw, id)
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
	// A submission recorded before workspaces were fenced names none; the receipt's own
	// workspace is then the one this execution runs in.
	var submission pb.MachineExecutionSubmit
	if proto.Unmarshal(link.Submission, &submission) != nil || submission.Offer == nil ||
		submission.ExpectedExecutionWorkspaceId != "" && receipt.ExecutionWorkspaceId != submission.ExpectedExecutionWorkspaceId ||
		receipt.SubmissionId != submission.SubmissionId ||
		submission.ReleaseRoot == nil && (!bytes.Equal(receipt.CaptureDigest, submission.CaptureDigest) ||
			!bytes.Equal(receipt.InvocationSpecDigest, submission.Offer.InvocationSpecDigest)) ||
		submission.ReleaseRoot != nil && (len(receipt.CaptureDigest) != 32 || len(receipt.InvocationSpecDigest) != 32) ||
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

// printableNote is a peer's word as a note may quote it: printable ASCII, bounded.
func printableNote(value string) string {
	runes := []rune(value)
	if len(runes) > 64 {
		runes = runes[:64]
	}
	for i, r := range runes {
		if r < 0x20 || r > 0x7e {
			runes[i] = '?'
		}
	}
	return string(runes)
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

func machineWorkActive(state string) bool {
	switch state {
	case "queued", "retrying", "running", "pausing", "canceling":
		return true
	default:
		return false
	}
}

// ObserveMachineExecution commits one authenticated page and its cursor together.
// Request rows and event ordinals are projections; the attempts table stays empty.
func (s *Store) ObserveMachineExecution(id string, state *pb.MachineExecutionState, page *pb.MachineExecutionEventPage) *exit.Error {
	return s.ObserveMachinePage(id, state, page, nil)
}

// ObserveMachinePage is ObserveMachineExecution for a page whose products this client now
// holds: each `product` entry is recorded as the Product `held` names for its sequence.
func (s *Store) ObserveMachinePage(id string, state *pb.MachineExecutionState, page *pb.MachineExecutionEventPage, held map[uint64]Product) *exit.Error {
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
	lost, err := machineExecutionLostIn(tx, id)
	if err != nil {
		return exit.Internalf("cannot inspect machine observation loss: %s", err)
	}
	if lost {
		return exit.Named(exit.Conflict, "machine_execution.state_lost", "cannot observe execution on a confirmed destroyed machine")
	}
	if page == nil || page.NextAfter > page.HeadSequence || page.HeadSequence > math.MaxInt64 || page.CompactedThrough > page.HeadSequence {
		return exit.New(exit.Conflict, "machine observation has an invalid event cursor")
	}
	// The machine is an independently upgraded peer: a state this client does not know is
	// neither terminal nor a refusal. The run keeps its projection, says so once, and its
	// events are recorded as ever; it never wedges on a word. A canceled run stays canceled:
	// a machine that accepted it before the cancel was known can only end it.
	nextState := observedMachineRequestState(state.State)
	var notes []string
	var current string
	if err := tx.QueryRow(`SELECT state FROM requests WHERE id=?`, id).Scan(&current); err != nil {
		return exit.Internalf("cannot read the run's state: %s", err)
	}
	if nextState == "" || current == "canceled" || link.Abandoned {
		nextState = current
	}
	var previous pb.MachineExecutionState
	// A snapshot older than the recorded one projects nothing, but its page's entries are
	// history all the same: they are recorded, never dropped with the snapshot.
	stale := false
	if len(link.ObservedState) > 0 {
		if proto.Unmarshal(link.ObservedState, &previous) != nil {
			return exit.Internalf("recorded machine state is unreadable")
		}
		stale = state.Sequence < previous.Sequence || state.Generation < previous.Generation
		if !stale && state.Generation == previous.Generation && state.AttemptOrdinal < previous.AttemptOrdinal {
			return exit.New(exit.Conflict, "machine observation regressed its attempt ordinal")
		}
	}
	if !stale && observedMachineRequestState(state.State) == "" && previous.State != state.State {
		notes = append(notes, fmt.Sprintf("the machine reports state %q, which this Creator does not know; the run is followed as it was", printableNote(state.State)))
	}
	cursor := uint64(link.RemoteCursor)
	retentionReleased := false
	// A replayed page cannot regress the durable cursor. Its state snapshot may
	// still be newer, so project it after ignoring already recorded event bytes.
	if page.NextAfter < cursor {
		page = &pb.MachineExecutionEventPage{NextAfter: cursor, HeadSequence: max(cursor, page.HeadSequence)}
	}
	// Only live progress is compacted on the machine: progress at or below CompactedThrough
	// may be gone, and every other event a page carries is retained history, recorded
	// whatever its sequence. A gap nothing explains is recorded and read past, never a stall.
	for _, event := range page.Events {
		if event == nil || event.Sequence <= cursor {
			continue
		}
		if event.Sequence > page.NextAfter {
			return exit.New(exit.Conflict, "machine returned an event past its page")
		}
		// One unreadable event is skipped with a note; the rest of the page is kept.
		if event.AtMs > math.MaxInt64 || event.AttemptOrdinal > math.MaxInt64 || event.Kind == "" || len(event.Kind) > 128 ||
			len(event.BodyCanonicalBytes) > 64<<10 || !json.Valid(event.BodyCanonicalBytes) {
			notes = append(notes, fmt.Sprintf("the machine's event %d was unreadable and is skipped", event.Sequence))
			cursor = event.Sequence
			continue
		}
		if event.Sequence-1 > max(cursor, page.CompactedThrough) {
			gap, _ := json.Marshal(map[string]uint64{"after": cursor, "before": event.Sequence})
			if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'machine.events_missing',?,?,?)`,
				id, event.AttemptOrdinal, string(gap), time.UnixMilli(int64(event.AtMs)).UTC().Format(time.RFC3339Nano)); err != nil {
				return exit.Internalf("cannot record missing machine events: %s", err)
			}
		}
		kind, payload := "machine."+event.Kind, event.BodyCanonicalBytes
		if event.Kind == "retention_released" {
			retentionReleased = true
		}
		if event.Kind == "running" && current != "canceled" && !link.Abandoned {
			kind = "run.in_progress"
		}
		if event.Kind == "product" {
			product, ok := held[event.Sequence]
			if !ok {
				// An output revision this client could not hold (a malformed entry) is skipped.
				notes = append(notes, fmt.Sprintf("the machine's output revision %d was unusable and is skipped", event.Sequence))
				cursor = event.Sequence
				continue
			}
			at := time.UnixMilli(int64(event.AtMs)).UTC().Format(time.RFC3339Nano)
			for _, item := range outputItemEvents(product) {
				raw, err := json.Marshal(item.payload)
				if err != nil {
					return exit.Internalf("cannot record an output revision: %s", err)
				}
				if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`,
					id, item.kind, event.AttemptOrdinal, string(raw), at); err != nil {
					return exit.Internalf("cannot record an output revision: %s", err)
				}
			}
			cursor = event.Sequence
			continue
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
	for _, note := range notes {
		if err := appendEventTx(tx, id, "request.warning", int64(state.AttemptOrdinal), map[string]any{"message": note}); err != nil {
			return exit.Internalf("cannot record a machine note: %s", err)
		}
	}
	if retentionReleased && !link.Abandoned {
		if _, err := tx.Exec(`UPDATE machine_executions SET pending_control=x'',cancel_requested=0 WHERE request_id=?`, id); err != nil {
			return exit.Internalf("cannot record Runtime retention release: %s", err)
		}
		if _, err := tx.Exec(`UPDATE requests SET retain_work=0 WHERE id=?`, id); err != nil {
			return exit.Internalf("cannot project Runtime retention release: %s", err)
		}
	}
	if stale {
		if _, err := tx.Exec(`UPDATE machine_executions SET remote_cursor=? WHERE request_id=?`, cursor, id); err != nil {
			return exit.Internalf("cannot update machine observation cursor: %s", err)
		}
		if err := tx.Commit(); err != nil {
			return exit.Internalf("cannot commit machine observation: %s", err)
		}
		return nil
	}
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(state)
	if err != nil {
		return exit.Internalf("cannot retain machine state: %s", err)
	}
	resetOutcome := !link.Abandoned && previous.AttemptOrdinal != 0 && previous.AttemptOrdinal != state.AttemptOrdinal
	if _, err := tx.Exec(`UPDATE machine_executions SET observed_state=?,remote_cursor=?,collected=?,outcome=CASE WHEN ? THEN x'' ELSE outcome END WHERE request_id=?`, raw, cursor, state.Collected, resetOutcome, id); err != nil {
		return exit.Internalf("cannot update machine observation cursor: %s", err)
	}
	if _, err := tx.Exec(`UPDATE requests SET state=?,ordinal=?,retain_work=CASE WHEN (? AND ?='succeeded') OR ?='canceled' THEN 0 ELSE retain_work END WHERE id=?`, nextState, state.AttemptOrdinal, state.Collected, nextState, nextState, id); err != nil {
		return exit.Internalf("cannot project machine execution status: %s", err)
	}
	// Custody follows the terminal: a caller waiting for the result wakes on this event.
	if state.Collected && !link.Collected {
		if err := appendEventTx(tx, id, MachineResultCollected, int64(state.AttemptOrdinal),
			map[string]any{"machine_execution": true, "state": state.State}); err != nil {
			return exit.Internalf("cannot record machine result collection: %s", err)
		}
	}
	// Finish the local idle clock atomically with dropping the active request
	// projection. Outcome collection happens later and must not leave an idle
	// release window or renew this clock on repeated terminal observations.
	// A state this client does not know says nothing about whether the work finished.
	if observedMachineRequestState(state.State) != "" && !machineWorkActive(state.State) && (len(link.ObservedState) == 0 ||
		machineWorkActive(previous.State) || state.AttemptOrdinal > previous.AttemptOrdinal) {
		if err := appendEventTx(tx, id, "client.machine_work_finished", int64(state.AttemptOrdinal),
			map[string]any{"machine_execution": true, "state": state.State}); err != nil {
			return exit.Internalf("cannot record machine work completion: %s", err)
		}
	}
	// A canceled run's terminal event comes with its outcome (RecordMachineOutcome): the
	// run's result is the fold of its products, collected like any other terminal.
	if !link.Abandoned && previous.State != state.State && state.State == "paused" {
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
		if link.Abandoned && !bytes.Equal(link.Outcome, raw) {
			var prior pb.AttemptOutcome
			if proto.Unmarshal(link.Outcome, &prior) == nil && prior.AttemptOrdinal != outcome.AttemptOrdinal {
				return recordAbandonedOutcome(tx, id, outcome.AttemptOrdinal, hex.EncodeToString(outcome.OutcomeDigest), raw)
			}
		}
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
	if link.Abandoned {
		return recordAbandonedOutcome(tx, id, outcome.AttemptOrdinal, hex.EncodeToString(outcome.OutcomeDigest), raw)
	}
	kind := "run.failed"
	if body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		kind = "run.completed"
	} else if body.Status == pb.OutcomeStatus_OUTCOME_STATUS_CANCELED {
		kind = "run.canceled"
	}
	// Every single output is done at the terminal, before it: the run carries its output.
	itemStatus := "completed"
	if body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		itemStatus = "incomplete"
	}
	output, problem := finishOutputs(tx, id, outcome.AttemptOrdinal, itemStatus)
	if problem != nil {
		return problem
	}
	if output == nil {
		output = []OutputItem{}
	}
	facts := map[string]any{"machine_execution": true, "status": strings.TrimPrefix(body.Status.String(), "OUTCOME_STATUS_"), "outputs": []any{}, "output": output}
	if body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
		facts["error"] = body.SafeMessage
	}
	payload, _ := json.Marshal(facts)
	// A run already announced canceled keeps that terminal; the machine's end is its `machine.outcome` entry.
	if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) SELECT ?,?,?,?,?
 WHERE NOT EXISTS(SELECT 1 FROM request_events WHERE request_id=? AND type='run.canceled')`, id, kind, outcome.AttemptOrdinal, string(payload), now(), id); err != nil {
		return exit.Internalf("cannot record machine terminal observation: %s", err)
	}
	// Runtime's measured memory sizes the next selection exactly as a local attempt's does.
	if problem := recordDeviceMemoryTx(tx, Terminal{RequestID: id, Attempt: int64(outcome.AttemptOrdinal),
		Status: strings.TrimPrefix(body.Status.String(), "OUTCOME_STATUS_"), Body: outcome.OutcomeCanonicalBytes}); problem != nil {
		return problem
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
	if link != nil && link.Abandoned {
		return exit.Named(exit.Conflict, "request.abandoned", "this run was abandoned locally; it cannot be resumed or submitted again")
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
	result, err := s.db.Exec(`UPDATE machine_executions SET pending_control=?,cancel_requested=CASE WHEN ? THEN 1 ELSE cancel_requested END WHERE request_id=? AND (length(pending_control)=0 OR pending_control=?) AND `+machineExecutionAdmissionOpen, raw, command.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL, id, raw)
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
	result, err := s.db.Exec(`UPDATE machine_executions SET pending_control=x'',cancel_requested=CASE WHEN ? THEN 0 ELSE cancel_requested END WHERE request_id=? AND pending_control=? AND `+machineExecutionAdmissionOpen, decoded.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL, id, command)
	if err != nil {
		return exit.Internalf("cannot acknowledge machine control: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		current, problem := s.MachineExecution(id)
		if problem == nil && current != nil && (len(current.PendingControl) == 0 || current.Abandoned) {
			return nil
		}
		return exit.New(exit.Conflict, "machine control acknowledgement changed its command")
	}
	return nil
}

// CancelMachineBeforeAcceptance records cancel intent in the transaction that reads acceptance.
// Unsent intent is canceled at once; a sent submission's cancel stays pending until the machine
// closes the key or returns a receipt. It reports a receipt recorded since the caller's read,
// whose cancel the caller sends as any accepted run's. A finished run is never reopened.
func (s *Store) CancelMachineBeforeAcceptance(id string) (bool, *exit.Error) {
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
		if len(link.Submission) > 0 && !link.SubmissionClosed {
			state, scope = "canceling", "machine_acceptance_unknown"
		}
		if err := projectCancellationTx(tx, id, state, scope); err != nil {
			return false, exit.Internalf("cannot project pending machine cancellation: %s", err)
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
	return appendEventTx(tx, id, StateEvent(state), 0, map[string]any{"scope": scope})
}

// CompleteMachineSubmissionClosure records the machine's durable promise that a
// key neither was nor can later be accepted. The original offer stays as history.
func (s *Store) CompleteMachineSubmissionClosure(id string, closure *pb.MachineSubmissionClosure) *exit.Error {
	if closure == nil || closure.Receipt != nil {
		return exit.New(exit.Conflict, "an accepted execution requires machine cancellation")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot record submission closure: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil {
		return exit.Internalf("cannot read closing submission: %s", err)
	}
	var submitted pb.MachineExecutionSubmit
	if link == nil || len(link.Receipt) > 0 || !link.CancelRequested || proto.Unmarshal(link.Submission, &submitted) != nil || submitted.Offer == nil ||
		closure.RequestId != id || closure.RequestId != submitted.Offer.RequestId || closure.SubmissionId != submitted.SubmissionId || closure.ExecutionWorkspaceId != submitted.ExpectedExecutionWorkspaceId {
		return exit.New(exit.Conflict, "submission closure differs from the pending cancellation")
	}
	if link.SubmissionClosed {
		return nil
	}
	if err := appendEventTx(tx, id, "machine.submission_closed", 0, map[string]any{"submission_id": closure.SubmissionId, "execution_workspace_id": closure.ExecutionWorkspaceId}); err != nil {
		return exit.Internalf("cannot retain closure evidence: %s", err)
	}
	if err := projectCancellationTx(tx, id, "canceled", "machine_submission_closed"); err != nil {
		return exit.Internalf("cannot settle closed submission: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit submission closure: %s", err)
	}
	return nil
}

// RefuseMachineSubmission ends unsent work or retains the machine's durable closure
// proof before ending a frozen submission. An arbitrary RPC failure is never proof.
func (s *Store) RefuseMachineSubmission(id, code, detail string, closure *pb.MachineSubmissionClosure) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot record machine submission refusal: %s", err)
	}
	defer tx.Rollback()
	link, err := scanMachineExecution(tx.QueryRow(`SELECT `+machineExecutionColumns+` FROM machine_executions WHERE request_id=?`, id))
	if err != nil || link == nil || len(link.Receipt) != 0 {
		return exit.New(exit.Conflict, "machine refusal contradicts accepted execution")
	}
	if len(link.Submission) > 0 {
		var submitted pb.MachineExecutionSubmit
		if closure == nil || closure.Receipt != nil || proto.Unmarshal(link.Submission, &submitted) != nil || submitted.Offer == nil || closure.RequestId != id || closure.RequestId != submitted.Offer.RequestId || closure.SubmissionId != submitted.SubmissionId || closure.ExecutionWorkspaceId != submitted.ExpectedExecutionWorkspaceId {
			return exit.New(exit.Conflict, "machine refusal has no matching submission closure")
		}
		if !link.SubmissionClosed {
			if err := appendEventTx(tx, id, "machine.submission_closed", 0, map[string]any{"submission_id": closure.SubmissionId, "execution_workspace_id": closure.ExecutionWorkspaceId}); err != nil {
				return exit.Internalf("cannot retain refusal closure evidence: %s", err)
			}
		}
	}
	if link.Abandoned {
		if err := tx.Commit(); err != nil {
			return exit.Internalf("cannot retain late closure evidence: %s", err)
		}
		return nil
	}
	if _, err := tx.Exec(`UPDATE requests SET state='refused',retain_work=0 WHERE id=?`, id); err != nil {
		return exit.Internalf("cannot retain machine refusal state: %s", err)
	}
	if err := appendEventTx(tx, id, "run.failed", 0, map[string]any{"error_type": code, "error": detail, "machine_accepted": false}); err != nil {
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
	_, err := s.db.Exec(`UPDATE machine_executions SET pending_control=x'' WHERE request_id=? AND pending_control=? AND `+machineExecutionAdmissionOpen, id, command)
	if err != nil {
		return exit.Internalf("cannot record rejected machine control: %s", err)
	}
	return nil
}
