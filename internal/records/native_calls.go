package records

import (
	"bytes"
	"database/sql"
	"errors"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// Fixed source/effect call admission shares the parent's bounded index namespace
// with ordinary package children. It is execution history, never a second job queue.
const nativeCallsDDL = `CREATE TABLE IF NOT EXISTS native_calls (
 id TEXT PRIMARY KEY, parent_request_id TEXT NOT NULL REFERENCES requests(id),
 call_index INTEGER NOT NULL CHECK(call_index>=0 AND call_index<32),
 kind TEXT NOT NULL CHECK(kind IN ('source','effect')), operation TEXT NOT NULL,
 intent_digest TEXT NOT NULL, request BLOB NOT NULL, frozen BLOB NOT NULL DEFAULT x'',
 state TEXT NOT NULL CHECK(state IN ('accepted','frozen','executing','succeeded','failed','canceled','stopped')),
 result BLOB NOT NULL DEFAULT x'', native_receipt BLOB NOT NULL DEFAULT x'',
 safe_code TEXT NOT NULL DEFAULT '', worker TEXT NOT NULL DEFAULT '',
 instance_id TEXT NOT NULL DEFAULT '', worker_boot_id TEXT NOT NULL DEFAULT '',
 parent_attempt INTEGER NOT NULL DEFAULT 0,
 cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
 UNIQUE(parent_request_id,call_index)
)`

type NativeCall struct {
	ID, ParentRequestID              string
	ParentAttempt                    int64
	CancelRequested                  bool
	CallIndex                        int64
	Kind, Operation, IntentDigest    string
	Request, Frozen                  []byte
	State                            string
	Result, NativeReceipt            []byte
	SafeCode                         string
	Worker, InstanceID, WorkerBootID string
}

const nativeCallColumns = `id,parent_request_id,call_index,kind,operation,intent_digest,request,frozen,state,result,native_receipt,safe_code,worker,instance_id,worker_boot_id,parent_attempt,cancel_requested`

func scanNativeCall(row interface{ Scan(...any) error }) (NativeCall, error) {
	var call NativeCall
	err := row.Scan(&call.ID, &call.ParentRequestID, &call.CallIndex, &call.Kind, &call.Operation, &call.IntentDigest, &call.Request, &call.Frozen, &call.State, &call.Result, &call.NativeReceipt, &call.SafeCode, &call.Worker, &call.InstanceID, &call.WorkerBootID, &call.ParentAttempt, &call.CancelRequested)
	return call, err
}
func (s *Store) NativeCall(parent string, index int64) (*NativeCall, *exit.Error) {
	call, err := scanNativeCall(s.db.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE parent_request_id=? AND call_index=?`, parent, index))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read native call: %s", err)
	}
	return &call, nil
}
func (s *Store) AcceptNativeCall(call NativeCall, parentAttempt int64, parentSpec, parentSession string) (NativeCall, bool, *exit.Error) {
	if call.ID == "" || len(call.ID) > 128 || call.ParentRequestID == "" || call.CallIndex < 0 || call.CallIndex >= 32 || parentAttempt <= 0 || len(call.Request) > 48*1024 || (call.Kind != "source" && call.Kind != "effect") || call.Operation == "" || len(call.Operation) > 128 {
		return NativeCall{}, false, exit.New(exit.Validation, "native call exceeds fixed identity bounds")
	}
	if _, err := canonical.Raw(call.IntentDigest); err != nil {
		return NativeCall{}, false, exit.New(exit.Validation, "native call digest is malformed")
	}
	raw, err := canonical.NormalizeJCS(call.Request)
	if err != nil || !bytes.Equal(raw, call.Request) || len(raw) == 0 || raw[0] != '{' {
		return NativeCall{}, false, exit.New(exit.Validation, "native call request is not canonical object")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return NativeCall{}, false, exit.Internalf("cannot begin native call: %s", err)
	}
	defer tx.Rollback()
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, call.ParentRequestID))
	if err != nil {
		return NativeCall{}, false, exit.Named(exit.Conflict, "native.parent_absent", "native call has no parent")
	}
	var owned bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND attempt=? AND session_id=? AND invocation_digest=? AND state IN ('offered','accepted','recovered_open'))`, parent.ID, parentAttempt, parentSession, parentSpec).Scan(&owned); err != nil {
		return NativeCall{}, false, exit.Internalf("cannot read native caller authority: %s", err)
	}
	if !owned || !parent.IsJob() || !parent.RetainWork || parent.State != "dispatching" || parent.Ordinal != parentAttempt {
		return NativeCall{}, false, exit.Named(exit.Conflict, "native.parent_stopped", "native call has no current private parent attempt")
	}
	var occupied bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests WHERE parent_request_id=? AND parent_call_index=?)`, parent.ID, call.CallIndex).Scan(&occupied); err != nil {
		return NativeCall{}, false, exit.Internalf("cannot inspect parent call index: %s", err)
	}
	if occupied {
		return NativeCall{}, false, exit.Named(exit.Conflict, "child.intent_changed", "parent index already names a package call")
	}
	existing, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE parent_request_id=? AND call_index=?`, parent.ID, call.CallIndex))
	if err == nil {
		if existing.ID != call.ID || existing.IntentDigest != call.IntentDigest || existing.Kind != call.Kind || existing.Operation != call.Operation || !bytes.Equal(existing.Request, call.Request) {
			return NativeCall{}, false, exit.Named(exit.Conflict, "child.intent_changed", "parent index already names another native call")
		}
		if existing.Kind == "source" && parentAttempt > existing.ParentAttempt && (existing.State == "failed" || existing.State == "stopped") {
			state := "accepted"
			if len(existing.Frozen) > 0 {
				state = "frozen"
			}
			if _, err := tx.Exec(`UPDATE native_calls SET state=?,parent_attempt=?,safe_code='' WHERE id=?`, state, parentAttempt, existing.ID); err != nil {
				return NativeCall{}, false, exit.Internalf("cannot reattach stopped native call: %s", err)
			}
			if err := tx.Commit(); err != nil {
				return NativeCall{}, false, exit.Internalf("cannot commit native reattachment: %s", err)
			}
			existing.State, existing.ParentAttempt, existing.SafeCode = state, parentAttempt, ""
		}
		return existing, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return NativeCall{}, false, exit.Internalf("cannot read native replay: %s", err)
	}
	_, err = tx.Exec(`INSERT INTO native_calls(id,parent_request_id,call_index,kind,operation,intent_digest,request,state,worker,parent_attempt) VALUES(?,?,?,?,?,?,?,'accepted',?,?)`, call.ID, parent.ID, call.CallIndex, call.Kind, call.Operation, call.IntentDigest, call.Request, parent.Worker, parentAttempt)
	if err != nil {
		return NativeCall{}, false, exit.Internalf("cannot record native call: %s", err)
	}
	if err = appendEventTx(tx, parent.ID, "native.accepted", parentAttempt, map[string]any{"call_index": call.CallIndex, "service_id": call.ID, "kind": call.Kind, "operation": call.Operation, "intent_digest": call.IntentDigest}); err != nil {
		return NativeCall{}, false, exit.Internalf("cannot journal native call: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return NativeCall{}, false, exit.Internalf("cannot commit native call: %s", err)
	}
	call.State = "accepted"
	call.Worker = parent.Worker
	call.ParentAttempt = parentAttempt
	return call, true, nil
}
func (s *Store) FreezeNativeCall(id string, frozen []byte) *exit.Error {
	raw, err := canonical.NormalizeJCS(frozen)
	if err != nil || !bytes.Equal(raw, frozen) || len(raw) > 1<<20 {
		return exit.New(exit.Validation, "native frozen intent is not bounded canonical data")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot freeze native call: %s", err)
	}
	defer tx.Rollback()
	call, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE id=?`, id))
	if err != nil {
		return exit.Named(exit.Conflict, "native.call_absent", "native call is absent")
	}
	if len(call.Frozen) > 0 {
		if !bytes.Equal(call.Frozen, frozen) {
			return exit.Named(exit.Conflict, "native.pin_changed", "native call already froze different intent")
		}
		return nil
	}
	if call.State != "accepted" {
		return exit.Named(exit.Conflict, "native.call_closed", "native call cannot freeze after terminal")
	}
	if _, err = tx.Exec(`UPDATE native_calls SET frozen=?,state='frozen' WHERE id=?`, frozen, id); err != nil {
		return exit.Internalf("cannot freeze native call: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return exit.Internalf("cannot commit native pin: %s", err)
	}
	return nil
}
func (s *Store) CompleteNativeCall(id string, result, receipt []byte) *exit.Error {
	return s.CompleteNativeCallAt(id, result, receipt, "", "")
}
func (s *Store) CompleteNativeCallAt(id string, result, receipt []byte, instance, boot string) *exit.Error {
	if receipt == nil {
		receipt = []byte{}
	}
	raw, err := canonical.NormalizeJCS(result)
	if err != nil || !bytes.Equal(raw, result) || len(raw) > 48*1024 || len(receipt) > 1<<20 {
		return exit.New(exit.Validation, "native completion exceeds canonical result bounds")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot complete native call: %s", err)
	}
	defer tx.Rollback()
	call, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE id=?`, id))
	if err != nil {
		return exit.Named(exit.Conflict, "native.call_absent", "native call is absent")
	}
	if call.State == "succeeded" {
		if !bytes.Equal(call.Result, result) || !bytes.Equal(call.NativeReceipt, receipt) {
			return exit.Named(exit.Conflict, "native.result_changed", "native completion contradicts prior result")
		}
		return nil
	}
	if call.State != "accepted" && call.State != "frozen" && call.State != "executing" {
		return exit.Named(exit.Conflict, "native.call_closed", "native call stopped before completion")
	}
	if _, err = tx.Exec(`UPDATE native_calls SET state='succeeded',result=?,native_receipt=?,instance_id=?,worker_boot_id=? WHERE id=?`, result, receipt, instance, boot, id); err != nil {
		return exit.Internalf("cannot record native completion: %s", err)
	}
	if err = appendEventTx(tx, call.ParentRequestID, "native.completed", 0, map[string]any{"call_index": call.CallIndex, "service_id": id}); err != nil {
		return exit.Internalf("cannot journal native completion: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return exit.Internalf("cannot commit native completion: %s", err)
	}
	return nil
}
func (s *Store) StopNativeCall(id, state, code string) *exit.Error {
	if (state != "failed" && state != "canceled" && state != "stopped") || len(code) > 128 {
		return exit.New(exit.Validation, "native stop is not a bounded terminal")
	}
	_, err := s.db.Exec(`UPDATE native_calls SET state=?,safe_code=? WHERE id=? AND state IN ('accepted','frozen','executing')`, state, code, id)
	if err != nil {
		return exit.Internalf("cannot stop native call: %s", err)
	}
	return nil
}

// StartNativeCall is the durable before-send marker. Replays never erase it.
func (s *Store) StartNativeCall(id string) *exit.Error {
	result, err := s.db.Exec(`UPDATE native_calls SET state='executing' WHERE id=? AND state IN ('frozen','executing')`, id)
	if err != nil {
		return exit.Internalf("cannot start native call: %s", err)
	}
	if n, _ := result.RowsAffected(); n == 0 {
		return exit.Named(exit.Conflict, "native.call_closed", "native call must freeze before execution")
	}
	return nil
}
func (s *Store) OwedNativeCalls(kind string) ([]NativeCall, *exit.Error) {
	if kind != "source" && kind != "effect" {
		return nil, exit.New(exit.Validation, "native call kind is not supported")
	}
	rows, err := s.db.Query(`SELECT `+nativeCallColumns+` FROM native_calls WHERE kind=? AND state IN ('accepted','frozen','executing') ORDER BY parent_request_id,call_index`, kind)
	if err != nil {
		return nil, exit.Internalf("cannot read owed native calls: %s", err)
	}
	defer rows.Close()
	var calls []NativeCall
	for rows.Next() {
		call, err := scanNativeCall(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode owed native call: %s", err)
		}
		calls = append(calls, call)
	}
	if err = rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish owed native calls: %s", err)
	}
	return calls, nil
}

// RequestNativeEffectCancel records cancellation of one awaited effect while
// retaining its executing marker for authoritative late-commit reconciliation.
func (s *Store) RequestNativeEffectCancel(parentID string, index, parentAttempt int64, parentSpec, parentSession, intent string) (*NativeCall, *exit.Error) {
	if parentID == "" || index < 0 || index >= 32 || parentAttempt <= 0 {
		return nil, exit.New(exit.Validation, "effect cancellation has invalid parent identity")
	}
	for _, value := range []string{parentSpec, intent} {
		raw, err := canonical.Raw(value)
		if err != nil {
			return nil, exit.New(exit.Validation, "effect cancellation has malformed identity")
		}
		spelled, _ := canonical.Spell(raw)
		if spelled != value {
			return nil, exit.New(exit.Validation, "effect cancellation identity is not canonical")
		}
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin effect cancellation: %s", err)
	}
	defer tx.Rollback()
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, parentID))
	if err != nil {
		return nil, exit.Named(exit.Conflict, "native.parent_absent", "effect cancellation has no parent")
	}
	var owned bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND attempt=? AND session_id=? AND invocation_digest=? AND state IN ('offered','accepted','recovered_open'))`, parentID, parentAttempt, parentSession, parentSpec).Scan(&owned); err != nil {
		return nil, exit.Internalf("cannot inspect effect cancellation authority: %s", err)
	}
	if !owned || !parent.IsJob() || !parent.RetainWork || parent.Ordinal != parentAttempt {
		return nil, exit.Named(exit.Conflict, "native.parent_stopped", "effect cancellation does not belong to the current parent attempt")
	}
	row, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE parent_request_id=? AND call_index=?`, parentID, index))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read canceled effect: %s", err)
	}
	if row.Kind != "effect" {
		return nil, nil
	}
	if row.IntentDigest != intent {
		return nil, exit.Named(exit.Conflict, "child.intent_changed", "effect cancellation names another accepted intent")
	}
	// Parent suspension interrupts Python awaits but retains logical effects.
	if parent.State == "pausing" || parent.State == "paused" || parent.State == "blocked" {
		return &row, nil
	}
	if row.State == "succeeded" || row.State == "failed" || row.State == "canceled" {
		return &row, nil
	}
	state, code := row.State, "publication.cancel_requested"
	if state != "executing" {
		state, code = "canceled", "publication.call_canceled"
	}
	if _, err := tx.Exec(`UPDATE native_calls SET cancel_requested=1,state=?,safe_code=? WHERE id=?`, state, code, row.ID); err != nil {
		return nil, exit.Internalf("cannot record effect cancellation: %s", err)
	}
	if err := appendEventTx(tx, parentID, "native.effect_cancel_requested", parentAttempt, map[string]any{"call_index": index, "service_id": row.ID, "intent_digest": intent}); err != nil {
		return nil, exit.Internalf("cannot journal effect cancellation: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit effect cancellation: %s", err)
	}
	row.CancelRequested, row.State, row.SafeCode = true, state, code
	return &row, nil
}

// StartNativeEffectWrite is the last owner-side authorization before each grant,
// PUT, finalize or CAS. Cancellation and this mutation reservation serialize here.
func (s *Store) StartNativeEffectWrite(id string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot authorize effect write: %s", err)
	}
	defer tx.Rollback()
	row, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE id=?`, id))
	if err != nil || row.Kind != "effect" {
		return exit.Named(exit.Conflict, "native.call_absent", "publication has no accepted effect")
	}
	if row.CancelRequested {
		return exit.Named(exit.Canceled, "publication.call_canceled", "the awaited publication call was canceled")
	}
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, row.ParentRequestID))
	if err != nil {
		return exit.Named(exit.Canceled, "publication.parent_stopped", "publication parent is absent")
	}
	if parent.State != "dispatching" {
		if parent.State == "canceling" || parent.State == "canceled" || parent.State == "releasing" {
			return exit.Named(exit.Canceled, "publication.parent_stopped", "stopped parent cannot issue publication writes")
		}
		return exit.Named(exit.Unavailable, "publication.parent_paused", "publication waits for its parent to resume")
	}
	if row.State != "frozen" && row.State != "executing" {
		return exit.Named(exit.Conflict, "native.call_closed", "publication must freeze before execution")
	}
	if _, err := tx.Exec(`UPDATE native_calls SET state='executing' WHERE id=?`, id); err != nil {
		return exit.Internalf("cannot reserve effect write: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit effect write reservation: %s", err)
	}
	return nil
}
