package records

// Fixed source/effect call admission shares the parent's bounded index namespace
// with ordinary package children. It is execution history, never a second job queue.
const nativeCallsDDL = `CREATE TABLE IF NOT EXISTS native_calls (
 id TEXT PRIMARY KEY, parent_request_id TEXT NOT NULL REFERENCES requests(id),
 call_index INTEGER NOT NULL CHECK(call_index>=0 AND call_index<4294967296),
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
