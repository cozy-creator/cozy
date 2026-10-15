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
