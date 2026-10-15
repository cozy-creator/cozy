package records

const maxActiveChildCalls = 32
const maxChildCallIndex = (1 << 32) - 1
const pendingChildStates = `'submitted','queued','dispatching','requeue_pending','finalizing','canceling'`
const pendingNativeStates = `'accepted','frozen','executing'`
const activeChildRequestIndex = `CREATE INDEX IF NOT EXISTS requests_active_children ON requests(parent_request_id) WHERE state IN (` + pendingChildStates + `)`
const activeNativeCallIndex = `CREATE INDEX IF NOT EXISTS native_active_calls ON native_calls(parent_request_id) WHERE state IN (` + pendingNativeStates + `)`
