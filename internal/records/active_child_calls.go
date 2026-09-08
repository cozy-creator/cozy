package records

import (
	"database/sql"

	"github.com/cozy-creator/cozy/internal/exit"
)

const maxActiveChildCalls = 32
const maxChildCallIndex = (1 << 32) - 1
const pendingChildStates = `'submitted','queued','dispatching','requeue_pending','finalizing','pausing','canceling'`
const pendingNativeStates = `'accepted','frozen','executing'`
const activeChildRequestIndex = `CREATE INDEX IF NOT EXISTS requests_active_children ON requests(parent_request_id) WHERE state IN (` + pendingChildStates + `)`
const activeNativeCallIndex = `CREATE INDEX IF NOT EXISTS native_active_calls ON native_calls(parent_request_id) WHERE state IN (` + pendingNativeStates + `)`

func requireActiveChildSlot(tx *sql.Tx, parent string) *exit.Error {
	var count int
	err := tx.QueryRow(`SELECT
      (SELECT COUNT(*) FROM requests WHERE parent_request_id=? AND state IN (`+pendingChildStates+`)) +
      (SELECT COUNT(*) FROM native_calls WHERE parent_request_id=? AND state IN (`+pendingNativeStates+`))`, parent, parent).Scan(&count)
	if err != nil {
		return exit.Internalf("cannot count active child calls: %s", err)
	}
	if count >= maxActiveChildCalls {
		return exit.Named(exit.Conflict, "child.active_limit", "a parent may have at most %d active calls", maxActiveChildCalls)
	}
	return nil
}

// ChildAt reads one accepted index without loading the parent's completed history.
func (s *Store) ChildAt(parent string, index int64) (*Request, *exit.Error) {
	row, err := scanRequest(s.db.QueryRow(`SELECT `+requestCols+` FROM requests WHERE parent_request_id=? AND parent_call_index=?`, parent, index))
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read accepted child call: %s", err)
	}
	return &row, nil
}
