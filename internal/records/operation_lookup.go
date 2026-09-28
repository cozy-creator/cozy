package records

import (
	"database/sql"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// This is an RPC obligation, not a computation cache. Runtime's workspace owns hits.
const operationLookupsDDL = `CREATE TABLE IF NOT EXISTS request_operation_lookups (
 request_id TEXT PRIMARY KEY REFERENCES requests(id),
 computation_digest TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('pending','miss','hit'))
)`

type OperationLookup struct{ Key, State string }

func (s *Store) OperationLookup(id string) (*OperationLookup, *exit.Error) {
	var row OperationLookup
	err := s.db.QueryRow(`SELECT computation_digest,state FROM request_operation_lookups WHERE request_id=?`, id).Scan(&row.Key, &row.State)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read pending operation lookup: %s", err)
	}
	return &row, nil
}

func (s *Store) BeginOperationLookup(id, key string) *exit.Error {
	if _, err := canonical.Raw(key); err != nil {
		return exit.New(exit.Validation, "operation lookup key is malformed")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin operation lookup: %s", err)
	}
	defer tx.Rollback()
	var allowed bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests WHERE id=? AND ordinal=0 AND child_reusable=1 AND state IN ('submitted','queued'))`, id).Scan(&allowed); err != nil {
		return exit.Internalf("cannot inspect operation lookup admission: %s", err)
	}
	if !allowed {
		return exit.Named(exit.Conflict, "operation.consumer_stopped", "stopped or offered request cannot start a cache lookup")
	}
	var qualified string
	if err := tx.QueryRow(`SELECT computation_digest FROM request_operation_contexts WHERE request_id=?`, id).Scan(&qualified); err != nil || qualified != key {
		return exit.Named(exit.Conflict, "operation.context_absent", "lookup requires the callee's bound numerical context")
	}
	if _, err := tx.Exec(`INSERT INTO request_operation_lookups(request_id,computation_digest,state) VALUES(?,?,'pending') ON CONFLICT(request_id) DO NOTHING`, id, key); err != nil {
		return exit.Internalf("cannot persist operation lookup intent: %s", err)
	}
	var held string
	if err := tx.QueryRow(`SELECT computation_digest FROM request_operation_lookups WHERE request_id=?`, id).Scan(&held); err != nil || held != key {
		return exit.Named(exit.Conflict, "operation.key_changed", "the operation lookup already names different computation")
	}
	if _, err := tx.Exec(`UPDATE request_operation_lookups SET state='pending' WHERE request_id=?`, id); err != nil {
		return exit.Internalf("cannot resume operation lookup: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit operation lookup: %s", err)
	}
	return nil
}

func (s *Store) CompleteOperationMiss(id, key string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_operation_lookups SET state='miss' WHERE request_id=? AND computation_digest=?`, id, key)
	if err != nil {
		return exit.Internalf("cannot record operation lookup miss: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.Named(exit.Conflict, "operation.lookup_absent", "the operation lookup has no matching pending intent")
	}
	return nil
}
