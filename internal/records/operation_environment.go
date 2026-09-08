package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// A numerical context is bound once after selecting the callee worker. It is
// distinct from the captured implementation and survives pending lookup replay.
const operationContextsDDL = `CREATE TABLE IF NOT EXISTS request_operation_contexts (
 request_id TEXT PRIMARY KEY REFERENCES requests(id),
 numerical_environment_digest TEXT NOT NULL,
 computation_digest TEXT NOT NULL
)`

type OperationContext struct{ NumericalEnvironment, Key string }

func QualifiedOperationKey(request Request, numerical string) (string, *exit.Error) {
	if _, err := canonical.Raw(numerical); err != nil {
		return "", exit.New(exit.Validation, "operation numerical identity is malformed")
	}
	computation, problem := OperationKey(request)
	if problem != nil {
		return "", problem
	}
	raw, _ := json.Marshal(map[string]string{"computation_digest": computation, "numerical_environment_digest": numerical})
	raw, _ = canonical.NormalizeJCS(raw)
	key, _ := canonical.Spell(canonical.Digest(raw))
	return key, nil
}

func scanOperationContext(row *sql.Row) (*OperationContext, *exit.Error) {
	var context OperationContext
	err := row.Scan(&context.NumericalEnvironment, &context.Key)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read operation numerical context: %s", err)
	}
	return &context, nil
}

func (s *Store) OperationContext(id string) (*OperationContext, *exit.Error) {
	return scanOperationContext(s.db.QueryRow(`SELECT numerical_environment_digest,computation_digest FROM request_operation_contexts WHERE request_id=?`, id))
}

func (s *Store) BindOperationContext(id, numerical string) (*OperationContext, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin operation numerical binding: %s", err)
	}
	defer tx.Rollback()
	request, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, id))
	if err != nil {
		return nil, exit.Internalf("cannot read operation consumer: %s", err)
	}
	key, problem := QualifiedOperationKey(request, numerical)
	if problem != nil {
		return nil, problem
	}
	held, problem := scanOperationContext(tx.QueryRow(`SELECT numerical_environment_digest,computation_digest FROM request_operation_contexts WHERE request_id=?`, id))
	if problem != nil {
		return nil, problem
	}
	if held != nil {
		if held.NumericalEnvironment != numerical || held.Key != key {
			return nil, exit.Named(exit.Conflict, "operation.numerical_environment_changed", "the selected callee environment changed after its operation context was bound; start a new script invocation")
		}
		return held, nil
	}
	if !request.ChildReusable || request.ParentRequestID == "" || (request.State != "submitted" && request.State != "queued") {
		return nil, exit.Named(exit.Conflict, "operation.consumer_stopped", "only an unoffered memoized call can bind its numerical context")
	}
	var prior string
	err = tx.QueryRow(`SELECT computation_digest FROM request_operation_lookups WHERE request_id=?`, id).Scan(&prior)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, exit.Internalf("cannot inspect pending operation identity: %s", err)
	}
	if err == nil && prior != key {
		return nil, exit.Named(exit.Conflict, "operation.key_changed", "an existing lookup is pinned to a different computation; start a new script invocation")
	}
	if _, err = tx.Exec(`INSERT INTO request_operation_contexts(request_id,numerical_environment_digest,computation_digest) VALUES(?,?,?)`, id, numerical, key); err != nil {
		return nil, exit.Internalf("cannot bind operation numerical context: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot persist operation numerical context: %s", err)
	}
	return &OperationContext{NumericalEnvironment: numerical, Key: key}, nil
}
