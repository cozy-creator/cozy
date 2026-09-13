package records

import (
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

type ExecutionCollection struct {
	RequestID            string `json:"request_id"`
	ExecutionGrantDigest string `json:"execution_grant_digest"`
	TerminalEventID      int64  `json:"terminal_event_id"`
	Attempt              int64  `json:"attempt"`
	Collected            bool   `json:"collected"`
	CollectionEventID    int64  `json:"collection_event_id,omitempty"`
}

type executionQuery interface{ QueryRow(string, ...any) *sql.Row }

func executionCollection(q executionQuery, id, grant string) (*ExecutionCollection, *exit.Error) {
	var parent, scope, state string
	var attempt int64
	if err := q.QueryRow(`SELECT parent_request_id,execution_grant_digest,state,ordinal FROM requests WHERE id=? AND kind='job'`, id).Scan(&parent, &scope, &state, &attempt); err != nil {
		if err == sql.ErrNoRows {
			return nil, exit.New(exit.NotFound, "execution root is absent")
		}
		return nil, exit.Internalf("cannot read execution collection root: %s", err)
	}
	if parent != "" || scope == "" || scope != grant {
		return nil, exit.New(exit.Credential, "collection is outside the current execution root grant")
	}
	if state != "succeeded" && state != "failed" && state != "canceled" && state != "blocked" {
		return nil, nil
	}
	receipt := &ExecutionCollection{RequestID: id, ExecutionGrantDigest: grant, Attempt: attempt}
	eventType := map[string]string{"succeeded": "request.completed", "failed": "request.failed", "canceled": "request.canceled", "blocked": "request.blocked"}[state]
	if err := q.QueryRow(`SELECT seq FROM request_events WHERE request_id=? AND attempt=? AND type=? ORDER BY seq DESC LIMIT 1`, id, attempt, eventType).Scan(&receipt.TerminalEventID); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, exit.Internalf("cannot read execution terminal identity: %s", err)
	}
	err := q.QueryRow(`SELECT seq FROM request_events WHERE request_id=? AND type='request.execution_collected' AND attempt=? AND json_extract(payload,'$.terminal_event_id')=? AND json_extract(payload,'$.execution_grant_digest')=? ORDER BY seq DESC LIMIT 1`, id, attempt, receipt.TerminalEventID, grant).Scan(&receipt.CollectionEventID)
	if err != nil && err != sql.ErrNoRows {
		return nil, exit.Internalf("cannot read execution collection receipt: %s", err)
	}
	receipt.Collected = err == nil
	return receipt, nil
}

func (s *Store) ExecutionCollection(id, grant string) (*ExecutionCollection, *exit.Error) {
	return executionCollection(s.db, id, grant)
}

// CollectExecution records an explicit client acknowledgement of this exact
// outcome. It releases no partial work or artifact ownership and changes no
// execution status. The existing event log is the collection receipt authority.
func (s *Store) CollectExecution(id, grant string, terminalEvent, attempt int64) (*ExecutionCollection, *exit.Error) {
	if terminalEvent <= 0 || attempt < 0 {
		return nil, exit.New(exit.Validation, "collection requires an exact terminal event and attempt")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin execution collection: %s", err)
	}
	defer tx.Rollback()
	receipt, problem := executionCollection(tx, id, grant)
	if problem != nil {
		return nil, problem
	}
	if receipt == nil || receipt.TerminalEventID != terminalEvent || receipt.Attempt != attempt {
		return nil, exit.New(exit.Conflict, "collection does not name the current execution outcome")
	}
	if !receipt.Collected {
		payload, _ := json.Marshal(map[string]any{"terminal_event_id": terminalEvent, "execution_grant_digest": grant})
		result, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'request.execution_collected',?,?,?)`, id, attempt, string(payload), now())
		if err != nil {
			return nil, exit.Internalf("cannot record execution collection: %s", err)
		}
		receipt.CollectionEventID, err = result.LastInsertId()
		if err != nil {
			return nil, exit.Internalf("cannot identify execution collection: %s", err)
		}
		receipt.Collected = true
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit execution collection: %s", err)
	}
	return receipt, nil
}
