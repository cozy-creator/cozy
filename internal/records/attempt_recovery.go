package records

import (
	"database/sql"
	"encoding/json"
	"errors"
	"github.com/cozy-creator/cozy/internal/archive"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// Recover rebinds an open obligation to its current peer. A sealed outcome keeps
// its execution identity: another worker replaying it did not execute it.
func (s *Store) Recover(requestID string, attempt int64, sessionID string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin attempt recovery: %s", err)
	}
	defer tx.Rollback()
	var state, digest, assignedSpec, previous string
	var body []byte
	err = tx.QueryRow(`SELECT state,terminal_digest,invocation_digest,session_id,COALESCE(terminal_body,x'') FROM attempts WHERE request_id=? AND attempt=?`, requestID, attempt).
		Scan(&state, &digest, &assignedSpec, &previous, &body)
	if errors.Is(err, sql.ErrNoRows) {
		return exit.New(exit.NotFound, "the worker reported a recovered attempt %s#%d this orchestrator never assigned", requestID, attempt)
	}
	if err != nil {
		return exit.Internalf("cannot read recovered attempt: %s", err)
	}
	switch state {
	case "preparing", "offered", "accepted", "recovered_open", "terminal", "closed":
	default:
		return exit.New(exit.Conflict, "attempt %s#%d cannot recover from %s", requestID, attempt, state)
	}
	nextState, nextSession := "recovered_open", sessionID
	if digest != "" || state == "terminal" || state == "closed" {
		nextState, nextSession = "terminal", previous
		if state == "closed" {
			nextState = "closed"
		}
		// Older Recover versions overwrote sealed sessions. Only an immutable,
		// fully bound observed boot can repair that history; absent evidence stays
		// absent and is never filled from whichever peer happens to replay it.
		var projection struct {
			Observation struct {
				Environment struct {
					Boot string `json:"worker_boot_id"`
				} `json:"environment"`
			} `json:"observation"`
		}
		if json.Unmarshal(body, &projection) == nil && projection.Observation.Environment.Boot != "" {
			doc, read := archive.Read(body, archive.TerminalBody)
			observedDigest, _ := canonical.Spell(canonical.Digest(body))
			if read != nil || observedDigest != digest || doc.Str("request_id") != requestID || doc.Int("attempt_ordinal") != attempt || doc.Str("invocation_spec_digest") != assignedSpec {
				return exit.Named(exit.Conflict, "attempt.execution_identity_unverified", "retained terminal cannot establish the original execution identity")
			}
			nextSession = projection.Observation.Environment.Boot
			if nextSession != previous {
				if err := appendEventTx(tx, requestID, "attempt.execution_identity_restored", attempt, map[string]any{"previous_session": previous, "execution_session": nextSession, "terminal_digest": digest}); err != nil {
					return exit.Internalf("cannot journal execution identity repair: %s", err)
				}
			}
		}
	}
	if _, err := tx.Exec(`UPDATE attempts SET state=?,session_id=? WHERE request_id=? AND attempt=?`, nextState, nextSession, requestID, attempt); err != nil {
		return exit.Internalf("cannot record recovered attempt: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit recovered attempt: %s", err)
	}
	return nil
}
