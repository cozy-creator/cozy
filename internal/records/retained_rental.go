package records

import (
	"database/sql"

	"github.com/cozy-creator/cozy/internal/exit"
)

func retainedRentalIDs(tx *sql.Tx, rental string) ([]string, *exit.Error) {
	rows, err := tx.Query(`SELECT id FROM requests WHERE worker=? AND retain_work=1 AND state IN (`+activeRequestStates+`) ORDER BY id`, rental)
	if err != nil {
		return nil, exit.Internalf("cannot read retained rental work: %s", err)
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot read retained rental owner: %s", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish retained rental census: %s", err)
	}
	return ids, nil
}

// RequestRetainedRentalAbandonment precedes an explicitly requested provider
// release. It closes transaction admission before the machine's bytes can vanish.
// Idle cleanup never calls this: it must continue respecting retained ownership.
func (s *Store) RequestRetainedRentalAbandonment(rental, actor string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin retained rental abandonment: %s", err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE rentals SET state='release_requested' WHERE id=? AND state NOT IN ('failed','released')`, rental); err != nil {
		return exit.Internalf("cannot fence released rental admission: %s", err)
	}
	ids, problem := retainedRentalIDs(tx, rental)
	if problem != nil {
		return problem
	}
	for _, id := range ids {
		changed, err := tx.Exec(`UPDATE requests SET state='canceling',control_revision=control_revision+1 WHERE id=? AND state NOT IN ('canceling','releasing')`, id)
		if err != nil {
			return exit.Internalf("cannot abandon retained rental request: %s", err)
		}
		if n, _ := changed.RowsAffected(); n == 0 {
			continue
		}
		if err := appendEventTx(tx, id, "request.cancel_requested", 0, map[string]any{"status": "canceling", "actor": actor, "rental": rental}); err != nil {
			return exit.Internalf("cannot journal retained rental abandonment: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit retained rental abandonment: %s", err)
	}
	return nil
}

// CompleteRetainedRentalAbandonment is called only after provider absence has
// been verified. It records storage loss as such, preserving every real prior
// outcome; it never fabricates a native artifact-finalization acknowledgement.
func (s *Store) CompleteRetainedRentalAbandonment(rental, actor string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin retained rental settlement: %s", err)
	}
	defer tx.Rollback()
	ids, problem := retainedRentalIDs(tx, rental)
	if problem != nil {
		return problem
	}
	for _, id := range ids {
		var state string
		if err := tx.QueryRow(`SELECT state FROM requests WHERE id=?`, id).Scan(&state); err != nil {
			return exit.Internalf("cannot inspect retained rental cancellation: %s", err)
		}
		if state != "canceling" && state != "releasing" {
			return exit.Named(exit.Conflict, "request.rental_abandonment_missing", "retained request %s has no explicit rental abandonment intent", id)
		}
		if _, err := tx.Exec(`UPDATE attempts SET state='closed',closed_at=CASE WHEN closed_at='' THEN ? ELSE closed_at END,
			terminal_status=CASE WHEN terminal_status='' THEN 'ABANDONED' ELSE terminal_status END,
			terminal_cause=CASE WHEN terminal_cause='' THEN 'EXECUTION_CONTEXT_LOST' ELSE terminal_cause END,
			safe_message=CASE WHEN safe_message='' THEN 'the owner explicitly released the retained rental' ELSE safe_message END
			WHERE request_id=? AND state IN (`+openAttemptStates+`)`, now(), id); err != nil {
			return exit.Internalf("cannot close released rental attempts: %s", err)
		}
		if _, err := tx.Exec(`UPDATE requests SET state='canceled' WHERE id=?`, id); err != nil {
			return exit.Internalf("cannot settle released rental request: %s", err)
		}
		if _, err := tx.Exec(`UPDATE request_model_transfers SET state='canceled',updated_at=? WHERE request_id=? AND state!='completed'`, now(), id); err != nil {
			return exit.Internalf("cannot settle released model work: %s", err)
		}
		if err := appendEventTx(tx, id, "request.canceled", 0, map[string]any{"status": "CANCELED", "actor": actor, "rental": rental, "error_type": "rental.released", "error": "the retained rental was explicitly released; its local intermediate bytes were discarded"}); err != nil {
			return exit.Internalf("cannot journal retained rental settlement: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit retained rental settlement: %s", err)
	}
	return nil
}
