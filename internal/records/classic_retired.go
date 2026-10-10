package records

import (
	"fmt"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ClassicRetiredCode names work accepted for the retired classic worker session: the local
// `cozy-runtime serve` this daemon spawned, or a rental's classic WorkerControl dispatch.
// Every request now runs as a machine execution.
const ClassicRetiredCode = "request.classic_retired"

// ClassicRetiredOutcome completes a weights finalization whose classic holder is gone; it
// carries no receipt.
const ClassicRetiredOutcome = "CLASSIC_RETIRED"

// RetireClassicWork ends, once at daemon start, every unfinished request that has no
// machine execution and is not the daemon's own model pass-through: it ran or would run
// on a classic worker session. Every open attempt closes, as ABANDONED unless it recorded a
// terminal, and nothing is revived. A request being canceled ends canceled; every other
// one fails with ClassicRetiredCode. Custody held for any classic run is forgotten.
func (s *Store) RetireClassicWork() ([]string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin classic retirement: %s", err)
	}
	defer tx.Rollback()
	rows, err := tx.Query(`SELECT id,state,ordinal FROM requests r WHERE state IN (` + activeRequestStates + `)
		AND NOT EXISTS(SELECT 1 FROM machine_executions m WHERE m.request_id=r.id)
		AND NOT (r.package='cozy/platform' AND r.entrypoint='model-pass-through')
		ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot read classic work: %s", err)
	}
	type retired struct {
		id, state string
		ordinal   int64
	}
	var found []retired
	for rows.Next() {
		var r retired
		if err := rows.Scan(&r.id, &r.state, &r.ordinal); err != nil {
			rows.Close()
			return nil, exit.Internalf("cannot read classic request: %s", err)
		}
		found = append(found, r)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, exit.Internalf("cannot finish classic census: %s", err)
	}
	const message = "this work was accepted for the retired classic worker; machines now run work as machine executions"
	// Every attempt row is a classic worker session's, and none is acknowledged again: each
	// still open closes here, settled request or not, and a recorded terminal keeps its outcome.
	if _, err := tx.Exec(`UPDATE attempts SET state='closed',
		closed_at=CASE WHEN closed_at='' THEN ? ELSE closed_at END,
		terminal_status=CASE WHEN terminal_status='' THEN 'ABANDONED' ELSE terminal_status END,
		terminal_cause=CASE WHEN terminal_cause='' THEN 'EXECUTION_CONTEXT_LOST' ELSE terminal_cause END,
		safe_message=CASE WHEN safe_message='' THEN ? ELSE safe_message END
		WHERE state IN (`+openAttemptStates+`)`, now(), message); err != nil {
		return nil, exit.Internalf("cannot close classic attempts: %s", err)
	}
	var ids []string
	for _, r := range found {
		state, event := "failed", "run.failed"
		if r.state == "canceling" {
			state, event = "canceled", "run.canceled"
		}
		if _, err := tx.Exec(`UPDATE requests SET state=?,retain_work=0 WHERE id=?`, state, r.id); err != nil {
			return nil, exit.Internalf("cannot settle classic request %s: %s", r.id, err)
		}
		if err := appendEventTx(tx, r.id, event, r.ordinal, map[string]any{
			"status": "FAILED", "error_type": ClassicRetiredCode, "error": message, "requeuing": false,
		}); err != nil {
			return nil, exit.Internalf("cannot journal classic retirement of %s: %s", r.id, err)
		}
		ids = append(ids, r.id)
	}
	// Custody a classic worker held for any run, finished or not, has no holder left to
	// release it: its pending weights finalizations and byte retentions are forgotten here,
	// and each store's own garbage collection reclaims the bytes.
	const classic = `SELECT r.id FROM requests r WHERE NOT EXISTS(SELECT 1 FROM machine_executions m WHERE m.request_id=r.id)`
	if _, err := tx.Exec(`UPDATE weights_finalizations SET result_outcome=?,completed_at=?
		WHERE completed_at='' AND request_id IN (`+classic+`)`, ClassicRetiredOutcome, now()); err != nil {
		return nil, exit.Internalf("cannot forget classic weights finalizations: %s", err)
	}
	if _, err := tx.Exec(`UPDATE native_artifact_retentions SET state='released'
		WHERE state!='released' AND owner_request_id IN (` + classic + `)`); err != nil {
		return nil, exit.Internalf("cannot forget classic artifact custody: %s", err)
	}
	if _, err := tx.Exec(`UPDATE request_weights_retentions SET state='released'
		WHERE state!='released' AND request_id IN (` + classic + `)`); err != nil {
		return nil, exit.Internalf("cannot forget classic weights custody: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit classic retirement: %s", err)
	}
	return ids, nil
}

// RentalEndedOutcome completes a weights finalization whose rental has ended; it carries
// no receipt.
const RentalEndedOutcome = "RENTAL_ENDED"

// ForgetEndedRentalCustody forgets, once at daemon start, custody tied to a rental this
// host knows has ended (released or failed, or already forgotten here). Its store went
// with the machine, so nothing can ever release it. Custody on a live rental stays.
func (s *Store) ForgetEndedRentalCustody() *exit.Error {
	const ended = `NOT EXISTS(SELECT 1 FROM rentals x WHERE x.id=%s AND x.state NOT IN ('released','failed'))`
	endedRequests := `SELECT r.id FROM requests r WHERE r.worker<>'' AND ` + fmt.Sprintf(ended, "r.worker")
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin ended-rental custody cleanup: %s", err)
	}
	defer tx.Rollback()
	for _, statement := range []struct {
		sql  string
		args []any
	}{
		{`UPDATE weights_finalizations SET result_outcome=?,completed_at=? WHERE completed_at='' AND request_id IN (` + endedRequests + `)`,
			[]any{RentalEndedOutcome, now()}},
		{`UPDATE request_weights_retentions SET state='released' WHERE state!='released' AND request_id IN (` + endedRequests + `)`, nil},
		{`UPDATE native_artifact_retentions SET state='released' WHERE state!='released' AND owner_worker NOT IN ('','local') AND ` +
			fmt.Sprintf(ended, "native_artifact_retentions.owner_worker"), nil},
	} {
		if _, err := tx.Exec(statement.sql, statement.args...); err != nil {
			return exit.Internalf("cannot forget ended-rental custody: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit ended-rental custody cleanup: %s", err)
	}
	return nil
}
