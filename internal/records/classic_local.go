package records

import (
	"fmt"

	"github.com/cozy-creator/cozy/internal/exit"
)

// ClassicLocalRetiredCode names work that only the retired classic local Runtime worker
// (`cozy-runtime serve` spawned by this daemon) could run or release.
const ClassicLocalRetiredCode = "request.classic_local_retired"

// ClassicLocalRetiredOutcome completes a weights finalization whose classic local holder
// is gone; it carries no receipt.
const ClassicLocalRetiredOutcome = "CLASSIC_LOCAL_RETIRED"

// RetireClassicLocalWork ends, once at daemon start, every unfinished request that ran or
// would run on the classic local worker: no rental and no machine execution. Open attempts
// close as ABANDONED and nothing is revived. A request being canceled ends canceled; every
// other one fails with ClassicLocalRetiredCode. Custody held for any classic local run is
// forgotten.
func (s *Store) RetireClassicLocalWork() ([]string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, exit.Internalf("cannot begin classic local retirement: %s", err)
	}
	defer tx.Rollback()
	rows, err := tx.Query(`SELECT id,state,ordinal FROM requests r WHERE worker='' AND rental=0
		AND state IN (` + activeRequestStates + `)
		AND NOT EXISTS(SELECT 1 FROM machine_executions m WHERE m.request_id=r.id)
		AND (EXISTS(SELECT 1 FROM attempts a WHERE a.request_id=r.id) OR retain_work=1)
		ORDER BY created_at,id`)
	if err != nil {
		return nil, exit.Internalf("cannot read classic local work: %s", err)
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
			return nil, exit.Internalf("cannot read classic local request: %s", err)
		}
		found = append(found, r)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return nil, exit.Internalf("cannot finish classic local census: %s", err)
	}
	const message = "this work needs the retired classic local Runtime worker; this computer now runs work through its machine"
	var ids []string
	for _, r := range found {
		if _, err := tx.Exec(`UPDATE attempts SET state='closed',closed_at=?,
			terminal_status=CASE WHEN terminal_status='' THEN 'ABANDONED' ELSE terminal_status END,
			terminal_cause=CASE WHEN terminal_cause='' THEN 'EXECUTION_CONTEXT_LOST' ELSE terminal_cause END,
			safe_message=CASE WHEN safe_message='' THEN ? ELSE safe_message END
			WHERE request_id=? AND state IN (`+openAttemptStates+`)`, now(), message, r.id); err != nil {
			return nil, exit.Internalf("cannot close classic local attempts of %s: %s", r.id, err)
		}
		state, event := "failed", "request.failed"
		if r.state == "canceling" {
			state, event = "canceled", "request.canceled"
		}
		if _, err := tx.Exec(`UPDATE requests SET state=?,retain_work=0 WHERE id=?`, state, r.id); err != nil {
			return nil, exit.Internalf("cannot settle classic local request %s: %s", r.id, err)
		}
		if err := appendEventTx(tx, r.id, event, r.ordinal, map[string]any{
			"status": "FAILED", "error_type": ClassicLocalRetiredCode, "error": message, "requeuing": false,
		}); err != nil {
			return nil, exit.Internalf("cannot journal classic local retirement of %s: %s", r.id, err)
		}
		ids = append(ids, r.id)
	}
	// Custody the classic local worker held for any run, finished or not, has no holder
	// left to release it: its pending weights finalizations and byte retentions are
	// forgotten here, and the store's own garbage collection reclaims the bytes.
	const classic = `SELECT r.id FROM requests r WHERE r.worker='' AND r.rental=0
		AND NOT EXISTS(SELECT 1 FROM machine_executions m WHERE m.request_id=r.id)`
	if _, err := tx.Exec(`UPDATE weights_finalizations SET result_outcome=?,completed_at=?
		WHERE completed_at='' AND request_id IN (`+classic+`)`, ClassicLocalRetiredOutcome, now()); err != nil {
		return nil, exit.Internalf("cannot forget classic local weights finalizations: %s", err)
	}
	if _, err := tx.Exec(`UPDATE native_artifact_retentions SET state='released'
		WHERE owner_worker='' AND state!='released' AND owner_request_id IN (` + classic + `)`); err != nil {
		return nil, exit.Internalf("cannot forget classic local artifact custody: %s", err)
	}
	if _, err := tx.Exec(`UPDATE request_weights_retentions SET state='released'
		WHERE state!='released' AND request_id IN (` + classic + `)`); err != nil {
		return nil, exit.Internalf("cannot forget classic local weights custody: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, exit.Internalf("cannot commit classic local retirement: %s", err)
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
