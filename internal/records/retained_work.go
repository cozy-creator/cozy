package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/exit"
)

func (s *Store) RetainedFailure(id string) (string, string, *exit.Error) {
	var raw string
	err := s.db.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND type='request.blocked' ORDER BY seq DESC LIMIT 1`, id).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", nil
	}
	if err != nil {
		return "", "", exit.Internalf("cannot read retained failure: %s", err)
	}
	var detail struct {
		ErrorType string `json:"error_type"`
		Error     string `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &detail); err != nil {
		return "", "", exit.Internalf("cannot decode retained failure: %s", err)
	}
	return detail.ErrorType, detail.Error, nil
}

// retainRetryTx acquires lineage and the same retained machine in the admission
// transaction. A concurrent cancellation wins before or after this commit, never
// between validating the predecessor and acquiring the descendant's rental hold.
func retainRetryTx(tx *sql.Tx, request *Request) *exit.Error {
	prior, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, request.RetryOf))
	if errors.Is(err, sql.ErrNoRows) {
		return exit.New(exit.NotFound, "retry predecessor %s is absent", request.RetryOf)
	}
	if err != nil {
		return exit.Internalf("cannot read retry predecessor: %s", err)
	}
	if !request.RetainWork || request.Kind != "job" || !prior.RetainWork || !prior.IsJob() ||
		(prior.State != "paused" && prior.State != "blocked") {
		return exit.Named(exit.Conflict, "request.retry_refused", "retry predecessor %s must retain stopped work; current state %s", prior.ID, prior.State)
	}
	var open int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`)`, prior.ID).Scan(&open); err != nil {
		return exit.Internalf("cannot inspect retry predecessor attempts: %s", err)
	}
	if open != 0 {
		return exit.Named(exit.Conflict, "request.retry_refused", "retry predecessor %s still has an open attempt", prior.ID)
	}
	if request.Worker != "" && request.Worker != prior.Worker {
		return exit.Named(exit.Conflict, "request.retry_worker_changed", "retry must retain predecessor %s's machine", prior.ID)
	}
	if prior.Worker != "" {
		var state string
		if err := tx.QueryRow(`SELECT state FROM rentals WHERE id=?`, prior.Worker).Scan(&state); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return exit.Named(exit.Conflict, "request.state_lost", "retry predecessor %s's retained rental is absent", prior.ID)
			}
			return exit.Internalf("cannot inspect retained rental: %s", err)
		}
		if state == "released" || state == "failed" || state == "release_requested" {
			return exit.Named(exit.Conflict, "request.state_lost", "retry predecessor %s's retained rental is %s", prior.ID, state)
		}
	}
	request.Worker, request.Rental, request.RentalRequired = prior.Worker, prior.Rental, prior.RentalRequired
	request.ReuseScope = prior.ReuseScope
	if request.ReuseScope == "" {
		request.ReuseScope = prior.ID
	}
	return nil
}

// RetainedState is an execution stop that still owns the request's exact inputs,
// intermediate outputs and machine. It is never a terminal attempt outcome.
func RetainedState(state string) bool {
	switch state {
	case "pausing", "paused", "blocked":
		return true
	}
	return false
}

func (s *Store) BlockRetainedWork(id, code, detail string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin retained failure: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE requests SET state='blocked' WHERE id=? AND retain_work=1
		AND state IN ('submitted','queued','dispatching','requeue_pending','finalizing','pausing','paused')`, id)
	if err != nil {
		return false, exit.Internalf("cannot retain failed work: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed == 0 {
		return false, nil
	}
	if err := appendEventTx(tx, id, "request.blocked", 0, map[string]any{
		"status": "blocked", "error_type": code, "error": detail, "requeuing": false,
	}); err != nil {
		return false, exit.Internalf("cannot journal retained failure: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit retained failure: %s", err)
	}
	return true, nil
}

// RequestPause commits intent before any stop frame, fencing concurrent dispatch.
// A request with an open attempt becomes paused only after its terminal is acked.
func (s *Store) RequestPause(id, actor string) (string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return "", exit.Internalf("cannot begin pause: %s", err)
	}
	defer tx.Rollback()
	var state, kind string
	var retain bool
	if err := tx.QueryRow(`SELECT state,kind,retain_work FROM requests WHERE id=?`, id).
		Scan(&state, &kind, &retain); err != nil {
		if err == sql.ErrNoRows {
			return "", exit.New(exit.NotFound, "request %s is absent", id)
		}
		return "", exit.Internalf("cannot read pause ownership: %s", err)
	}
	if kind != "job" || !retain {
		return "", exit.Named(exit.Conflict, "request.not_resumable", "request %s was not submitted with retained work", id)
	}
	if state == "paused" || state == "pausing" {
		return state, nil
	}
	if settledRequestState(state) || state == "canceling" || state == "releasing" || state == "blocked" || state == "finalizing" {
		return "", exit.Named(exit.Conflict, "request.pause_refused", "request %s cannot pause from %s", id, state)
	}
	next := "pausing"
	if _, err := tx.Exec(`UPDATE requests SET state=?,control_revision=control_revision+1 WHERE id=?`, next, id); err != nil {
		return "", exit.Internalf("cannot record pause: %s", err)
	}
	if err := appendEventTx(tx, id, "request."+next, 0, map[string]any{"actor": actor, "status": next}); err != nil {
		return "", exit.Internalf("cannot journal pause: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return "", exit.Internalf("cannot commit pause: %s", err)
	}
	return next, nil
}

func (s *Store) CompleteRequestPause(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin pause completion: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE requests SET state='paused' WHERE id=? AND state='pausing'
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`))
		AND NOT EXISTS(SELECT 1 FROM request_model_transfers WHERE request_id=? AND state='materializing')`, id, id, id)
	if err != nil {
		return false, exit.Internalf("cannot finish pause: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed == 0 {
		return false, nil
	}
	if err := appendEventTx(tx, id, "request.paused", 0, map[string]any{"status": "paused"}); err != nil {
		return false, exit.Internalf("cannot journal paused state: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit paused state: %s", err)
	}
	return true, nil
}

// ResumeRequest changes only scheduling disposition; every execution input and
// historical attempt stays untouched. Dispatch alone allocates the next ordinal.
func (s *Store) ResumeRequest(id, actor string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin resume: %s", err)
	}
	defer tx.Rollback()
	var state string
	var retain bool
	var open int
	if err := tx.QueryRow(`SELECT state,retain_work,(SELECT COUNT(*) FROM attempts
		WHERE request_id=r.id AND state IN (`+openAttemptStates+`)) FROM requests r WHERE id=?`, id).
		Scan(&state, &retain, &open); err != nil {
		return false, exit.Internalf("cannot read resume ownership: %s", err)
	}
	if !retain || state != "paused" || open != 0 {
		return false, exit.Named(exit.Conflict, "request.resume_refused",
			"request %s must be paused with all attempts closed; current state %s", id, state)
	}
	if _, err := tx.Exec(`UPDATE requests SET state='queued',control_revision=control_revision+1 WHERE id=? AND state='paused'`, id); err != nil {
		return false, exit.Internalf("cannot queue resumed request: %s", err)
	}
	if err := appendEventTx(tx, id, "request.resumed", 0, map[string]any{"actor": actor, "status": "queued"}); err != nil {
		return false, exit.Internalf("cannot journal resume: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit resume: %s", err)
	}
	return true, nil
}

// RentalRetainsWork covers every dependent request, not just the request that
// originally paid for a shared rental.
func (s *Store) RentalRetainsWork(id string) (bool, *exit.Error) {
	var found bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests WHERE worker=? AND retain_work=1
		AND state IN (`+activeRequestStates+`) AND state!='releasing')`, id).Scan(&found)
	if err != nil {
		return false, exit.Internalf("cannot read retained rental ownership: %s", err)
	}
	return found, nil
}

func (s *Store) RequestRetainedCancellation(id, actor string) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin retained cancellation: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE requests SET state='canceling',control_revision=control_revision+1 WHERE id=? AND retain_work=1
		AND state NOT IN (`+settledRequestStates+`,'canceling','releasing')`, id)
	if err != nil {
		return exit.Internalf("cannot record retained cancellation: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed == 0 {
		return nil
	}
	if err := appendEventTx(tx, id, "request.cancel_requested", 0, map[string]any{"actor": actor, "status": "canceling"}); err != nil {
		return exit.Internalf("cannot journal retained cancellation: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit retained cancellation: %s", err)
	}
	return nil
}

// RecordRetainedFinalization supplies a disposition after the stopped attempt was
// already acknowledged. The original invocation and writer identity stay intact.
func (s *Store) RecordRetainedFinalization(f WeightsFinalization) *exit.Error {
	_, err := s.db.Exec(`INSERT INTO weights_finalizations(request_id,attempt,instance_id,
		owner_scope,invocation_digest,output_slot,disposition,receipt_digest,scratch_root_id,recorded_at)
		SELECT ?,?,?,?,?,?,'ABANDON_UNCOMMITTED','','',?
		WHERE EXISTS(SELECT 1 FROM requests WHERE id=? AND state='canceling' AND retain_work=1)
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`))
		ON CONFLICT(request_id,invocation_digest,output_slot) DO NOTHING`,
		f.RequestID, f.Attempt, f.InstanceID, f.OwnerScope, f.InvocationDigest, f.OutputSlot,
		now(), f.RequestID, f.RequestID)
	if err != nil {
		return exit.Internalf("cannot record retained artifact abandonment: %s", err)
	}
	return nil
}

func (s *Store) ReleaseRetainedWork(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin retained cancellation completion: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE requests SET state='releasing' WHERE id=? AND state='canceling'
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`))
		AND NOT EXISTS(SELECT 1 FROM weights_finalizations WHERE request_id=? AND completed_at='')`, id, id, id)
	if err != nil {
		return false, exit.Internalf("cannot complete retained cancellation: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed == 0 {
		return false, nil
	}
	if _, err := tx.Exec(`UPDATE request_model_transfers SET state='canceled',updated_at=?
		WHERE request_id=? AND state!='completed'`, now(), id); err != nil {
		return false, exit.Internalf("cannot abandon retained source transfer: %s", err)
	}
	if err := appendEventTx(tx, id, "request.releasing", 0, map[string]any{"status": "canceling"}); err != nil {
		return false, exit.Internalf("cannot journal retained cancellation completion: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit retained cancellation completion: %s", err)
	}
	return true, nil
}

func (s *Store) CompleteRetainedCancellation(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin cancellation settlement: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(`UPDATE requests SET state='canceled' WHERE id=? AND state='releasing'`, id)
	if err != nil {
		return false, exit.Internalf("cannot settle retained cancellation: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed == 0 {
		return false, nil
	}
	if err := appendEventTx(tx, id, "request.canceled", 0, map[string]any{"status": "CANCELED", "cause": "CLIENT_CANCELED"}); err != nil {
		return false, exit.Internalf("cannot journal canceled request: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit canceled request: %s", err)
	}
	return true, nil
}
