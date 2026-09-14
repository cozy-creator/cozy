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
	retainedResult := prior.RetainsLocalOutputs()
	if prior.State == "succeeded" && !retainedResult {
		if err := tx.QueryRow(retainedDescendantsSQL, prior.ID).Scan(&retainedResult); err != nil {
			return exit.Internalf("cannot inspect predecessor child custody: %s", err)
		}
	}
	retained, stopped := prior.RetainWork, prior.State == "paused" || prior.State == "blocked" || prior.State == "succeeded" && retainedResult
	if request.MachineExecutionObserver {
		var machineRetained bool
		if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM machine_executions e JOIN requests r ON r.id=e.request_id WHERE r.id=? AND length(e.receipt)>0 AND length(e.pending_control)=0 AND e.cancel_requested=0 AND `+machineExecutionOwed+`)`, prior.ID).Scan(&machineRetained); err != nil {
			return exit.Internalf("cannot inspect predecessor machine custody: %s", err)
		}
		retained = machineRetained
		stopped = machineRetained && (prior.State == "failed" || prior.State == "paused" || prior.State == "succeeded")
	}
	if !request.RetainWork || request.Kind != "job" || !retained || !prior.IsJob() || !stopped {
		return exit.Named(exit.Conflict, "request.retry_refused", "retry predecessor %s must retain stopped work; current state %s", prior.ID, prior.State)
	}
	var open int
	if err := tx.QueryRow(`WITH RECURSIVE family(id) AS (SELECT ? UNION ALL SELECT r.id FROM requests r JOIN family f ON r.parent_request_id=f.id)
		SELECT COUNT(*) FROM attempts WHERE request_id IN (SELECT id FROM family) AND state IN (`+openAttemptStates+`)`, prior.ID).Scan(&open); err != nil {
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
	if request.RequestedRental != "" && request.RequestedRental != prior.RequestedRental {
		return exit.Named(exit.Conflict, "request.retry_rental_changed", "retry cannot change requested rental")
	}
	request.RequestedRental = prior.RequestedRental
	request.Worker, request.Rental, request.RentalRequired = prior.Worker, prior.Rental, prior.RentalRequired
	request.AttentionKernel = prior.AttentionKernel
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

func (r Request) RetainsLocalOutputs() bool {
	return r.RetainWork && (r.ChildArtifacts || (r.WeightsOutputs != "" && r.WeightsOutputs != "[]")) &&
		(r.ModelTransfer == nil || r.ModelTransfer.Destination == "")
}

func (s *Store) BlockRetainedWork(id, code, detail string) (bool, *exit.Error) {
	return s.blockRetainedWork(id, code, detail, false)
}

func (s *Store) BlockLostRetainedWork(id, detail string) (bool, *exit.Error) {
	return s.blockRetainedWork(id, "request.state_lost", detail, true)
}

func (s *Store) blockRetainedWork(id, code, detail string, includeStopped bool) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin retained failure: %s", err)
	}
	defer tx.Rollback()
	states := `'submitted','queued','dispatching','requeue_pending','finalizing'`
	if includeStopped {
		states += `,'pausing','paused'`
	}
	result, err := tx.Exec(`UPDATE requests SET state='blocked' WHERE id=? AND retain_work=1
		AND state IN (`+states+`)`, id)
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
	var state, kind, parent string
	var retain bool
	if err := tx.QueryRow(`SELECT state,kind,parent_request_id,retain_work FROM requests WHERE id=?`, id).
		Scan(&state, &kind, &parent, &retain); err != nil {
		if err == sql.ErrNoRows {
			return "", exit.New(exit.NotFound, "request %s is absent", id)
		}
		return "", exit.Internalf("cannot read pause ownership: %s", err)
	}
	if (kind != "job" && parent == "") || !retain {
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
		AND NOT EXISTS(SELECT 1 FROM request_operation_lookups WHERE request_id=requests.id AND state='pending')
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`))
		AND NOT EXISTS(SELECT 1 FROM request_model_transfers WHERE request_id=? AND state='materializing')
		AND NOT EXISTS(WITH RECURSIVE family(id) AS (SELECT id FROM requests WHERE parent_request_id=? UNION ALL SELECT r.id FROM requests r JOIN family f ON r.parent_request_id=f.id)
		SELECT 1 FROM requests r JOIN family f ON r.id=f.id WHERE r.state NOT IN ('paused','blocked',`+settledRequestStates+`)
		OR EXISTS(SELECT 1 FROM attempts a WHERE a.request_id=r.id AND a.state IN (`+openAttemptStates+`)))`, id, id, id, id)
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
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests r LEFT JOIN request_model_transfers t ON t.request_id=r.id WHERE r.worker=? AND r.retain_work=1
		AND ((r.state IN (`+activeRequestStates+`) AND r.state!='releasing') OR
		(r.state='succeeded' AND (r.child_artifacts=1 OR r.weights_outputs NOT IN ('','[]')) AND COALESCE(json_extract(t.intent,'$.destination'),'')='')))`, id).Scan(&found)
	if err != nil {
		return false, exit.Internalf("cannot read retained rental ownership: %s", err)
	}
	return found, nil
}

func (s *Store) RequestRetainedCancellation(id, actor string) *exit.Error {
	_, problem := s.requestRetainedCancellation(id, actor, "")
	return problem
}

// The child may complete after the scheduler reads it. The parent's cancellation
// never changes a success that won that race, including an ordinal-zero cache hit.
func (s *Store) RequestDescendantCancellation(parent, child, actor string) (bool, *exit.Error) {
	return s.requestRetainedCancellation(child, actor, parent)
}

func (s *Store) requestRetainedCancellation(id, actor, parent string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin retained cancellation: %s", err)
	}
	defer tx.Rollback()
	statement := `UPDATE requests SET state='canceling',control_revision=control_revision+1 WHERE id=? AND retain_work=1
		AND state NOT IN ('failed','canceled','refused','abandoned','canceling','releasing')`
	args := []any{id}
	if parent != "" {
		statement = requestDescendantsCTE + statement + ` AND state!='succeeded' AND id IN (SELECT id FROM descendants)
			AND EXISTS(SELECT 1 FROM requests WHERE id=? AND state='canceling')`
		args = []any{parent, id, parent}
	}
	result, err := tx.Exec(statement, args...)
	if err != nil {
		return false, exit.Internalf("cannot record retained cancellation: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed == 0 {
		return false, nil
	}
	if _, err := tx.Exec(`UPDATE request_model_transfers SET state='canceling',updated_at=? WHERE request_id=? AND state NOT IN ('completed','canceled')`, now(), id); err != nil {
		return false, exit.Internalf("cannot fence retained source and publication work: %s", err)
	}
	if err := appendEventTx(tx, id, "request.cancel_requested", 0, map[string]any{"actor": actor, "status": "canceling"}); err != nil {
		return false, exit.Internalf("cannot journal retained cancellation: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit retained cancellation: %s", err)
	}
	return true, nil
}

// RecordRetainedFinalization supplies a disposition after the stopped attempt was
// already acknowledged. The original invocation and writer identity stay intact.
func (s *Store) RecordRetainedFinalization(f WeightsFinalization) *exit.Error {
	return s.recordWorkFinalization(f, "")
}

func (s *Store) RecordSuccessfulFinalization(root string, f WeightsFinalization) *exit.Error {
	if root == "" {
		return exit.Internalf("successful finalization has no root intent")
	}
	return s.recordWorkFinalization(f, root)
}

func (s *Store) recordWorkFinalization(f WeightsFinalization, releaseRoot string) *exit.Error {
	_, err := s.db.Exec(requestDescendantsCTE+`INSERT INTO weights_finalizations(request_id,attempt,instance_id,
		owner_scope,invocation_digest,output_slot,disposition,receipt_digest,scratch_root_id,recorded_at)
		SELECT ?,?,?,?,?,?,'ABANDON_UNCOMMITTED','','',?
		WHERE EXISTS(SELECT 1 FROM requests r WHERE r.id=? AND r.retain_work=1 AND
		 (r.state='canceling' OR (r.state='succeeded' AND (EXISTS(SELECT 1 FROM successful_work_releases w,json_each(w.members) m
		 WHERE w.request_id=? AND w.state='draining' AND m.value=r.id
		 AND EXISTS(SELECT 1 FROM requests root WHERE root.id=w.request_id AND root.state='succeeded'))
		 OR (r.id IN (SELECT id FROM descendants) AND EXISTS(SELECT 1 FROM requests root WHERE root.id=? AND root.state='canceling'))))))
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`))
		ON CONFLICT(request_id,invocation_digest,output_slot) DO NOTHING`,
		releaseRoot, f.RequestID, f.Attempt, f.InstanceID, f.OwnerScope, f.InvocationDigest, f.OutputSlot,
		now(), f.RequestID, releaseRoot, releaseRoot, f.RequestID)
	if err != nil {
		return exit.Internalf("cannot record retained artifact abandonment: %s", err)
	}
	return nil
}

// The cancelled ancestor owns this cleanup. Success remains the child's result,
// including a validated memo hit with no execution attempt. The existing parent
// cancellation and retain_work bit make the cut replayable without another ledger.
func (s *Store) ReleaseCompletedChildWork(parent, child string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin completed child release: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(requestDescendantsCTE+`UPDATE requests SET retain_work=0
		WHERE id=? AND state='succeeded' AND retain_work=1 AND id IN (SELECT id FROM descendants)
		AND EXISTS(SELECT 1 FROM requests WHERE id=? AND state='canceling')
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=requests.id AND state IN (`+openAttemptStates+`))
		AND NOT EXISTS(SELECT 1 FROM request_operation_lookups WHERE request_id=requests.id AND state='pending')
		AND NOT EXISTS(SELECT 1 FROM request_weights_retentions WHERE request_id=requests.id AND state!='released')
		AND NOT EXISTS(SELECT 1 FROM native_artifact_retentions WHERE consumer_id=requests.id AND state!='released')
		AND NOT EXISTS(SELECT 1 FROM weights_finalizations WHERE request_id=requests.id AND completed_at='')`, parent, child, parent)
	if err != nil {
		return false, exit.Internalf("cannot release completed child ownership: %s", err)
	}
	changed, _ := result.RowsAffected()
	if changed > 0 {
		if err := appendEventTx(tx, child, "request.work_released", 0, map[string]any{"status": "SUCCEEDED", "parent_request_id": parent}); err != nil {
			return false, exit.Internalf("cannot journal completed child release: %s", err)
		}
	}
	var ready bool
	if err := tx.QueryRow(requestDescendantsCTE+`SELECT EXISTS(SELECT 1 FROM requests
		WHERE id=? AND state='succeeded' AND retain_work=0 AND id IN (SELECT id FROM descendants))
		AND EXISTS(SELECT 1 FROM requests WHERE id=? AND state='canceling')`, parent, child, parent).Scan(&ready); err != nil {
		return false, exit.Internalf("cannot read completed child release: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit completed child release: %s", err)
	}
	return ready, nil
}

func (s *Store) ReleaseRetainedWork(id string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin retained cancellation completion: %s", err)
	}
	defer tx.Rollback()
	result, err := tx.Exec(requestDescendantsCTE+`UPDATE requests SET state='releasing' WHERE id=? AND state='canceling'
		AND NOT EXISTS(SELECT 1 FROM request_operation_lookups WHERE request_id=requests.id AND state='pending')
		AND NOT EXISTS(SELECT 1 FROM request_weights_retentions WHERE request_id=requests.id AND state!='released')
		AND NOT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND state IN (`+openAttemptStates+`))
		AND NOT EXISTS(SELECT 1 FROM weights_finalizations WHERE request_id=? AND completed_at='')
		AND NOT EXISTS(SELECT 1 FROM requests r JOIN descendants d ON d.id=r.id
		  WHERE r.state IN (`+activeRequestStates+`) OR (r.state='succeeded' AND r.retain_work=1))`, id, id, id, id)
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
