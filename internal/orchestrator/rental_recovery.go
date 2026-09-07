package orchestrator

import (
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// RENTAL FAILURE RECOVERY.
//
// Pinning a request to one rental is intended (owner ruling 2026-09-03): the daemon
// chooses a machine and the request goes there. What was missing is the other arm — when
// that machine dies, the work it was holding must come back and be replanned somewhere
// else. Without it a rental that reached a terminal failed state kept its pin forever, and
// the deadlock was mutual: the requests could never be routed because they named a dead
// rental, and the rental could never be released because `RentalRunCounts` still counted
// them. Observed live 2026-09-04: rental pr-183abac284d1e16f5f0a (nitian) sat `failed`
// holding one in-flight and two queued anima requests while the orchestrator bought a
// second pod, served an identical request on it, and idle-released it out from under the
// three that were still waiting.
//
// THE TRIGGER IS AN OBSERVATION, NEVER A CLOCK. Recovery runs because the hub says the
// rental is terminally failed and its provider resource has been reclaimed — not because
// an attempt has been open for a while. An attempt at 300 s on a healthy pod is a long
// job; the same attempt on a reclaimed pod is stranded. Only the second is actionable, and
// only the rental's state can tell them apart.
//
// THE TWO CASES ARE NOT THE SAME, and they get different answers:
//
//	queued, never dispatched   the pin is released and routing replans it. Nothing ran, so
//	                           no requeue life is charged — the same rule `RequeueForCapacity`
//	                           already applies to a worker that answered "not now".
//	accepted by a worker       the attempt is closed as lost and the request goes through
//	                           the ORDINARY requeue budget. It may have partially executed,
//	                           so re-offering it costs a life and a request out of lives
//	                           fails saying so, rather than being retried silently or
//	                           stranded silently.
//
// "Replanned if possible" needs an honest else-branch, and it already has one: an unpinned
// --rental request runs `selectOrStart`, which acquires a rental or fails the request with
// the acquisition's own reason. Recovery never leaves a request in a state where nothing
// will speak about it again.
// RecoverLostWork is the sweep, and it asks the question from the REQUEST side.
//
// The first cut of this keyed on OBSERVING a rental go `failed`, which has a hole: once the
// rental RECORD is gone — `cozy rental end`, or any reconcile that forgets a released row —
// there is no object left to observe, so nothing ever looked at the work again. That is
// exactly how req-b2df33d17e663a8a1e047246 stayed `in_progress` for sixteen hours against a
// container destroyed four minutes into its life. A request row always exists while the
// request is active, so the request is what the sweep must enumerate.
func (c *Orchestrator) RecoverLostWork() {
	orphaned, problem := c.opt.Store.OrphanedRentalWork()
	if problem != nil {
		c.logf("cannot read work pinned to lost rentals: %s", problem.Message)
		return
	}
	for _, req := range orphaned {
		c.recoverPinned(req, req.Worker, c.lostRentalCause(req.Worker))
	}
}

// lostRentalCause says WHY the rental cannot serve, in the words the operator will see.
func (c *Orchestrator) lostRentalCause(rentalID string) string {
	row, problem := c.opt.Store.RentalRow(rentalID)
	if problem != nil || row == nil {
		return "the rental record is gone"
	}
	if row.Failure.Code != "" {
		return row.Failure.Code
	}
	return row.State
}

// recoverPinned answers for ONE request. The open attempt is looked up on the request
// itself rather than on the rental's worker slot: the slot is derived from the rental id
// and a rental whose pod was reclaimed may have no worker record left to ask.
//
// Three attempt shapes arrive here and only one of them is the in-flight case:
//
//	preparing / offered      never crossed to the worker. Aborted, which returns the
//	                         request to `submitted`, and then released like any queued row.
//	accepted / recovered_open  crossed, and may have executed. Stranded and re-offered
//	                         under the requeue budget.
//	terminal                 the worker already committed a real outcome that has not been
//	                         acked. It is LEFT ALONE. Replacing a recorded terminal with a
//	                         synthetic failure would publish `request.failed` over a run
//	                         that may have succeeded.
func (c *Orchestrator) recoverPinned(req records.Request, rentalID, cause string) {
	attempts, problem := c.opt.Store.Attempts(req.ID)
	if problem != nil {
		c.logf("%s: cannot read attempts while recovering rental %s: %s",
			req.ID, rentalID, problem.Message)
		return
	}
	for _, attempt := range attempts {
		switch attempt.State {
		case "terminal":
			c.logf("%s#%d holds an unacked terminal from lost rental %s; leaving its recorded outcome alone",
				req.ID, attempt.Attempt, rentalID)
			return
		case "preparing", "offered":
			// Nothing crossed the worker boundary, so nothing is owed and nothing is
			// charged. Abort returns the request to `submitted` and the queued arm below
			// releases its pin.
			if e := c.opt.Store.AbortDispatch(req.ID, attempt.Attempt, attempt.SessionID,
				"the rented machine was lost before this assignment was offered ("+cause+")"); e != nil {
				c.logf("%s#%d could not be aborted on lost rental %s: %s",
					req.ID, attempt.Attempt, rentalID, e.Message)
				return
			}
			c.settleDispatch(req.ID, uint64(attempt.Attempt), false)
			req.State = "submitted"
		case "accepted", "recovered_open":
			// THE IN-FLIGHT CASE. Its worker accepted it and owes a terminal that can never
			// arrive: the pod holding the journal was destroyed. Close it as lost and
			// re-offer under the budget.
			abandoned, problem := c.opt.Store.AbandonLostAttempt(req.ID, attempt.Attempt,
				"the rented machine was lost before this attempt reported a terminal ("+cause+")",
				records.RequeueAfterLoss)
			if problem != nil {
				c.logf("%s#%d could not be abandoned: %s", req.ID, attempt.Attempt, problem.Message)
				return
			}
			if !abandoned {
				return
			}
			c.settleDispatch(req.ID, uint64(attempt.Attempt), false)
			c.emit(req.ID, "request.attempt_failed", uint64(attempt.Attempt), map[string]any{
				"status": "FAILED", "cause": "RENTAL_LOST", "error_type": "rental.lost",
				"error":     "the rented machine " + rentalID + " was lost while this attempt was running (" + cause + ")",
				"outputs":   []any{},
				"requeuing": true,
			})
			// THE PIN GOES TOO, and it must go BEFORE the requeue. Both arms release it —
			// they differ only in what the retry costs, never in where it may run. A
			// requeue that kept the pin would re-dispatch the request straight back at the
			// corpse, spend a life discovering the machine is still gone, and do it again
			// until the budget ran out.
			if _, problem := c.opt.Store.UnpinRentalWork(req.ID, rentalID); problem != nil {
				c.logf("%s could not be released from lost rental %s: %s",
					req.ID, rentalID, problem.Message)
				return
			}
			c.logf("%s#%d was running on lost rental %s; re-offering it under the requeue budget",
				req.ID, attempt.Attempt, rentalID)
			// Charges a life on purpose: this attempt may have partially executed.
			c.Requeue(req.ID, "the rented machine it was running on was lost")
			return
		}
	}
	// THE QUEUED CASE. No attempt is executing, so there is nothing to replay and nothing
	// to charge. Release the pin and let routing choose again.
	released, problem := c.opt.Store.UnpinRentalWork(req.ID, rentalID)
	if problem != nil {
		c.logf("%s could not be released from rental %s: %s", req.ID, rentalID, problem.Message)
		return
	}
	if !released {
		// It is neither queued nor holding an attempt this arm can decide for. Say so
		// rather than leaving silence.
		c.logf("%s is %s on lost rental %s and was not recovered by either arm",
			req.ID, req.State, rentalID)
		return
	}
	req.Worker = ""
	c.emit(req.ID, "request.queued", 0, map[string]any{
		"reason": "the rented machine " + rentalID + " was lost before this request started (" + cause + ")",
	})
	c.logf("%s was queued on lost rental %s; replanning it", req.ID, rentalID)
	if !c.enqueue(req.ID) {
		return
	}
	c.selectOrStart(req)
	go c.drain()
}

// rentalCanServe answers whether the rental a request is pinned to could still take it.
// A row that is absent, terminally failed, or released cannot, and that is a durable fact
// this host can read without asking anything remote — which matters, because the thing it
// would have to ask is the thing that is gone.
func (c *Orchestrator) rentalCanServe(rentalID string) bool {
	if rentalID == "" {
		return false
	}
	row, problem := c.opt.Store.RentalRow(rentalID)
	if problem != nil {
		// An unreadable store is not an observation about the pod. Leave the attempt alone.
		return true
	}
	return row != nil && row.State != "failed" && row.State != "released"
}

// CancelLostAttempt settles a request whose attempt can no longer be reached, as CANCELED,
// from local durable state alone.
//
// Teardown reaches here when the control stream that held an attempt is gone. Recovery
// wants a requeue in that situation; teardown wants the opposite, because the operator
// asked for everything to stop and replanning the work into a daemon that is shutting down
// answers a different question than the one they asked.
//
// The attempt's closure is what makes the cancellation reachable at all: both
// `CancelQueuedRequest` and `FailQueuedRequest` refuse a request holding an open attempt,
// which is precisely how one lost attempt made `cozy down` AND `cozy down --all` refuse and
// left the daemon killable only by signal.
func (c *Orchestrator) CancelLostAttempt(requestID string, attempt int64, reason string) *exit.Error {
	canceled, problem := c.opt.Store.AbandonLostAttempt(requestID, attempt, reason,
		records.CancelAfterLoss)
	if problem != nil {
		return problem
	}
	if !canceled {
		// Not an open attempt this arm owns. The ordinary queued cancellation is then the
		// right verb, and it is the caller's existing path.
		return c.CancelQueued(requestID, "cozy down --all")
	}
	c.settleDispatch(requestID, uint64(attempt), false)
	c.forget(requestID)
	c.frames.forget(requestID)
	c.logf("%s#%d was canceled with its execution context already gone: %s",
		requestID, attempt, reason)
	c.signalClosed(requestWaitKey(requestID),
		exit.New(exit.Canceled, "%s was canceled; its execution context was gone", requestID))
	c.RetryOutputExport(requestID)
	return nil
}

// An explicitly rented pod is still owned when no request needs it. Restore its
// existing signed control claim on restart so owner absence cannot reclaim it.
// Each attachment uses the normal resolver and runs outside fleet acquisition.
func (c *Orchestrator) resumeManualRentals() *exit.Error {
	if c.opt.Rentals == nil {
		return nil
	}
	rows, problem := c.opt.Store.Rentals()
	if problem != nil {
		return problem
	}
	for _, row := range rows {
		retained, problem := c.opt.Store.RentalRetainsWork(row.ID)
		if problem != nil {
			return problem
		}
		if (row.ManagedRequestID != "" && !retained) || !records.RentalReadyState(row.State) {
			continue
		}
		go c.resumeManualRental(row.ID)
	}
	return nil
}

func (c *Orchestrator) resumeManualRental(id string) {
	for {
		c.mu.Lock()
		closing := c.closing
		c.mu.Unlock()
		if closing {
			return
		}
		row, problem := c.opt.Store.RentalRow(id)
		if problem != nil {
			c.logf("rental %s control reattachment cannot read ownership: %s", id, problem.Message)
			return
		}
		retained, retainedProblem := c.opt.Store.RentalRetainsWork(id)
		if retainedProblem != nil {
			return
		}
		if row == nil || (row.ManagedRequestID != "" && !retained) || !records.RentalReadyState(row.State) {
			return
		}
		if _, _, _, problem = c.EnsureRental(id); problem == nil {
			return
		}
		c.logf("rental %s control reattachment refused: %s", id, problem.Message)
		if problem.Code != exit.Unavailable {
			return
		}
		c.mu.Lock()
		worker := c.workers[rentalInstanceID(id)]
		connecting := worker != nil && !worker.exited && !worker.stopping
		c.mu.Unlock()
		if connecting {
			return // the existing connection loop owns transport recovery
		}
		timer := time.NewTimer(ReportCadence)
		select {
		case <-c.done:
			timer.Stop()
			return
		case <-timer.C:
		}
	}
}
