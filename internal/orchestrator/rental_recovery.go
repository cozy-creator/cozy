package orchestrator

import (
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
func (c *Orchestrator) RecoverRentalWork(rentalID, cause string) {
	if rentalID == "" {
		return
	}
	pinned, problem := c.opt.Store.PinnedRentalWork(rentalID)
	if problem != nil {
		c.logf("cannot read the work pinned to rental %s: %s", rentalID, problem.Message)
		return
	}
	if len(pinned) == 0 {
		return
	}
	c.logf("rental %s is %s; recovering %d pinned request(s)", rentalID, cause, len(pinned))
	for _, req := range pinned {
		c.recoverPinned(req, rentalID, cause)
	}
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
			stranded, problem := c.opt.Store.StrandRentalAttempt(req.ID, attempt.Attempt,
				"the rented machine was lost before this attempt reported a terminal ("+cause+")")
			if problem != nil {
				c.logf("%s#%d could not be stranded: %s", req.ID, attempt.Attempt, problem.Message)
				return
			}
			if !stranded {
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
