package orchestrator

import (
	"context"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"time"
)

// Older hosts discard weights custody on OutcomeAck, so they cannot acknowledge
// a retained stop even when their execution/cancellation protocol is compatible.
const RetainedWorkWireMinor uint32 = 39

func (c *Orchestrator) CancelRetainedRequest(id, actor string) *exit.Error {
	if problem := c.opt.Store.RequestRetainedCancellation(id, actor); problem != nil {
		return problem
	}
	c.forget(id)
	if problem := c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_CLIENT); problem != nil {
		return problem
	}
	go c.finishRetainedCancellation(id)
	return nil
}

func (c *Orchestrator) finishRetainedCancellation(id string) {
	request, problem := c.opt.Store.RequestRow(id)
	if problem != nil || request == nil || (request.State != "canceling" && request.State != "releasing") {
		return
	}
	if request.State == "releasing" {
		c.finishRetainedRelease(*request)
		return
	}
	attempts, problem := c.opt.Store.Attempts(id)
	if problem != nil {
		return
	}
	for _, attempt := range attempts {
		if openAttempt(attempt.State) || attempt.State == "preparing" {
			_ = c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_CLIENT)
			return
		}
	}
	// Walk newest first: a repeated invocation has one transaction per slot and its
	// latest recorded worker is the current holder after an ordinary recovery.
	seen := map[string]bool{}
	for i := len(attempts) - 1; i >= 0; i-- {
		attempt := attempts[i]
		if attempt.State != "closed" {
			continue
		}
		outputs, problem := decodeWeightsOutputs(attempt.WeightsOutputs)
		if problem != nil {
			c.logf("request %s retained cancellation: %s", id, problem.Message)
			return
		}
		for _, output := range outputs {
			key := attempt.InvocationDigest + "/" + output.OutputID
			if seen[key] {
				continue
			}
			seen[key] = true
			if problem := c.opt.Store.RecordRetainedFinalization(records.WeightsFinalization{
				RequestID: id, Attempt: attempt.Attempt, InstanceID: attempt.InstanceID,
				OwnerScope: recordOwnerID, InvocationDigest: attempt.InvocationDigest, OutputSlot: output.OutputID,
			}); problem != nil {
				c.logf("request %s retained cancellation: %s", id, problem.Message)
				return
			}
		}
	}
	for _, attempt := range attempts {
		pending, problem := c.opt.Store.PendingWeightsFinalizations(id, attempt.Attempt)
		if problem != nil {
			return
		}
		if len(pending) > 0 {
			c.mu.Lock()
			var session *session
			if w := c.workers[attempt.InstanceID]; w != nil && w.snapshotAcknowledged {
				session = c.sessions[w.bootID]
			}
			c.mu.Unlock()
			if session != nil {
				_, _ = c.sendPendingWeightsFinalizations(session, id, attempt.Attempt)
			} else if request.Worker != "" {
				_, _, _, _ = c.EnsureRental(request.Worker)
			}
			c.retryRetainedCancellation(id)
			return
		}
	}
	if request.ModelTransfer != nil && c.opt.ModelTransfers != nil {
		pending, problem := c.opt.Store.RetriedSourceCustodyPending(id)
		if problem != nil || pending {
			c.retryRetainedCancellation(id)
			return
		}
		if problem := c.opt.ModelTransfers.AbandonModelTransferPublications(context.Background(), id); problem != nil {
			c.retryRetainedCancellation(id)
			return
		}
		if problem := c.opt.ModelTransfers.ReleaseCheckpoints(context.Background(), id); problem != nil {
			c.retryRetainedCancellation(id)
			return
		}
	}
	changed, problem := c.opt.Store.ReleaseRetainedWork(id)
	if problem != nil || !changed {
		return
	}
	request.State = "releasing"
	c.finishRetainedRelease(*request)
}

func (c *Orchestrator) finishRetainedRelease(request records.Request) {
	id := request.ID
	if request.Worker != "" {
		row, problem := c.opt.Store.RentalRow(request.Worker)
		if problem != nil {
			c.retryRetainedCancellation(id)
			return
		}
		if row != nil && row.ManagedRequestID != "" && row.State != "released" && row.State != "failed" {
			retained, problem := c.opt.Store.RentalRetainsWork(row.ID)
			if problem != nil {
				c.retryRetainedCancellation(id)
				return
			}
			queued, running, problem := c.opt.Store.RentalRunCounts(row.ID)
			if problem != nil {
				c.retryRetainedCancellation(id)
				return
			}
			if queued == 0 && running == 0 && !retained {
				if problem := c.releaseManagedNow(request); problem != nil {
					c.retryRetainedCancellation(id)
					return
				}
				row, problem = c.opt.Store.RentalRow(request.Worker)
				if problem != nil || (row != nil && row.State != "released" && row.State != "failed") {
					c.retryRetainedCancellation(id)
					return
				}
			}
		}
	}
	changed, problem := c.opt.Store.CompleteRetainedCancellation(id)
	if problem != nil || !changed {
		return
	}
	request.State = "canceled"
	c.cleanupRequestAssets(request)
	c.reclaimTmp(id)
	c.cleanupPublication(request)
	c.RetryOutputExport(id)
	c.signalClosed(requestWaitKey(id), exit.New(exit.Canceled, "request %s was canceled", id))
}

func (c *Orchestrator) retryRetainedCancellation(id string) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if !closing {
		time.AfterFunc(ReportCadence, func() { c.finishRetainedCancellation(id) })
	}
}

// PauseRequest stops execution after durably recording the retention intent. A
// lost connection leaves pausing owed; reconnect sends the same digest-fenced stop.
func (c *Orchestrator) PauseRequest(id, actor string) *exit.Error {
	state, problem := c.opt.Store.RequestPause(id, actor)
	if problem != nil {
		return problem
	}
	c.forget(id)
	if state == "pausing" {
		return c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_DRAIN)
	}
	return nil
}

func (c *Orchestrator) stopRetainedAttempt(id string, reason pb.CancelReason) *exit.Error {
	attempts, problem := c.opt.Store.Attempts(id)
	if problem != nil {
		return problem
	}
	for _, attempt := range attempts {
		switch attempt.State {
		case "preparing":
			if problem := c.opt.Store.AbortDispatch(id, attempt.Attempt, attempt.SessionID, "request execution stopped by its owner"); problem != nil {
				return problem
			}
			c.settleDispatch(id, uint64(attempt.Attempt), false)
		case "offered", "accepted", "recovered_open":
			if problem := c.Cancel(id, uint64(attempt.Attempt), reason, ClientCancelGraceMS); problem != nil && problem.Code != exit.Unavailable {
				return problem
			}
		}
	}
	_, problem = c.opt.Store.CompleteRequestPause(id)
	return problem
}

func (c *Orchestrator) ResumeRequest(id, actor string) *exit.Error {
	request, problem := c.opt.Store.RequestRow(id)
	if problem != nil {
		return problem
	}
	if request == nil {
		return exit.New(exit.NotFound, "request %s is absent", id)
	}
	if request.Worker != "" && !c.rentalCanServe(request.Worker) {
		return exit.Named(exit.Conflict, "request.state_lost", "request %s's retained rental is unavailable; its work cannot be claimed as resumed", id)
	}
	if _, problem := c.opt.Store.ResumeRequest(id, actor); problem != nil {
		return problem
	}
	request, problem = c.opt.Store.RequestRow(id)
	if problem != nil || request == nil {
		return problem
	}
	go func() {
		if _, problem := c.activateRecorded(*request); problem != nil {
			c.logf("request %s resume activation: %s", id, problem.Message)
		}
	}()
	return nil
}

// restoreRetainedWork reconstructs control ownership without queueing paused work.
func (c *Orchestrator) restoreRetainedWork() *exit.Error {
	rows, problem := c.opt.Store.ActiveRequests()
	if problem != nil {
		return problem
	}
	for _, request := range rows {
		if request.State == "pausing" {
			if problem := c.stopRetainedAttempt(request.ID, pb.CancelReason_CANCEL_REASON_DRAIN); problem != nil {
				c.logf("request %s pause recovery: %s", request.ID, problem.Message)
			}
		}
		if request.RetainWork && (request.State == "canceling" || request.State == "releasing") {
			go c.finishRetainedCancellation(request.ID)
		}
	}
	return nil
}
