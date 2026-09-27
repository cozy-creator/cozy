package orchestrator

import (
	"context"
	"github.com/cozy-creator/cozy/internal/canonical"
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
	return c.finishRequestedCancellation(id)
}

func (c *Orchestrator) finishRequestedCancellation(id string) *exit.Error {
	c.forget(id)
	c.cancelTransfer(id)
	if problem := c.cancelChildCalls(id, id); problem != nil {
		return problem
	}
	if problem := c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_CLIENT); problem != nil {
		return problem
	}
	c.finishRetainedCancellation(id)
	return nil
}

func (c *Orchestrator) runRetainedCancellation(ctx context.Context, id string) {
	request, problem := c.opt.Store.RequestRow(id)
	if problem != nil || request == nil || (request.State != "canceling" && request.State != "releasing") {
		return
	}
	if request.State == "releasing" {
		c.finishRetainedRelease(*request)
		return
	}
	if problem := c.cancelChildCalls(id, id); problem != nil {
		c.retryRetainedCancellation(id)
		return
	}
	ready, problem := c.releaseCompletedChildCalls(ctx, id, id)
	if problem != nil || !ready {
		if problem != nil {
			c.logf("request %s completed child cleanup remains pending: %s", id, problem.Message)
		}
		c.retryRetainedCancellation(id)
		return
	}
	attempts, problem := c.opt.Store.Attempts(id)
	if problem != nil {
		return
	}
	ready, problem = c.releaseAbandonedWork(ctx, *request, attempts, "")
	if problem != nil || !ready {
		_ = c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_CLIENT)
		c.retryRetainedCancellation(id)
		return
	}
	if problem := c.ackReleasedRetainedAttempts(ctx, *request, attempts); problem != nil {
		c.retryRetainedCancellation(id)
		return
	}
	changed, problem := c.opt.Store.ReleaseRetainedWork(id)
	if problem != nil || !changed {
		c.retryRetainedCancellation(id)
		return
	}
	request.State = "releasing"
	c.finishRetainedRelease(*request)
}

// Cancellation and completed-descendant cleanup drop the same request-owned
// byte holds. The latter uses its ancestor's cancellation to authorize native
// finalization, while keeping the child's successful execution immutable.
func (c *Orchestrator) releaseAbandonedWork(ctx context.Context, request records.Request, attempts []records.Attempt, releaseRoot string) (bool, *exit.Error) {
	id := request.ID
	if _, problem := c.lookupOperationPending(ctx, request, true); problem != nil {
		return false, problem
	}
	borrowed, problem := c.opt.Store.PendingArtifactBorrowers(id)
	if problem != nil || borrowed {
		return false, problem
	}
	for _, attempt := range attempts {
		if openAttempt(attempt.State) || attempt.State == "preparing" {
			return false, nil
		}
	}
	if problem := c.releaseChildRetentions(ctx, id, false); problem != nil {
		return false, problem
	}
	ready, problem := c.settleRetainedWeights(request, attempts, releaseRoot)
	if problem != nil || !ready {
		return false, problem
	}
	if problem := c.releaseOriginalDerivedResults(ctx, id); problem != nil {
		return false, problem
	}
	if request.ModelTransfer != nil {
		pending, problem := c.opt.Store.RetriedSourceCustodyPending(id)
		if problem != nil || pending {
			return false, problem
		}
		if problem := c.releaseRetainedSource(ctx, request); problem != nil {
			return false, problem
		}
		if c.opt.ModelTransfers != nil {
			if problem := c.opt.ModelTransfers.AbandonModelTransferPublications(ctx, id); problem != nil {
				return false, problem
			}
			if problem := c.opt.ModelTransfers.ReleaseCheckpoints(ctx, id); problem != nil {
				return false, problem
			}
		}
	}
	return true, nil
}

// A retained terminal remains a Host obligation after the attempt first closes.
// Release that same terminal only after its native/source ownership is settled.
// Reconnect replays this ordinary ACK if the connection dies before delivery.
func (c *Orchestrator) ackReleasedRetainedAttempts(ctx context.Context, request records.Request, attempts []records.Attempt) *exit.Error {
	if len(attempts) == 0 {
		return nil
	}
	if request.Worker != "" {
		rental, problem := c.opt.Store.RentalRow(request.Worker)
		if problem != nil {
			return problem
		}
		if rental != nil && (rental.State == "released" || rental.State == "failed") {
			return nil
		}
	}
	s, problem := c.workspaceControlContext(ctx, request.Worker)
	if problem != nil {
		return problem
	}
	for _, attempt := range attempts {
		if attempt.State != "closed" || attempt.TerminalID == "" {
			continue
		}
		invocation, err := canonical.Raw(attempt.InvocationDigest)
		if err != nil {
			return exit.Internalf("retained attempt has malformed invocation identity")
		}
		outcome, err := canonical.Raw(attempt.TerminalDigest)
		if err != nil {
			return exit.Internalf("retained attempt has malformed terminal identity")
		}
		ack := &pb.AttemptOutcomeAck{RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID, RequestId: request.ID, AttemptOrdinal: uint64(attempt.Attempt), InvocationSpecDigest: invocation, OutcomeId: attempt.TerminalID, OutcomeDigest: outcome, RetainWork: false}
		if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_OutcomeAck{OutcomeAck: ack}}) {
			return exit.Unavailablef("released terminal acknowledgement awaits its workspace connection")
		}
	}
	return nil
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
				release := c.opt.ReleaseRetainedRental
				if release == nil {
					release = c.opt.ReleaseManagedRental
				}
				if release == nil {
					c.retryRetainedCancellation(id)
					return
				}
				if _, problem := release(row.ID); problem != nil {
					c.logf("request %s retained rental release remains pending: %s", id, problem.Message)
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
	if problem := c.pauseChildCalls(id); problem != nil {
		return problem
	}
	if state == "pausing" {
		problem := c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_DRAIN)
		c.retryRetainedPause(id)
		return problem
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
	request, problem := c.opt.Store.RequestRow(id)
	if problem != nil {
		return problem
	}
	// Cancellation reconciles a pending memo lookup in its tracked cleanup pass,
	// where the shutdown context can stop a disconnected native peer.
	if request != nil && request.State == "pausing" {
		if _, problem := c.lookupOperationPending(nil, *request, true); problem != nil {
			return problem
		}
	}
	if request != nil && request.State == "pausing" && request.Worker != "" && request.ModelTransfer != nil && request.ModelTransfer.HasAcquisition() {
		c.pauseRetainedSource(*request)
		return nil
	}
	c.mu.Lock()
	preparing := c.transferDispatching[id] || c.transferRunning[id] || c.localTransfers[id] != nil || c.checkpointUploads[id] != nil
	c.mu.Unlock()
	if preparing {
		return nil
	}
	_, problem = c.opt.Store.CompleteRequestPause(id)
	return problem
}

func (c *Orchestrator) retryRetainedPause(id string) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing {
		return
	}
	time.AfterFunc(ReportCadence, func() {
		request, problem := c.opt.Store.RequestRow(id)
		if problem != nil || request == nil || request.State != "pausing" {
			return
		}
		_ = c.stopRetainedAttempt(id, pb.CancelReason_CANCEL_REASON_DRAIN)
		c.retryRetainedPause(id)
	})
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
	successful, problem := c.opt.Store.PendingSuccessfulWorkReleases()
	if problem != nil {
		return problem
	}
	for _, id := range successful {
		c.finishSuccessfulWorkRelease(id)
	}
	completed, problem := c.opt.Store.CompletedNativeByteRecipients()
	if problem != nil {
		return problem
	}
	for _, id := range completed {
		go c.finishNativeByteRecipients(id)
	}
	rows, problem := c.opt.Store.ActiveRequests()
	if problem != nil {
		return problem
	}
	for _, request := range rows {
		if request.State == "finalizing" && request.ModelTransfer == nil && request.RetainsLocalOutputs() {
			go c.finishClosedNativeRootResult(request.ID, request.Ordinal)
		}
		if request.State == "pausing" {
			if problem := c.stopRetainedAttempt(request.ID, pb.CancelReason_CANCEL_REASON_DRAIN); problem != nil {
				c.logf("request %s pause recovery: %s", request.ID, problem.Message)
			}
			c.retryRetainedPause(request.ID)
		}
		if request.RetainWork && (request.State == "canceling" || request.State == "releasing") {
			c.finishRetainedCancellation(request.ID)
		}
	}
	return nil
}

// Exact older-slot finalization is shared by cancellation and successful release.
// ACK(false) alone cannot abandon an unfinished derived writer.
func (c *Orchestrator) settleRetainedWeights(request records.Request, attempts []records.Attempt, releaseRoot string) (bool, *exit.Error) {
	id := request.ID
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
			return false, problem
		}
		for _, output := range outputs {
			key := attempt.InvocationDigest + "/" + output.OutputID
			if seen[key] {
				continue
			}
			seen[key] = true
			finalization := records.WeightsFinalization{
				RequestID: id, Attempt: attempt.Attempt, InstanceID: attempt.InstanceID,
				OwnerScope: recordOwnerID, InvocationDigest: attempt.InvocationDigest, OutputSlot: output.OutputID,
			}
			var problem *exit.Error
			if releaseRoot == "" {
				problem = c.opt.Store.RecordRetainedFinalization(finalization)
			} else {
				problem = c.opt.Store.RecordSuccessfulFinalization(releaseRoot, finalization)
			}
			if problem != nil {
				return false, problem
			}
		}
	}
	for _, attempt := range attempts {
		pending, problem := c.opt.Store.PendingWeightsFinalizations(id, attempt.Attempt)
		if problem != nil {
			return false, problem
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
			return false, nil
		}
	}
	return true, nil
}
