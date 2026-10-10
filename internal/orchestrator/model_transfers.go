package orchestrator

import (
	"fmt"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (c *Orchestrator) runModelPassThrough(req records.Request) {
	// This continuation owns only privileged source/destination I/O for the one
	// ordinary attempt-zero request. It is restartable from the sidecar and has no
	// scheduler, graph, retry budget, route, id namespace, or terminal of its own.
	c.mu.Lock()
	if c.transferRunning[req.ID] {
		c.mu.Unlock()
		return
	}
	c.transferRunning[req.ID] = true
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.transferRunning, req.ID)
		c.mu.Unlock()
	}()
	if c.opt.ModelTransfers == nil {
		c.logf("model transfer %s remains submitted: this daemon has no transfer owner", req.ID)
		return
	}
	transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
	if problem == nil && transfer != nil && (transfer.State == "completed" || transfer.State == "failed") {
		_, problem = c.opt.Store.SettleModelTransferRequest(req.ID, 0)
		if problem != nil {
			c.logf("pass-through model transfer %s settlement failed: %s", req.ID, problem.Message)
		}
		c.forgetTransferProgress(req.ID)
		return
	}
	moved := false
	if problem == nil && transfer != nil {
		before := transferMark(transfer)
		ctx, leave := c.joinTransfer(req.ID)
		problem = c.opt.ModelTransfers.PassThrough(ctx, req.ID, transfer.ModelTransferIntent)
		canceled := ctx.Err() != nil
		leave()
		if canceled {
			// Canceled: the cancel settled the request; nothing here retries or fails it.
			c.forgetTransferProgress(req.ID)
			return
		}
		after, _ := c.opt.Store.ModelTransferOf(req.ID)
		moved = after != nil && transferMark(after) != before
	}
	if problem != nil {
		// An unavailable source or destination is tried again only after an attempt that
		// moved the transfer to its next durable step; one that moved nothing fails it.
		if !permanentTransferFailure(problem) && moved {
			time.AfterFunc(2*time.Second, func() { c.runModelPassThrough(req) })
			return
		}
		_ = c.opt.Store.FailModelTransfer(req.ID, problem.ErrName(), problem.Message)
		_, _ = c.opt.Store.SettleModelTransferRequest(req.ID, 0)
		c.forgetTransferProgress(req.ID)
		c.signalClosed(requestWaitKey(req.ID), problem)
		return
	}
	current, readProblem := c.opt.Store.RequestRow(req.ID)
	if readProblem != nil || current == nil || current.State == "canceled" {
		return
	}
	if _, problem := c.opt.Store.SettleModelTransferRequest(req.ID, 0); problem != nil {
		c.logf("pass-through model transfer %s settlement failed: %s", req.ID, problem.Message)
	}
	c.forgetTransferProgress(req.ID)
}

// ResumeModelTransfers restarts the daemon's own model transfers a prior daemon left owed.
func (c *Orchestrator) ResumeModelTransfers() *exit.Error {
	owed, problem := c.opt.Store.ModelTransfersOwed()
	if problem != nil {
		return problem
	}
	for _, transfer := range owed {
		request, problem := c.opt.Store.RequestRow(transfer.RequestID)
		if problem != nil || request == nil || request.State == "canceled" || !passThrough(*request) {
			continue
		}
		go c.runModelPassThrough(*request)
	}
	return nil
}

func (c *Orchestrator) forgetTransferProgress(requestID string) {
	c.ForgetPhase(requestID)
}

func permanentTransferFailure(problem *exit.Error) bool {
	if problem == nil {
		return false
	}
	switch problem.Code {
	case exit.Unavailable, exit.Deadline, exit.Capacity:
		return false
	default:
		return true
	}
}

// transferMark is what a transfer has durably done: its step and the models and checkpoints
// it holds.
func transferMark(t *records.ModelTransfer) string {
	return fmt.Sprint(t.State, t.Models, t.Checkpoints)
}
