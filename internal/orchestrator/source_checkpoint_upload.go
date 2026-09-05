package orchestrator

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
)

type sourceCheckpointUpload struct{ again bool }

// A single uploader follows the durable observed heads while conversion and
// source downloads continue. Coalescing wakeups never coalesces custody facts.
func (c *Orchestrator) kickSourceCheckpointUpload(requestID string) {
	if c.opt.ModelTransfers == nil {
		return
	}
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return
	}
	if running := c.sourceUploads[requestID]; running != nil {
		running.again = true
		c.mu.Unlock()
		return
	}
	running := &sourceCheckpointUpload{}
	c.sourceUploads[requestID] = running
	c.mu.Unlock()
	go func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go func() {
			select {
			case <-c.done:
				cancel()
			case <-ctx.Done():
			}
		}()
		for {
			c.mu.Lock()
			running.again = false
			c.mu.Unlock()
			problem := c.syncSourceCheckpoint(ctx, requestID)
			if problem == nil {
				c.signalTransfer(requestID)
			}
			c.mu.Lock()
			again := running.again && problem == nil && !c.closing
			if !again {
				delete(c.sourceUploads, requestID)
			}
			closing := c.closing
			c.mu.Unlock()
			if again {
				continue
			}
			if problem != nil && !closing {
				if permanentTransferFailure(problem) && problem.ErrName() != "model_transfer.source_checkpoint_superseded" {
					transfer, _ := c.opt.Store.ModelTransferOf(requestID)
					if transfer != nil && transfer.State != "completed" && transfer.State != "canceled" {
						_ = c.opt.Store.FailModelTransfer(requestID, problem.ErrName(), problem.Message)
						c.signalTransfer(requestID)
					}
					c.logf("model transfer %s: source checkpoint custody refused (%s): %s", requestID, problem.ErrName(), problem.Message)
					return
				}
				c.logf("model transfer %s: source checkpoint custody pending (%s): %s",
					requestID, problem.ErrName(), problem.Message)
				time.AfterFunc(2*time.Second, func() { c.kickSourceCheckpointUpload(requestID) })
			}
			return
		}
	}()
}

func (c *Orchestrator) syncSourceCheckpoint(ctx context.Context, requestID string) *exit.Error {
	transfer, problem := c.opt.Store.ModelTransferOf(requestID)
	if problem != nil || transfer == nil {
		return problem
	}
	if transfer.State == "completed" || transfer.State == "canceled" {
		return c.opt.ModelTransfers.ReleaseSourceCheckpoints(ctx, requestID)
	}
	if transfer.State == "failed" {
		return nil // failed work retains its recovery holds until explicit cancellation
	}
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || request == nil {
		return problem
	}
	host, problem := c.sourceCheckpointHost(*request)
	if problem != nil {
		return problem
	}
	return c.opt.ModelTransfers.SyncSourceCheckpoints(ctx, requestID, host)
}
