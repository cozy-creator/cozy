package orchestrator

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

type sourceCheckpointUpload struct{ again bool }

// Conversion overlaps upload, but producer dispatch consumes only recoverable
// inputs. A refused producer therefore cannot interrupt unfinished source custody.
func (c *Orchestrator) awaitSourceInputCustody(req records.Request, bootID string) *exit.Error {
	for {
		transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
		if problem != nil {
			return problem
		}
		if transfer == nil {
			return exit.Internalf("source input custody has no model transfer owner")
		}
		current, problem := c.opt.Store.RequestRow(req.ID)
		if problem != nil {
			return problem
		}
		if current == nil || current.Worker != req.Worker {
			return exit.Unavailablef("source input placement changed before checkpoint custody was confirmed")
		}
		if transfer.State == "failed" {
			return exit.Named(exit.Failed, transfer.ErrorCode, "%s", transfer.SafeError)
		}
		if transfer.State == "canceled" || current.State == "canceled" {
			return exit.New(exit.Canceled, "model transfer %s was canceled", req.ID)
		}
		session, problem := c.rentalControl(req.Worker)
		if problem != nil {
			return problem
		}
		if session.bootID != bootID {
			return exit.Unavailablef("source input worker changed before checkpoint custody was confirmed")
		}
		progress, problem := c.opt.Store.ModelSourceProgress(req.ID)
		if problem != nil {
			return problem
		}
		if len(progress) != len(transfer.SourceProfiles) {
			return exit.Named(exit.Structural, "model_transfer.source_checkpoint_missing",
				"prepared source inputs did not report a recoverable checkpoint for every slot")
		}
		ready := true
		var held, total uint64
		for _, slot := range progress {
			if slot.WorkerBootID != bootID {
				return exit.Named(exit.Structural, "model_transfer.source_checkpoint_missing",
					"prepared source slot %s has no checkpoint observation from its current worker", slot.Observed.Slot)
			}
			total += uint64(slot.Observed.Bytes)
			if slot.Acknowledged != nil {
				held += uint64(slot.Acknowledged.Bytes)
			}
			ready = ready && slot.Acknowledged != nil && *slot.Acknowledged == slot.Observed
		}
		if ready {
			return nil
		}
		c.ObservePhase(req.ID, PhaseSample{Name: PhasePreparing, Detail: "retaining converted source checkpoints in Tensorhub",
			HasBytes: true, Moved: held, Total: total})
		c.kickSourceCheckpointUpload(req.ID)
		if problem := c.waitTransfer(context.Background(), req.ID); problem != nil {
			return problem
		}
	}
}

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
			again := running.again && !c.closing
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
	if !transfer.HasAcquisition() {
		return nil
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
