package orchestrator

import (
	"context"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

type checkpointUpload struct {
	again   bool
	intents map[string]*pb.WeightsTransactionStatus
}

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
		if transfer.State == "canceling" || transfer.State == "canceled" || current.State == "canceled" {
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
			if !current.RetainWork {
				ready = ready && slot.Acknowledged != nil && *slot.Acknowledged == slot.Observed
			}
		}
		if ready {
			return nil
		}
		c.ObservePhase(req.ID, PhaseSample{Name: PhasePreparing, Detail: "retaining converted source checkpoints in Tensorhub",
			HasBytes: true, Moved: held, Total: total})
		c.kickCheckpointUpload(req.ID)
		if problem := c.waitTransfer(context.Background(), req.ID); problem != nil {
			return problem
		}
	}
}

// A single uploader follows the durable observed heads while conversion and
// source downloads continue. Coalescing wakeups never coalesces custody facts.
func (c *Orchestrator) kickCheckpointUpload(requestID string, statuses ...*pb.WeightsTransactionStatus) {
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return
	}
	running, exists := c.checkpointUploads[requestID]
	if !exists {
		running = &checkpointUpload{intents: make(map[string]*pb.WeightsTransactionStatus)}
		c.checkpointUploads[requestID] = running
	}
	for _, row := range statuses {
		running.intents[row.OutputSlot] = row
	}
	if exists {
		running.again = true
		c.mu.Unlock()
		return
	}
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
			intents := running.intents
			running.intents = make(map[string]*pb.WeightsTransactionStatus)
			c.mu.Unlock()
			problem := c.syncCheckpointUploads(ctx, requestID)
			if problem == nil {
				for _, intent := range intents {
					if intent.IntentReady {
						continue
					}
					if problem = c.readyWeightsCheckpoint(ctx, requestID, intent); problem != nil {
						break
					}
				}
			}
			if problem == nil {
				c.signalTransfer(requestID)
			}
			c.mu.Lock()
			if problem != nil {
				for slot, intent := range intents {
					if running.intents[slot] == nil {
						running.intents[slot] = intent
					}
				}
			}
			again := running.again && !c.closing
			if !again {
				delete(c.checkpointUploads, requestID)
			}
			closing := c.closing
			c.mu.Unlock()
			if again {
				continue
			}
			if problem != nil && !closing {
				if permanentTransferFailure(problem) && problem.ErrName() != "model_transfer.source_checkpoint_superseded" && problem.ErrName() != "model_transfer.checkpoint_superseded" {
					transfer, _ := c.opt.Store.ModelTransferOf(requestID)
					if transfer != nil && transfer.State != "completed" && transfer.State != "canceling" && transfer.State != "canceled" {
						_ = c.opt.Store.FailModelTransfer(requestID, problem.ErrName(), problem.Message)
						c.signalTransfer(requestID)
					}
					c.logf("model transfer %s: source checkpoint custody refused (%s): %s", requestID, problem.ErrName(), problem.Message)
					return
				}
				c.logf("model transfer %s: source checkpoint custody pending (%s): %s",
					requestID, problem.ErrName(), problem.Message)
				time.AfterFunc(2*time.Second, func() {
					rows := make([]*pb.WeightsTransactionStatus, 0, len(running.intents))
					for _, intent := range running.intents {
						rows = append(rows, intent)
					}
					c.kickCheckpointUpload(requestID, rows...)
				})
			}
			return
		}
	}()
}

func (c *Orchestrator) syncCheckpointUploads(ctx context.Context, requestID string) *exit.Error {
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || request == nil {
		return problem
	}
	// Private retained work owns bytes on its selected machine. Publication of
	// final outputs is a separate explicit operation; progress never implies an
	// upload to Tensorhub, including source and weights checkpoints.
	if request.RetainWork {
		return nil
	}
	transfer, problem := c.opt.Store.ModelTransferOf(requestID)
	if problem != nil || transfer == nil {
		return problem
	}
	if c.opt.ModelTransfers == nil {
		return nil
	}
	if transfer.State == "completed" || transfer.State == "canceled" || transfer.State == "canceling" {
		return c.opt.ModelTransfers.ReleaseCheckpoints(ctx, requestID)
	}
	if transfer.State == "failed" {
		return nil // failed work retains its recovery holds until explicit cancellation
	}
	host, problem := c.checkpointHost(*request)
	if problem != nil {
		return problem
	}
	return c.opt.ModelTransfers.SyncCheckpoints(ctx, requestID, host)
}
