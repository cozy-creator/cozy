package orchestrator

import (
	"bytes"
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func (c *Orchestrator) controlRetainedSource(ctx context.Context, request records.Request, paused bool) *exit.Error {
	if request.ControlRevision == 0 {
		return nil
	}
	s, problem := c.rentalControl(request.Worker)
	if problem != nil {
		return problem
	}
	if s.host == nil || s.claim == nil {
		return exit.Unavailablef("source control awaits the claimed original Host")
	}
	selection, err := canonical.Raw(request.ModelTransfer.SourceSelection)
	if err != nil {
		return exit.Internalf("source control selection is malformed")
	}
	result, err := s.host.ModelSourceControl(ctx, &pb.ModelSourceControlCall{Claim: s.claim, OperationId: request.ID,
		SourceSelectionDigest: selection, ControlRevision: uint64(request.ControlRevision), Paused: paused})
	if err != nil {
		return exit.Unavailablef("source control awaits its original Host acknowledgement")
	}
	if result == nil || result.OperationId != request.ID || !bytes.Equal(result.SourceSelectionDigest, selection) || result.ControlRevision != uint64(request.ControlRevision) || result.Paused != paused {
		return exit.Named(exit.Structural, "request.source_control_changed", "source control acknowledgement changed its operation, selection, or revision")
	}
	return nil
}

// Begin one asynchronous drain. The CLI acknowledges pause intent immediately;
// the durable paused state waits for both Host and local orchestration to drain.
func (c *Orchestrator) pauseRetainedSource(request records.Request) {
	c.mu.Lock()
	if c.sourcePauseRunning[request.ID] || c.closing {
		c.mu.Unlock()
		return
	}
	c.sourcePauseRunning[request.ID] = true
	c.mu.Unlock()
	go func() {
		defer func() { c.mu.Lock(); delete(c.sourcePauseRunning, request.ID); c.mu.Unlock() }()
		s, problem := c.rentalControl(request.Worker)
		if problem != nil {
			return
		}
		if problem := c.controlRetainedSource(s.ctx, request, true); problem != nil {
			c.logf("request %s source pause pending: %s", request.ID, problem.Message)
			return
		}
		if problem := c.opt.Store.PauseModelSourceMaterialization(request.ID, request.ControlRevision); problem != nil {
			return
		}
		c.signalTransfer(request.ID)
		c.mu.Lock()
		preparing := c.transferDispatching[request.ID] || c.transferRunning[request.ID] || c.localTransfers[request.ID] != nil || c.checkpointUploads[request.ID] != nil
		c.mu.Unlock()
		if !preparing {
			_, _ = c.opt.Store.CompleteRequestPause(request.ID)
		}
	}()
}
