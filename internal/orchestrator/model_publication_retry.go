package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (c *Orchestrator) RetryModelTransferPublication(requestID, actor string) *exit.Error {
	if _, problem := c.opt.Store.RetryModelTransferPublication(requestID, actor); problem != nil {
		return problem
	}
	return c.resumeModelTransferPublication(requestID)
}

func (c *Orchestrator) resumeModelTransferPublication(requestID string) *exit.Error {
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil {
		return problem
	}
	if request == nil {
		return exit.New(exit.NotFound, "publication request is absent")
	}
	attempt, problem := c.opt.Store.AttemptRow(requestID, request.Ordinal)
	if problem != nil {
		return problem
	}
	if attempt == nil {
		return exit.New(exit.Conflict, "publication has no retained attempt")
	}
	if request.Worker == "" {
		c.kickRecoveredLocalTransfer(requestID, attempt.Attempt)
	} else {
		c.kickModelTransferFinalizer(requestID, attempt.Attempt)
		if _, problem := c.rentalControl(request.Worker); problem != nil {
			c.selectOrStart(*request)
		}
	}
	return nil
}

func (c *Orchestrator) publicationCompleteOrCanceled(requestID string) bool {
	transfer, problem := c.opt.Store.ModelTransferOf(requestID)
	return problem == nil && transfer != nil && (transfer.State == "completed" || transfer.State == "canceled")
}

func (c *Orchestrator) retainedPublication(request records.Request, attempt records.Attempt) bool {
	return request.ModelTransfer != nil && attempt.TerminalStatus == "SUCCEEDED" && !c.publicationCompleteOrCanceled(request.ID)
}
