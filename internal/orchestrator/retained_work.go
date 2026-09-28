package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
)

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
