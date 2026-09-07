package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (c *Orchestrator) pauseChildCalls(parent string) *exit.Error {
	children, problem := c.opt.Store.Children(parent)
	if problem != nil {
		return problem
	}
	for _, child := range children {
		if records.Settled(child.State) {
			continue
		}
		if child.State == "blocked" {
			if problem := c.pauseChildCalls(child.ID); problem != nil {
				return problem
			}
			continue
		}
		if child.State == "canceling" || child.State == "releasing" || child.State == "finalizing" {
			continue
		}
		if problem := c.PauseRequest(child.ID, "parent transaction stopped"); problem != nil {
			return problem
		}
	}
	return nil
}

func (c *Orchestrator) cancelChildCalls(parent string) *exit.Error {
	children, problem := c.opt.Store.Children(parent)
	if problem != nil {
		return problem
	}
	for _, child := range children {
		if records.Settled(child.State) && !(child.State == "succeeded" && child.RetainsLocalOutputs()) {
			continue
		}
		if problem := c.CancelRetainedRequest(child.ID, "parent transaction abandoned"); problem != nil {
			return problem
		}
	}
	return nil
}
