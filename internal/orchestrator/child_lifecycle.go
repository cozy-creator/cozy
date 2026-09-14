package orchestrator

import (
	"context"
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

func (c *Orchestrator) cancelChildCalls(root, parent string) *exit.Error {
	children, problem := c.opt.Store.Children(parent)
	if problem != nil {
		return problem
	}
	for _, child := range children {
		if child.State == "succeeded" {
			// The parent abandons its remaining custody, not a completed result.
			// Still stop unfinished grandchildren of a successful caller.
			if problem := c.cancelChildCalls(root, child.ID); problem != nil {
				return problem
			}
			continue
		}
		retaining, problem := c.opt.Store.RequestRetaining(child)
		if problem != nil {
			return problem
		}
		if records.Settled(child.State) && !retaining {
			continue
		}
		if child.State == "canceling" || child.State == "releasing" {
			c.finishRetainedCancellation(child.ID)
			continue
		}
		changed, problem := c.opt.Store.RequestDescendantCancellation(root, child.ID, "parent transaction abandoned")
		if problem != nil {
			return problem
		}
		if changed {
			if problem := c.finishRequestedCancellation(child.ID); problem != nil {
				return problem
			}
		}
	}
	return nil
}

// The cancelled root is the existing durable cleanup intent. A restart walks the
// same immutable child edges and resumes native releases before clearing custody.
func (c *Orchestrator) releaseCompletedChildCalls(ctx context.Context, root, parent string) (bool, *exit.Error) {
	children, problem := c.opt.Store.Children(parent)
	if problem != nil {
		return false, problem
	}
	for _, child := range children {
		if !records.Settled(child.State) {
			return false, nil
		}
		ready, problem := c.releaseCompletedChildCalls(ctx, root, child.ID)
		if problem != nil || !ready {
			return false, problem
		}
		if child.State != "succeeded" {
			continue
		}
		attempts, problem := c.opt.Store.Attempts(child.ID)
		if problem != nil {
			return false, problem
		}
		if child.RetainWork {
			ready, problem := c.releaseAbandonedWork(ctx, child, attempts, root)
			if problem != nil || !ready {
				return false, problem
			}
			ready, problem = c.opt.Store.ReleaseCompletedChildWork(root, child.ID)
			if problem != nil || !ready {
				return false, problem
			}
		}
		// retain_work=false is durable before ACK(false). Snapshot reconciliation
		// also replays that disposition if the process dies between these calls.
		if problem := c.ackReleasedRetainedAttempts(ctx, child, attempts); problem != nil {
			return false, problem
		}
		child.RetainWork = false
		c.cleanupRequestAssets(child)
		c.reclaimTmp(child.ID)
	}
	return true, nil
}
