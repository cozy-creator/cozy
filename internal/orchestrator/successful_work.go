package orchestrator

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// New no-artifact roots are armed at admission. An absent/old schema or an
// explicitly returned artifact keeps the existing conservative retention path.
func (c *Orchestrator) beginSuccessfulWorkRelease(req records.Request, ordinal int64) bool {
	if req.ParentRequestID != "" || req.State != "succeeded" {
		return false
	}
	intent, problem := c.opt.Store.SuccessfulWorkRelease(req.ID)
	if problem != nil || intent == nil || intent.State == "complete" || intent.State == "deferred" {
		return false
	}
	a, problem := c.opt.Store.AttemptRow(req.ID, ordinal)
	if problem != nil || a == nil || a.State != "closed" || a.TerminalStatus != "SUCCEEDED" {
		return false
	}
	doc, err := canonical.Read(a.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil || req.RetainsLocalOutputs() || len(doc.Sub("output_manifest").List("outputs")) != 0 || len(doc.List("weights_receipts")) != 0 {
		_ = c.opt.Store.DeferSuccessfulWorkRelease(req.ID, "returned or unverified output custody")
		return false
	}
	started, problem := c.opt.Store.BeginSuccessfulWorkRelease(req, *a)
	if problem != nil {
		c.logf("request %s successful cleanup intent: %s", req.ID, problem.Message)
		return false
	}
	return started
}

func (c *Orchestrator) finishSuccessfulWorkRelease(id string) {
	c.startRetainedCleanup(id, c.runSuccessfulWorkRelease)
}

func (c *Orchestrator) retrySuccessfulWorkRelease(id string) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if !closing {
		time.AfterFunc(ReportCadence, func() { c.finishSuccessfulWorkRelease(id) })
	}
}

func (c *Orchestrator) runSuccessfulWorkRelease(ctx context.Context, id string) {
	root, problem := c.opt.Store.RequestRow(id)
	if problem != nil || root == nil {
		return
	}
	if root.State != "succeeded" {
		if root.State == "canceling" || root.State == "releasing" {
			c.retryRetainedCancellation(id)
		}
		return
	}
	intent, problem := c.opt.Store.SuccessfulWorkRelease(id)
	if problem != nil || intent == nil {
		return
	}
	if intent.State == "armed" {
		if !c.beginSuccessfulWorkRelease(*root, root.Ordinal) {
			return
		}
		intent, problem = c.opt.Store.SuccessfulWorkRelease(id)
		if problem != nil || intent == nil {
			return
		}
	}
	if intent.State != "draining" && intent.State != "release_work" {
		return
	}
	a, problem := c.opt.Store.AttemptRow(id, intent.Attempt)
	if problem != nil || a == nil || a.State != "closed" || a.TerminalStatus != "SUCCEEDED" ||
		root.Ordinal != intent.Attempt || root.BodyDigest != intent.BodyDigest || root.PlanID != intent.PlanID ||
		a.InvocationDigest != intent.Invocation || a.TerminalID != intent.TerminalID || a.TerminalDigest != intent.TerminalDigest {
		return
	}
	members, problem := c.opt.Store.SuccessfulWorkFamily(id)
	if problem != nil {
		c.retrySuccessfulWorkRelease(id)
		return
	}
	if intent.State == "draining" {
		// A caught child failure does not make that child's unfinished work unused.
		// Defer this first slice rather than abandoning a failed/paused descendant.
		for _, member := range members {
			if member.State != "succeeded" || member.ModelTransfer != nil {
				_ = c.opt.Store.DeferSuccessfulWorkRelease(id, "retained or unsupported descendant")
				c.afterAck(*root, *a, nil)
				return
			}
			attempts, e := c.opt.Store.Attempts(member.ID)
			if e != nil {
				c.retrySuccessfulWorkRelease(id)
				return
			}
			for _, attempt := range attempts {
				if openAttempt(attempt.State) || attempt.State == "preparing" {
					c.retrySuccessfulWorkRelease(id)
					return
				}
			}
			pending, e := c.opt.Store.PendingArtifactBorrowers(member.ID)
			if e != nil || pending {
				c.retrySuccessfulWorkRelease(id)
				return
			}
			if _, e := c.lookupOperationPending(ctx, member, true); e != nil {
				c.retrySuccessfulWorkRelease(id)
				return
			}
		}
		for _, member := range members {
			if ctx.Err() != nil {
				return
			}
			if e := c.releaseChildRetentions(ctx, member.ID, false); e != nil {
				c.retrySuccessfulWorkRelease(id)
				return
			}
			attempts, e := c.opt.Store.Attempts(member.ID)
			if e != nil {
				c.retrySuccessfulWorkRelease(id)
				return
			}
			ready, e := c.settleRetainedWeights(member, attempts, id)
			if e != nil || !ready {
				c.retrySuccessfulWorkRelease(id)
				return
			}
			if e := c.releaseOriginalDerivedResults(ctx, member.ID); e != nil {
				c.retrySuccessfulWorkRelease(id)
				return
			}
		}
		ready, e := c.opt.Store.ReadySuccessfulWorkRelease(id, members)
		if e != nil || !ready {
			c.retrySuccessfulWorkRelease(id)
			return
		}
	}
	// release_work is durable before the first false ACK; normal snapshot recovery
	// also observes retain_work=false and replays that exact terminal disposition.
	for _, member := range members {
		attempts, e := c.opt.Store.Attempts(member.ID)
		if e != nil {
			c.retrySuccessfulWorkRelease(id)
			return
		}
		if e := c.ackReleasedRetainedAttempts(ctx, member, attempts); e != nil {
			c.retrySuccessfulWorkRelease(id)
			return
		}
	}
	if e := c.opt.Store.CompleteSuccessfulWorkRelease(id); e != nil {
		c.retrySuccessfulWorkRelease(id)
		return
	}
	if current, e := c.opt.Store.RequestRow(id); e == nil && current != nil {
		c.afterAck(*current, *a, nil)
	}
}
