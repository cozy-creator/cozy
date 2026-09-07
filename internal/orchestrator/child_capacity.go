package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// jobExecutionRole assigns the separate CPU orchestration slot from captured
// dependency facts. A package never chooses this role in its invocation payload.
func (c *Orchestrator) jobExecutionRole(req records.Request, spec WorkerLaunchSpec) (WorkerLaunchSpec, *exit.Error) {
	if !spec.IsJob() {
		return spec, nil
	}
	plans := append([]*JobPlan(nil), spec.Placement.Jobs...)
	plan := *plans[0]
	plans[0] = &plan
	spec.Placement.Jobs = plans
	bound, problem := c.opt.Store.HasChildBindings(req.InstallID)
	if problem != nil {
		return spec, problem
	}
	plan.Orchestration = false
	plan.OrchestrationParent = nil
	if bound {
		if !req.RetainWork || req.NeedsAccelerator || len(req.Models) > 0 || len(plan.WeightsOutputs) > 0 {
			return spec, exit.Named(exit.Structural, "child.orchestration_resources", "an invocable composition must keep its parent CPU-only and delegate model work to children")
		}
		plan.Orchestration = req.Worker != ""
	}
	if req.Worker != "" && req.ParentRequestID != "" {
		if plan.Orchestration {
			return spec, exit.Named(exit.Unavailable, "child.nested_orchestration", "one rented worker supports one CPU parent and one ordinary child")
		}
		parent, problem := c.retainedOrchestrationParent(req)
		if problem != nil {
			return spec, problem
		}
		plan.OrchestrationParent = parent
	}
	return spec, nil
}

func (c *Orchestrator) retainedOrchestrationParent(child records.Request) (*JobPlan, *exit.Error) {
	parent, problem := c.opt.Store.RequestRow(child.ParentRequestID)
	if problem != nil || parent == nil || parent.State != "dispatching" || parent.Worker != child.Worker {
		return nil, exit.Named(exit.Conflict, "child.parent_stopped", "child preparation no longer has its active parent on this rental")
	}
	attempt, problem := c.opt.Store.AttemptRow(parent.ID, parent.Ordinal)
	if problem != nil || attempt == nil || !openAttempt(attempt.State) {
		return nil, exit.Named(exit.Conflict, "child.parent_stopped", "child preparation has no open parent attempt")
	}
	doc, err := canonical.Read(attempt.InvocationCanonical, &pb.InvocationSpec{})
	if err != nil {
		return nil, exit.Internalf("the parent invocation is not canonical")
	}
	c.mu.Lock()
	worker := c.workers[rentalInstanceID(child.Worker)]
	var plan *JobPlan
	if worker != nil && worker.spec.IsJob() {
		plan = worker.spec.Placement.Jobs[0]
		if !plan.Orchestration {
			plan = plan.OrchestrationParent
		}
		if plan != nil {
			copy := *plan
			plan = &copy
		}
	}
	c.mu.Unlock()
	if plan == nil || !plan.Orchestration || plan.OrchestrationParent != nil || plan.NeedsAccelerator || plan.DescriptorID != parent.PlanID || plan.BuildID != doc.Sub("job").Str("build_id") {
		return nil, exit.Named(exit.Conflict, "child.parent_slot_unavailable", "the rental has no exact retained CPU orchestration parent")
	}
	return plan, nil
}
