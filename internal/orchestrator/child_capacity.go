package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// jobExecutionRole can place a resource-free caller in the separate CPU slot.
// Captured dependencies make child calls available; they do not require delegation
// or prevent an ordinary job from using Models, weights outputs, or a device.
func (c *Orchestrator) jobExecutionRole(req records.Request, spec WorkerLaunchSpec) (WorkerLaunchSpec, *exit.Error) {
	if !spec.IsJob() {
		return spec, nil
	}
	plans := append([]*JobPlan(nil), spec.Placement.Jobs...)
	plan := *plans[0]
	plans[0] = &plan
	spec.Placement.Jobs = plans
	// The composition PARENT takes the CPU slot; a captured callee of the same install
	// is an ordinary job, which is what a package awaiting its own sibling requires.
	bound, problem := c.opt.Store.CompositionParent(req.InstallID, req.Entrypoint)
	if problem != nil {
		return spec, problem
	}
	plan.Orchestration = false
	plan.OrchestrationParent = nil
	if bound && req.RetainWork && !req.NeedsAccelerator && len(req.OwnModels()) == 0 && len(plan.WeightsOutputs) == 0 {
		plan.Orchestration = req.Worker != ""
		if plan.Orchestration {
			raw, _, err := canonical.Identity(c.jobDirective(&plan))
			if err != nil {
				return spec, exit.Internalf("cannot encode CPU parent contract: %s", err)
			}
			if problem := c.opt.Store.CaptureOrchestrationDirective(req.ID, raw); problem != nil {
				return spec, problem
			}
		}
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
	var directive pb.JobDirective
	if canonical.Unmarshal(parent.OrchestrationDirective, &directive) != nil || !directive.Orchestration || directive.OrchestrationParent != nil || directive.ResourceCaps.GetDeviceRequired() || directive.DeviceCount != 0 || directive.JobDescriptorId != parent.PlanID || directive.InstallationId != doc.Sub("job").Str("installation_id") {
		return nil, exit.Named(exit.Conflict, "child.parent_slot_unavailable", "the rental has no exact retained CPU orchestration parent")
	}
	return &JobPlan{Function: parent.Entrypoint, DescriptorID: directive.JobDescriptorId, InstallationID: directive.InstallationId, Orchestration: true, FrozenDirective: &directive}, nil
}
