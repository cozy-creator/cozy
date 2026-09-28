package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

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
