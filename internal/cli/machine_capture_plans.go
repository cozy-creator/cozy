package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// bindServingPlan records the binding the machine authored for the request's callable in
// the placement it prepared: an inference root is submitted against exactly that binding.
func (m *machineRuns) bindServingPlan(request records.Request, installationID string, prepared *pb.DesiredPlacementSet) (records.Request, *exit.Error) {
	if validateMachinePrepared(prepared) != nil {
		return request, exit.New(exit.Conflict, "machine preparation returned no verified placement for inference")
	}
	_, planID, problem := orchestrator.ServingPlacement(prepared.PlacementSetCanonicalBytes, installationID, request)
	if problem != nil {
		return request, problem
	}
	if planID != request.PlanID {
		if problem := m.store.BindRequestPlan(request.ID, planID); problem != nil {
			return request, problem
		}
		request.PlanID = planID
	}
	return request, nil
}

// Both published and synced-source jobs use the executing installation's
// declaration. The client's SDK never supplies a competing root descriptor.
func machineJobPlan(ctx context.Context, connection *machineConnection, request records.Request, installationID string, job *launch.Entrypoint) (*orchestrator.JobPlan, *exit.Error) {
	plan := &orchestrator.JobPlan{
		Function: request.Entrypoint, DescriptorID: job.DescriptorID, InstallationID: installationID,
		Outputs: launch.AssetPaths(job.Result), NeedsAccelerator: request.NeedsAccelerator,
		RSSCap: orchestrator.DefaultJobRSSCap, AcceleratorDeclared: job.Accelerator != nil,
		CPUSlotModelInputs: true,
	}
	for _, output := range job.WeightsOutputs {
		plan.Outputs = append(plan.Outputs, output.OutputID)
		plan.WeightsOutputs = append(plan.WeightsOutputs, orchestrator.WeightsOutput{OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes})
	}
	return plan, nil
}
