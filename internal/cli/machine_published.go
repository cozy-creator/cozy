package cli

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Every machine prepares a published release the same way, and a job or an inference root
// is submitted against that preparation.
func (m *machineRuns) publishedSubmission(ctx context.Context, request records.Request, connection *machineConnection) (*pb.MachineExecutionSubmit, *exit.Error) {
	if request.Rental && request.Worker == "" {
		return nil, exit.New(exit.Conflict, "published machine execution requires a pinned rental")
	}
	began := time.Now()
	prepared, problem := connection.preparePublished(ctx, request)
	if problem != nil {
		return nil, problem
	}
	m.submissionStage(request.ID, "package_preparation", preparedDetail(request.Package, request.Release, prepared), began)
	installed := prepared.InstalledPackage
	if installed == nil || installed.InstallationId == "" || installed.Package != request.Package || installed.Release != request.Release {
		return nil, exit.New(exit.Conflict, "published preparation returned another installed package")
	}
	iface, problem := launch.DecodePackageInterface(installed.PackageInterface)
	if problem != nil {
		return nil, problem
	}
	job, problem := iface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	if (job.Kind == "job") != request.IsJob() {
		return nil, exit.New(exit.Conflict, "installed callable changed its kind")
	}
	if job.Kind == "job" {
		request.PlanID = job.DescriptorID
	} else if request, problem = m.bindServingPlan(request, installed.InstallationId, prepared.DesiredPlacementSet); problem != nil {
		return nil, problem
	}
	capture := &pb.MachineExecutionCapture{
		RootInstallationId: installed.InstallationId,
		InstalledPackages:  []*pb.InstalledPackage{installed},
	}
	addPublishedBindings(capture, captureName{ID: installed.InstallationId}, captureName{ID: installed.InstallationId}, iface, true)
	if problem := m.capturePublishedDependencies(ctx, request, connection, capture, prepared.LockedRequirements); problem != nil {
		return nil, problem
	}

	sort.Slice(capture.Bindings, func(i, j int) bool {
		a, b := capture.Bindings[i], capture.Bindings[j]
		if bindingCaller(a) != bindingCaller(b) {
			return bindingCaller(a) < bindingCaller(b)
		}
		if a.Module != b.Module {
			return a.Module < b.Module
		}
		return a.Export < b.Export
	})
	raw, digest, err := canonical.Identity(capture)
	if err != nil {
		return nil, exit.Internalf("cannot encode published execution capture: %s", err)
	}
	began = time.Now()
	byteInputs, problem := m.stageMachineInputs(ctx, request, connection)
	if problem != nil {
		return nil, problem
	}
	if len(byteInputs) > 0 {
		m.submissionStage(request.ID, "inputs", fmt.Sprintf("%d input(s)", len(byteInputs)), began)
	}
	frozen := localpackage.ExecutionCapture{Canonical: raw, Digest: digest}
	if !request.IsJob() {
		return orchestrator.MachineServingSubmission(request, frozen, prepared.DesiredPlacementSet, byteInputs)
	}
	plan, problem := machineJobPlan(ctx, connection, request, installed.InstallationId, job)
	if problem != nil {
		return nil, problem
	}
	return orchestrator.MachineJobSubmission(request, frozen, plan, byteInputs)
}

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
	workspace, problem := currentExecutionWorkspace(ctx, connection)
	if problem != nil {
		return nil, problem
	}
	plan := &orchestrator.JobPlan{
		Function: request.Entrypoint, DescriptorID: job.DescriptorID, InstallationID: installationID,
		Outputs: launch.AssetPaths(job.Result), NeedsAccelerator: request.NeedsAccelerator,
		RSSCap: orchestrator.DefaultJobRSSCap, AcceleratorDeclared: job.Accelerator != nil,
		CPUSlotModelInputs: workspace.CpuSlotModelInputs,
	}
	for _, output := range job.WeightsOutputs {
		plan.Outputs = append(plan.Outputs, output.OutputID)
		plan.WeightsOutputs = append(plan.WeightsOutputs, orchestrator.WeightsOutput{OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes})
	}
	return plan, nil
}
