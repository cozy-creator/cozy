package cli

import (
	"context"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Local installed code and rented machines use the same execution path: the machine
// prepares the release, and a job or an inference root is submitted against that preparation.
func (m *machineRuns) publishedSubmission(ctx context.Context, request records.Request, connection *machineConnection) (*pb.MachineExecutionSubmit, *exit.Error) {
	if (request.Rental && request.Worker == "") || connection.preparePublished == nil {
		return nil, exit.New(exit.Conflict, "published machine execution requires a local install or pinned rental")
	}
	if connection.wireMinor < pb.PublishedMachineCaptureWireMinor {
		return nil, exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "published machine execution requires the current installed-package protocol")
	}
	prepared, problem := connection.preparePublished(ctx, request)
	if problem != nil {
		return nil, problem
	}
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
	addPublishedBindings(capture, installed.InstallationId, installed.InstallationId, iface, true)
	if problem := m.capturePublishedDependencies(ctx, request, connection, capture, prepared.LockedRequirements); problem != nil {
		return nil, problem
	}

	sort.Slice(capture.Bindings, func(i, j int) bool {
		a, b := capture.Bindings[i], capture.Bindings[j]
		if a.CallerInstallationId != b.CallerInstallationId {
			return a.CallerInstallationId < b.CallerInstallationId
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
	byteInputs, problem := m.stageMachineInputs(ctx, request, connection)
	if problem != nil {
		return nil, problem
	}
	frozen := localpackage.ExecutionCapture{Canonical: raw, Digest: digest}
	if !request.IsJob() {
		return orchestrator.MachineServingSubmission(request, frozen, prepared.DesiredPlacementSet, byteInputs)
	}
	return orchestrator.MachineJobSubmission(request, frozen, machineJobPlan(request, installed.InstallationId, job), byteInputs)
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
func machineJobPlan(request records.Request, installationID string, job *launch.Entrypoint) *orchestrator.JobPlan {
	plan := &orchestrator.JobPlan{
		Function: request.Entrypoint, DescriptorID: job.DescriptorID, InstallationID: installationID,
		Outputs: launch.AssetPaths(job.Result), NeedsAccelerator: request.NeedsAccelerator,
		RSSCap: orchestrator.DefaultJobRSSCap,
	}
	for _, output := range job.WeightsOutputs {
		plan.Outputs = append(plan.Outputs, output.OutputID)
		plan.WeightsOutputs = append(plan.WeightsOutputs, orchestrator.WeightsOutput{OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes})
	}
	return plan
}
