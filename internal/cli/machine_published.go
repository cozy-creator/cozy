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

// Local installed code and explicit rented machines use the same execution path.
// Rented roots with external placement keep their existing coordinator.
func (m *machineRuns) publishedSubmission(ctx context.Context, request records.Request, connection *machineConnection) (*pb.MachineExecutionSubmit, *exit.Error) {
	if (request.Rental && request.RequestedRental == "" && request.Worker == "") || connection.preparePublished == nil {
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
	if job.Kind != "job" {
		return nil, exit.New(exit.Conflict, "installed callable is not a job")
	}
	request.PlanID = job.DescriptorID
	if problem := m.resolver.captureMachineInterface(request, iface); problem != nil {
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
	if connection.wireMinor >= pb.CapturedModelDefaultsWireMinor {
		m.resolver.captureDefaultRows(capture, request.Package, installed.InstallationId, iface, request.Rental, connection.publicOrigin)
	}
	raw, digest, err := canonical.Identity(capture)
	if err != nil {
		return nil, exit.Internalf("cannot encode published execution capture: %s", err)
	}
	plan := &orchestrator.JobPlan{
		Function: request.Entrypoint, DescriptorID: request.PlanID, InstallationID: installed.InstallationId,
		Outputs: launch.AssetPaths(job.Result), NeedsAccelerator: request.NeedsAccelerator,
		RSSCap: orchestrator.DefaultJobRSSCap,
	}
	for _, output := range job.WeightsOutputs {
		plan.WeightsOutputs = append(plan.WeightsOutputs, orchestrator.WeightsOutput{OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes})
	}
	byteInputs, problem := m.stageMachineInputs(ctx, request, connection)
	if problem != nil {
		return nil, problem
	}
	return orchestrator.MachineJobSubmission(request, localpackage.ExecutionCapture{Canonical: raw, Digest: digest}, plan, byteInputs)
}
