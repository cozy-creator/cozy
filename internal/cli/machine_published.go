package cli

import (
	"bytes"
	"context"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
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
		return nil, exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "published machine execution requires actual Runtime protocol 54")
	}
	var iface *launch.PackageInterface
	if request.Rental {
		ref, problem := hub.ParseRef(request.Package)
		if problem != nil {
			return nil, problem
		}
		detail, problem := m.resolver.catalog.PackageRelease(ctx, ref, request.Release)
		if problem != nil {
			return nil, problem
		}
		if _, problem := detail.Requirements(); problem != nil {
			return nil, problem
		}
		if detail.Release.Release != request.Release {
			return nil, exit.New(exit.Conflict, "published machine root changed its release")
		}
		iface, problem = launch.DecodePackageInterface(detail.PackageInterface)
		if problem != nil {
			return nil, problem
		}
	} else {
		facts, problem := m.resolver.installFacts(request.InstallID)
		if problem != nil {
			return nil, problem
		}
		if facts.Install.SourceKind != "tensorhub" || facts.Install.Package != request.Package || facts.Install.Version != request.Release {
			return nil, exit.New(exit.Conflict, "published local root changed its immutable install")
		}
		iface = facts.PackageInterface
	}
	job, problem := iface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	if job.Kind != "job" || job.DescriptorID != request.PlanID {
		return nil, exit.New(exit.Conflict, "published machine root changed its captured declaration")
	}
	prepared, problem := connection.preparePublished(ctx, request)
	if problem != nil {
		return nil, problem
	}
	var set pb.PlacementSet
	if err := canonical.Unmarshal(prepared.PlacementSetCanonicalBytes, &set); err != nil || len(set.Placements) != 1 {
		return nil, exit.New(exit.Conflict, "published preparation requires exactly one placement")
	}
	placement := set.Placements[0]
	if placement.GetPackage().GetPackage() != request.Package || placement.GetPackage().GetRelease() != request.Release || placement.GetDevelopment() != nil || placement.Environment == nil || placement.PackageInterface == nil {
		return nil, exit.New(exit.Conflict, "published preparation changed its package selection")
	}
	_, codeDigest, err := canonical.Identity(placement.Environment)
	if err != nil || !bytes.Equal(codeDigest, placement.EnvironmentDigest) || !bytes.Equal(placement.PackageInterface.Digest, canonical.Digest(iface.Raw)) || placement.PackageInterface.Length != uint64(len(iface.Raw)) {
		return nil, exit.New(exit.Conflict, "published preparation differs from its immutable catalog release")
	}
	if problem := m.resolver.captureMachineInterface(request, iface); problem != nil {
		return nil, problem
	}
	capture := &pb.MachineExecutionCapture{
		RootRevisionDigest: codeDigest,
		PublishedRevisions: []*pb.PublishedPackageRevision{{Package: placement.GetPackage(), Environment: placement.Environment, PackageInterface: placement.PackageInterface}},
	}
	for _, callable := range append(append([]launch.Entrypoint(nil), iface.Jobs...), iface.Entrypoints...) {
		if callable.Invocable == nil {
			continue
		}
		capture.Bindings = append(capture.Bindings, &pb.MachineCallableBinding{
			CallerRevisionDigest: codeDigest, CalleeRevisionDigest: codeDigest,
			InterfaceDigest: placement.PackageInterface.Digest,
			Module:          callable.Invocable.Module, Export: callable.Invocable.Export, Entrypoint: callable.Name,
		})
	}
	sort.Slice(capture.Bindings, func(i, j int) bool {
		a, b := capture.Bindings[i], capture.Bindings[j]
		if a.Module != b.Module {
			return a.Module < b.Module
		}
		return a.Export < b.Export
	})
	if connection.wireMinor >= pb.CapturedModelDefaultsWireMinor {
		m.resolver.captureDefaultRows(capture, request.Package, codeDigest, iface, request.Rental)
	}
	raw, digest, err := canonical.Identity(capture)
	if err != nil {
		return nil, exit.Internalf("cannot encode published execution capture: %s", err)
	}
	buildID, err := canonical.Spell(codeDigest)
	if err != nil {
		return nil, exit.New(exit.Conflict, "published preparation has no Environment identity")
	}
	plan := &orchestrator.JobPlan{
		Function: request.Entrypoint, DescriptorID: request.PlanID, BuildID: buildID,
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
