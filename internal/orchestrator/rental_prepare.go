package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

// PrepareRentalPackage waits for the worker's verified preparation receipt.
// It installs code and only the explicitly selected models; it does not activate
// a serving placement or replace other packages already loaded on the worker.
func (c *Orchestrator) PrepareRentalPackage(ctx context.Context, instance string, ref *pb.DownloadPackageRef, models []*pb.DownloadModelRef) *exit.Error {
	if ref == nil || c.opt.RentalPackageSet == nil || c.opt.RentalPrepareFacts == nil {
		return exit.New(exit.Unavailable, "rental preparation has no package download authority")
	}
	w, s, problem := c.localControlContext(ctx, instance)
	if problem != nil {
		return problem
	}
	downloads, problem := c.opt.RentalPackageSet([]*pb.DownloadPackageRef{ref}, models)
	if problem != nil {
		return problem
	}
	facts, problem := c.opt.RentalPrepareFacts(ctx, w.spec.Connection, ref)
	if problem != nil {
		return problem
	}
	if problem := requireAdapterDownloadPeer(s, downloads); problem != nil {
		return problem
	}
	call := &pb.PreparePackageSetCall{
		Claim: s.claim, SupportsModelMaterializationRecovery: true,
		PackageSet:  &pb.DesiredPackageSet{DownloadDelegation: downloads},
		Application: facts.Application, ModelSlotPaths: facts.ModelSlotPaths,
		ImageInventory: facts.ImageInventory, PythonRequires: facts.PythonRequires,
		PythonVersion: hostruntime.PythonMinor(facts.PythonVersion), LockedRequirements: facts.LockedRequirements,
		PackageInterface: facts.PackageInterface,
	}
	result := c.runHostPrepare(s, w, 0, hostLabel("explicit_prepare", ref.Package),
		func(context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
			return s.host.PreparePackageSet(ctx, call)
		})
	if result.fault != nil {
		return result.fault
	}
	if result.refusal != "" {
		return exit.Named(exit.Structural, "worker.prepare_refused", "%s", result.refusal).WithCause(result.code)
	}
	if result.err != nil {
		return exit.Unavailablef("worker model preparation was interrupted: %s; repeat the same preparation to resume", result.err)
	}
	if result.set == nil {
		return exit.Unavailablef("worker model preparation ended without its verified receipt; repeat the same preparation to resume")
	}
	return nil
}

// PrepareRentalModels fills the worker's managed TensorFS store without installing
// code or creating an inference placement. The Host verifies each full closure.
func (c *Orchestrator) PrepareRentalModels(ctx context.Context, instance string, models []*pb.DownloadModelRef) *exit.Error {
	if len(models) == 0 || c.opt.RentalPackageSet == nil {
		return exit.Usagef("model preparation requires exact model selections")
	}
	w, s, problem := c.localControlContext(ctx, instance)
	if problem != nil {
		return problem
	}
	downloads, problem := c.opt.RentalPackageSet(nil, models)
	if problem != nil {
		return problem
	}
	if problem := requireAdapterDownloadPeer(s, downloads); problem != nil {
		return problem
	}
	result := c.runHostPrepare(s, w, 0, "explicit_model_download", func(context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
		return s.host.PreparePackageSet(ctx, &pb.PreparePackageSetCall{Claim: s.claim, SupportsModelMaterializationRecovery: true, PackageSet: &pb.DesiredPackageSet{DownloadDelegation: downloads}})
	})
	if result.fault != nil {
		return result.fault
	}
	if result.refusal != "" {
		return exit.Named(exit.Structural, "worker.prepare_refused", "%s", result.refusal).WithCause(result.code)
	}
	if result.err != nil || result.set == nil {
		return exit.Unavailablef("worker model download ended without its verified receipt; the queued selection will retry")
	}
	return nil
}
