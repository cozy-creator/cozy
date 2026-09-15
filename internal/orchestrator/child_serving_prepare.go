package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

// A modeled child must not activate the intermediate code-only PlacementSet.
// Prepare code, join its exact model inputs, then publish one complete placement
// while retaining the CPU parent whose canonical invocation authorized this call.
func (c *Orchestrator) prepareChildServing(instance string, req records.Request, revision localpackage.Revision) *exit.Error {
	_, current, problem := c.localControl(instance)
	if problem != nil {
		return problem
	}
	if problem := requireAdapterPeer(current, hasModelAdapters(req.Models)); problem != nil {
		return problem
	}
	if problem := requireMixedModelInputs(current, mixedModelInputs(req.Models)); problem != nil {
		return problem
	}
	prepared, problem := c.prepareUnpublishedPackage(instance, req, revision, false)
	if problem != nil {
		return problem
	}
	w, s, problem := c.localControl(instance)
	if problem != nil {
		return problem
	}
	w.localMu.Lock()
	defer w.localMu.Unlock()
	parent, problem := c.activeParentFor(req)
	if problem != nil || parent == nil {
		return exit.Unavailablef("child serving preparation lost its active CPU parent")
	}
	c.mu.Lock()
	problem = c.rentalPreparationAllowedLocked(w, req)
	c.mu.Unlock()
	if problem != nil {
		return problem
	}
	var selected *pb.DesiredPrivatePlacementSet
	if len(req.Models) > 0 {
		native, problem := c.nativeServingModels(req)
		if problem != nil {
			return problem
		}
		digest, err := canonical.Raw(revision.Digest)
		if err != nil {
			return exit.Internalf("child code revision is not canonical")
		}
		var downloadSet []byte
		if models := downloadModelRefs(req.Models); len(models) > 0 {
			if c.opt.RentalPackageSet == nil {
				return exit.Named(exit.Unavailable, "rental.package_set_signer_missing", "this Cozy daemon has no package_set signer")
			}
			downloadSet, problem = c.opt.RentalPackageSet(nil, models)
			if problem != nil {
				return problem
			}
		}
		selected = &pb.DesiredPrivatePlacementSet{OperationId: req.ID, LocalRevisionDigest: digest, NativeModels: native, DownloadDelegation: downloadSet}
	}
	rev := c.nextRevision()
	c.mu.Lock()
	w.hostPrepareSeq++
	seq := w.hostPrepareSeq
	w.revision, w.desiredRefusal, w.orchestrationParent = rev, nil, parent
	w.desiredLocal, w.desiredPackages, w.desiredModels, w.desiredDownloadSets = nil, nil, nil, nil
	w.desiredUnpublishedPlacement, w.desiredEpoch = cloneUnpublishedPlacementSet(selected), s.epoch
	c.mu.Unlock()
	label := hostLabel("child_serving", req.ID)
	if selected != nil {
		call := &pb.PreparePrivatePlacementCall{SupportsModelMaterializationRecovery: true, Claim: s.claim, PrivatePlacementSet: selected}
		result := c.runHostPrepare(s, w, seq, label,
			func(ctx context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
				return s.host.PreparePrivatePlacement(ctx, call)
			})
		if !c.settleHostPrepare(s, w, seq, label, result) {
			if result.fault != nil {
				return result.fault
			}
			if result.refusal != "" {
				return exit.Named(exit.Structural, "worker.prepare_refused", "%s", result.refusal)
			}
			return exit.Unavailablef("child model preparation awaits its exact result")
		}
		prepared = result.set
	}
	c.convergePrepared(s, w, seq, rev, label, prepared)
	return nil
}
