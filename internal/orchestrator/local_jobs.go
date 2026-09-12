package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

// A unpublished package job needs its prepared environment and job records. Preparing it
// must not activate a serving placement or remove a running orchestration parent.
func (c *Orchestrator) prepareLocalJob(instanceID string, request records.Request, revision localpackage.Revision) (*pb.DesiredPlacementSet, *exit.Error) {
	selected, transfer, problem := localSelection(request.ID, revision)
	if problem != nil {
		return nil, problem
	}
	w, s, problem := c.localControl(instanceID)
	if problem != nil {
		return nil, problem
	}
	w.localMu.Lock()
	defer w.localMu.Unlock()
	if request.LocalPackageUploadedBootID != s.bootID {
		if problem := c.transferLocalPackage(instanceID, request.ID, transfer); problem != nil {
			return nil, problem
		}
		w, s, problem = c.localControl(instanceID)
		if problem != nil {
			return nil, problem
		}
		if !c.localTransferVerified(request.ID, revision.Digest, s) {
			return nil, exit.Unavailablef("unpublished package job files await verification on the current worker")
		}
		if problem := c.opt.Store.MarkLocalPackageUploaded(request.ID, revision.Digest, s.bootID); problem != nil {
			return nil, problem
		}
	}
	if s.host == nil || s.claim == nil {
		return nil, exit.Unavailablef("unpublished package job preparation awaits the claimed Host")
	}
	if problem := requireLocalPackageCapacity(s, len(selected.Files)); problem != nil {
		return nil, problem
	}
	call := &pb.PrepareLocalPackageCall{Claim: s.claim, LocalPackageSet: selected}
	result := c.runHostPrepare(s, w, 0, hostLabel("local_job", request.ID),
		func(ctx context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
			return s.host.PrepareLocalPackage(ctx, call)
		})
	if result.fault != nil {
		return nil, result.fault
	}
	if result.refusal != "" {
		return nil, exit.Named(exit.Structural, "worker.prepare_refused", "unpublished package job preparation refused: %s", result.refusal)
	}
	if result.err != nil || result.set == nil {
		return nil, exit.Unavailablef("unpublished package job preparation ended before its exact result")
	}
	// Preparing code does not download a job's Model inputs. Reuse the private
	// model preparation, but do not activate its result as a serving placement.
	// Native operation inputs keep their existing custody and are not downloads.
	if models := downloadModelRefs(request.Models); len(models) > 0 {
		if c.opt.RentalPackageSet == nil {
			return nil, exit.Unavailablef("unpublished package job models require a rental download set")
		}
		downloads, problem := c.opt.RentalPackageSet(nil, models)
		if problem != nil {
			return nil, problem
		}
		call := &pb.PreparePrivatePlacementCall{Claim: s.claim, PrivatePlacementSet: &pb.DesiredPrivatePlacementSet{
			OperationId: selected.OperationId, LocalRevisionDigest: selected.Package.LocalRevisionDigest, DownloadDelegation: downloads}}
		result = c.runHostPrepare(s, w, 0, hostLabel("local_job_models", request.ID),
			func(ctx context.Context) (grpc.ServerStreamingClient[pb.PrepareEvent], error) {
				return s.host.PreparePrivatePlacement(ctx, call)
			})
		if result.fault != nil {
			return nil, result.fault
		}
		if result.refusal != "" {
			return nil, exit.Named(exit.Structural, "worker.prepare_refused", "unpublished package job model preparation refused: %s", result.refusal)
		}
		if result.err != nil || result.set == nil {
			return nil, exit.Unavailablef("unpublished package job model preparation ended before its exact result")
		}
	}
	c.mu.Lock()
	current := c.sessions[s.bootID] == s && c.workers[instanceID] == w
	if current {
		delete(c.localTransfers, request.ID)
	}
	c.mu.Unlock()
	if !current {
		return nil, exit.Unavailablef("unpublished package job preparation belonged to a superseded worker session")
	}
	return result.set, nil
}
