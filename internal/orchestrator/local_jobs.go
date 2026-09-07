package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

// A private job needs its prepared environment and job records. Preparing it
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
			return nil, exit.Unavailablef("private job files await verification on the current worker")
		}
		if problem := c.opt.Store.MarkLocalPackageUploaded(request.ID, revision.Digest, s.bootID); problem != nil {
			return nil, problem
		}
	}
	if s.host == nil || s.claim == nil {
		return nil, exit.Unavailablef("private job preparation awaits the claimed Host")
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
		return nil, exit.Named(exit.Structural, "worker.prepare_refused", "private job preparation refused: %s", result.refusal)
	}
	if result.err != nil || result.set == nil {
		return nil, exit.Unavailablef("private job preparation ended before its exact result")
	}
	c.mu.Lock()
	current := c.sessions[s.bootID] == s && c.workers[instanceID] == w
	if current {
		delete(c.localTransfers, request.ID)
	}
	c.mu.Unlock()
	if !current {
		return nil, exit.Unavailablef("private job preparation belonged to a superseded worker session")
	}
	return result.set, nil
}
