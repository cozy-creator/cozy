package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func (c *Orchestrator) PruneOperationCache(rental string) (uint32, uint64, bool, *exit.Error) {
	s, problem := c.workspaceControl(rental)
	if problem != nil {
		return 0, 0, false, problem
	}
	workspace, problem := s.operationWorkspace()
	if problem != nil {
		return 0, 0, false, problem
	}
	result, err := workspace.PruneOperationCache(s.ctx, &pb.PruneOperationCacheCall{Claim: s.claim})
	if err != nil {
		return 0, 0, false, operationCacheProblem(err)
	}
	if result == nil {
		return 0, 0, false, exit.Named(exit.Structural, "operation.prune_reply_absent", "Host returned no cache pruning observation")
	}
	return result.RemovedEntries, result.ReclaimedBytes, result.StoreBusy, nil
}
