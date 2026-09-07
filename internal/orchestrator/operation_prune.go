package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func (c *Orchestrator) PruneOperationCache(rental string) (uint32, uint64, bool, *exit.Error) {
	if _, _, _, problem := c.EnsureRental(rental); problem != nil {
		return 0, 0, false, problem
	}
	s, problem := c.rentalControl(rental)
	if problem != nil {
		return 0, 0, false, problem
	}
	if s.host == nil || s.claim == nil {
		return 0, 0, false, exit.Unavailablef("operation cache pruning requires the claimed private Host")
	}
	result, err := s.host.PruneOperationCache(s.ctx, &pb.PruneOperationCacheCall{Claim: s.claim})
	if err != nil {
		return 0, 0, false, operationCacheProblem(err)
	}
	if result == nil {
		return 0, 0, false, exit.Named(exit.Structural, "operation.prune_reply_absent", "Host returned no cache pruning observation")
	}
	return result.RemovedEntries, result.ReclaimedBytes, result.StoreBusy, nil
}
