package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

// The Host authenticates and forwards to the Runtime workspace authority.
type operationWorkspace interface {
	RecordOperationResult(context.Context, *pb.RecordOperationResultCall, ...grpc.CallOption) (*pb.RecordOperationResultResult, error)
	LookupOperation(context.Context, *pb.LookupOperationCall, ...grpc.CallOption) (*pb.LookupOperationResult, error)
	PruneOperationCache(context.Context, *pb.PruneOperationCacheCall, ...grpc.CallOption) (*pb.PruneOperationCacheResult, error)
}

func (s *session) operationWorkspace() (operationWorkspace, *exit.Error) {
	if s == nil || s.claim == nil {
		return nil, exit.Unavailablef("operation workspace requires a current worker claim")
	}
	if s.host != nil {
		return s.host, nil
	}
	if s.preparation != nil {
		return s.preparation, nil
	}
	return nil, exit.Unavailablef("the local Runtime does not expose the operation workspace service")
}
