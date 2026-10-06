package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// CheckpointHost supplies the claimed pod's bounded byte-transfer lane.
// The publication owner supplies exact subjects and short-lived grants only.
type CheckpointHost struct {
	BootID   string
	Page     func(context.Context, *pb.CheckpointPageRequest) (*pb.CheckpointPageResult, *exit.Error)
	Transfer func(context.Context, *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error)
}
