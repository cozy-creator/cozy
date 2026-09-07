package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Probe the pinned peer before any Claim or preparation can change ownership.
// PodHost returns its intersection with the actual Runtime; a local worker is
// queried directly on the same RuntimePreparation service.
func probeWorkerProtocol(ctx context.Context, connection grpc.ClientConnInterface, remote bool) *exit.Error {
	ctx, cancel := context.WithTimeout(ctx, hub.Timeout)
	defer cancel()
	var info *pb.ProtocolInfoResult
	var err error
	if remote {
		info, err = pb.NewPodHostClient(connection).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	} else {
		info, err = pb.NewRuntimePreparationClient(connection).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	}
	if err != nil {
		if status.Code(err) != codes.Unimplemented && status.Code(err) != codes.FailedPrecondition {
			return exit.Unavailablef("worker protocol probe is temporarily unavailable")
		}
		return exit.Named(exit.Conflict, "worker.protocol_incompatible", "worker has no compatible read-only protocol probe")
	}
	if info == nil || info.MinimumWireMinor == 0 || info.MinimumWireMinor > info.WireMinor ||
		info.WireMinor < pb.MinCompatibleWireMinor || pb.WireMinor < info.MinimumWireMinor {
		return exit.Named(exit.Conflict, "worker.protocol_incompatible", "worker and Creator protocol ranges do not overlap")
	}
	return nil
}
