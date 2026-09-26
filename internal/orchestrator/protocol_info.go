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
		if status.Code(err) == codes.FailedPrecondition {
			return exit.Named(exit.Conflict, "worker.protocol_incompatible", "worker protocol probe refused: %s", status.Convert(err).Message())
		}
		return exit.Named(exit.Conflict, "worker.protocol_incompatible", "worker has no compatible read-only protocol probe; update the worker")
	}
	return ValidateWorkerProtocol(info, remote)
}

// ValidateWorkerProtocol checks the execution range independently of Host-only
// features. A compatible Runtime need not implement the rental idle contract.
func ValidateWorkerProtocol(info *pb.ProtocolInfoResult, rental bool) *exit.Error {
	if info == nil || info.MinimumWireMinor == 0 || info.MinimumWireMinor > info.WireMinor {
		return exit.Named(exit.Conflict, "worker.protocol_incompatible", "worker reported an invalid supported protocol range; update the worker")
	}
	if info.WireMinor < pb.MinCompatibleWireMinor || pb.WireMinor < info.MinimumWireMinor {
		return exit.Named(exit.Conflict, "worker.protocol_incompatible",
			"Creator supports worker protocol %d–%d; this worker supports %d–%d. Update %s",
			pb.MinCompatibleWireMinor, pb.WireMinor, info.MinimumWireMinor, info.WireMinor,
			protocolUpgradeTarget(info))
	}
	if rental && !info.SupportsRentalKeepalive {
		return exit.Named(exit.Conflict, "worker.rental_idle_guard_required",
			"this rental worker lacks the reliable active-work reporting or manual keepalive required by the mandatory 15-minute idle shutdown; update the rental worker image")
	}
	return nil
}

func protocolUpgradeTarget(info *pb.ProtocolInfoResult) string {
	if info.MinimumWireMinor > pb.WireMinor {
		return "the local cozy-creator CLI"
	}
	return "the worker Runtime"
}

// RentalProtocolInfo reads the pinned PodHost's supported range without a Claim.
// PodHost answers its intersection with the installed Runtime. This maintenance
// probe does not admit execution or require the Creator execution range.
func RentalProtocolInfo(ctx context.Context, remote *WorkerConnection) (*pb.ProtocolInfoResult, *exit.Error) {
	if remote == nil || remote.CACert == "" {
		return nil, exit.New(exit.Credential, "the protocol probe requires a pinned rental")
	}
	conn, err := dialWorker(remote.Addr, remote)
	if err != nil {
		return nil, exit.Unavailablef("cannot open the pinned rental's protocol probe")
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(ctx, hub.Timeout)
	defer cancel()
	info, err := pb.NewPodHostClient(conn).ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil {
		return nil, exit.Unavailablef("the rental's protocol probe failed: %s", status.Convert(err).Message())
	}
	return info, validateMaintenanceProtocol(info)
}

// Maintenance uses only the signed Claim and closed snapshot surfaces retained
// since the rental keepalive contract. It never acknowledges the snapshot or
// sends execution/preparation messages. Future hard cuts remain refused.
func validateMaintenanceProtocol(info *pb.ProtocolInfoResult) *exit.Error {
	if info == nil || info.MinimumWireMinor == 0 || info.MinimumWireMinor > info.WireMinor || info.WireMinor < pb.RentalKeepaliveWireMinor || info.MinimumWireMinor > pb.WireMinor {
		return exit.Named(exit.Conflict, "worker.protocol_incompatible", "worker has no supported maintenance protocol range")
	}
	if !info.SupportsRentalKeepalive {
		return exit.Named(exit.Conflict, "worker.rental_idle_guard_required", "rental maintenance requires reliable active-work reporting and manual keepalive")
	}
	return nil
}
