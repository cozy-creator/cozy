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

// Probe the pinned peer before Claim. Its range is recorded, never a reason to refuse the
// connection or Claim (worker-protocol VERSIONING): each operation gates on what it uses,
// and a peer without the probe answers an unknown range. Only the rental idle guard is a
// connection-level requirement, because idle release is unsafe without it.
func probeWorkerProtocol(ctx context.Context, connection grpc.ClientConnInterface, remote bool) (*pb.ProtocolInfoResult, *exit.Error) {
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
			return nil, exit.Unavailablef("worker protocol probe is temporarily unavailable")
		}
		return nil, nil
	}
	if remote && !info.SupportsRentalKeepalive {
		return info, rentalIdleGuardRequired()
	}
	return info, nil
}

func rentalIdleGuardRequired() *exit.Error {
	return exit.Named(exit.Conflict, "worker.rental_idle_guard_required",
		"this rental worker lacks the reliable active-work reporting or manual keepalive required by the mandatory 15-minute idle shutdown; update the rental worker image")
}

// ValidateWorkerProtocol gates ordinary preparation and execution on the peer's range.
// A peer outside it fails only that operation, naming the component to update.
func ValidateWorkerProtocol(info *pb.ProtocolInfoResult, rental bool) *exit.Error {
	if info == nil || info.MinimumWireMinor == 0 || info.MinimumWireMinor > info.WireMinor {
		return exit.Named(exit.Conflict, pb.CapabilityUnavailableCode, "worker reported no usable protocol range").
			WithRemedy("update the worker Runtime (`cozy rental update <rental>` for a rental)")
	}
	if info.WireMinor < pb.MinCompatibleWireMinor || pb.WireMinor < info.MinimumWireMinor {
		return exit.Named(exit.Conflict, pb.CapabilityUnavailableCode,
			"Creator executes worker protocol %d–%d; this worker supports %d–%d",
			pb.MinCompatibleWireMinor, pb.WireMinor, info.MinimumWireMinor, info.WireMinor).
			WithRemedy("update %s; other work on this machine continues", protocolUpgradeTarget(info.MinimumWireMinor))
	}
	if rental && !info.SupportsRentalKeepalive {
		return rentalIdleGuardRequired()
	}
	return nil
}

func protocolUpgradeTarget(peerMinimum uint32) string {
	if peerMinimum > pb.WireMinor {
		return "the local cozy CLI"
	}
	return "the worker Runtime (`cozy rental update <rental>` for a rental)"
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

// Maintenance uses only the signed Claim and snapshot surfaces, which no wire minor
// refuses. It needs the keepalive capability that makes idle release safe.
func validateMaintenanceProtocol(info *pb.ProtocolInfoResult) *exit.Error {
	if info == nil || info.WireMinor < pb.RentalKeepaliveWireMinor || !info.SupportsRentalKeepalive {
		return exit.Named(exit.Conflict, "worker.rental_idle_guard_required", "rental maintenance requires reliable active-work reporting and manual keepalive")
	}
	return nil
}
