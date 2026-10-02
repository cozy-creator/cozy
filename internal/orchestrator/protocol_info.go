package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/status"
)

// ValidateWorkerProtocol checks usable protocol metadata for new work. Minor
// numbers never reject a peer; capabilities and individual RPCs decide support.
func ValidateWorkerProtocol(info *pb.ProtocolInfoResult, rental string) *exit.Error {
	update := "update the local Runtime"
	if rental != "" {
		update = "run `cozy rental update " + rental + "`"
	}
	refuse := func(format string, args ...any) *exit.Error {
		return exit.Named(exit.Conflict, pb.CapabilityUnavailableCode,
			format+"; %s to run new work there (its other work continues)", append(args, update)...)
	}
	if info == nil || info.MinimumWireMinor > info.WireMinor {
		return refuse("worker reported no usable protocol range")
	}
	// Minor numbers describe peer evolution, not operation support. Each new
	// execution checks the actual workspace capabilities and affected RPCs.
	return nil
}

// RentalProtocolInfo reads the pinned PodHost's supported range without a Claim.
// PodHost answers its intersection with the installed Runtime, or its own range when
// they share none. This maintenance probe admits no execution.
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
	return info, nil
}
