package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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
