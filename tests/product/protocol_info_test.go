package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestLocalRuntimeSkewDoesNotRequireRentalFeatures(t *testing.T) {
	info := &pb.ProtocolInfoResult{WireMinor: pb.MinCompatibleWireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor}
	if problem := orchestrator.ValidateWorkerProtocol(info, ""); problem != nil {
		t.Fatalf("local floor Runtime required a Host-only feature: %v", problem)
	}
	if problem := orchestrator.ValidateWorkerProtocol(info, podRental); problem == nil || problem.ErrName() != "worker.rental_idle_guard_required" {
		t.Fatalf("an old rental Host bypassed its mandatory idle guard: %v", problem)
	}
}
