package orchestrator

import (
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPeerMinorNumbersDoNotRefuseAvailableOperations(t *testing.T) {
	for _, rangeInfo := range []*pb.ProtocolInfoResult{{WireMinor: 0, MinimumWireMinor: 0}, {WireMinor: 5, MinimumWireMinor: 0}, {WireMinor: 72, MinimumWireMinor: 0}, {WireMinor: 1000, MinimumWireMinor: 800}} {
		if problem := ValidateWorkerProtocol(rangeInfo, ""); problem != nil {
			t.Fatalf("peer %v refused before operation negotiation: %v", rangeInfo, problem)
		}
	}
}
