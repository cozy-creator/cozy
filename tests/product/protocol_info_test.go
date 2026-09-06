package orchestrator

import (
	"context"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type protocolProbeConnection struct {
	minor, floor uint32
	absent       bool
	calls        []string
}

func (c *protocolProbeConnection) Invoke(_ context.Context, method string, _ any, reply any, _ ...grpc.CallOption) error {
	c.calls = append(c.calls, method)
	if c.absent {
		return status.Error(codes.Unimplemented, "old peer")
	}
	response := reply.(*pb.ProtocolInfoResult)
	response.WireMinor, response.MinimumWireMinor = c.minor, c.floor
	return nil
}
func (c *protocolProbeConnection) NewStream(context.Context, *grpc.StreamDesc, string, ...grpc.CallOption) (grpc.ClientStream, error) {
	panic("probe opened an ownership stream")
}

func TestProtocolRangeProbeHasNoOwnershipSideEffect(t *testing.T) {
	for _, remote := range []bool{false, true} {
		for _, row := range []struct {
			name             string
			minor, floor     uint32
			absent, accepted bool
		}{
			{"current", pb.WireMinor, pb.MinCompatibleWireMinor, false, true},
			{"additive", pb.WireMinor + 1, pb.MinCompatibleWireMinor, false, true},
			{"old", pb.MinCompatibleWireMinor - 1, 1, false, false},
			{"new hardcut", pb.WireMinor + 1, pb.WireMinor + 1, false, false},
			{"invalid", 1, 2, false, false}, {"missing", 0, 0, true, false},
		} {
			c := &protocolProbeConnection{minor: row.minor, floor: row.floor, absent: row.absent}
			problem := probeWorkerProtocol(t.Context(), c, remote)
			if (problem == nil) != row.accepted {
				t.Fatalf("%s remote=%v: %v", row.name, remote, problem)
			}
			path := pb.RuntimePreparation_ProtocolInfo_FullMethodName
			if remote {
				path = pb.PodHost_ProtocolInfo_FullMethodName
			}
			if len(c.calls) != 1 || c.calls[0] != path {
				t.Fatalf("probe mutated or reached wrong service: %v", c.calls)
			}
		}
	}
}
