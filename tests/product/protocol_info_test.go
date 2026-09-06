package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Exercise the real owner on pinned TLS: incompatible peers must receive no Claim.
func TestProtocolRangeProbeHasNoOwnershipSideEffect(t *testing.T) {
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
		t.Run(row.name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			var probes, claims atomic.Int64
			pod := &fakePod{controlKey: public}
			pod.protocolInfo = func(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
				probes.Add(1)
				if row.absent {
					return nil, status.Error(codes.Unimplemented, "old peer")
				}
				return &pb.ProtocolInfoResult{WireMinor: row.minor, MinimumWireMinor: row.floor}, nil
			}
			pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
				if frame.GetClaim() != nil {
					claims.Add(1)
				}
				return false, nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "protocol-range-"+records.NewID("proof"), rentalWiring(connection, private))
			_, _, _, problem := o.c.EnsureRental(podRental)
			if (problem == nil) != row.accepted {
				t.Fatalf("wrong range result: %v", problem)
			}
			if probes.Load() == 0 {
				t.Fatal("owner skipped read-only probe")
			}
			if row.accepted && claims.Load() != 1 {
				t.Fatalf("compatible peer received %d claims", claims.Load())
			}
			if !row.accepted && claims.Load() != 0 {
				t.Fatal("incompatible peer received ownership mutation")
			}
		})
	}
}
