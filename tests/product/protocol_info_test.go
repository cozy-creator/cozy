package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
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
		unsafe           bool
	}{
		{"wire59", 59, 59, false, true, false},
		{"unsafe rental Host", 59, 59, false, false, true},
		{"current", pb.WireMinor, pb.MinCompatibleWireMinor, false, true, false},
		{"additive", pb.WireMinor + 1, pb.MinCompatibleWireMinor, false, true, false},
		{"old", pb.MinCompatibleWireMinor - 1, 1, false, false, false},
		{"new hardcut", pb.WireMinor + 1, pb.WireMinor + 1, false, false, false},
		{"invalid", 1, 2, false, false, false}, {"missing", 0, 0, true, false, false},
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
				return &pb.ProtocolInfoResult{WireMinor: row.minor, MinimumWireMinor: row.floor, SupportsRentalKeepalive: !row.unsafe}, nil
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
			if row.unsafe && (problem == nil || problem.ErrName() != "worker.rental_idle_guard_required") {
				t.Fatalf("unsafe Host did not get a feature-specific refusal: %v", problem)
			}
			if !row.accepted && !row.unsafe && !row.absent && row.floor > 0 && row.floor <= row.minor {
				if !strings.Contains(problem.Message, fmt.Sprintf("%d–%d", row.floor, row.minor)) ||
					!strings.Contains(problem.Message, fmt.Sprintf("%d–%d", pb.MinCompatibleWireMinor, pb.WireMinor)) {
					t.Fatalf("range refusal omitted supported versions: %v", problem)
				}
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

func TestLocalRuntimeSkewDoesNotRequireRentalFeatures(t *testing.T) {
	info := &pb.ProtocolInfoResult{WireMinor: 59, MinimumWireMinor: 59}
	if problem := orchestrator.ValidateWorkerProtocol(info, false); problem != nil {
		t.Fatalf("local Runtime59 required a Host-only feature: %v", problem)
	}
	if problem := orchestrator.ValidateWorkerProtocol(info, true); problem == nil || problem.ErrName() != "worker.rental_idle_guard_required" {
		t.Fatalf("an old rental Host bypassed its mandatory idle guard: %v", problem)
	}
}
