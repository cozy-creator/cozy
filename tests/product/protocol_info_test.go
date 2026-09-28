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

// Exercise the real owner on pinned TLS. A peer's protocol range, or its absence, never
// refuses the Claim (worker-protocol fd6d9a92); only the rental idle guard, which makes
// idle release safe, keeps a Claim from a Host.
func TestProtocolRangeProbeHasNoOwnershipSideEffect(t *testing.T) {
	for _, row := range []struct {
		name             string
		minor, floor     uint32
		absent, accepted bool
		unsafe           bool
	}{
		{"floor", pb.MinCompatibleWireMinor, pb.MinCompatibleWireMinor, false, true, false},
		{"unsafe rental Host", pb.MinCompatibleWireMinor, pb.MinCompatibleWireMinor, false, false, true},
		{"current", pb.WireMinor, pb.MinCompatibleWireMinor, false, true, false},
		{"additive", pb.WireMinor + 1, pb.MinCompatibleWireMinor, false, true, false},
		{"old", pb.MinCompatibleWireMinor - 1, 1, false, true, false},
		{"new floor", pb.WireMinor + 1, pb.WireMinor + 1, false, true, false},
		{"invalid", 1, 2, false, true, false}, {"missing", 0, 0, true, true, false},
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
			if row.unsafe && (problem == nil || problem.ErrName() != "worker.rental_idle_guard_required" || !strings.Contains(problem.Message, "active-work reporting")) {
				t.Fatalf("unsafe Host did not get a feature-specific refusal: %v", problem)
			}
			if row.floor > 0 && row.floor <= row.minor && (row.minor < pb.MinCompatibleWireMinor || row.floor > pb.WireMinor) {
				// Execution on a peer outside the range fails that operation alone, naming both ranges.
				problem := orchestrator.ValidateWorkerProtocol(&pb.ProtocolInfoResult{WireMinor: row.minor, MinimumWireMinor: row.floor}, podRental)
				if problem == nil || problem.ErrName() != pb.CapabilityUnavailableCode ||
					!strings.Contains(problem.Message, fmt.Sprintf("%d–%d", row.floor, row.minor)) ||
					!strings.Contains(problem.Message, fmt.Sprintf("%d–%d", pb.MinCompatibleWireMinor, pb.WireMinor)) {
					t.Fatalf("an execution outside the range was not a named capability failure: %v", problem)
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
	info := &pb.ProtocolInfoResult{WireMinor: pb.MinCompatibleWireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor}
	if problem := orchestrator.ValidateWorkerProtocol(info, ""); problem != nil {
		t.Fatalf("local floor Runtime required a Host-only feature: %v", problem)
	}
	if problem := orchestrator.ValidateWorkerProtocol(info, podRental); problem == nil || problem.ErrName() != "worker.rental_idle_guard_required" {
		t.Fatalf("an old rental Host bypassed its mandatory idle guard: %v", problem)
	}
}
