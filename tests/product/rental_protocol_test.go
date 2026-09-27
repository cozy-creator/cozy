package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// An older pod keeps its Claim, snapshots and keepalive (no wire minor refuses them); only
// new preparation and execution route elsewhere.
func TestRentalReuseRequiresNegotiatedCurrentProtocol(t *testing.T) {
	for _, test := range []struct {
		name  string
		minor uint32
		older bool
	}{
		{"minimum compatible", pb.MinCompatibleWireMinor, false},
		{"current", pb.WireMinor, false},
		{"newer compatible", pb.WireMinor + 1, false},
		{"below minimum", pb.MinCompatibleWireMinor - 1, true},
	} {
		name, minor, older := test.name, test.minor, test.older
		t.Run(name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public, wireMinor: minor}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "protocol-"+name, rentalWiring(connection, private))
			_, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			for _, job := range []bool{false, true} {
				reason, _ := o.c.RentalStanding(podRental, job)
				if older && reason != "protocol_unsupported" {
					t.Fatalf("older claimed pod remained eligible (job=%t): %q", job, reason)
				} else if !older && reason != "" {
					t.Fatalf("current claimed pod excluded (job=%t): %q", job, reason)
				}
			}
		})
	}
}

func TestIdleControlAcceptsMinorSkew(t *testing.T) {
	for _, minor := range []uint32{pb.MinCompatibleWireMinor, pb.WireMinor, pb.WireMinor + 1, pb.MinCompatibleWireMinor - 1} {
		public, private, err := ed25519.GenerateKey(rand.Reader)
		must(t, err)
		pod := &fakePod{controlKey: public, wireMinor: minor}
		connection, _ := startFakePod(t, t.TempDir(), pod)
		var options orchestrator.Options
		rentalWiring(connection, private)(&options)
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		control, problem := orchestrator.DialIdleControl(ctx, connection, options.RentalClaimProof, nil)
		if control != nil {
			must(t, control.Close())
		}
		cancel()
		fatal(t, problem)
		if control == nil {
			t.Fatalf("minor %d did not open idle control", minor)
		}
		pod.mu.Lock()
		mutated := len(pod.offers) + len(pod.prepares) + len(pod.desired)
		pod.mu.Unlock()
		if mutated != 0 {
			t.Fatalf("idle control prepared or dispatched work for minor %d", minor)
		}
	}
}

func TestQueuedPinToOlderWorkerReplansWithoutOffering(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, wireMinor: pb.MinCompatibleWireMinor - 1, serve: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	var acquisitions atomic.Int64
	o := hostOwner(t, "old-pin", rentalWiring(connection, private), func(options *orchestrator.Options) {
		options.RentalFleet = func(records.Request) (string, *exit.Error) { return "fleet retained", nil }
		options.AcquireManagedRental = func(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			if req.Worker != "" {
				t.Errorf("acquisition retained obsolete pin: %s", req.Worker)
			}
			acquisitions.Add(1)
			return orchestrator.PlacementDecision{}, "", exit.Unavailablef("new capacity held by test")
		}
	})
	_, _, _, problem := o.c.EnsureRental(podRental)
	fatal(t, problem)
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "old-pin", Package: "acme/old-pin", Entrypoint: "tile", PlanID: podPlanID("acme/old-pin"), Release: "1.0.0", Payload: []byte(`{"size":16}`), Outputs: []string{"image"}, Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "replanning from negotiated old peer", func() bool { return acquisitions.Load() > 0 })
	row, problem := o.store.RequestRow(id)
	fatal(t, problem)
	if row == nil || row.Worker != "" {
		t.Fatalf("queued request did not release its old pin: %#v", row)
	}
	attempts, problem := o.store.Attempts(id)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatalf("replanning created attempts: %v", attempts)
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.offers) != 0 || len(pod.prepares) != 0 {
		t.Fatalf("old peer received new work: offers=%d prepares=%d", len(pod.offers), len(pod.prepares))
	}
}
