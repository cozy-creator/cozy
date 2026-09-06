package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"sync/atomic"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// An older pod can remain attached to preserve its data and settle old work,
// but new directives are authored at this client's current wire minor.
func TestRentalReuseRequiresNegotiatedCurrentProtocol(t *testing.T) {
	for _, older := range []bool{false, true} {
		name := "current"
		minor := uint32(pb.WireMinor)
		if older {
			name = "older"
			minor--
		}
		t.Run(name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public, wireMinor: minor}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "protocol-"+name, rentalWiring(connection, private))
			_, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			for _, job := range []bool{false, true} {
				kept, excluded := o.c.ModeCompatibleRentalsWithExclusions([]string{podRental}, job)
				if older {
					if len(kept) != 0 || len(excluded) != 1 || excluded[0].RentalID != podRental || excluded[0].Reason != "protocol_unsupported" {
						t.Fatalf("older claimed pod remained eligible (job=%t): kept=%v excluded=%v", job, kept, excluded)
					}
				} else if len(kept) != 1 || kept[0] != podRental || len(excluded) != 0 {
					t.Fatalf("current claimed pod excluded (job=%t): kept=%v excluded=%v", job, kept, excluded)
				}
			}
		})
	}
}

func TestQueuedPinToOlderWorkerReplansWithoutOffering(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, wireMinor: pb.WireMinor - 1, serve: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	var acquisitions atomic.Int64
	o := hostOwner(t, "old-pin", rentalWiring(connection, private), func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) { return "fleet retained", nil }
		options.AcquireManagedRental = func(req records.Request) (orchestrator.RentalDecision, string, *exit.Error) {
			if req.Worker != "" {
				t.Errorf("acquisition retained obsolete pin: %s", req.Worker)
			}
			acquisitions.Add(1)
			return orchestrator.RentalDecision{}, "", exit.Unavailablef("new capacity held by test")
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
