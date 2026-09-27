package producttest

import (
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestWakeQueuePreservesUnassignedRequestedRentalAffinities(t *testing.T) {
	var mu sync.Mutex
	seen := map[string]bool{}
	recording := false
	o := hostOwner(t, "pinned-preparation-wake", func(opt *orchestrator.Options) {
		opt.RentalFleet = func(records.Request) (string, *exit.Error) { return "existing requested rentals", nil }
		opt.AcquireManagedRental = func(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			mu.Lock()
			if recording {
				seen[req.RequestedRental] = true
			}
			mu.Unlock()
			return orchestrator.PlacementDecision{}, "", exit.Unavailablef("fixture preparation is not ready")
		}
	})
	for _, rental := range []string{"wanted-a", "wanted-b"} {
		_, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: rental,
			Package: "proof/wake", Entrypoint: "tile", Release: "1.0.0", Payload: []byte(`{}`),
			Rental: true, RentalRequired: true, RequestedRental: rental})
		fatal(t, problem)
	}
	mu.Lock()
	recording = true
	mu.Unlock()
	// The wake hands each requested rental's queue to that rental's scheduler, which asks
	// its own placement question asynchronously.
	o.c.WakeQueue()
	waitUntil(t, "the wake asks every independent requested rental", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return seen["wanted-a"] && seen["wanted-b"]
	})
}
