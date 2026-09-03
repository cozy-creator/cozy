package producttest

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// A Hub outage is capacity weather, not the request's answer. The durable queue may be woken any
// number of times without settling or duplicating the request.
func TestRentalHubOutageKeepsTheRequestQueued(t *testing.T) {
	var observations atomic.Int64
	o := hostOwner(t, "rental-hub-outage", func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) {
			observations.Add(1)
			return "", exit.Unavailablef("Tensorhub is temporarily unreachable")
		}
		options.AcquireManagedRental = func(records.Request) (orchestrator.RentalDecision, string, *exit.Error) {
			t.Fatal("rental acquisition ran without a readable fleet")
			return orchestrator.RentalDecision{}, "", nil
		}
	})

	spec := fakeSpec("outage-recovery", "0")
	request := submission(planIDOf(t, spec), "fake/outage-recovery", "rental-hub-outage-1",
		map[string]any{"outage": true})
	request.Rental = true
	requestID, _, problem := o.c.Submit(request)
	fatal(t, problem)
	if row, _ := o.store.RequestRow(requestID); row == nil || row.State == "failed" || row.State == "canceled" || row.State == "completed" {
		t.Fatalf("Hub outage settled the durable request instead of queueing it: %#v", row)
	}

	o.c.WakeQueue()
	deadline := time.Now().Add(5 * time.Second)
	for observations.Load() < 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if observations.Load() < 2 {
		t.Fatal("a fleet observation did not re-ask the queued request")
	}
	if row, _ := o.store.RequestRow(requestID); row == nil || row.State == "failed" || row.State == "canceled" || row.State == "completed" {
		t.Fatalf("retrying the outage changed the request's durable state: %#v", row)
	}

	fatal(t, o.c.CancelQueued(requestID, "test cleanup"))
	attempts, problem := o.store.Attempts(requestID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatalf("outage retries minted %d attempt(s) without capacity: %#v", len(attempts), attempts)
	}
}
