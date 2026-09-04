package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// cl-121: a worker saying "no room right now" must not spend one of a request's lives.
//
// Measured on the live daemon, run 228: FOUR attempts on one request, every one ending
// before execution reached the device, and the request settled `failed` having produced
// nothing. The requeue budget exists so a request whose attempts keep FAULTING terminates
// instead of dispatching forever — but a worker that answered NO_CAPACITY has not
// attempted anything. Charging it turns capacity pressure into a death sentence with a
// three-strike counter.
//
// The local path has drawn this line all along: `device_envelope_held` in selectOrStart
// keeps the durable request on the queue and charges nothing, because a later
// idle-capacity report re-enters select-or-start. This is that rule reaching the path a
// rented pod takes.

// TestACapacityRefusalDoesNotSpendALife drives the real worker protocol. The peer refuses
// every offer NO_CAPACITY — a journaled pre-execution outcome, exactly as a real worker
// whose device lane has no free seat answers — far more times than the whole budget
// allows, and the request must still be alive and still owed.
func TestACapacityRefusalDoesNotSpendALife(t *testing.T) {
	o := hostOwner(t, "no-capacity")
	spec := fakeSpec("no-capacity", "0", "--arm", "no-capacity")
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))

	requestID, _, e := o.c.Submit(submission(planID, "fake/no-capacity", "crowded",
		map[string]any{"n": 1}))
	fatal(t, e)

	// Well past MaxRequeues worth of refusals. A budget that charged them would have
	// settled the request several refusals ago.
	want := orchestrator.MaxRequeues + 3
	waitUntil(t, "more refusals than the whole requeue budget", func() bool {
		return countEvents(o, "NO_CAPACITY") >= want
	})

	row, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if row == nil {
		t.Fatal("the request row disappeared")
	}
	if settled := row.State == "failed" || row.State == "succeeded" || row.State == "abandoned"; settled {
		t.Fatalf("%d capacity refusals settled the request as %q; nothing was ever executed",
			want, row.State)
	}
	// THE BUDGET IS UNTOUCHED. This is the assertion that fails on the old code: every
	// refusal charged a life, so the counter would read MaxRequeues and the request would
	// be gone.
	if row.Requeues != 0 {
		t.Fatalf("capacity refusals charged %d of %d lives; a machine saying 'not now' is "+
			"not a failed attempt", row.Requeues, orchestrator.MaxRequeues)
	}

	// And it is still ACTIVE — owed capacity, not quietly forgotten. The exact state is
	// whichever point of the offer/refuse/requeue cycle this instant caught, so the
	// assertion is on the set the request must stay inside rather than on one member of
	// it: pinning a racing cycle to a single state would be a flaky test asserting a
	// coincidence.
	switch row.State {
	case "submitted", "queued", "dispatching", "requeue_pending":
	default:
		t.Fatalf("after %d capacity refusals the request is %q, which is not an active state",
			want, row.State)
	}

	// AND THE REFUSALS KEEP COMING, which is the other half of "not charged": the request
	// is still being offered capacity rather than parked out of the cycle entirely.
	before := countEvents(o, "NO_CAPACITY")
	waitUntil(t, "the request is still being offered after its budget would have run out",
		func() bool { return countEvents(o, "NO_CAPACITY") > before })
	again, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	if again.Requeues != 0 {
		t.Fatalf("a later capacity refusal charged %d lives", again.Requeues)
	}
}

// TestOnlyNoCapacityIsTreatedAsCapacityPressure keeps the allow-list honest. The polarity
// is the point: a cause not positively known to mean "not now" keeps paying, because a
// request that waits forever on a permanent condition is the exact failure the budget
// exists to prevent.
func TestOnlyNoCapacityIsTreatedAsCapacityPressure(t *testing.T) {
	for _, c := range []struct {
		status, cause string
		want          bool
	}{
		{"REFUSED", "NO_CAPACITY", true},
		{"REFUSED", "ADMISSION_EPOCH_STALE", false},
		{"REFUSED", "UNKNOWN_PLACEMENT", false},
		{"REFUSED", "PLACEMENT_NOT_DISPATCHABLE", false},
		{"FAILED", "EXECUTOR_FAULT", false},
		{"FAILED", "NO_CAPACITY", false},
		{"ABANDONED", "", false},
		{"REFUSED", "SOMETHING_NOBODY_HAS_WRITTEN_YET", false},
	} {
		if got := orchestrator.CapacityRefusal(c.status, c.cause); got != c.want {
			t.Errorf("CapacityRefusal(%q,%q) = %t, want %t", c.status, c.cause, got, c.want)
		}
	}
}
