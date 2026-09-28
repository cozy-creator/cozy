package producttest

import (
	"testing"
)

// NO_CAPACITY is not a failure (owner ruling 2026-09-28): a worker saying "no room right
// now" has attempted nothing, so the request stays queued and waits for capacity, with no
// counter that could ever settle it. Measured live on run 228 before cl-121: four "not now"
// refusals settled a run that had executed nothing.
//
// TestACapacityRefusalWaitsQueued drives the real worker protocol. The peer refuses every
// offer NO_CAPACITY — a journaled pre-execution outcome, exactly as a real worker whose
// device lane has no free seat answers — many times over, and the request must still be
// alive, still owed, and never failed.
func TestACapacityRefusalWaitsQueued(t *testing.T) {
	o := hostOwner(t, "no-capacity")
	spec := fakeSpec("no-capacity", "0", "--arm", "no-capacity")
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID, ""))

	requestID, _, e := o.c.Submit(submission(planID, "fake/no-capacity", "crowded",
		map[string]any{"n": 1}))
	fatal(t, e)

	const want = 6
	waitUntil(t, "repeated capacity refusals", func() bool {
		return countEvents(o, "NO_CAPACITY") >= want
	})

	// Still ACTIVE — owed capacity, not quietly forgotten. The exact state is whichever
	// point of the offer/refuse/requeue cycle this instant caught, so the assertion is on
	// the set the request must stay inside rather than on one member of it.
	row, problem := o.store.RequestRow(requestID)
	fatal(t, problem)
	switch row.State {
	case "submitted", "queued", "dispatching", "requeue_pending":
	default:
		t.Fatalf("after %d capacity refusals the request is %q, which is not an active state", want, row.State)
	}
	events, problem := o.store.EventsAfter(requestID, 0, 1000)
	fatal(t, problem)
	for _, event := range events {
		if event.Type == "request.failed" {
			t.Fatalf("a capacity refusal failed the request: %v", event.Payload)
		}
	}

	// AND THE REFUSALS KEEP COMING: the request is still being offered capacity rather than
	// parked out of the cycle entirely.
	before := countEvents(o, "NO_CAPACITY")
	waitUntil(t, "the request is still being offered capacity", func() bool {
		return countEvents(o, "NO_CAPACITY") > before
	})
}
