package producttest

import (
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

// cl-186. `EnsurePlacementReady` waits on observations and on nothing else, which is right
// — a cold start is not a timeout — but the set of observations it could end on named
// every way the WORKER can fail and no way the REQUEST can. A preparation that stalled
// (2026-09-08: two pods sat fifteen minutes at zero bytes on a presign whose URLs arrived
// already spent) left the loop spinning forever, holding `starting[slot]`; cancelling the
// request did not reach it, every later request for the slot returned early from
// selectOrStart and parked on "no attached rental has a DISPATCHABLE placement", and the
// only recovery anyone found was `cozy rental end`. Two paid H100s went that way.
//
// The wait now ends when the only reason to keep waiting is gone. That is an observation,
// not a clock.
func TestAPlacementWaitEndsWhenItsRequestDoes(t *testing.T) {
	o := hostOwner(t, "wait-outlives")
	spec := fakeSpec("wait-outlives", "10")
	planID := planIDOf(t, spec)
	requestID, _, e := o.c.Submit(submission(planID, "fake/wait-outlives", "wait-outlives-1",
		map[string]any{"n": 1}))
	fatal(t, e)
	fatal(t, o.store.SettleRequest(requestID, "canceled"))
	row, e := o.store.RequestRow(requestID)
	fatal(t, e)
	if row == nil || !records.Settled(row.State) {
		t.Fatalf("the request is %+v; the arm needs it settled", row)
	}

	// The worker this wait names is not there — the stalled-preparation shape reduced to
	// its essential: nothing about the WORKER will ever make this wait return, and the
	// request that started it is over.
	done := make(chan string, 1)
	go func() {
		problem := o.c.EnsurePlacementReady("no-such-instance", planID, requestID)
		if problem == nil {
			done <- ""
			return
		}
		done <- problem.ErrName()
	}()
	select {
	case name := <-done:
		if name != "request.settled_while_preparing" {
			t.Fatalf("the wait ended on %q; a settled request ends its own wait", name)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the wait outlived the request it was started for")
	}

	// The control: with no request owning it, the answer is unchanged — this branch is
	// the only thing that moved.
	problem := o.c.EnsurePlacementReady("no-such-instance", planID, "")
	if problem == nil || strings.Contains(problem.ErrName(), "settled") {
		t.Fatalf("an unowned wait answered %v; it must still answer about the worker", problem)
	}
}
