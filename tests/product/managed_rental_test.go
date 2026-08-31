package producttest

import (
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalRequestRequiresTheExplicitAcquisitionSeam(t *testing.T) {
	owner := hostOwner(t, "managed-rental-gate")
	requested := submission("sha256:plan", "proof/package", "rental-gate", map[string]any{})
	requested.Rental = true
	id, attempt, problem := owner.c.Submit(requested)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("rental request dispatched attempt %d without acquisition", attempt)
	}
	row := waitRequestState(t, owner, id, "failed")
	if !row.Rental || row.Worker != "" {
		t.Fatalf("rental request lost placement intent: %+v", row)
	}
	events, problem := owner.store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if !eventHasError(events, "rental.acquisition_unavailable") {
		t.Fatalf("rental request did not stop at the acquisition seam: %+v", events)
	}
	changedMode := requested
	changedMode.Rental = false
	if _, _, problem := owner.c.Submit(changedMode); problem == nil ||
		!strings.Contains(problem.Message, "different body") {
		t.Fatalf("changed placement mode reused one orchestrator identity: %v", problem)
	}

	local := submission("sha256:plan", "proof/package", "local-gate", map[string]any{})
	localID, attempt, problem := owner.c.Submit(local)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("local request dispatched attempt %d without local capacity", attempt)
	}
	waitRequestState(t, owner, localID, "failed")
	events, problem = owner.store.EventsAfter(localID, 0, 100)
	fatal(t, problem)
	if eventHasError(events, "rental.acquisition_unavailable") {
		t.Fatalf("local request entered rental acquisition: %+v", events)
	}
}

func eventHasError(events []records.Event, name string) bool {
	for _, event := range events {
		if event.Payload["error_type"] == name {
			return true
		}
	}
	return false
}

func waitRequestState(t *testing.T, owner *owner, id, state string) *records.Request {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		row, problem := owner.store.RequestRow(id)
		fatal(t, problem)
		if row != nil && row.State == state {
			return row
		}
		time.Sleep(10 * time.Millisecond)
	}
	row, problem := owner.store.RequestRow(id)
	fatal(t, problem)
	t.Fatalf("request %s did not reach %s: %+v", id, state, row)
	return nil
}
