package producttest

import (
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestBudgetedRequestStopsBeforePaidRentalWithoutPackagePreparation(t *testing.T) {
	owner := hostOwner(t, "budgeted-rental-gate")
	budgeted := submission("sha256:plan", "proof/package", "budgeted-gate", map[string]any{})
	budgeted.MaxCostUSDMicros = 2_000_000
	id, attempt, problem := owner.c.Submit(budgeted)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("budgeted request dispatched attempt %d without package preparation", attempt)
	}
	row := waitRequestState(t, owner, id, "failed")
	if row.MaxCostUSDMicros != 2_000_000 || row.Worker != "" {
		t.Fatalf("budgeted request lost local-first intent: %+v", row)
	}
	events, problem := owner.store.EventsAfter(id, 0, 100)
	fatal(t, problem)
	if !eventHasError(events, "rental.package_preparation_unavailable") {
		t.Fatalf("budgeted request did not stop at the pre-purchase gate: %+v", events)
	}

	zero := submission("sha256:plan", "proof/package", "zero-budget-gate", map[string]any{})
	zeroID, attempt, problem := owner.c.Submit(zero)
	fatal(t, problem)
	if attempt != 0 {
		t.Fatalf("zero-budget request dispatched attempt %d without local capacity", attempt)
	}
	waitRequestState(t, owner, zeroID, "failed")
	events, problem = owner.store.EventsAfter(zeroID, 0, 100)
	fatal(t, problem)
	if eventHasError(events, "rental.package_preparation_unavailable") {
		t.Fatalf("zero-budget request entered automatic rental path: %+v", events)
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
