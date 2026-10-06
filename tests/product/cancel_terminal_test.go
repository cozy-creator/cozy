package producttest

import (
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"testing"
)

// The store half of that race: the intent survives for the observer to send.
func TestCancelAfterAcceptanceKeepsItsIntent(t *testing.T) {
	store, request := pendingNativeFixture(t)
	fatal(t, store.AcceptRunV1(request.ID, &v1.RunState{Id: request.ID, Number: 1, Attempt: 1, State: "running"}))
	accepted, problem := store.RequestMachineCancellation(request.ID, "")
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if !accepted || !link.CancelRequested || row.State != "canceling" {
		t.Fatalf("a cancel after acceptance was lost (accepted %v, intent kept %v, run %s)", accepted, link.CancelRequested, row.State)
	}
}
