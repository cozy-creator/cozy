package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func assertDurableCancel(t *testing.T, store *records.Store, id, expectedActor string) {
	t.Helper()
	link, problem := store.MachineExecution(id)
	fatal(t, problem)
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	actor, _, _, problem := store.CancelAttribution(id)
	fatal(t, problem)
	if link == nil || row == nil {
		t.Fatal("cancel lost its local request or machine link")
	}
	if !link.CancelRequested || row.State != "canceling" || actor != expectedActor {
		t.Fatalf("cancel was not durably attributed before the machine replied: cancel=%v state=%s actor=%q", link.CancelRequested, row.State, actor)
	}
	events, problem := store.EventsAfter(id, 0, 1000)
	fatal(t, problem)
	count := 0
	for _, event := range events {
		if event.Type == "request.cancel_requested" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("cancellation attribution duplicated: %d", count)
	}
}

func TestMachineCancellationIntentAndActorAreAtomic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	request, receipt := machineObserverRecord(t, store)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	_, err = db.Exec(`CREATE TRIGGER reject_cancel_actor BEFORE INSERT ON request_events
 WHEN NEW.type='request.cancel_requested' BEGIN SELECT RAISE(ABORT,'actor storage failed'); END`)
	must(t, err)
	if _, problem := store.RequestMachineCancellation(request.ID, "cozy run cancel"); problem == nil {
		t.Fatal("failed actor persistence acknowledged cancellation")
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if link.CancelRequested || row.State != "dispatching" {
		t.Fatalf("failed attribution partially committed intent: cancel=%v state=%s", link.CancelRequested, row.State)
	}
	_, err = db.Exec(`DROP TRIGGER reject_cancel_actor`)
	must(t, err)
	_, problem = store.RequestMachineCancellation(request.ID, "cozy run cancel")
	fatal(t, problem)
	_, problem = store.RequestMachineCancellation(request.ID, "another cancel caller")
	fatal(t, problem)
	assertDurableCancel(t, store, request.ID, "cozy run cancel")
	// Old running snapshots do not erase pending intent; an actual successful
	// outcome still wins if it completed before the machine could apply the cancel.
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	assertDurableCancel(t, store, request.ID, "cozy run cancel")
	state.State, state.Generation = "succeeded", 2
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	row, problem = store.RequestRow(request.ID)
	fatal(t, problem)
	if row.State != "succeeded" {
		t.Fatalf("cancel intent replaced a real successful outcome: %s", row.State)
	}
}
