package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A cancel's intent and its actor are one record: when the actor cannot be stored, nothing of
// the cancel is, and a second caller adds no second actor. A running state the machine sends
// before it applies the cancel does not erase the intent; a success it reached first still wins.
func TestACancelAndItsActorAreRecordedTogether(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	request, _, problem := store.Submit(records.Request{ID: "cancel-durable", IdemKey: "cancel-durable", Package: "local/proof",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("c"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": "local"}))
	running := &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}
	fatal(t, store.AcceptRunV1(request.ID, running))

	durable := func(actor string) {
		t.Helper()
		link, problem := store.MachineExecution(request.ID)
		fatal(t, problem)
		row, problem := store.RequestRow(request.ID)
		fatal(t, problem)
		recorded, _, _, problem := store.CancelAttribution(request.ID)
		fatal(t, problem)
		events, problem := store.EventsAfter(request.ID, 0, 1000)
		fatal(t, problem)
		actors := 0
		for _, event := range events {
			if event.Type == "request.cancel_requested" {
				actors++
			}
		}
		if !link.CancelRequested || row.State != "canceling" || recorded != actor || actors != 1 {
			t.Fatalf("the cancel is not one durable record: intent %v, state %s, actor %q, %d actors", link.CancelRequested, row.State, recorded, actors)
		}
	}

	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	_, err = db.Exec(`CREATE TRIGGER reject_cancel_actor BEFORE INSERT ON request_events
 WHEN NEW.type='request.cancel_requested' BEGIN SELECT RAISE(ABORT,'actor storage failed'); END`)
	must(t, err)
	if _, problem := store.RequestMachineCancellation(request.ID, "cozy run cancel"); problem == nil {
		t.Fatal("a cancel whose actor was not stored was acknowledged")
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if link.CancelRequested || row.State != "dispatching" {
		t.Fatalf("a cancel was half recorded: intent %v, state %s", link.CancelRequested, row.State)
	}
	_, err = db.Exec(`DROP TRIGGER reject_cancel_actor`)
	must(t, err)

	_, problem = store.RequestMachineCancellation(request.ID, "cozy run cancel")
	fatal(t, problem)
	_, problem = store.RequestMachineCancellation(request.ID, "another cancel caller")
	fatal(t, problem)
	durable("cozy run cancel")
	fatal(t, store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: 2, Event: &v1.RunEvent_State{State: running}}, nil))
	durable("cozy run cancel")
	fatal(t, store.RecordRunOutcomeV1(request.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"}}))
	row, problem = store.RequestRow(request.ID)
	fatal(t, problem)
	if row.State != "succeeded" {
		t.Fatalf("the cancel replaced the run's real success: %s", row.State)
	}
}
