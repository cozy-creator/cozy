package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// rentedRunV1 records a run sent over cozy.machine.v1 to rental pr-owned-machine; accepted
// there when accepted, else its acceptance never reached this client.
func rentedRunV1(t *testing.T, store *records.Store, label string, accepted bool) records.Request {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "lost-" + label, IdemKey: "lost-" + label, Package: "local/rented",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true,
		Rental: true, RequestedRental: "pr-owned-machine"})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-owned-machine"))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": "pr-owned-machine"}))
	if accepted {
		fatal(t, store.AcceptRunV1(request.ID, &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	return *row
}

func rentalLossStore(t *testing.T) *records.Store {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	for _, id := range []string{"pr-owned-machine", "pr-unrelated"} {
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: id, SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
			HourlyRateUSDMicros: 100_000, State: "ready", Hub: "https://hub.example"}))
	}
	return store
}

// A rental confirmed gone settles its runs from the records: it ends what they owe without
// inventing an acceptance or an outcome, a canceled run ends canceled and any other failed,
// and nothing the machine might still say afterwards brings one back.
func TestAnEndedRentalSettlesItsRunsWithoutInventingTheirOutcome(t *testing.T) {
	for _, arm := range []struct {
		name             string
		accepted, cancel bool
		want             string
	}{
		{"sent canceled", false, true, "canceled"}, {"accepted canceled", true, true, "canceled"},
		{"sent", false, false, "failed"}, {"accepted", true, false, "failed"},
	} {
		t.Run(arm.name, func(t *testing.T) {
			store := rentalLossStore(t)
			request := rentedRunV1(t, store, "run", arm.accepted)
			if arm.cancel {
				_, problem := store.RequestMachineCancellation(request.ID, "cozy run cancel")
				fatal(t, problem)
			}
			// Another rental ending, or none yet, discharges nothing: the run may be running.
			_, problem := store.ForgetRental("pr-unrelated")
			fatal(t, problem)
			if owed, problem := store.MachineExecutionOwesWork(request.ID); problem != nil || !owed {
				t.Fatalf("before its machine is known gone the run owes it work: %v %v", owed, problem)
			}

			_, problem = store.ForgetRental("pr-owned-machine")
			fatal(t, problem)
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			lost, problem := store.MachineExecutionLost(request.ID)
			fatal(t, problem)
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			held, problem := store.Obligations()
			fatal(t, problem)
			if row.State != arm.want || !lost || owed || len(held) != 0 {
				t.Fatalf("the run of an ended rental is %s (lost %v, owed %v, holding %+v); want %s", row.State, lost, owed, held, arm.want)
			}
			if accepted, problem := store.RunV1(request.ID); problem != nil || accepted != arm.accepted {
				t.Fatalf("the loss changed the run's acceptance to %v %v", accepted, problem)
			}
			events, problem := store.EventsAfter(request.ID, 0, 1000)
			fatal(t, problem)
			for _, event := range events {
				if event.Type == "run.completed" || event.Payload["scope"] == "before_machine_submission" {
					t.Fatalf("the loss invented %s %v", event.Type, event.Payload)
				}
			}

			// A late word from the gone machine changes nothing, and a cancel after the end
			// is refused rather than turning the run back into canceling.
			_ = store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: 9, Event: &v1.RunEvent_State{State: &v1.RunState{
				Id: request.ID, Number: 1, State: "running", Attempt: 1}}}, nil)
			if _, problem := store.RequestMachineCancellation(request.ID, "late cancel"); problem == nil || problem.Code != exit.Conflict {
				t.Fatalf("a cancel after the machine was gone was taken: %v", problem)
			}
			row, problem = store.RequestRow(request.ID)
			fatal(t, problem)
			owed, problem = store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if row.State != arm.want || owed {
				t.Fatalf("after the end the run became %s (owed %v)", row.State, owed)
			}
		})
	}
}

// A run that succeeded but whose result could not be written here waits for its machine to
// hand it over; once its rental is gone, it no longer holds the daemon.
func TestAnEndedRentalReleasesASucceededRunItCouldNotHandOver(t *testing.T) {
	store := rentalLossStore(t)
	request := rentedRunV1(t, store, "uncollected", true)
	fatal(t, store.RecordRunOutcomeV1(request.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"},
		Export: true, Refused: exit.New(exit.Unavailable, "the output folder was full")}))
	holds := func() bool {
		t.Helper()
		held, problem := store.Obligations()
		fatal(t, problem)
		owed, problem := store.MachineExecutionOwesWork(request.ID)
		fatal(t, problem)
		for _, obligation := range held {
			owed = owed || obligation.ID == request.ID
		}
		return owed
	}
	if !holds() {
		t.Fatal("a succeeded run whose result is still on its machine holds nothing")
	}
	_, problem := store.ForgetRental("pr-owned-machine")
	fatal(t, problem)
	if holds() {
		t.Fatal("a succeeded run of an ended rental still holds the daemon")
	}
}
