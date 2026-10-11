package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A rental holding a cozy.machine.v1 run is owed until this client holds the run's result:
// sent but unconfirmed, running, paused, or ended with its collection refused. Releasing it
// earlier would destroy work only that machine has.
func TestRuntimeObligationsPreventPrematureRentalRelease(t *testing.T) {
	for _, scenario := range []struct {
		name, state            string
		sent, ended, collected bool
		owed                   bool
	}{
		{name: "unsent", owed: true},
		{name: "sent_unconfirmed", sent: true, owed: true},
		{name: "running", sent: true, state: "running", owed: true},
		{name: "paused", sent: true, state: "paused", owed: true},
		{name: "ended_not_collected", sent: true, state: "running", ended: true, owed: true},
		{name: "ended_and_collected", sent: true, state: "running", ended: true, collected: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			request, _, problem := store.Submit(records.Request{ID: "job-owed", IdemKey: "owed", Package: "local/example", Entrypoint: "main",
				Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, "pr-owned-machine"))
			if scenario.sent {
				_, problem = store.MarkRunV1Sent(request.ID)
				fatal(t, problem)
			}
			if scenario.state != "" {
				fatal(t, store.AcceptRunV1(request.ID, "pr-owned-machine", &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
				fatal(t, store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: 1, Event: &v1.RunEvent_State{State: &v1.RunState{
					Id: request.ID, Number: 1, State: scenario.state, Sequence: 1, Attempt: 1}}}, nil))
			}
			if scenario.ended {
				end := records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded", Result: []byte(`{}`)}}
				if !scenario.collected {
					end.Refused = exit.Named(exit.Unavailable, "output.write_failed", "this computer could not write the run's output")
				}
				fatal(t, store.RecordRunOutcomeV1(request.ID, end))
			}
			rental := records.Rental{ID: "pr-owned-machine", ManagedRequestID: request.ID, State: "ready"}
			owed, problem := orchestrator.RentalOwedBy(store, rental)
			fatal(t, problem)
			if owed != scenario.owed {
				t.Fatalf("rental owed=%v, want %v", owed, scenario.owed)
			}
			spent, problem := orchestrator.RentalSpent(store, rental)
			fatal(t, problem)
			if scenario.owed && spent {
				t.Fatal("a rental holding a run became spent capacity")
			}
			attempts, problem := store.Attempts(request.ID)
			fatal(t, problem)
			if len(attempts) != 0 {
				t.Fatal("release protection depended on a local attempt")
			}
		})
	}
}

// The same on a real machine: a job running on a rental holds it, and the result this client
// collected frees it.
func TestARentalRunningAV1JobIsOwedUntilItsResultIsHeld(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", restartProject(t)); code != 0 {
		t.Fatalf("installing the restart package [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/restart-proof/slow", "seconds=4", "--rental=tessa", "--json", "--idempotency-key", "owed"); code != 0 {
		t.Fatalf("submitting the job [exit %d]\n%s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey("owed")
	fatal(t, problem)
	owed := func() bool {
		t.Helper()
		rental, problem := store.RentalRow(parityRental)
		fatal(t, problem)
		owed, problem := orchestrator.RentalOwedBy(store, *rental)
		fatal(t, problem)
		return owed
	}
	eventually(t, root, "the job running on the rental", func() bool {
		row, problem := store.RequestRow(request.ID)
		return problem == nil && row != nil && row.State == "dispatching"
	})
	if !owed() {
		t.Fatal("a rental running a job was not owed")
	}
	if code, out := runCozy(t, root, "run", "watch", request.ID, "--json"); code != 0 || !strings.Contains(out, `"value":4`) {
		t.Fatalf("the job did not complete [exit %d]\n%s", code, out)
	}
	eventually(t, root, "the rental freed once the result is held", func() bool { return !owed() })
}
