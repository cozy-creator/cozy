package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Unfinished work that only the retired classic worker session could run ends once,
// named, at Calcifer start, as does a model transfer Calcifer once ran itself; machine
// executions stay.
func TestCalciferStartRetiresClassicWork(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	queued := recordPrivateTransaction(t, store, "classic-queued", "")
	failed := recordPrivateTransaction(t, store, "classic-failed", "")
	_, problem = store.FailQueuedRequest(failed.ID, retainedFailure("source.not_ready", "input is not available yet"))
	fatal(t, problem)
	// A settled run whose classic worker never acknowledged its terminal.
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: queued.Package, WorkerID: "worker"}))
	unacked := offerChildParent(t, store, recordPrivateTransaction(t, store, "classic-unacked", ""))
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: unacked.ID, Attempt: 1, SessionID: "private-boot",
		InvocationDigest: childDigest("1"), TerminalID: "terminal-unacked", TerminalDigest: childDigest("2"),
		Status: "SUCCEEDED", RequestState: "succeeded", Body: []byte(`{}`)})
	fatal(t, problem)
	machine := records.Request{ID: "req-machine-local", IdemKey: "idem-machine-local",
		BodyDigest: queued.BodyDigest, Package: queued.Package, Entrypoint: queued.Entrypoint, Release: queued.Release, PlanID: queued.PlanID,
		LocalInstallationID: queued.LocalInstallationID, Payload: queued.Payload, MachineExecutionObserver: true}
	_, _, problem = store.Submit(machine)
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(machine.ID, "local"))
	transfer := records.Request{ID: "req-pass-through", IdemKey: "idem-pass-through", BodyDigest: queued.BodyDigest,
		Package: "cozy/platform", Entrypoint: "model-pass-through",
		PlanID: "sha256:" + strings.Repeat("0", 64), Payload: []byte("{}")}
	_, _, problem = store.Submit(transfer)
	fatal(t, problem)
	store.Close()

	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	retired, problem := store.RetireClassicWork()
	fatal(t, problem)
	if len(retired) != 2 {
		t.Fatalf("retired %v, want the unfinished classic request and Calcifer's old model transfer", retired)
	}
	attempt, problem := store.AttemptRow(unacked.ID, 1)
	fatal(t, problem)
	if attempt.State != "closed" || attempt.TerminalStatus != "SUCCEEDED" {
		t.Fatalf("an unacknowledged classic terminal stayed open or lost its outcome: %+v", attempt)
	}
	for id, want := range map[string]string{failed.ID: "failed", queued.ID: "failed", unacked.ID: "succeeded", machine.ID: "", transfer.ID: "failed"} {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if want == "" {
			if row == nil || row.State == "failed" {
				t.Fatalf("machine work was retired: %+v", row)
			}
			continue
		}
		if row == nil || row.State != want {
			t.Fatalf("%s = %+v, want %s", id, row, want)
		}
	}
	events, problem := store.EventsAfter(queued.ID, 0, 100)
	fatal(t, problem)
	last := events[len(events)-1]
	if last.Type != "run.failed" || last.Payload["error_type"] != records.ClassicRetiredCode {
		t.Fatalf("classic retirement was not named: %+v", last)
	}
	again, problem := store.RetireClassicWork()
	fatal(t, problem)
	if len(again) != 0 {
		t.Fatalf("retirement was not idempotent: %v", again)
	}
}
