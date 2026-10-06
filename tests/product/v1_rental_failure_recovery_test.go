package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// The owner's rulings as behaviour: "when that pod fails it should recover those jobs and
// schedule them elsewhere if possible", and a request waits in an outbox until a worker
// confirms it; after that it is no longer ours, and a worker failure means the user resubmits.
// When a rental is proven failed, what its machine confirmed over cozy.machine.v1 decides:
//
//	never sent, or sent and unconfirmed   back in the outbox and placed again, same identity
//	selected with --rental                settled with the lost rental's cause; it may not move
//	confirmed                             failed with the machine's loss, saying so
//	private transaction                   fails as retained work; its bytes died with the pod
//	succeeded                             keeps its success: a real outcome is never overwritten
//
// A rental that is only unreachable (degraded) proves nothing: its unconfirmed run waits on it.
//
// The daemon is the real process against a hub that moves the rental to `failed` the way
// Tensorhub does, only after proving provider absence.
func TestARentalsFailureRecoversItsRuns(t *testing.T) {
	root := filepath.Join(scratchBase, "v1-rental-failure-recovery")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = os.RemoveAll(root)
		}
	})
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hubURL+"\n"+
		"tensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")
	hub.packageReleases = map[string]any{"fake/lost@1": rentalReleaseFacts()}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	// The pod is degraded while it holds the work: nothing can attach to it, so the runs wait
	// on it rather than failing for an unrelated transport reason first.
	for id, name := range map[string]string{"rental-lost": "nitian", "rental-unreachable": "kochiya"} {
		hub.add(id, name)
		hub.setState(id, "degraded", "")
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: id, MachineName: name, SKU: "cpu",
			AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "degraded", Hub: hubURL, Address: "127.0.0.1:1"}))
	}
	type arm struct {
		kind, selected, rental        string
		sent, accepted, retained, won bool
	}
	arms := map[string]arm{
		"req-lost-queued-a": {}, "req-lost-queued-b": {},
		"req-lost-selected":  {selected: "rental-lost"},
		"req-lost-sent":      {sent: true},
		"req-lost-running":   {sent: true, accepted: true},
		"job-lost-retained":  {kind: "job", sent: true, accepted: true, retained: true},
		"req-lost-succeeded": {sent: true, accepted: true, won: true},
		"req-unreachable":    {sent: true, rental: "rental-unreachable"},
	}
	for id, arm := range arms {
		if arm.rental == "" {
			arm.rental = "rental-lost"
			arms[id] = arm
		}
		body, _ := canonical.Spell(canonical.Digest([]byte(id)))
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: "idem-" + id, BodyDigest: body, Package: "fake/lost",
			Release: "1", Entrypoint: "generate", Kind: arm.kind, Payload: []byte("{}"), Rental: true, Worker: arm.rental,
			RequestedRental: arm.selected, RetainWork: arm.retained, MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, arm.rental))
		if arm.sent {
			if send, problem := store.MarkRunV1Sent(id); problem != nil || !send {
				t.Fatalf("%s was not sent: %v %v", id, send, problem)
			}
		}
		if arm.accepted {
			fatal(t, store.AcceptRunV1(id, arm.rental, &v1.RunState{Id: id, Number: 1, State: "running", Attempt: 1}))
		}
		if arm.won {
			// Its result is still on the machine: this client could not write it yet.
			fatal(t, store.RecordRunOutcomeV1(id, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"},
				Export: true, Refused: exit.New(exit.Unavailable, "the output folder was full")}))
		}
	}

	startDaemonProcess(t, root)
	// THE MACHINE DIES. That state, not an elapsed clock, is the whole trigger.
	hub.setState("rental-lost", "failed", "readiness.receipt_conflict")

	// Recovered means nothing still waits on the corpse, and each released run was placed
	// again: after its release it waits for a machine and says so.
	replanned := func(id string) bool {
		events, problem := store.EventsAfter(id, 0, 1000)
		fatal(t, problem)
		released := false
		for _, event := range events {
			released = released || event.Type == "request.queued" && event.Payload["machine_id"] == "rental-lost"
			if released && event.Type == "request.parked" {
				return true
			}
		}
		return false
	}
	deadline := time.Now().Add(60 * time.Second)
	for {
		queued, running, problem := store.RentalRunCounts("rental-lost")
		fatal(t, problem)
		if queued == 0 && running == 0 && replanned("req-lost-queued-a") && replanned("req-lost-queued-b") && replanned("req-lost-sent") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("recovery did not finish after 60s: %d queued, %d running\n%s", queued, running, tail(logPath))
		}
		time.Sleep(500 * time.Millisecond)
	}

	for id, arm := range arms {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		link, problem := store.MachineExecution(id)
		fatal(t, problem)
		owed, problem := store.MachineExecutionOwesWork(id)
		fatal(t, problem)
		errType, _, errText, problem := store.SettledFailure(id)
		fatal(t, problem)
		events, problem := store.EventsAfter(id, 0, 100)
		fatal(t, problem)
		said := map[string]map[string]any{}
		for _, event := range events {
			said[event.Type] = event.Payload
		}
		switch {
		case arm.rental == "rental-unreachable":
			if settled(row.State) || link.MachineID != "rental-unreachable" || said["request.queued"] != nil || !owed {
				t.Fatalf("%s left a machine that is only unreachable: %+v link=%q events=%v", id, row, link.MachineID, said)
			}
		case !arm.accepted && arm.selected == "" && !arm.retained:
			queued := said["request.queued"]
			if settled(row.State) || link.MachineID != "" || row.Worker != "" || row.Machine != "nitian" ||
				queued["machine_id"] != "rental-lost" || !strings.Contains(fmt.Sprint(queued["reason"]), "readiness.receipt_conflict") {
				t.Fatalf("%s was not released to be placed again: %+v link=%q events=%v\n%s", id, row, link.MachineID, said, tail(logPath))
			}
			if row.ID != id {
				t.Fatalf("%s was placed again under another identity: %s", id, row.ID)
			}
		case row.Machine != "nitian":
			t.Fatalf("%s lost the machine word it ran against: %q", id, row.Machine)
		case arm.won:
			if row.State != "succeeded" {
				t.Fatalf("%s: its machine's failure overwrote a real success: %+v", id, row)
			}
		case arm.retained:
			if row.State != "failed" || row.RetainWork || errType != "request.state_lost" || errText != records.LostRetainedWorkMessage || owed {
				t.Fatalf("%s: retained work did not fail as lost: %+v %s %s owed=%v", id, row, errType, errText, owed)
			}
		default:
			lost := said["client.machine_lost"]
			if row.State != "failed" || owed || errType != "machine_execution.state_lost" || lost["cause"] != "readiness.receipt_conflict" {
				t.Fatalf("%s did not settle with its machine's loss: %+v %s %v owed=%v\n%s", id, row, errType, lost, owed, tail(logPath))
			}
		}
	}

	// A failed machine has left the current fleet; its diagnosis survives.
	if code, out := runCozy(t, root, "rental", "list", "--json", "--full"); code != 0 || strings.Contains(out, "rental-lost") {
		t.Fatalf("failed rental remained in the current fleet [exit %d]: %s", code, out)
	}
	stored, problem := store.RentalRow("rental-lost")
	fatal(t, problem)
	if stored == nil || stored.MachineName != "nitian" || stored.State != "failed" || stored.Failure.Code != "readiness.receipt_conflict" {
		t.Fatalf("failed rental history lost its diagnosis: %+v", stored)
	}
}
