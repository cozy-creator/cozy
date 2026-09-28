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
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// TestRentalFailureRecovery is the owner's ruling as behaviour: "it's fine for cozy-daemon
// to assign work to specific pods, but when that pod fails it should recover those jobs and
// schedule them elsewhere if possible".
//
// Observed live 2026-09-04: rental pr-183abac284d1e16f5f0a (nitian) went `failed` with
// readiness.receipt_conflict while holding one in-flight and two queued requests, and
// nothing released them. Every rented call is now a Runtime execution linked to its
// machine, so the arms are drawn by what crossed to that machine, not by a local attempt:
//
//	offer never sent        released and placed again; nothing ran, nothing is charged
//	offer sent              lost with the machine that may have run it, saying so
//	selected with --rental  settled with the lost rental's cause; it may not move
//	private transaction     fails as retained work; its bytes died with the pod
//
// The daemon is the real process against a hub that moves the rental to `failed` the way
// Tensorhub does, only after proving provider absence.
func TestRentalFailureRecovery(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-failure-recovery")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	hub := newFakeRentalHub(t, 0)
	port := hub.port()
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")

	hub.packageReleases = map[string]any{"fake/lost@1": rentalReleaseFacts()}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	// The pod is degraded while it holds the work: nothing can attach to it, so the runs
	// wait on it rather than failing for an unrelated transport reason first.
	hub.add("rental-lost", "nitian")
	hub.setState("rental-lost", "degraded", "")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "rental-lost", MachineName: "nitian", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "degraded", Hub: hubURL, Address: "127.0.0.1:1",
	}))
	type arm struct {
		kind, selected string
		sent, accepted bool
		retained       bool
	}
	arms := map[string]arm{
		"req-lost-queued-a": {}, "req-lost-queued-b": {},
		"req-lost-selected": {selected: "rental-lost"},
		"req-lost-sent":     {sent: true},
		"req-lost-running":  {sent: true, accepted: true},
		"job-lost-retained": {kind: "job", sent: true, accepted: true, retained: true},
	}
	for id, arm := range arms {
		body, _ := canonical.Spell(canonical.Digest([]byte(id)))
		request, _, problem := store.Submit(records.Request{ID: id, IdemKey: "idem-" + id,
			BodyDigest: body, Package: "fake/lost", Release: "1", Entrypoint: "generate",
			Kind: arm.kind, Payload: []byte("{}"), Rental: true, Worker: "rental-lost",
			RequestedRental: arm.selected, RetainWork: arm.retained, MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, "rental-lost"))
		if !arm.sent {
			continue
		}
		capture, spec := []byte(`{"capture":"`+id+`"}`), []byte(`{"invocation":"`+id+`"}`)
		submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
			CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
			Offer: &pb.AttemptOffer{RequestId: id, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
		fatal(t, store.RecordMachineSubmission(id, submission))
		if !arm.accepted {
			continue
		}
		receipt := &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: request.IdemKey,
			CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest,
			AcceptedAtMs: 1000, WorkerId: "worker", WorkerBootId: "boot", ExecutionWorkspaceId: "workspace"}
		fatal(t, store.AcceptMachineExecution(id, receipt))
		fatal(t, store.ObserveMachineExecution(id, &pb.MachineExecutionState{RequestId: id, WorkerId: "worker",
			WorkerBootId: "boot", ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "running"},
			&pb.MachineExecutionEventPage{}))
	}
	queued, running, problem := store.RentalRunCounts("rental-lost")
	fatal(t, problem)
	if queued != 4 || running != 2 {
		t.Fatalf("the rental should hold 4 queued and 2 running before it fails, got %d/%d", queued, running)
	}

	startDaemonProcess(t, root)
	// THE MACHINE DIES. That state — not an elapsed clock — is the whole trigger.
	hub.setState("rental-lost", "failed", "readiness.receipt_conflict")

	// Recovered means nothing still waits on the corpse, and each released run was placed
	// again. This hub offers no machine, which is weather: the run waits and says so.
	replanned := func(id string) bool {
		return strings.Contains(lastEventField(t, store, id, "request.parked", "reason"), "offered no CPU product")
	}
	deadline := time.Now().Add(60 * time.Second)
	for {
		queued, running, problem = store.RentalRunCounts("rental-lost")
		fatal(t, problem)
		if queued == 0 && running == 0 && replanned("req-lost-queued-a") && replanned("req-lost-queued-b") {
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
		if errType == "rental.inference_runtime_owned" {
			t.Fatalf("%s: failed as classic dispatch: %+v %s", id, row, errType)
		}
		switch {
		case !arm.sent && arm.selected == "":
			queued := said["request.queued"]
			if settled(row.State) || link.MachineID != "" || row.Worker != "" || row.Machine != "nitian" ||
				queued["machine_id"] != "rental-lost" || !strings.Contains(fmt.Sprint(queued["reason"]), "readiness.receipt_conflict") {
				t.Fatalf("%s was not released to be placed again: %+v link=%q events=%v\n%s",
					id, row, link.MachineID, said, tail(logPath))
			}
		case row.Machine != "nitian":
			t.Fatalf("%s lost the machine word it ran against: %q", id, row.Machine)
		case arm.retained:
			if row.State != "failed" || row.RetainWork || errType != "request.state_lost" || errText != records.LostRetainedWorkMessage || owed {
				t.Fatalf("%s: retained work did not fail as lost: %+v %s %s owed=%v", id, row, errType, errText, owed)
			}
		default:
			lost := said["client.machine_lost"]
			if row.State != "failed" || owed || errType != "machine_execution.state_lost" ||
				lost["cause"] != "readiness.receipt_conflict" || lost["had_acceptance_receipt"] != arm.accepted {
				t.Fatalf("%s did not settle with the machine's loss: %+v %s %v owed=%v", id, row, errType, lost, owed)
			}
		}
	}
	if log, err := os.ReadFile(logPath); err != nil || strings.Contains(string(log), "inference_runtime_owned") {
		t.Fatalf("rented work reached classic dispatch [%v]:\n%s", err, tail(logPath))
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

// TestRentalFailureKeepsARecordedTerminal is the arm that must NOT fire. An attempt whose
// worker already committed a terminal holds a real outcome that has not been acked yet.
// Recovery must leave it completely alone: replacing it with a synthetic failure would
// publish `request.failed` over a run that succeeded, which the dispatch code calls the
// worst failure class this system has.
func TestRentalFailureKeepsARecordedTerminal(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-failure-terminal")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()

	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "rental-terminal", MachineName: "gone", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: "http://127.0.0.1:1",
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "rental-terminal.pem"),
	}))
	if _, _, problem := store.Submit(records.Request{
		ID: "req-terminal", IdemKey: "idem-req-terminal",
		BodyDigest: "sha256:" + strings.Repeat("e5", 32),
		Package:    "fake/lost", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true, Worker: "rental-terminal",
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	session := "session-terminal"
	fatal(t, store.SpawnWorker(records.WorkerProcess{
		InstanceID: "ins-terminal", Package: "fake/lost", WorkerID: "remote", Devices: []string{"cpu"}}))
	digest := "sha256:" + strings.Repeat("f6", 32)
	attempt, problem := store.Dispatch(records.Attempt{
		RequestID: "req-terminal", SessionID: session, InstanceID: "ins-terminal",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch("req-terminal", attempt, session))
	fatal(t, store.Accepted("req-terminal", attempt, session))
	applied, problem := store.AcceptTerminal(records.Terminal{
		RequestID: "req-terminal", Attempt: attempt, SessionID: session,
		InvocationDigest: digest, TerminalID: "out-req-terminal",
		TerminalDigest: "sha256:" + strings.Repeat("07", 32),
		Status:         "SUCCEEDED", Cause: "COMPLETED"})
	fatal(t, problem)
	if !applied {
		t.Fatal("the terminal was not recorded")
	}

	// The store refuses to strand it, whatever the caller believes about the machine.
	stranded, problem := store.AbandonLostAttempt("req-terminal", attempt, "the pod is gone",
		records.RequeueAfterLoss)
	fatal(t, problem)
	if stranded {
		t.Fatal("a recorded terminal was overwritten with a synthetic failure")
	}
	row := attemptRow(t, store, "req-terminal", attempt)
	if row.TerminalStatus != "SUCCEEDED" {
		t.Fatalf("the committed outcome changed to %q", row.TerminalStatus)
	}
}

func attemptRow(t *testing.T, store *records.Store, requestID string, attempt int64) records.Attempt {
	t.Helper()
	attempts, problem := store.Attempts(requestID)
	fatal(t, problem)
	for _, a := range attempts {
		if a.Attempt == attempt {
			return a
		}
	}
	t.Fatalf("%s#%d has no attempt row", requestID, attempt)
	return records.Attempt{}
}

func settled(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}
