package producttest

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestMachineAbandonmentPreservesUncertainEvidenceAndRental(t *testing.T) {
	for _, mode := range []string{"unknown", "unknown-canceling", "accepted"} {
		t.Run(mode, func(t *testing.T) {
			accepted := mode == "accepted"
			store, request, receipt := machineObserverFixture(t)
			fatal(t, store.RecordRental(records.Rental{ID: "pr-owned-machine", State: "ready", MachineName: "retained", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, ManagedRequestID: request.ID, Hub: "http://127.0.0.1:1"}))
			if accepted {
				fatal(t, store.AcceptMachineExecution(request.ID, receipt))
				fatal(t, store.RecordMachineControl(request.ID, &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "pending-cancel", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}))
			} else if mode == "unknown-canceling" {
				_, problem := store.RequestMachineCancellation(request.ID, "")
				fatal(t, problem)
			}
			before, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			changed, problem := store.AbandonMachineExecution(request.ID, "explicit owner")
			fatal(t, problem)
			if !changed {
				t.Fatal("owner intent not recorded")
			}
			changed, problem = store.AbandonMachineExecution(request.ID, "repeated owner")
			fatal(t, problem)
			if changed {
				t.Fatal("replay changed abandoned run")
			}
			after, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if !after.Abandoned || after.Collected != before.Collected || after.SubmissionClosed != before.SubmissionClosed || after.CancelRequested != before.CancelRequested ||
				!bytes.Equal(after.Submission, before.Submission) || !bytes.Equal(after.Receipt, before.Receipt) || !bytes.Equal(after.PendingControl, before.PendingControl) || !bytes.Equal(after.Outcome, before.Outcome) {
				t.Fatal("local abandonment fabricated or discarded remote evidence")
			}
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if owed {
				t.Fatal("abandoned local observer still owes reconciliation")
			}
			lost, problem := store.MachineExecutionLost(request.ID)
			fatal(t, problem)
			if lost {
				t.Fatal("local abandonment claimed machine destruction")
			}
			rented, problem := store.RentalRow("pr-owned-machine")
			fatal(t, problem)
			if rented == nil || rented.State != "ready" {
				t.Fatal("local abandonment changed rental lifecycle")
			}
			// These are the fleet release and replacement predicates, not the
			// observer's local escape. Remote uncertainty still owns the machine.
			live, problem := store.RentalHasLiveMachineExecutions(rented.ID)
			fatal(t, problem)
			owed, problem = orchestrator.RentalOwedBy(store, *rented)
			fatal(t, problem)
			spent, problem := orchestrator.RentalSpent(store, *rented)
			fatal(t, problem)
			if !live || !owed || spent {
				t.Fatalf("abandonment opened fleet release/replacement: live=%v owed=%v spent=%v", live, owed, spent)
			}
			var submission pb.MachineExecutionSubmit
			must(t, proto.Unmarshal(after.Submission, &submission))
			if store.RecordMachineSubmission(request.ID, &submission) == nil {
				t.Fatal("abandoned submission could be replayed")
			}
			if store.LinkMachineExecution(request.ID, "another-machine") == nil {
				t.Fatal("abandoned run moved to another machine")
			}
			events, problem := store.EventsAfter(request.ID, 0, 100)
			fatal(t, problem)
			count := 0
			for _, event := range events {
				if event.Type == "client.machine_abandoned" {
					count++
					if event.Payload["actor"] != "explicit owner" || event.Payload["remote_stop_confirmed"] != false || event.Payload["acceptance_unknown"] == accepted {
						t.Fatal("abandonment lost its actor or uncertainty")
					}
				}
			}
			if count != 1 {
				t.Fatal("abandonment was not idempotent")
			}
		})
	}
}

func TestMachineAbandonmentRetainsLateFactsWithoutReopening(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	_, problem := store.RequestMachineCancellation(request.ID, "")
	fatal(t, problem)
	_, problem = store.AbandonMachineExecution(request.ID, "owner")
	fatal(t, problem)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 1}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "running", BodyCanonicalBytes: []byte(`{}`)}}}))
	state.State, state.Sequence = "succeeded", 2
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 2, HeadSequence: 2}))
	body, err := canonical.Bytes(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: "sha256:" + hex.EncodeToString(receipt.InvocationSpecDigest), Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, SafeMessage: "real late completion"})
	must(t, err)
	outcome := &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest, OutcomeId: "late-outcome", OutcomeDigest: canonical.Digest(body), OutcomeCanonicalBytes: body}
	fatal(t, store.RecordMachineOutcome(request.ID, outcome))
	fatal(t, store.RecordMachineOutcome(request.ID, outcome))
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if row.State != "abandoned" || len(link.Receipt) == 0 || len(link.Outcome) == 0 || !link.CancelRequested {
		t.Fatal("late facts changed owner intent or were discarded")
	}
	if problem := store.RecordMachineControl(request.ID, &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "resume", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RESUME}); problem == nil {
		t.Fatal("abandoned run resumed")
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	late := 0
	for _, event := range events {
		if event.Type == "run.in_progress" || event.Type == "run.completed" {
			t.Fatal("late remote fact reopened local terminal")
		}
		if event.Type == "client.abandoned_machine_outcome" {
			late++
		}
	}
	if late != 1 {
		t.Fatal("late outcome evidence not retained idempotently")
	}
}

func TestMachineAbandonmentDoesNotRewritePriorTerminal(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "succeeded", Sequence: 1}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1}))
	_, problem := store.AbandonMachineExecution(request.ID, "owner")
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if row.State != "succeeded" {
		t.Fatal("abandonment replaced an existing terminal")
	}
}

func TestRunCancelAbandonCLIIsLocalAndSurvivesDaemonRestart(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	request, _ := machineObserverRecord(t, store)
	_, problem = store.RequestMachineCancellation(request.ID, "")
	fatal(t, problem)
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	store.Close()
	svc := startDaemonProcess(t, root)
	refusal := svc.call(t, http.MethodPost, "/v1/local/requests/"+request.ID+"/abandon", map[string]any{})
	if refusal.Status != http.StatusBadRequest {
		t.Fatalf("unnamed abandonment: %s", refusal.brief())
	}
	// A separate ordinary cancel observer is already waiting on unknown remote
	// acceptance. Explicit local abandonment must wake it without inventing stop.
	waiter := exec.Command(cozyBin, "run", "cancel", request.ID, "--await", "--json")
	waiter.Env = childEnv(t, root)
	var waitingOutput bytes.Buffer
	waiter.Stdout, waiter.Stderr = &waitingOutput, &waitingOutput
	setProcessGroup(waiter)
	must(t, waiter.Start())
	t.Cleanup(func() { _ = killGroup(waiter.Process.Pid) })
	waited := make(chan error, 1)
	go func() { waited <- waiter.Wait() }()
	reader, problem := records.OpenReadOnly(layout.DB)
	fatal(t, problem)
	observationBudget := time.After(10 * time.Second)
	for {
		actor, _, _, problem := reader.CancelAttribution(request.ID)
		fatal(t, problem)
		if actor != "" {
			break
		}
		select {
		case <-waited:
			t.Fatal("unknown acceptance completed cancel before abandonment")
		case <-observationBudget:
			t.Fatal("cancel observer did not record its intent")
		case <-time.After(10 * time.Millisecond):
		}
	}
	reader.Close()
	code, out := runCozy(t, root, "run", "cancel", request.ID, "--abandon", "--json")
	if code != 0 {
		t.Fatalf("ordinary abandon [%d]: %s", code, out)
	}
	var result struct {
		AbandonedLocally    bool `json:"abandoned_locally"`
		RemoteStopConfirmed bool `json:"remote_stop_confirmed"`
		RentalReleased      bool `json:"rental_released"`
	}
	must(t, json.Unmarshal([]byte(out), &result))
	if !result.AbandonedLocally || result.RemoteStopConfirmed || result.RentalReleased {
		t.Fatalf("CLI claimed a remote side effect: %s", out)
	}
	select {
	case <-waited:
		if strings.Contains(waitingOutput.String(), `"status":"canceled"`) {
			t.Fatal("local abandonment reported confirmed remote cancellation")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("explicit abandonment left the cancel observer waiting")
	}
	if code, out := runCozy(t, root, "run", "cancel", request.ID, "--abandon", "--await"); code == 0 || !strings.Contains(out, "cannot be combined") {
		t.Fatalf("abandon awaited remote cancellation [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("stop isolated observer [%d]: %s", code, out)
	}
	<-svc.exited
	svc = startDaemonProcess(t, root)
	row := svc.call(t, http.MethodGet, "/v1/requests/"+request.ID+"?recorded=1", nil)
	if row.Status != http.StatusOK || !strings.Contains(string(row.Body), `"abandoned_locally":true`) {
		t.Fatalf("restart forgot abandonment: %s", row.brief())
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("stop restarted observer [%d]: %s", code, out)
	}
	<-svc.exited
	store, problem = records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if after == nil || !after.Abandoned || !bytes.Equal(after.Submission, before.Submission) || len(after.Receipt) != 0 || after.SubmissionClosed || !after.CancelRequested {
		t.Fatal("ordinary CLI/restart lost unknown acceptance evidence")
	}
}

func TestMachineAbandonmentKeepsLateClosureWithoutAnotherTerminal(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	_, problem := store.RequestMachineCancellation(request.ID, "")
	fatal(t, problem)
	_, problem = store.AbandonMachineExecution(request.ID, "owner")
	fatal(t, problem)
	closure := &pb.MachineSubmissionClosure{RequestId: request.ID, SubmissionId: request.IdemKey, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId}
	if store.RefuseMachineSubmission(request.ID, "unproven", "still unknown", nil) == nil {
		t.Fatal("abandonment substituted for nonacceptance proof")
	}
	fatal(t, store.RefuseMachineSubmission(request.ID, "late-refusal", "real closure arrived", closure))
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if row.State != "abandoned" || !link.SubmissionClosed || len(link.Receipt) != 0 {
		t.Fatal("late closure changed owner intent or lost its evidence")
	}
	live, problem := store.RentalHasLiveMachineExecutions("pr-owned-machine")
	fatal(t, problem)
	if live {
		t.Fatal("proven nonacceptance retained the remote execution fence")
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	fatal(t, problem)
	terminals := 0
	for _, event := range events {
		if event.Type == "run.failed" {
			terminals++
		}
	}
	if terminals != 1 {
		t.Fatal("late refusal added another terminal")
	}
}

func TestMachineAbandonmentKeepsPendingCancelAfterLateRelease(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	command := &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "original-cancel", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}
	fatal(t, store.RecordMachineControl(request.ID, command))
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	_, problem = store.AbandonMachineExecution(request.ID, "owner")
	fatal(t, problem)
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 2, AttemptOrdinal: 1, State: "canceled", Sequence: 1}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "retention_released", BodyCanonicalBytes: []byte(`{}`)}}}))
	fatal(t, store.CompleteMachineControl(request.ID, before.PendingControl))
	fatal(t, store.RejectMachineControl(request.ID, before.PendingControl))
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if !after.CancelRequested || !bytes.Equal(after.PendingControl, before.PendingControl) {
		t.Fatal("late cleanup discarded the abandoned control evidence")
	}
	// A real Runtime release removes the fleet fence while local evidence stays.
	live, problem := store.RentalHasLiveMachineExecutions("pr-owned-machine")
	fatal(t, problem)
	owed, problem := store.RentalHasMachineObligations("pr-owned-machine")
	fatal(t, problem)
	if live || owed {
		t.Fatal("authoritative Runtime release did not discharge remote lifecycle custody")
	}
}
