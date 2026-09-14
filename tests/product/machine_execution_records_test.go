package producttest

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func machineObserverFixture(t *testing.T) (*records.Store, records.Request, *pb.MachineExecutionReceipt) {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	request, receipt := machineObserverRecord(t, store)
	return store, request, receipt
}

func machineObserverRecord(t *testing.T, store *records.Store) (records.Request, *pb.MachineExecutionReceipt) {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "job-observed", IdemKey: "observed", Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-owned-machine"))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{
		SubmissionId: request.IdemKey, CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)},
	}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: 1000, WorkerId: "worker", WorkerBootId: "boot-1", ExecutionWorkspaceId: "persistent-workspace"}
	return request, receipt
}

type retentionReleaseMachine struct {
	store   *records.Store
	state   *pb.MachineExecutionState
	control atomic.Int32
}

func (m *retentionReleaseMachine) Refresh(context.Context, records.Request) *exit.Error { return nil }

func (m *retentionReleaseMachine) Control(_ context.Context, request records.Request, action string) *exit.Error {
	if action != "cancel" {
		return exit.Usagef("expected cancellation")
	}
	m.control.Add(1)
	m.state.Sequence++
	return m.store.ObserveMachineExecution(request.ID, m.state, &pb.MachineExecutionEventPage{
		NextAfter: m.state.Sequence, HeadSequence: m.state.Sequence,
		Events: []*pb.MachineExecutionEvent{{Sequence: m.state.Sequence, AttemptOrdinal: 1, AtMs: 1002, Kind: "retention_released", BodyCanonicalBytes: []byte(`{}`)}},
	})
}

func TestMachineCancellationAfterDeadlineReleasesRetainedWork(t *testing.T) {
	o := hostOwner(t, "machine-deadline-cancellation")
	request, receipt := machineObserverRecord(t, o.store)
	fatal(t, o.store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: "worker", WorkerBootId: "boot-1", ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "canceled", Sequence: 1}
	fatal(t, o.store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1,
		Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "terminal", BodyCanonicalBytes: []byte(`{}`)}},
	}))
	machine := &retentionReleaseMachine{store: o.store, state: state}
	defer publicationControlAPI(t, o, func(options *api.Options) { options.MachineExecutions = machine })()
	code, out := runCozy(t, o.root, "run", "cancel", request.ID, "--json")
	if code != 0 || !strings.Contains(out, `"changed":true`) || machine.control.Load() != 1 {
		t.Fatalf("terminal cancellation skipped Runtime retention release: [%d] %s controls=%d", code, out, machine.control.Load())
	}
	owed, problem := o.store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if owed {
		t.Fatal("explicit Runtime retention release remained owed")
	}
}

func TestMachineObserverCannotOwnAttemptsBeforeOrAfterAcceptance(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	for _, accepted := range []bool{false, true} {
		if accepted {
			fatal(t, store.AcceptMachineExecution(request.ID, receipt))
		}
		_, problem := store.Dispatch(records.Attempt{RequestID: request.ID, InstanceID: "local-worker", SessionID: "local-session", InvocationDigest: childDigest("3"), InvocationCanonical: []byte(`{}`)})
		if problem == nil {
			t.Fatalf("machine observation minted a local attempt (accepted=%v)", accepted)
		}
		attempts, problem := store.Attempts(request.ID)
		fatal(t, problem)
		if len(attempts) != 0 {
			t.Fatalf("machine observer retained %d local attempts", len(attempts))
		}
	}
}

func TestMachineRetryUsesRetainedAuthorityAfterErrorCollection(t *testing.T) {
	store, prior, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(prior.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: prior.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "failed", Collected: true}
	fatal(t, store.ObserveMachineExecution(prior.ID, state, &pb.MachineExecutionEventPage{}))
	request := records.Request{ID: "job-retry-machine", IdemKey: "retry-machine", Kind: "job", Package: prior.Package, Entrypoint: prior.Entrypoint, Payload: []byte(`{}`), BodyDigest: childDigest("6"), RetainWork: true, RetryOf: prior.ID, MachineExecutionObserver: true}
	retried, fresh, problem := store.Submit(request)
	fatal(t, problem)
	if !fresh || retried.RetryOf != prior.ID {
		t.Fatal("collected error prevented retry against retained machine authority")
	}
	fatal(t, store.RecordMachineControl(prior.ID, &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: prior.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "cancel-before-retry", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}))
	request.ID, request.IdemKey = "job-retry-after-cancel", "retry-after-cancel"
	if _, _, problem := store.Submit(request); problem == nil {
		t.Fatal("retry raced past pending authoritative cancellation")
	}
}

func TestExplicitClientShutdownRequiresDurableMachineAcceptance(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	for _, accepted := range []bool{false, true} {
		if accepted {
			fatal(t, store.AcceptMachineExecution(request.ID, receipt))
		}
		blocked, problem := store.ClientShutdownObligations()
		fatal(t, problem)
		if (len(blocked) == 0) != accepted {
			t.Fatalf("explicit disconnect ignored durable acceptance: accepted=%v obligations=%+v", accepted, blocked)
		}
		all, problem := store.Obligations()
		fatal(t, problem)
		if len(all) == 0 {
			t.Fatal("explicit disconnect weakened automatic idle retention")
		}
	}
	command := &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "pause-before-disconnect", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_PAUSE}
	fatal(t, store.RecordMachineControl(request.ID, command))
	blocked, problem := store.ClientShutdownObligations()
	fatal(t, problem)
	if len(blocked) == 0 {
		t.Fatal("explicit disconnect abandoned a pending control")
	}
}

func TestMachineAcceptanceCannotChangeDestinationOrCapture(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	if problem := store.LinkMachineExecution(request.ID, "pr-another-machine"); problem == nil {
		t.Fatal("ambiguous submission moved to another machine")
	}
	foreign := proto.Clone(receipt).(*pb.MachineExecutionReceipt)
	foreign.PublicationAuthorizationId = "9a4c3c53-564b-4497-8398-ac0f55bcc2cc"
	if problem := store.AcceptMachineExecution(request.ID, foreign); problem == nil {
		t.Fatal("accepted publication authority absent from the frozen submission")
	}
	foreign = proto.Clone(receipt).(*pb.MachineExecutionReceipt)
	foreign.CaptureDigest = bytes.Repeat([]byte{7}, 32)
	if problem := store.AcceptMachineExecution(request.ID, foreign); problem == nil {
		t.Fatal("accepted another captured program")
	}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	fatal(t, store.AcceptMachineExecution(request.ID, proto.Clone(receipt).(*pb.MachineExecutionReceipt)))
	foreign = proto.Clone(receipt).(*pb.MachineExecutionReceipt)
	foreign.ExecutionWorkspaceId = "replacement-empty-workspace"
	if problem := store.AcceptMachineExecution(request.ID, foreign); problem == nil {
		t.Fatal("replaced accepted Runtime workspace identity")
	}
}

func TestMachineObservationReplaysAcrossWorkerRestartWithoutLocalAttempts(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: "worker", WorkerBootId: "boot-1", ExecutionWorkspaceId: "persistent-workspace", Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 2}
	page := &pb.MachineExecutionEventPage{NextAfter: 2, HeadSequence: 2, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "accepted", BodyCanonicalBytes: []byte(`{}`)}, {Sequence: 2, AttemptOrdinal: 1, AtMs: 1002, Kind: "started", BodyCanonicalBytes: []byte(`{}`)}}}
	fatal(t, store.ObserveMachineExecution(request.ID, state, page))
	fatal(t, store.ObserveMachineExecution(request.ID, state, page))
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link.RemoteCursor != 2 {
		t.Fatalf("event cursor = %d", link.RemoteCursor)
	}
	state.WorkerBootId, state.State, state.Sequence = "boot-2", "succeeded", 3
	page = &pb.MachineExecutionEventPage{NextAfter: 3, HeadSequence: 3, Events: []*pb.MachineExecutionEvent{{Sequence: 3, AttemptOrdinal: 1, AtMs: 1003, Kind: "terminal", BodyCanonicalBytes: []byte(`{"status":"SUCCEEDED"}`)}}}
	fatal(t, store.ObserveMachineExecution(request.ID, state, page))
	stale := proto.Clone(state).(*pb.MachineExecutionState)
	stale.State, stale.Sequence = "running", 2
	fatal(t, store.ObserveMachineExecution(request.ID, stale, &pb.MachineExecutionEventPage{NextAfter: 2, HeadSequence: 2}))
	observed, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if observed.State != "succeeded" || observed.Ordinal != 1 {
		t.Fatalf("stale observation changed completion: %+v", observed)
	}
	wrong := proto.Clone(state).(*pb.MachineExecutionState)
	wrong.ExecutionWorkspaceId = "new-empty-disk"
	if problem := store.ObserveMachineExecution(request.ID, wrong, page); problem == nil {
		t.Fatal("worker restart silently replaced retained execution journal")
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("observing remote history created local execution authority")
	}
}

func TestMachineCancellationPreservesAmbiguousAcceptance(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	handled, problem := store.CancelMachineBeforeAcceptance(request.ID)
	fatal(t, problem)
	if !handled {
		t.Fatal("outbound submission lost its pending cancellation")
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if row.State != "canceling" {
		t.Fatalf("ambiguous submission was falsely settled: %s", row.State)
	}
	// The delayed receipt is still recorded, so the same real execution can be
	// canceled remotely instead of silently abandoning a running job.
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if !link.CancelRequested || len(link.Receipt) == 0 {
		t.Fatal("late acceptance dropped explicit cancellation intent")
	}
	command := &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "cancel-after-lost-reply", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}
	fatal(t, store.RecordMachineControl(request.ID, command))
	fatal(t, store.RecordMachineControl(request.ID, command))
	changed := proto.Clone(command).(*pb.MachineExecutionControl)
	changed.ExpectedGeneration++
	if problem := store.RecordMachineControl(request.ID, changed); problem == nil {
		t.Fatal("a retry changed the pending command's generation")
	}
	link, problem = store.MachineExecution(request.ID)
	fatal(t, problem)
	fatal(t, store.CompleteMachineControl(request.ID, link.PendingControl))
	fatal(t, store.CompleteMachineControl(request.ID, link.PendingControl))
	link, problem = store.MachineExecution(request.ID)
	fatal(t, problem)
	if !link.CancelRequested || len(link.PendingControl) == 0 {
		t.Fatal("control acknowledgement was mistaken for native retention release")
	}
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: "worker", WorkerBootId: "boot-1", ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 2, AttemptOrdinal: 1, State: "canceled", Sequence: 1}
	page := &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1002, Kind: "retention_released", BodyCanonicalBytes: []byte(`{"collected":false}`)}}}
	fatal(t, store.ObserveMachineExecution(request.ID, state, page))
	link, problem = store.MachineExecution(request.ID)
	fatal(t, problem)
	if link.CancelRequested || len(link.PendingControl) != 0 {
		t.Fatal("authoritative retention release did not finish cancellation")
	}
}

func TestRuntimeObligationsPreventPrematureRentalRelease(t *testing.T) {
	for _, scenario := range []struct {
		name, state                         string
		accepted, collected, released, owed bool
	}{
		{name: "ambiguous_intake", owed: true},
		{name: "running", state: "running", accepted: true, owed: true},
		{name: "paused", state: "paused", accepted: true, owed: true},
		{name: "failed_after_error_collection", state: "failed", accepted: true, collected: true, owed: true},
		{name: "success_before_collection", state: "succeeded", accepted: true, owed: true},
		{name: "canceled_before_native_cleanup", state: "canceled", accepted: true, owed: true},
		{name: "successful_collection", state: "succeeded", accepted: true, collected: true},
		{name: "canceled_after_native_cleanup", state: "canceled", accepted: true, released: true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			store, request, receipt := machineObserverFixture(t)
			if scenario.accepted {
				fatal(t, store.AcceptMachineExecution(request.ID, receipt))
				state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: "worker", WorkerBootId: "boot-1", ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: scenario.state, Collected: scenario.collected, Sequence: 1}
				kind := "observed"
				if scenario.released {
					kind = "retention_released"
				}
				page := &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: kind, BodyCanonicalBytes: []byte(`{}`)}}}
				fatal(t, store.ObserveMachineExecution(request.ID, state, page))
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
				t.Fatal("retained Runtime execution became spent capacity")
			}
			attempts, problem := store.Attempts(request.ID)
			fatal(t, problem)
			if len(attempts) != 0 {
				t.Fatal("release protection depended on a local attempt")
			}
		})
	}
}

func TestMachineInvocationCarriesFrozenDeadlineAndRefusesUnstagedInputs(t *testing.T) {
	request := records.Request{ID: "job-deadline", IdemKey: "deadline", Kind: "job", Package: "local/example", Entrypoint: "main", Org: "local", Payload: []byte(`{}`), PlanID: childDigest("2"), LocalPackageDigest: childDigest("3"), DeadlineUnixMS: 1900000000123}
	plan := &orchestrator.JobPlan{Function: "main", DescriptorID: request.PlanID}
	capture := localpackage.ExecutionCapture{Canonical: []byte(`{}`), Digest: canonical.Digest([]byte(`{}`))}
	submission, problem := orchestrator.MachineJobSubmission(request, capture, plan, nil)
	fatal(t, problem)
	var spec pb.InvocationSpec
	must(t, canonical.Unmarshal(submission.Offer.InvocationSpecCanonicalBytes, &spec))
	if spec.DeadlineUnixMs != request.DeadlineUnixMS || submission.MaxAttempts != uint32(orchestrator.MaxRequeues+1) {
		t.Fatal("machine execution dropped its deadline or retry bound")
	}
	if !submission.PreparedState.GetJob().Orchestration {
		t.Fatal("CPU captured root occupied its managed children's device lane")
	}
	request.Models = []records.ModelRef{{Slot: "model", Manifest: childDigest("4"), ManifestLength: 123}}
	if _, problem := orchestrator.MachineJobSubmission(request, capture, plan, nil); problem == nil || problem.Code != exit.Structural {
		t.Fatalf("unstaged inputs would retry indefinitely: %v", problem)
	}
}

func TestMachineCancellationBeforeTransmissionCreatesNoRemotePromise(t *testing.T) {
	store, _, _ := machineObserverFixture(t)
	request, _, problem := store.Submit(records.Request{ID: "job-unsent", IdemKey: "unsent", Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("2"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	handled, problem := store.CancelMachineBeforeAcceptance(request.ID)
	fatal(t, problem)
	if !handled {
		t.Fatal("unsent intent was not canceled")
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if row.State != "canceled" || len(link.Submission) != 0 || len(link.Receipt) != 0 {
		t.Fatal("unsent cancellation invented machine acceptance")
	}
	capture, spec := []byte(`{}`), []byte(`{}`)
	if problem := store.RecordMachineSubmission(request.ID, &pb.MachineExecutionSubmit{
		SubmissionId: request.IdemKey, CaptureDigest: canonical.Digest(capture), CaptureCanonicalBytes: capture,
		Offer: &pb.AttemptOffer{RequestId: request.ID, InvocationSpecDigest: canonical.Digest(spec), InvocationSpecCanonicalBytes: spec},
	}); problem == nil {
		t.Fatal("late preparation revived an already canceled unsent request")
	}
}
