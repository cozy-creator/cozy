package producttest

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func machineObserverFixture(t *testing.T) (*records.Store, records.Request, *pb.MachineExecutionReceipt) {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
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
	return store, request, receipt
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

func TestMachineAcceptanceCannotChangeDestinationOrCapture(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	if problem := store.LinkMachineExecution(request.ID, "pr-another-machine"); problem == nil {
		t.Fatal("ambiguous submission moved to another machine")
	}
	foreign := proto.Clone(receipt).(*pb.MachineExecutionReceipt)
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
