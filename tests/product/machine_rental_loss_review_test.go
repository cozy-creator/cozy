package producttest

import (
	"bytes"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestDestroyedUnacceptedExecutionCannotReturnToCanceling(t *testing.T) {
	store, request, _ := machineObserverFixture(t)
	_, problem := store.ForgetRental("pr-owned-machine")
	fatal(t, problem)
	before, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if before.State != "failed" {
		t.Fatal(before.State)
	}
	_, _ = store.RequestMachineCancellation(request.ID, "")
	after, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	held, problem := store.Obligations()
	fatal(t, problem)
	if after.State == "canceling" || owed || len(held) != 0 {
		t.Fatalf("late cancellation revived destroyed execution: before=%s after=%s owes_work=%v obligations=%+v", before.State, after.State, owed, held)
	}
}

func TestEndedRentalDoesNotTurnRemoteModelRetentionIntoLocalCustody(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	_, envelope := retainedModelEnvelope(t)
	observedLossOutcome(t, store, request, receipt, envelope)
	model := envelope.RetainedModels[0]
	hold := records.MachineModelRetention{OutcomeID: "loss-outcome", Artifact: model.ModelArtifactCanonicalBytes, TransactionID: model.Retention.WeightsTransactionId, ReceiptDigest: model.Retention.TensorfsReceiptDigest, SourceRetentionID: model.Retention.RetentionId, RetentionID: childDigest("a")}
	fatal(t, store.FreezeMachineModelRetention(request.ID, hold))
	fatal(t, store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "held"))
	before, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	_, problem = store.ForgetRental("pr-owned-machine")
	fatal(t, problem)
	after, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	holds, problem := store.MachineModelRetentions(request.ID)
	fatal(t, problem)
	outputs, problem := store.VisibleOutputs(request.ID)
	fatal(t, problem)
	if after.Collected || !bytes.Equal(before.Outcome, after.Outcome) || len(holds) != 1 || holds[0].State != "held" || len(outputs) != 0 {
		t.Fatal("remote Model hold became fabricated local custody or a release receipt")
	}
}

func observedLossOutcome(t *testing.T, store *records.Store, request records.Request, receipt *pb.MachineExecutionReceipt, result *pb.ResultEnvelope) {
	t.Helper()
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "succeeded"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	spec, _ := canonical.Spell(receipt.InvocationSpecDigest)
	raw, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: spec, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, Result: result})
	must(t, err)
	fatal(t, store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest, OutcomeId: "loss-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: raw}))
}

func TestEndedSucceededArtifactRequestDoesNotKeepShutdownBlocked(t *testing.T) {
	store, original, originalReceipt := machineObserverFixture(t)
	link, problem := store.MachineExecution(original.ID)
	fatal(t, problem)
	request := original
	request.ID, request.IdemKey = "job-retained", "retained"
	request.RetainWork, request.ChildArtifacts, request.MachineExecutionObserver = true, true, true
	request, _, problem = store.Submit(request)
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-owned-machine"))
	var submission pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(link.Submission, &submission))
	submission.SubmissionId, submission.Offer.RequestId = request.IdemKey, request.ID
	fatal(t, store.RecordMachineSubmission(request.ID, &submission))
	receipt := proto.Clone(originalReceipt).(*pb.MachineExecutionReceipt)
	receipt.RequestId, receipt.SubmissionId = request.ID, request.IdemKey
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "succeeded"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	fatal(t, store.RecordMachineControl(request.ID, &pb.MachineExecutionControl{Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}, CommandId: "cancel-retained-result", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}))
	_, problem = store.ForgetRental("pr-owned-machine")
	fatal(t, problem)
	held, problem := store.Obligations()
	fatal(t, problem)
	after, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if len(held) != 0 {
		t.Fatalf("destroyed successful retained work still holds the daemon: retain=%v state=%s obligations=%+v", after.RetainWork, after.State, held)
	}
	if after.State != "succeeded" {
		t.Fatal("loss rewrote the historical successful execution")
	}
}
