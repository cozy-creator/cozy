package producttest

import (
	"bytes"
	"os"
	"path/filepath"
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
	_, _ = store.CancelMachineBeforeAcceptance(request.ID)
	after, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	blocked, problem := store.ClientShutdownObligations()
	fatal(t, problem)
	if after.State == "canceling" || owed || len(blocked) != 0 {
		t.Fatalf("late cancellation revived destroyed execution: before=%s after=%s owes_work=%v shutdown=%+v", before.State, after.State, owed, blocked)
	}
}

func TestEndedRentalExposesOnlyVerifiedLocalFileCopies(t *testing.T) {
	for _, copied := range []bool{false, true} {
		t.Run(map[bool]string{false: "uncopied", true: "copied"}[copied], func(t *testing.T) {
			store, request, receipt := machineObserverFixture(t)
			observedLossOutcome(t, store, request, receipt, nil)
			path := filepath.Join(t.TempDir(), "received.txt")
			data := []byte("locally retained output\n")
			must(t, os.WriteFile(path, data, 0600))
			digest, _ := canonical.Spell(canonical.Digest(data))
			file := records.MachineFileResult{OutcomeID: "loss-outcome", RetentionID: childDigest("8"),
				Source: records.ByteOutput{RequestID: request.ID, Attempt: 1, OutputID: "file", Digest: digest, Length: int64(len(data)), MimeType: "text/plain", ProducerRootID: childDigest("5"), ReceiptDigest: childDigest("6"), ManifestID: childDigest("7"), ManifestLength: 123, ContentBytes: int64(len(data))},
				Output: records.Output{OutputID: "file", MediaID: "local-media", Path: path, Digest: digest, Length: int64(len(data)), MimeType: "text/plain"}}
			_, problem := store.FreezeMachineFileResult(request.ID, file)
			fatal(t, problem)
			if copied {
				fatal(t, store.AdvanceMachineFileResult(request.ID, file.RetentionID, "copied"))
			}
			before, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			_, problem = store.ForgetRental("pr-owned-machine")
			fatal(t, problem)
			outputs, problem := store.VisibleOutputs(request.ID)
			fatal(t, problem)
			media, id, attempt, problem := store.Media(file.Output.MediaID)
			fatal(t, problem)
			if copied {
				if len(outputs) != 1 || outputs[0] != file.Output || media == nil || *media != file.Output || id != request.ID || attempt != 1 {
					t.Fatalf("verified local file vanished with remote ACK: outputs=%+v media=%+v", outputs, media)
				}
			} else if len(outputs) != 0 || media != nil {
				t.Fatal("an incomplete/unverified local file gained custody because its machine disappeared")
			}
			after, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			files, problem := store.MachineFileResults(request.ID)
			fatal(t, problem)
			wantState := "pending"
			if copied {
				wantState = "copied"
			}
			if after.Collected || !bytes.Equal(before.Outcome, after.Outcome) || len(files) != 1 || files[0].State != wantState || files[0].Copied != copied {
				t.Fatal("local visibility fabricated remote collection/release or changed the exact terminal")
			}
		})
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
	blocked, problem := store.ClientShutdownObligations()
	fatal(t, problem)
	after, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if len(blocked) != 0 {
		t.Fatalf("destroyed successful retained work still blocks shutdown: retain=%v state=%s obligations=%+v", after.RetainWork, after.State, blocked)
	}
	if after.State != "succeeded" {
		t.Fatal("loss rewrote the historical successful execution")
	}
}
