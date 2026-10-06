package producttest

import (
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func retainedModelEnvelope(t *testing.T) (json.RawMessage, *pb.ResultEnvelope) {
	t.Helper()
	schema := json.RawMessage(`{"input":"model"}`)
	artifact := records.ModelArtifact{ProducerRequestID: "runtime-child-with-no-client-row", OutputSlot: "model", Manifest: records.ArtifactObjectRef{Digest: childDigest("6"), Length: 123}, TensorFSReceiptDigest: childDigest("7")}
	raw, err := json.Marshal(artifact)
	must(t, err)
	raw, err = canonical.NormalizeJCS(raw)
	must(t, err)
	receipt, err := canonical.Raw(artifact.TensorFSReceiptDigest)
	must(t, err)
	return schema, &pb.ResultEnvelope{ResultSchemaDigest: canonical.Digest(schema), InlineResult: raw, RetainedModels: []*pb.RetainedModelResult{{ModelArtifactCanonicalBytes: raw, Retention: &pb.DerivedRetentionRequest{WeightsTransactionId: childDigest("8"), TensorfsReceiptDigest: receipt, RetentionId: childDigest("9")}}}}
}

func TestCollectedModelHoldKeepsRentalOwedUntilExactRelease(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "succeeded"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	_, envelope := retainedModelEnvelope(t)
	spec, _ := canonical.Spell(receipt.InvocationSpecDigest)
	raw, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: spec, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, Result: envelope})
	must(t, err)
	fatal(t, store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest, OutcomeId: "native-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: raw}))
	model := envelope.RetainedModels[0]
	hold := records.MachineModelRetention{OutcomeID: "native-outcome", Artifact: model.ModelArtifactCanonicalBytes, TransactionID: model.Retention.WeightsTransactionId, ReceiptDigest: model.Retention.TensorfsReceiptDigest, SourceRetentionID: model.Retention.RetentionId, RetentionID: childDigest("a")}
	wrong := hold
	wrong.OutcomeID = "another-outcome"
	if problem := store.FreezeMachineModelRetention(request.ID, wrong); problem == nil {
		t.Fatal("recipient hold bound itself to an unobserved outcome")
	}
	fatal(t, store.FreezeMachineModelRetention(request.ID, hold))
	fatal(t, store.FreezeMachineModelRetention(request.ID, hold))
	fatal(t, store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "held"))
	state.Collected, state.State, state.Sequence = true, "canceled", 1
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "retention_released", BodyCanonicalBytes: []byte(`{}`)}}}))
	owed, problem := store.RentalHasMachineObligations("pr-owned-machine")
	fatal(t, problem)
	if !owed {
		t.Fatal("Runtime root release erased independently collected model custody")
	}
	fatal(t, store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "releasing"))
	fatal(t, store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "released"))
	owed, problem = store.RentalHasMachineObligations("pr-owned-machine")
	fatal(t, problem)
	if owed {
		t.Fatal("exact recipient release did not clear the rental obligation")
	}
	if problem := store.FreezeMachineModelRetention(request.ID, hold); problem == nil {
		t.Fatal("collection replay revived explicitly released custody")
	}
}
