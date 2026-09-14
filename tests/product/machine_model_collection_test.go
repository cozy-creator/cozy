package producttest

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
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

func TestMachineModelCollectionRequiresCompleteHashedDescriptors(t *testing.T) {
	schema, envelope := retainedModelEnvelope(t)
	models, native, problem := launch.ValidateMachineModelResults(schema, envelope)
	fatal(t, problem)
	if !native || len(models) != 1 || models[0].Pointer != "" || models[0].Artifact.ProducerRequestID != "runtime-child-with-no-client-row" {
		t.Fatal("root borrowed artifact was not validated without a Creator producer row")
	}
	for name, change := range map[string]func(*pb.ResultEnvelope){
		"schema":           func(value *pb.ResultEnvelope) { value.ResultSchemaDigest = bytes.Repeat([]byte{1}, 32) },
		"missing metadata": func(value *pb.ResultEnvelope) { value.RetainedModels = nil },
		"wrong pointer":    func(value *pb.ResultEnvelope) { value.RetainedModels[0].ResultPointer = "/model" },
		"changed artifact": func(value *pb.ResultEnvelope) { value.RetainedModels[0].ModelArtifactCanonicalBytes = []byte(`{}`) },
		"changed receipt": func(value *pb.ResultEnvelope) {
			value.RetainedModels[0].Retention.TensorfsReceiptDigest = bytes.Repeat([]byte{1}, 32)
		},
		"missing actual transaction": func(value *pb.ResultEnvelope) { value.RetainedModels[0].Retention.WeightsTransactionId = "" },
		"missing root custody":       func(value *pb.ResultEnvelope) { value.RetainedModels[0].Retention.RetentionId = "" },
		"duplicate": func(value *pb.ResultEnvelope) {
			value.RetainedModels = append(value.RetainedModels, proto.Clone(value.RetainedModels[0]).(*pb.RetainedModelResult))
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := proto.Clone(envelope).(*pb.ResultEnvelope)
			change(changed)
			if _, _, problem := launch.ValidateMachineModelResults(schema, changed); problem == nil {
				t.Fatal("changed model custody descriptor was accepted")
			}
		})
	}
}

func TestMachineModelCollectionUsesSchemaPointersNotLookalikeFields(t *testing.T) {
	_, envelope := retainedModelEnvelope(t)
	schema := json.RawMessage(`{"fields":[{"name":"models/~","type":{"list":{"input":"model"}},"wire":"required"}],"tag":null,"tag_field":""}`)
	inline, err := json.Marshal(map[string]any{"models/~": []json.RawMessage{envelope.InlineResult}})
	must(t, err)
	inline, err = canonical.NormalizeJCS(inline)
	must(t, err)
	envelope.InlineResult, envelope.ResultSchemaDigest = inline, canonical.Digest(schema)
	envelope.RetainedModels[0].ResultPointer = "/models~1~0/0"
	models, native, problem := launch.ValidateMachineModelResults(schema, envelope)
	fatal(t, problem)
	if !native || len(models) != 1 || models[0].Pointer != "/models~1~0/0" {
		t.Fatal("model list or RFC6901 escaping changed")
	}
	schema = json.RawMessage(`{"fields":[],"tag":null,"tag_field":""}`)
	envelope.ResultSchemaDigest = canonical.Digest(schema)
	if _, _, problem := launch.ValidateMachineModelResults(schema, envelope); problem == nil {
		t.Fatal("a model-shaped value outside the schema gained custody")
	}
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
