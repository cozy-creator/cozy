package producttest

import (
	"encoding/json"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestMachineAssetResultChecksConsumedFinalMetadata(t *testing.T) {
	schema := json.RawMessage(`{"fields":[{"name":"image","type":{"asset":"image"},"wire":"required","asset_bound":{"max_bytes":100,"media_types":["image/png"]}}]}`)
	schema, err := canonical.NormalizeJCS(schema)
	must(t, err)
	asset := map[string]any{"asset_ref": childDigest("4"), "digest": childDigest("4"), "kind": "image", "media_type": "image/png", "size_bytes": 79}
	envelope := func(value map[string]any) *pb.ResultEnvelope {
		raw, err := json.Marshal(map[string]any{"image": value})
		must(t, err)
		raw, err = canonical.NormalizeJCS(raw)
		must(t, err)
		return &pb.ResultEnvelope{InlineResult: raw, ResultSchemaDigest: canonical.Digest(schema)}
	}
	requireResultUsable(t, schema, envelope(asset))
	// A newer Runtime's additive metadata and schema spelling are not this host's facts.
	additive := map[string]any{"worker_note": "newer runtime"}
	for k, v := range asset {
		additive[k] = v
	}
	evolved := envelope(additive)
	evolved.ResultSchemaDigest = canonical.Digest([]byte(`{"fields":[],"newer":true}`))
	requireResultUsable(t, schema, evolved)
	for name, change := range map[string]func(map[string]any){
		"missing digest": func(a map[string]any) { delete(a, "digest") },
		"changed digest": func(a map[string]any) { a["digest"] = childDigest("5") },
		"wrong kind":     func(a map[string]any) { a["kind"] = "file" },
		"wrong media":    func(a map[string]any) { a["media_type"] = "text/plain" },
		"oversized":      func(a map[string]any) { a["size_bytes"] = 101 },
		"fractional":     func(a map[string]any) { a["size_bytes"] = 1.5 },
	} {
		t.Run(name, func(t *testing.T) {
			copy := map[string]any{}
			for k, v := range asset {
				copy[k] = v
			}
			change(copy)
			if drift, problem := launch.ValidateMachineResult(schema, envelope(copy)); problem != nil || drift.Usable("image") {
				t.Fatalf("changed final asset metadata was accepted: %v %v", drift, problem)
			}
		})
	}
}

func TestMachineFileCustodySurvivesReplyLossWithoutExecutionRows(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	file := records.MachineFileResult{OutcomeID: "file-outcome", RetentionID: childDigest("8"), Source: records.ByteOutput{RequestID: request.ID, Attempt: 1, OutputID: "image", Digest: childDigest("4"), Length: 16, MimeType: "image/png", ProducerRootID: childDigest("5"), ReceiptDigest: childDigest("6"), ManifestID: childDigest("7"), ManifestLength: 123, ContentBytes: 16}, Output: records.Output{OutputID: "image", MediaID: "med-original", Path: "/client/private-copy", Digest: childDigest("4"), Length: 16, MimeType: "image/png"}}
	if _, problem := store.FreezeMachineFileResult(request.ID, file); problem == nil {
		t.Fatal("file hold was frozen without an observed outcome")
	}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "succeeded"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	spec, _ := canonical.Spell(receipt.InvocationSpecDigest)
	raw, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: spec, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	fatal(t, store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest, OutcomeId: file.OutcomeID, OutcomeDigest: digest, OutcomeCanonicalBytes: raw}))
	first, problem := store.FreezeMachineFileResult(request.ID, file)
	fatal(t, problem)
	file.Output.MediaID = "med-retry"
	replayed, problem := store.FreezeMachineFileResult(request.ID, file)
	fatal(t, problem)
	if replayed.Output.MediaID != first.Output.MediaID {
		t.Fatal("lost retain reply minted a new media handle")
	}
	changed := file
	changed.Source.ReceiptDigest = childDigest("9")
	if _, problem := store.FreezeMachineFileResult(request.ID, changed); problem == nil {
		t.Fatal("replayed file changed native custody")
	}
	if outputs, problem := store.VisibleOutputs(request.ID); problem != nil || len(outputs) != 0 {
		t.Fatal("uncollected file became visible")
	}
	fatal(t, store.AdvanceMachineFileResult(request.ID, file.RetentionID, "copied"))
	fatal(t, store.AppendEvent(request.ID, "machine.retention_released", 1, map[string]any{}))
	if owed, problem := store.MachineExecutionOwesWork(request.ID); problem != nil || !owed {
		t.Fatal("recipient hold was forgotten after root release")
	}
	fatal(t, store.AdvanceMachineFileResult(request.ID, file.RetentionID, "released"))
	state.Collected = true
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	outputs, problem := store.VisibleOutputs(request.ID)
	fatal(t, problem)
	if len(outputs) != 1 || outputs[0].Path != file.Output.Path || outputs[0].MediaID != first.Output.MediaID {
		t.Fatal("collected output did not project its exact copied receipt")
	}
	media, requestID, attempt, problem := store.Media(first.Output.MediaID)
	fatal(t, problem)
	if media == nil || requestID != request.ID || attempt != 1 || media.Digest != file.Output.Digest {
		t.Fatal("opaque media lookup lost observed outcome binding")
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("file collection invented Creator execution attempts")
	}
	if owed, problem := store.MachineExecutionOwesWork(request.ID); problem != nil || owed {
		t.Fatal("released file hold still owes rental custody")
	}
}

// requireResultUsable fails unless every declared output of the result was read.
func requireResultUsable(t *testing.T, schema json.RawMessage, envelope *pb.ResultEnvelope) {
	t.Helper()
	drift, problem := launch.ValidateMachineResult(schema, envelope)
	fatal(t, problem)
	if len(drift.Failed) > 0 {
		t.Fatalf("a valid result was refused: %v", drift.Warnings())
	}
}
