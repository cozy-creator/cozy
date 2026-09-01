package producttest

import (
	"context"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/transfer"
)

func TestModelTransferInstructionIsSourceFirstAndPlacementIsFrozen(t *testing.T) {
	upload := modeltransfer.Instruction{Kind: "model-upload", Source: "hf://acme/model@deadbeef",
		Destination: "acme/model", Producer: "acme/tools@v2/quantize", Placement: "rental-only"}
	raw, err := upload.Bytes()
	must(t, err)
	replayed, err := modeltransfer.ParseInstruction(raw)
	must(t, err)
	if replayed != upload || !strings.HasPrefix(upload.ID(), "modeltransfer-") {
		t.Fatalf("upload instruction changed on replay: %+v", replayed)
	}
	download := upload
	download.Kind, download.Destination, download.Placement = "model-download", "local/model", ""
	if !strings.HasPrefix(download.ID(), "modeltransfer-") || download.ID() == upload.ID() {
		t.Fatal("transfer kind and destination are not part of canonical run identity")
	}
}

func TestModelTransferResourceNeedsHaveOneStrictParser(t *testing.T) {
	needs, err := modeltransfer.ParseResourceNeeds(1, "ram64g,sm90+,vram80g,cuda13.0+")
	if err != nil || needs != (modeltransfer.ResourceNeeds{GPUCount: 1, MinSM: 90,
		VRAMGB: 80, RAMGB: 64}) {
		t.Fatalf("H3 resource needs = %+v, %v", needs, err)
	}
	for _, invalid := range []string{"cudaevil+", "sm90", "vram80g+", "ram0g"} {
		if _, err := modeltransfer.ParseResourceNeeds(1, invalid); err == nil {
			t.Errorf("invalid resource %q was accepted", invalid)
		}
	}
}

func TestModelTransferFinalizationCancelAndPartialOutputsStayVisible(t *testing.T) {
	store, problem := records.Open(t.TempDir() + "/records.db")
	fatal(t, problem)
	defer store.Close()
	request := transferRequest("job-transfer-cancel-finalize")
	request.ModelTransfer.Outputs = append(request.ModelTransfer.Outputs,
		records.ModelTransferOutput{Name: "second", RequiredContract: &records.ModelTransferContract{
			TopologyDigest: digest("c"), Encodings: []string{digest("d")}}})
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	fatal(t, store.BeginModelTransferMaterialization(request.ID))
	fatal(t, store.CompleteModelTransferMaterialization(request.ID, nil))
	fatal(t, store.BeginModelTransferFinalization(request.ID))
	fatal(t, store.SettleRequest(request.ID, "finalizing"))
	artifact := records.ModelTransferArtifact{RequestID: request.ID, Attempt: 0,
		OutputSlot: "model", ManifestID: digest("e"), ManifestLength: 8,
		Evidence: []byte("proof")}
	fatal(t, store.RecordModelTransferArtifact(artifact))
	fatal(t, store.CompleteModelTransferOutput(request.ID, 0, "model", "publish-proof"))
	fatal(t, store.RequestModelTransferCancellation(request.ID))
	state, problem := store.SettleModelTransferRequest(request.ID, 0)
	fatal(t, problem)
	if state != "canceled" {
		t.Fatalf("finalizing cancellation settled %s", state)
	}
	artifacts, problem := store.ModelTransferArtifacts(request.ID, 0)
	fatal(t, problem)
	if len(artifacts) != 1 || artifacts[0].FinalID != "publish-proof" ||
		artifacts[0].ManifestID != digest("e") {
		t.Fatalf("retained partial output disappeared: %+v", artifacts)
	}
	events, problem := store.EventsAfter(request.ID, 0, 10)
	fatal(t, problem)
	if len(events) != 1 || events[0].Type != "request.canceled" {
		t.Fatalf("canceled finalization event = %+v", events)
	}
}

func TestFailedAndCanceledProducerAttemptsTerminalizeTransferBeforeAck(t *testing.T) {
	for _, test := range []struct{ status, sidecar, request string }{
		{"FAILED", "failed", "failed"}, {"CANCELED", "canceled", "canceled"},
	} {
		t.Run(strings.ToLower(test.status), func(t *testing.T) {
			store, problem := records.Open(t.TempDir() + "/records.db")
			fatal(t, problem)
			defer store.Close()
			request := transferRequest("job-terminal-" + strings.ToLower(test.status))
			_, _, problem = store.Submit(request)
			fatal(t, problem)
			fatal(t, store.CompleteModelTransferMaterialization(request.ID, nil))
			worker := "worker-" + strings.ToLower(test.status)
			fatal(t, store.AttachWorker(records.WorkerProcess{InstanceID: worker,
				Package: request.Package, WorkerID: worker}))
			attempt, problem := store.Dispatch(records.Attempt{RequestID: request.ID,
				InstanceID: worker, SessionID: "boot", InvocationDigest: digest("1"),
				InvocationCanonical: []byte(`{}`)})
			fatal(t, problem)
			fatal(t, store.OfferDispatch(request.ID, attempt, "boot"))
			applied, problem := store.AcceptTerminal(records.Terminal{RequestID: request.ID,
				Attempt: attempt, SessionID: "boot", InvocationDigest: digest("1"),
				TerminalID: "terminal", TerminalDigest: digest("2"), Status: test.status,
				Cause: test.status, SafeMessage: "producer stopped", RequestState: "finalizing",
				EventType: "request.finalizing", EventPayload: map[string]any{"status": "FINALIZING"}})
			fatal(t, problem)
			if !applied {
				t.Fatal("producer terminal was not applied")
			}
			transferRow, problem := store.ModelTransferOf(request.ID)
			fatal(t, problem)
			if transferRow.State != test.sidecar {
				t.Fatalf("%s attempt left sidecar %s", test.status, transferRow.State)
			}
			settled, problem := store.SettleModelTransferRequest(request.ID, attempt)
			fatal(t, problem)
			if settled != test.request {
				t.Fatalf("%s attempt settled request %s", test.status, settled)
			}
		})
	}
}

func TestPublicationValidationRefusesChangedOpenAndFinalize(t *testing.T) {
	manifest := hub.Object{ID: digest("a"), Length: 11}
	blob := hub.Object{ID: digest("b"), Length: 29}
	declared := []hub.Object{manifest, blob}
	opened := hub.OpenPublicationResponse{Publication: hub.Session{Operation: "publish-proof",
		State: "open", Objects: []hub.Transfer{{ObjectID: manifest.ID, Length: manifest.Length,
			State: "accepted"}, {ObjectID: blob.ID, Length: blob.Length, State: "claimed"}}}}
	totals, problem := transfer.ValidateOpenedPublication(opened, "publish-proof", declared)
	fatal(t, problem)
	changed := opened
	changed.Publication.Operation = "other"
	if _, problem := transfer.ValidateOpenedPublication(changed, "publish-proof", declared); problem == nil {
		t.Fatal("changed publication operation was accepted")
	}
	evidence := []byte("evidence")
	checkpoint := hub.CheckpointPublication{PublishID: "publish-proof", CheckpointID: manifest.ID,
		Manifest: hub.ManifestRef{SHA256: strings.TrimPrefix(manifest.ID, "sha256:"), Length: manifest.Length},
		Objects:  1, Bytes: blob.Length, CheckpointEvidenceBase64: hub.B64(evidence), State: "checkpointed"}
	fatal(t, transfer.ValidateFinalizedCheckpoint(checkpoint, "publish-proof", manifest.ID,
		manifest.Length, evidence, totals))
	checkpoint.Manifest.Length++
	if problem := transfer.ValidateFinalizedCheckpoint(checkpoint, "publish-proof", manifest.ID,
		manifest.Length, evidence, totals); problem == nil {
		t.Fatal("changed finalized Manifest length was accepted")
	}
}

func TestPassThroughTransferResumesAndCancelCannotBeOverwritten(t *testing.T) {
	t.Run("restart", func(t *testing.T) {
		store, problem := records.Open(t.TempDir() + "/records.db")
		fatal(t, problem)
		defer store.Close()
		request := passThroughRequest("job-pass-restart")
		_, _, problem = store.Submit(request)
		fatal(t, problem)
		layout, problem := home.Open(t.TempDir())
		fatal(t, problem)
		owner := &passThroughOwner{store: store}
		controller, problem := orchestrator.Open(orchestrator.Options{
			Cfg: config.Config{}, Layout: layout, Store: store, ModelTransfers: owner})
		fatal(t, problem)
		defer controller.Close(0)
		_, _, problem = controller.Reconcile()
		fatal(t, problem)
		waitTransferRequestState(t, store, request.ID, "succeeded")
	})

	t.Run("wrong-platform-function-stays-owed", func(t *testing.T) {
		store, problem := records.Open(t.TempDir() + "/records.db")
		fatal(t, problem)
		defer store.Close()
		request := passThroughRequest("job-pass-wrong-function")
		request.Entrypoint = "other"
		_, _, problem = store.Submit(request)
		fatal(t, problem)
		owed, problem := store.Owed()
		fatal(t, problem)
		if len(owed) != 1 || owed[0].ID != request.ID {
			t.Fatalf("non-pass-through cozy/platform request disappeared on restart: %+v", owed)
		}
	})

	t.Run("cancel", func(t *testing.T) {
		store, problem := records.Open(t.TempDir() + "/records.db")
		fatal(t, problem)
		defer store.Close()
		layout, problem := home.Open(t.TempDir())
		fatal(t, problem)
		owner := &passThroughOwner{store: store, started: make(chan struct{}), release: make(chan struct{})}
		controller, problem := orchestrator.Open(orchestrator.Options{
			Cfg: config.Config{}, Layout: layout, Store: store, ModelTransfers: owner})
		fatal(t, problem)
		defer controller.Close(0)
		intent := passThroughRequest("").ModelTransfer
		id, _, fresh, problem := controller.SubmitDetail(orchestrator.Submission{
			Kind: "job", Package: "cozy/platform", Entrypoint: "model-pass-through",
			Payload: []byte("{}"), IdemKey: "pass-cancel", BodyDigest: digest("c"),
			ModelTransfer: intent})
		fatal(t, problem)
		if !fresh {
			t.Fatal("pass-through request replayed unexpectedly")
		}
		<-owner.started
		canceled, problem := store.CancelQueuedRequest(id, map[string]any{"status": "CANCELED"})
		fatal(t, problem)
		if !canceled {
			t.Fatal("pass-through request was not cancelable before attempt zero")
		}
		close(owner.release)
		waitTransferRequestState(t, store, id, "canceled")
		time.Sleep(50 * time.Millisecond)
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if row.State != "canceled" {
			t.Fatalf("pass-through completion overwrote cancellation with %s", row.State)
		}
	})
}

type passThroughOwner struct {
	store   *records.Store
	started chan struct{}
	release chan struct{}
}

func (o *passThroughOwner) MaterializeLocal(context.Context, string,
	records.ModelTransferIntent) ([]orchestrator.ModelRef, *exit.Error) {
	return nil, exit.Unavailablef("unused")
}

func (o *passThroughOwner) RefreshRemoteSource(context.Context,
	records.ModelTransferIntent) ([]orchestrator.ModelSourceCapability, *exit.Error) {
	return nil, exit.Unavailablef("unused")
}

func (o *passThroughOwner) Finalize(context.Context, string,
	orchestrator.ModelTransferMover) *exit.Error {
	return exit.Unavailablef("unused")
}

func (o *passThroughOwner) PassThrough(_ context.Context, requestID string,
	_ records.ModelTransferIntent) *exit.Error {
	if o.started != nil {
		close(o.started)
		<-o.release
		request, problem := o.store.RequestRow(requestID)
		if problem != nil {
			return problem
		}
		if request.State == "canceled" {
			return exit.New(exit.Canceled, "canceled")
		}
	}
	if problem := o.store.BeginModelTransferMaterialization(requestID); problem != nil {
		return problem
	}
	if problem := o.store.CompleteModelTransferMaterialization(requestID, nil); problem != nil {
		return problem
	}
	if problem := o.store.BeginModelTransferFinalization(requestID); problem != nil {
		return problem
	}
	if problem := o.store.RecordModelTransferArtifact(records.ModelTransferArtifact{
		RequestID: requestID, Attempt: 0, OutputSlot: "model", ManifestID: digest("d"),
		ManifestLength: 8, Evidence: []byte("proof")}); problem != nil {
		return problem
	}
	if problem := o.store.CompleteModelTransferOutput(requestID, 0, "model", "local:proof"); problem != nil {
		return problem
	}
	return o.store.CompleteModelTransfer(requestID, map[string]string{"model": digest("d")})
}

func passThroughRequest(id string) records.Request {
	request := transferRequest(id)
	request.Package, request.Entrypoint = "cozy/platform", "model-pass-through"
	request.ModelTransfer.SourceProfiles = nil
	request.ModelTransfer.Outputs = []records.ModelTransferOutput{{Name: "model"}}
	return request
}

func waitTransferRequestState(t *testing.T, store *records.Store, id, state string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if row != nil && row.State == state {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("request %s did not reach %s", id, state)
}

func TestModelTransferSidecarIsTransactionalAndCanonical(t *testing.T) {
	store, problem := records.Open(t.TempDir() + "/records.db")
	fatal(t, problem)
	defer store.Close()
	request := transferRequest("job-transfer-normalize")
	request.ModelTransfer.Outputs = []records.ModelTransferOutput{
		{Name: "z", RequiredContract: &records.ModelTransferContract{
			TopologyDigest: digest("d"), Encodings: []string{digest("c"), digest("b")}}},
		{Name: "a", RequiredContract: &records.ModelTransferContract{
			TopologyDigest: digest("a"), Encodings: []string{digest("f")}}},
	}
	created, fresh, problem := store.Submit(request)
	fatal(t, problem)
	if !fresh || created.ID != request.ID {
		t.Fatalf("ordinary request submission = %+v fresh=%t", created, fresh)
	}
	transfer, problem := store.ModelTransferOf(request.ID)
	fatal(t, problem)
	if transfer == nil || transfer.Outputs[0].Name != "a" ||
		!reflect.DeepEqual(transfer.Outputs[1].RequiredContract.Encodings,
			[]string{digest("b"), digest("c")}) {
		t.Fatalf("sidecar intent was not normalized: %+v", transfer)
	}
	rows, problem := store.Requests("", 10)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].ID != request.ID {
		t.Fatalf("model transfer created a parallel lifecycle: %+v", rows)
	}
}

func TestModelTransferSourceStatusIsMonotoneAndCanceledIsAbsorbing(t *testing.T) {
	store, problem := records.Open(t.TempDir() + "/records.db")
	fatal(t, problem)
	defer store.Close()
	request := transferRequest("job-transfer-source")
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	fatal(t, store.BeginModelTransferMaterialization(request.ID))
	base := records.ModelTransferSourceStatus{RequestID: request.ID, Member: "weights.safetensors",
		ObjectID: digest("1"), Length: 100, CapabilityRevision: 1,
		State: "accepted", Transferred: 10}
	fatal(t, store.RecordModelTransferSourceStatus(base))
	base.State, base.Transferred = "downloading", 50
	fatal(t, store.RecordModelTransferSourceStatus(base))
	regressed := base
	regressed.Transferred = 40
	if problem := store.RecordModelTransferSourceStatus(regressed); problem == nil {
		t.Fatal("source progress regressed within one capability revision")
	}
	base.State, base.Transferred = "verified", 100
	fatal(t, store.RecordModelTransferSourceStatus(base))
	late := base
	late.State, late.Transferred = "downloading", 100
	if problem := store.RecordModelTransferSourceStatus(late); problem == nil {
		t.Fatal("verified source accepted a late downloading frame")
	}
	canceled, problem := store.CancelQueuedRequest(request.ID, map[string]any{"status": "CANCELED"})
	fatal(t, problem)
	if !canceled {
		t.Fatal("queued transfer cancellation did not settle")
	}
	if problem := store.CompleteModelTransferMaterialization(request.ID, nil); problem == nil {
		t.Fatal("canceled transfer accepted materialization completion")
	}
	transfer, problem := store.ModelTransferOf(request.ID)
	fatal(t, problem)
	if transfer.State != "canceled" {
		t.Fatalf("canceled sidecar changed to %s", transfer.State)
	}
}

func TestModelTransferArtifactsAreAttemptScopedAndGrantOrdered(t *testing.T) {
	store, problem := records.Open(t.TempDir() + "/records.db")
	fatal(t, problem)
	defer store.Close()
	request := transferRequest("job-transfer-artifacts")
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	for attempt, manifest := range map[int64]string{1: digest("2"), 2: digest("3")} {
		artifact := records.ModelTransferArtifact{RequestID: request.ID, Attempt: attempt,
			OutputSlot: "model", ManifestID: manifest, ManifestLength: 8,
			Evidence: []byte("evidence"), InvocationDigest: digest("4"),
			TransactionID: "tx", ReceiptDigest: digest("5"), Receipt: []byte("receipt"),
			Objects: []records.ModelTransferObject{{ObjectID: manifest, Length: 8, SourceRef: "manifest"}}}
		fatal(t, store.RecordModelTransferArtifact(artifact))
	}
	first, problem := store.ModelTransferArtifact(request.ID, 1, "model")
	fatal(t, problem)
	second, problem := store.ModelTransferArtifact(request.ID, 2, "model")
	fatal(t, problem)
	if first.ManifestID == second.ManifestID {
		t.Fatal("different attempt outputs collapsed under one slot")
	}
	status := records.ModelTransferObject{RequestID: request.ID, Attempt: 2, OutputSlot: "model",
		ObjectID: second.ManifestID, Length: 8, OperationID: "publish", GrantRevision: 1,
		UpdateSequence: 2, State: "uploading", Transferred: 4}
	fatal(t, store.RecordModelTransferObjectStatus(status))
	late := status
	late.GrantRevision, late.UpdateSequence = 1, 3
	fatal(t, store.RecordModelTransferObjectStatus(late))
	regressedState := late
	regressedState.UpdateSequence, regressedState.State = 4, "accepted"
	if problem := store.RecordModelTransferObjectStatus(regressedState); problem == nil {
		t.Fatal("same-grant object status regressed from uploading to accepted")
	}
	newGrant := status
	newGrant.GrantRevision, newGrant.UpdateSequence = 2, 1
	fatal(t, store.RecordModelTransferObjectStatus(newGrant))
	oldGrant := status
	oldGrant.UpdateSequence = 4
	if problem := store.RecordModelTransferObjectStatus(oldGrant); problem == nil {
		t.Fatal("late old-grant status overwrote a newer grant revision")
	}
}

func TestModelTransferTerminalAndEventCommitTogether(t *testing.T) {
	store, problem := records.Open(t.TempDir() + "/records.db")
	fatal(t, problem)
	defer store.Close()
	request := transferRequest("job-transfer-terminal")
	request.Package, request.Entrypoint = "cozy/platform", "model-pass-through"
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	fatal(t, store.BeginModelTransferMaterialization(request.ID))
	fatal(t, store.CompleteModelTransferMaterialization(request.ID, nil))
	fatal(t, store.BeginModelTransferFinalization(request.ID))
	fatal(t, store.RecordModelTransferArtifact(records.ModelTransferArtifact{RequestID: request.ID,
		Attempt: 0, OutputSlot: "model", ManifestID: digest("6"), ManifestLength: 8,
		Evidence: []byte("evidence")}))
	fatal(t, store.CompleteModelTransferOutput(request.ID, 0, "model", "local:proof"))
	fatal(t, store.CompleteModelTransfer(request.ID, map[string]string{"model": digest("6")}))
	state, problem := store.SettleModelTransferRequest(request.ID, 0)
	fatal(t, problem)
	if state != "succeeded" {
		t.Fatalf("settled state = %s", state)
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	events, problem := store.EventsAfter(request.ID, 0, 20)
	fatal(t, problem)
	if row.State != "succeeded" || len(events) != 1 || events[0].Type != "request.completed" {
		t.Fatalf("terminal/event diverged: request=%s events=%+v", row.State, events)
	}
}

func transferRequest(id string) records.Request {
	return records.Request{ID: id, IdemKey: "idem-" + id, BodyDigest: digest("9"),
		Package: "acme/producer", Entrypoint: "produce", PlanID: digest("8"),
		Payload: []byte("{}"), Kind: "job", ArtifactOutputs: "[]",
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload",
			Destination: "acme/model", Source: "hf://acme/model@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			SourceSelection: digest("7"), SourceFiles: []records.ModelTransferSourceFile{{
				Member: "weights.safetensors", SHA256: strings.Repeat("1", 64), Length: 100}},
			SourceProfiles: map[string]string{"source": "proof/model/1"},
			Outputs: []records.ModelTransferOutput{{Name: "model",
				RequiredContract: &records.ModelTransferContract{TopologyDigest: digest("a"),
					Encodings: []string{digest("b")}}}}}}
}

func digest(character string) string { return "sha256:" + strings.Repeat(character, 64) }

func TestOrdinaryProducerCarriesMultipleSourceProfilesAndNamedContracts(t *testing.T) {
	raw := []byte(`{"application":"h3:app","entrypoints":[],"format":"cozy.package.descriptor/1","jobs":[{"artifact_outputs":[{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"bf16-full","required_contract":{"encodings":["sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"topology_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}},{"max_bytes":4096,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8-adaln-pruned","required_contract":{"encodings":["sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"],"topology_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}}],"models":[{"class":"H3Dits","component_use":{},"path":"four-lane.models.dits","source_profile":"hf/minimax-h3/native-dual-bf16/1","stamps":{}},{"class":"H3Shared","component_use":{},"path":"four-lane.models.shared","source_profile":"hf/minimax-h3/shared-bf16/1","stamps":{}}],"name":"four-lane","publishes":false,"request":{"fields":[]},"resources":{"gpu_count":1,"placement":"single_node","requires":"sm90+,vram80g,ram64g"},"result":{"fields":[]}}]}`)
	descriptor, problem := launch.DecodeDescriptor(raw)
	fatal(t, problem)
	job, problem := descriptor.Function("four-lane")
	fatal(t, problem)
	if len(job.Models) != 2 || len(job.ArtifactOutputs) != 2 ||
		job.Models[0].SourceProfile == job.Models[1].SourceProfile ||
		job.ArtifactOutputs[1].RequiredContract == nil {
		t.Fatalf("ordinary producer hook lost its typed inputs or outputs: %+v", job)
	}
}
