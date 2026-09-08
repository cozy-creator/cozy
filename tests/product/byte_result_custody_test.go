package producttest

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Native receipt integrity is proved by Runtime's real TensorFS peer tests. This
// regression exercises Creator's actual transaction and capability ownership.
func TestChildByteResultHasIndependentRecipientAndCannotBeBorrowedByHash(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "byte-parent", ""))
	child := records.Request{ID: "req-byte-producer", IdemKey: "byte-producer", BodyDigest: childDigest("3"), Package: "local/operation", Entrypoint: "report", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5")}
	child, _, problem = store.SubmitChild(child, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	child = offerChildParent(t, store, child)
	output := records.ByteOutput{RequestID: child.ID, Attempt: 1, OutputID: "report", Digest: childDigest("6"), Length: 65537, MimeType: "application/json", ProducerRootID: childDigest("7"), ReceiptDigest: childDigest("8"), ManifestID: childDigest("9"), ManifestLength: 192, ContentBytes: 65537}
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: child.ID, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "byte-done", TerminalDigest: childDigest("a"), Status: "SUCCEEDED", RequestState: "succeeded", Body: []byte(`{}`), ByteOutputs: []records.ByteOutput{output}})
	fatal(t, problem)
	fatal(t, store.Closed(child.ID, 1))
	outputs, problem := store.ByteOutputs(child.ID, 1)
	fatal(t, problem)
	if len(outputs) != 1 || outputs[0] != output {
		t.Fatal("terminal byte custody did not commit exactly")
	}
	assets, problem := store.ReceivedByteAssets(parent.ID)
	fatal(t, problem)
	if len(assets) != 0 {
		t.Fatal("producer output was granted before recipient retention")
	}
	hold, problem := store.ReserveByteResult(child.ID, output)
	fatal(t, problem)
	if hold.State != "pending" || hold.ArtifactKind != "tree" {
		t.Fatal("result lacks independent pending retention")
	}
	replay, problem := store.ReserveByteResult(child.ID, output)
	fatal(t, problem)
	if replay != hold {
		t.Fatal("lost reply changed recipient intent")
	}
	changed := output
	changed.ManifestLength++
	if _, problem := store.ReserveByteResult(child.ID, changed); problem == nil {
		t.Fatal("changed native subject accepted")
	}
	fatal(t, store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot"))
	assets, problem = store.ReceivedByteAssets(parent.ID)
	fatal(t, problem)
	if len(assets) != 1 || assets[0].LocalPath != "" || assets[0].Native == nil {
		t.Fatal("byte grant fabricated a local path or was absent")
	}
	downstream := records.Request{ID: "req-byte-reader", IdemKey: "byte-reader", BodyDigest: childDigest("b"), Package: "local/operation", Entrypoint: "read", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 1, ChildIntentDigest: childDigest("c"), ChildTargetDigest: childDigest("d"), Assets: assets}
	downstream.Assets[0].FieldPath = "report"
	recorded, _, problem := store.SubmitChild(downstream, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if recorded.Assets[0].Native.RetentionID == hold.RetentionID {
		t.Fatal("downstream borrowed parent retention instead of acquiring an input hold")
	}
	retained, problem := store.NativeArtifactRetentions(recorded.ID)
	fatal(t, problem)
	if len(retained) != 1 || retained[0].Kind != "input" || retained[0].State != "pending" {
		t.Fatal("input hold was not part of child admission")
	}
	stranger := offerChildParent(t, store, recordPrivateTransaction(t, store, "byte-stranger", ""))
	forged := downstream
	forged.ID = "req-byte-forged"
	forged.IdemKey = "byte-forged"
	forged.ParentRequestID = stranger.ID
	forged.ParentCallIndex = 0
	if _, _, problem := store.SubmitChild(forged, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("another parent borrowed a known byte digest")
	}
	fatal(t, store.BeginNativeArtifactRelease(child.ID, false))
	if store.ConfirmNativeArtifact(hold.RetentionID, "private-worker", "private-boot") == nil {
		t.Fatal("late ACK resurrected release")
	}
	fatal(t, store.CompleteNativeArtifactRelease(hold.RetentionID))
	if _, problem := store.ReserveByteResult(child.ID, output); problem == nil {
		t.Fatal("released result was reacquired")
	}
	received, problem := store.ReceivedByteAssets(parent.ID)
	fatal(t, problem)
	if len(received) != 0 {
		t.Fatal("released result stayed in parent grants")
	}
	retained, problem = store.NativeArtifactRetentions(recorded.ID)
	fatal(t, problem)
	if len(retained) != 1 || retained[0].State != "pending" {
		t.Fatal("releasing producer recipient erased separate downstream obligation")
	}
}

func TestTreeResultDiscoveryAndNativeInputPreserveManifestLength(t *testing.T) {
	ep := &launch.Entrypoint{Request: launch.Struct{Fields: []launch.Field{{Name: "source", Type: json.RawMessage(`{"input":"tree"}`)}}}, Result: launch.Struct{Fields: []launch.Field{{Name: "corpus", Type: json.RawMessage(`{"input":"tree"}`)}}}}
	paths := launch.AssetPaths(ep.Result)
	if len(paths) != 1 || paths[0] != "corpus" {
		t.Fatalf("tree output was not declared: %v", paths)
	}
	source := records.ByteOutput{Digest: childDigest("1"), Length: 123, ContentBytes: 1 << 20, MimeType: "application/vnd.cozy.tree-manifest"}
	parent := []records.AssetBinding{{Digest: source.Digest, Length: source.Length, MediaType: source.MimeType, Native: &records.ByteAssetBinding{Output: source, RetentionID: childDigest("2")}}}
	payload, _ := json.Marshal(map[string]string{"source": source.Digest})
	inherited, problem := launch.InheritChildAssets(ep, payload, parent)
	fatal(t, problem)
	if len(inherited) != 1 || inherited[0].Length != 123 || inherited[0].Native.Output.ContentBytes != 1<<20 {
		t.Fatalf("manifest length was confused with payload capacity: %+v", inherited)
	}
}

func TestByteMemoAdoptionPreservesOriginalOutcomeAndExactRecipient(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	first := offerChildParent(t, store, recordPrivateTransaction(t, store, "byte-cache-first", ""))
	source := offerChildParent(t, store, operationHistory(t, store, "byte-cache-source", first))
	b := records.ByteOutput{RequestID: source.ID, Attempt: 1, OutputID: "report", Digest: childDigest("1"), Length: 65537, MimeType: "application/json", ProducerRootID: childDigest("2"), ReceiptDigest: childDigest("3"), ManifestID: childDigest("4"), ManifestLength: 192, ContentBytes: 65537}
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: source.ID, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "byte-cache-done", TerminalDigest: childDigest("5"), Status: "SUCCEEDED", RequestState: "succeeded", Body: []byte(`{}`), ByteOutputs: []records.ByteOutput{b}})
	fatal(t, problem)
	fatal(t, store.Closed(source.ID, 1))
	second := offerChildParent(t, store, recordPrivateTransaction(t, store, "byte-cache-second", ""))
	consumer := operationHistory(t, store, "byte-cache-consumer", second)
	key, problem := records.OperationKey(consumer)
	fatal(t, problem)
	fatal(t, store.BeginOperationLookup(consumer.ID, key))
	attempt, problem := store.AttemptRow(source.ID, 1)
	fatal(t, problem)
	cached := records.CachedOperation{Key: key, SourceRequestID: source.ID, SourceAttempt: 1, InvocationDigest: attempt.InvocationDigest, OutcomeID: attempt.TerminalID, OutcomeDigest: attempt.TerminalDigest, OutcomeBody: attempt.TerminalBody}
	if store.AdoptCachedOperation(consumer.ID, cached) == nil {
		t.Fatal("byte cache hit without independent retention was accepted")
	}
	h := records.NativeArtifactRetention{ArtifactKind: "tree", ProducerAttempt: 1, ProducerOutputID: b.OutputID, ContentBytes: b.ContentBytes, ConsumerID: consumer.ID, ParentRequestID: second.ID, Kind: "result", Slot: b.OutputID, ProducerID: source.ID, ManifestID: b.ManifestID, ManifestLength: b.ManifestLength, ReceiptDigest: b.ReceiptDigest, TransactionID: b.ProducerRootID, OwnerRequestID: source.ID, RetentionID: childDigest("6"), InstanceID: "private-worker", WorkerBootID: "private-boot", State: "held"}
	wrong := h
	wrong.ContentBytes++
	cached.ByteRetentions = []records.NativeArtifactRetention{wrong}
	if store.AdoptCachedOperation(consumer.ID, cached) == nil {
		t.Fatal("byte cache hit changed content capacity")
	}
	cached.ByteRetentions = []records.NativeArtifactRetention{h}
	fatal(t, store.AdoptCachedOperation(consumer.ID, cached))
	row, problem := store.RequestRow(consumer.ID)
	fatal(t, problem)
	if row.Ordinal != 0 || row.ReusedFrom != source.ID || row.State != "succeeded" {
		t.Fatal("byte cache invented an attempt or lost provenance")
	}
	replay, problem := store.ReserveByteResult(consumer.ID, b)
	fatal(t, problem)
	if replay.RetentionID != h.RetentionID {
		t.Fatal("watcher substituted another root after cache adoption")
	}
	received, problem := store.ReceivedByteAssets(second.ID)
	fatal(t, problem)
	if len(received) != 1 || received[0].Native.Output.RequestID != source.ID {
		t.Fatal("cache hit lost original byte producer")
	}
}
