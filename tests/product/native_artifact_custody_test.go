package producttest

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

// This checks Creator admission/custody records. TensorFS separately proves that
// the original observed native receipt really owns its payload on the worker.
func TestNativeArtifactChildAdmissionUsesRealServiceProvenanceAndIndependentHold(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "native-artifact", ""))
	source := records.NativeCall{ID: "source-native-artifact", ParentRequestID: parent.ID, CallIndex: 0, Kind: "source", Operation: "convert_cozytensors", IntentDigest: childDigest("4"), Request: []byte(`{"profile":"checkpoint"}`)}
	_, _, problem = store.AcceptNativeCall(source, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	receipt := []byte(`{"manifest":{"length":100,"sha256":"` + strings.Repeat("2", 64) + `"},"transaction_id":"` + childDigest("3") + `"}`)
	receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
	artifact := records.ModelArtifact{ProducerRequestID: source.ID, OutputSlot: "model", Manifest: records.ArtifactObjectRef{Digest: childDigest("2"), Length: 100}, TensorFSReceiptDigest: receiptDigest}
	result, _ := json.Marshal(artifact)
	result, _ = canonical.NormalizeJCS(result)
	fatal(t, store.CompleteNativeCallAt(source.ID, result, receipt, "private-worker", "private-boot"))
	provenance, problem := store.ResolveArtifactSource(artifact)
	fatal(t, problem)
	if provenance.Weights != nil || provenance.NativeServiceID != source.ID || provenance.TransactionID != childDigest("3") || provenance.OwnerRequestID != parent.ID {
		t.Fatalf("invented package provenance: %+v", provenance)
	}
	allowed, problem := store.ArtifactSourceAllowed(parent, *provenance)
	fatal(t, problem)
	if !allowed {
		t.Fatal("own native result not authorized")
	}
	changed := artifact
	changed.Manifest.Length++
	if _, problem := store.ResolveArtifactSource(changed); problem == nil {
		t.Fatal("changed native artifact accepted")
	}
	payload, _ := json.Marshal(map[string]any{"source": artifact})
	payload, _ = canonical.NormalizeJCS(payload)
	child := records.Request{ID: "req-native-artifact-consumer", IdemKey: "native-artifact-consumer", BodyDigest: childDigest("5"), Package: "local/operation", Entrypoint: "quantize", Kind: "job", Payload: payload, Models: []records.ModelRef{{Slot: "source", Model: "native/model", Manifest: artifact.Manifest.Digest, ManifestLength: artifact.Manifest.Length}}, ParentRequestID: parent.ID, ParentCallIndex: 1, ChildIntentDigest: childDigest("6"), ChildTargetDigest: childDigest("7")}
	recorded, _, problem := store.SubmitChild(child, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	holds, problem := store.NativeArtifactRetentions(recorded.ID)
	fatal(t, problem)
	if len(holds) != 1 || holds[0].State != "pending" || holds[0].ProducerID != source.ID {
		t.Fatalf("no independent pending native hold: %+v", holds)
	}
	fatal(t, store.ConfirmNativeArtifact(holds[0].RetentionID, "private-worker", "private-boot"))
	fatal(t, store.BeginNativeArtifactRelease(recorded.ID, true))
	if store.ConfirmNativeArtifact(holds[0].RetentionID, "private-worker", "private-boot") == nil {
		t.Fatal("late acquire resurrected releasing hold")
	}
	fatal(t, store.CompleteNativeArtifactRelease(holds[0].RetentionID))
	if _, problem := store.ReserveNativeArtifact(recorded.ID, recorded.ID, "input", "result/source", artifact); problem == nil {
		t.Fatal("released hold was reacquired")
	}
	unrelated := recordPrivateTransaction(t, store, "unrelated-native-artifact", "")
	allowed, problem = store.ArtifactSourceAllowed(unrelated, *provenance)
	fatal(t, problem)
	if allowed {
		t.Fatal("another private scope borrowed an unreceived native result")
	}
}
