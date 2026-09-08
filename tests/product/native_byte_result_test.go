package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

// Custody is committed before the value becomes grantable; no terminal or fake
// child request is invented for a service running inside an ordinary parent.
func TestNativeByteServiceUsesOriginalAttemptAndIndependentReceivedHold(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "native-byte", ""))
	call := records.NativeCall{ID: "source-files", ParentRequestID: parent.ID, CallIndex: 80, Kind: "source", Operation: "source_files", IntentDigest: childDigest("4"), Request: []byte(`{}`)}
	_, _, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	receipt := []byte("native receipt is opaque to Creator")
	digest, _ := canonical.Spell(canonical.Digest(receipt))
	spec, _ := canonical.Raw(childDigest("1"))
	output := records.ByteOutput{RequestID: parent.ID, Attempt: 1, OutputID: "runtime.source_files.80", Digest: childDigest("6"), Length: 100, MimeType: "application/vnd.cozy.tree-manifest", ReceiptDigest: digest, ManifestID: childDigest("6"), ManifestLength: 100, ContentBytes: 1024}
	output.ProducerRootID = records.NativeByteProducerRoot("cozy-local-client", parent.ID, 1, spec, output.OutputID)
	complete := func(b records.ByteOutput, producerSpec string) (records.NativeArtifactRetention, error) {
		h, p := store.CompleteNativeByteCall(call.ID, 1, childDigest("1"), "private-boot", producerSpec, b, []byte(`{"files":{}}`), receipt, "private-worker")
		if p != nil {
			return h, p
		}
		return h, nil
	}
	if _, err := complete(output, childDigest("2")); err == nil {
		t.Fatal("forged producer spec accepted")
	}
	if value, problem := store.NativeByteOutput(call.ID); problem != nil || value != nil {
		t.Fatal("refusal reserved bytes")
	}
	held, err := complete(output, childDigest("1"))
	must(t, err)
	if held.State != "pending" || held.ConsumerID != parent.ID || held.ParentRequestID != parent.ID {
		t.Fatalf("wrong native recipient: %+v", held)
	}
	assets, problem := store.ReceivedByteAssets(parent.ID)
	fatal(t, problem)
	if len(assets) != 0 {
		t.Fatal("unretained source was exposed")
	}
	visible, problem := store.ByteOutputs(parent.ID, 1)
	fatal(t, problem)
	if len(visible) != 0 {
		t.Fatal("native service pretended to be a parent return field")
	}
	again, err := complete(output, childDigest("1"))
	must(t, err)
	if again != held {
		t.Fatal("lost reply changed held intent")
	}
	changed := output
	changed.ContentBytes++
	if _, err := complete(changed, childDigest("1")); err == nil {
		t.Fatal("changed completed source accepted")
	}
	fatal(t, store.ConfirmNativeArtifact(held.RetentionID, "private-worker", "private-boot"))
	assets, problem = store.ReceivedByteAssets(parent.ID)
	fatal(t, problem)
	if len(assets) != 1 || assets[0].Native == nil {
		t.Fatal("completed retained source was not granted")
	}
	child := records.Request{ID: "req-native-byte-reader", IdemKey: "native-byte-reader", BodyDigest: childDigest("b"), Package: "local/operation", Entrypoint: "read", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 81, ChildIntentDigest: childDigest("c"), ChildTargetDigest: childDigest("d"), Assets: assets}
	child.Assets[0].FieldPath = "metadata"
	downstream, _, problem := store.SubmitChild(child, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if downstream.Assets[0].Native.RetentionID == held.RetentionID {
		t.Fatal("child borrowed source hold")
	}
	fatal(t, store.BeginNativeArtifactRelease(parent.ID, false))
	if store.ConfirmNativeArtifact(held.RetentionID, "private-worker", "private-boot") == nil {
		t.Fatal("late ACK resurrected hold")
	}
	if _, err := complete(output, childDigest("1")); err == nil {
		t.Fatal("released parent reacquired same native output")
	}
	hold, problem := store.NativeArtifactRetentions(downstream.ID)
	fatal(t, problem)
	if len(hold) != 1 || hold[0].State != "pending" {
		t.Fatal("parent release erased independent child input")
	}
}
