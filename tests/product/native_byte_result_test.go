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

func TestNativeSourceRetryPreservesCancellationAndOriginalByteProducer(t *testing.T) {
	for _, state := range []string{"failed", "stopped", "canceled"} {
		t.Run(state, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
			parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "native-retry", ""))
			call := records.NativeCall{ID: "source-retry", ParentRequestID: parent.ID, CallIndex: 1, Kind: "source", Operation: "source_files", IntentDigest: childDigest("4"), Request: []byte(`{}`)}
			_, _, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
			fatal(t, problem)
			fatal(t, store.StopNativeCall(call.ID, state, "interrupted"))
			_, problem = store.AcceptTerminal(records.Terminal{RequestID: parent.ID, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "failed", TerminalDigest: childDigest("a"), Status: "FAILED", RequestState: "requeue_pending", Body: []byte(`{}`)})
			fatal(t, problem)
			fatal(t, store.Closed(parent.ID, 1))
			_, _, _, problem = store.BeginRequeue(parent.ID, 3, false)
			fatal(t, problem)
			ordinal, problem := store.Dispatch(records.Attempt{RequestID: parent.ID, InstanceID: "private-worker", SessionID: "private-boot", InvocationDigest: childDigest("2"), InvocationCanonical: []byte(`{}`)})
			fatal(t, problem)
			if ordinal != 2 {
				t.Fatal("retry changed original attempt")
			}
			fatal(t, store.OfferDispatch(parent.ID, 2, "private-boot"))
			again, _, problem := store.AcceptNativeCall(call, 2, childDigest("2"), "private-boot")
			fatal(t, problem)
			if state != "failed" {
				if again.State != state {
					t.Fatal("final cancellation was reopened")
				}
				return
			}
			if again.State != "accepted" {
				t.Fatal("interruption did not preserve retryable source intent")
			}
			receipt := []byte("native original A1 receipt")
			digest, _ := canonical.Spell(canonical.Digest(receipt))
			spec, _ := canonical.Raw(childDigest("1"))
			output := records.ByteOutput{RequestID: parent.ID, Attempt: 1, OutputID: "runtime.source_files.1", Digest: childDigest("6"), Length: 100, MimeType: "application/vnd.cozy.tree-manifest", ReceiptDigest: digest, ManifestID: childDigest("6"), ManifestLength: 100, ContentBytes: 1024}
			output.ProducerRootID = records.NativeByteProducerRoot("cozy-local-client", parent.ID, 1, spec, output.OutputID)
			hold, problem := store.CompleteNativeByteCall(call.ID, 2, childDigest("2"), "private-boot", childDigest("1"), output, []byte(`{"files":{}}`), receipt, "private-worker")
			fatal(t, problem)
			if hold.ProducerAttempt != 1 {
				t.Fatal("reattachment rebound A1 producer to A2")
			}
			changed := output
			changed.Attempt = 2
			if _, problem := store.CompleteNativeByteCall(call.ID, 2, childDigest("2"), "private-boot", childDigest("2"), changed, []byte(`{"files":{}}`), receipt, "private-worker"); problem == nil {
				t.Fatal("replay replaced original producer")
			}
		})
	}
}
