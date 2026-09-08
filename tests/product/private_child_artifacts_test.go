package producttest

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPrivateModelArtifactAdmissionAcquiresCustodyBeforeExecution(t *testing.T) {
	for _, kind := range []string{"job", "serving"} {
		t.Run(kind, func(t *testing.T) { privateModelArtifactAdmission(t, kind) })
	}
}

func privateModelArtifactAdmission(t *testing.T, kind string) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "artifact-parent", ""))
	producer, _, problem := store.SubmitChild(records.Request{ID: "req-artifact-producer", IdemKey: "artifact-producer", Kind: "job", Package: "local/source", Entrypoint: "source", Payload: []byte(`{}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3"), ChildArtifacts: true, WeightsOutputs: `[{"output_id":"weights"}]`}, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	producer = offerChildParent(t, store, producer)
	nativeReceipt := childDigest("4")
	protocolReceipt := &pb.WeightsReceipt{OwnerAuthorityScope: "owner", RequestId: producer.ID, InvocationSpecDigest: childDigest("1"), OutputSlot: "weights", WeightsTransactionId: childDigest("5"), TensorfsReceiptDigest: nativeReceipt, TensorfsReceiptCanonicalBytes: []byte(`{}`)}
	receipt, _, err := canonical.Identity(protocolReceipt)
	must(t, err)
	receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
	weights := records.ModelTransferWeights{RequestID: producer.ID, Attempt: 1, OutputSlot: "weights", ManifestID: childDigest("6"), ManifestLength: 321, InvocationDigest: childDigest("1"), TransactionID: childDigest("5"), ReceiptDigest: receiptDigest, Receipt: receipt}
	fatal(t, store.RecordModelTransferWeights(weights))
	closeChild(t, store, producer, "SUCCEEDED", "succeeded")
	artifact := records.ModelArtifact{ProducerRequestID: producer.ID, OutputSlot: "weights", Manifest: records.ArtifactObjectRef{Digest: weights.ManifestID, Length: 321}, TensorFSReceiptDigest: nativeReceipt}
	_, problem = store.ArtifactOutput(artifact)
	fatal(t, problem)
	forged := artifact
	forged.TensorFSReceiptDigest = receiptDigest
	if _, problem := store.ArtifactOutput(forged); problem == nil {
		t.Fatal("protocol receipt digest was accepted as native receipt authority")
	}
	payload, err := json.Marshal(map[string]any{"model": artifact})
	must(t, err)
	var callArguments []byte
	if kind == "serving" {
		callArguments, err = canonical.NormalizeJCS(append(append([]byte(`{"models":`), payload...), []byte(`,"payload":{"prompt":"test"}}`)...))
		must(t, err)
		payload = []byte(`{"prompt":"test"}`)
	}
	consumer, _, problem := store.SubmitChild(records.Request{ID: "req-artifact-consumer", IdemKey: "artifact-consumer", Kind: kind, Package: "local/candidate", Entrypoint: "run", Payload: payload, BodyDigest: childDigest("7"), ParentRequestID: parent.ID, ParentCallIndex: 1, ChildIntentDigest: childDigest("8"), ChildTargetDigest: childDigest("9"), Models: []records.ModelRef{{Slot: "model", Manifest: artifact.Manifest.Digest, ManifestLength: artifact.Manifest.Length}}}, 1, childDigest("1"), "private-boot", callArguments)
	fatal(t, problem)
	if kind == "serving" {
		retained, problem := store.ChildArguments(consumer)
		fatal(t, problem)
		if string(retained) != string(callArguments) || string(consumer.Payload) != `{"prompt":"test"}` {
			t.Fatal("serving admission changed its payload or lost the original call")
		}
	}
	retentions, problem := store.WeightsRetentions(consumer.ID)
	fatal(t, problem)
	if len(retentions) != 1 || retentions[0].State != "pending" || retentions[0].ProducerRequestID != producer.ID || retentions[0].ProducerAttempt != 1 {
		t.Fatalf("input admission failed to acquire durable original custody: %+v", retentions)
	}
	borrowed, problem := store.PendingArtifactBorrowers(producer.ID)
	fatal(t, problem)
	if !borrowed {
		t.Fatal("original cleanup could overtake accepted consumer")
	}
	fatal(t, store.RequestRetainedCancellation(producer.ID, "explicit owner abandonment"))
	_, problem = store.RecordWeightsRetention(retentions[0])
	fatal(t, problem)
	fatal(t, store.ConfirmWeightsRetention(retentions[0].RetentionID, "private-worker", "private-boot"))
	borrowed, problem = store.PendingArtifactBorrowers(producer.ID)
	fatal(t, problem)
	if borrowed {
		t.Fatal("independent native hold did not release predecessor cleanup barrier")
	}
	_, problem = store.ReleaseRetainedWork(producer.ID)
	fatal(t, problem)
	_, problem = store.CompleteRetainedCancellation(producer.ID)
	fatal(t, problem)
	held, problem := store.ArtifactHasCustody(producer.ID, 1, "weights", parent.ReuseScope)
	fatal(t, problem)
	if !held {
		t.Fatal("original disposal lost independent consumer custody")
	}
	fatal(t, store.BeginWeightsRetentionRelease(consumer.ID, false))
	fatal(t, store.CompleteWeightsRetentionRelease(retentions[0].RetentionID))
	held, problem = store.ArtifactHasCustody(producer.ID, 1, "weights", parent.ReuseScope)
	fatal(t, problem)
	if held {
		t.Fatal("released final artifact hold remained spendable")
	}
	if _, problem := store.RecordWeightsRetention(retentions[0]); problem == nil {
		t.Fatal("released artifact retention was resurrected")
	}
}

func TestOnlySchemaDeclaredModelResultsAcquireNativeCustody(t *testing.T) {
	ordinary := launch.Struct{Fields: []launch.Field{{Name: "tensorfs_receipt_digest", Type: json.RawMessage(`"str"`)}, {Name: "manifest", Type: json.RawMessage(`"str"`)}}}
	if paths := launch.ModelArtifactPaths(ordinary); len(paths) != 0 {
		t.Fatal("ordinary result field spellings became native ownership")
	}
	native := launch.Struct{Fields: []launch.Field{{Name: "candidate", Type: json.RawMessage(`{"union":["null",{"input":"model"}]}`)}, {Name: "history", Type: json.RawMessage(`{"list":{"input":"model"}}`)}}}
	if paths := launch.ModelArtifactPaths(native); !reflect.DeepEqual(paths, [][]string{{"candidate"}, {"history", "*"}}) {
		t.Fatalf("typed native artifact paths = %#v", paths)
	}
	if paths := launch.ModelArtifactPaths(launch.Struct{Input: "model"}); len(paths) != 1 || len(paths[0]) != 0 {
		t.Fatalf("direct ModelArtifact result path = %#v", paths)
	}
}

func TestModelArtifactIsClosedAndDigestExact(t *testing.T) {
	valid := records.ModelArtifact{ProducerRequestID: "producer", OutputSlot: "weights", Manifest: records.ArtifactObjectRef{Digest: childDigest("a"), Length: 42}, TensorFSReceiptDigest: childDigest("b")}
	raw, err := json.Marshal(valid)
	must(t, err)
	if _, problem := records.DecodeModelArtifact(raw); problem != nil {
		t.Fatal(problem)
	}
	for _, mutation := range []string{`{"producer_request_id":"p","output_slot":"w","manifest":{"digest":"bad","length":1},"tensorfs_receipt_digest":"bad"}`, `{"producer_request_id":"p","output_slot":"w","manifest":null,"tensorfs_receipt_digest":"bad"}`} {
		if _, problem := records.DecodeModelArtifact([]byte(mutation)); problem == nil {
			t.Fatal("forged model handle passed its closed schema")
		}
	}
}

func TestPrivateCompletedScriptRetainsDelegatedArtifacts(t *testing.T) {
	for _, artifacts := range []bool{false, true} {
		t.Run(map[bool]string{false: "scalar", true: "native_artifact"}[artifacts], func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
			parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "no-result-parent", ""))
			child, _, problem := store.SubmitChild(records.Request{ID: "req-delegated-result", IdemKey: "delegated-result", Kind: "job", Package: "local/source", Entrypoint: "compute", Payload: []byte(`{}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3"), ChildArtifacts: artifacts}, 1, childDigest("1"), "private-boot", nil)
			fatal(t, problem)
			closeChild(t, store, child, "SUCCEEDED", "succeeded")
			closeChild(t, store, parent, "SUCCEEDED", "succeeded")
			parentRow, problem := store.RequestRow(parent.ID)
			fatal(t, problem)
			retaining, problem := store.RequestRetaining(*parentRow)
			fatal(t, problem)
			if retaining != artifacts {
				t.Fatalf("parent with no return value retained=%t, expected child artifact custody=%t", retaining, artifacts)
			}
			if artifacts {
				fatal(t, store.RequestRetainedCancellation(child.ID, "release retained child"))
				_, problem = store.ReleaseRetainedWork(child.ID)
				fatal(t, problem)
				_, problem = store.CompleteRetainedCancellation(child.ID)
				fatal(t, problem)
				retaining, problem = store.RequestRetaining(*parentRow)
				fatal(t, problem)
				if retaining {
					t.Fatal("completed parent still retained an abandoned child")
				}
			}
		})
	}
}
