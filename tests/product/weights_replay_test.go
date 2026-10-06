package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
)

// An exact replay of a weights receipt with its object inventory is the same fact: the
// stored rows also carry transfer progress the frame never did, and that must not read
// as a changed identity. A changed object is still a conflict.
func TestWeightsReceiptReplayWithObjectsIsIdempotent(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "replay-parent", ""))
	producer, _, problem := store.SubmitChild(records.Request{ID: "req-replay-producer", IdemKey: "replay-producer", Kind: "job", Package: "local/source", Entrypoint: "source", Payload: []byte(`{}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3"), ChildArtifacts: true, WeightsOutputs: `[{"output_id":"weights"}]`}, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	producer = offerChildParent(t, store, producer)
	receipt, _, err := fixtureIdentity(map[string]any{"owner_authority_scope": "owner", "request_id": producer.ID, "invocation_spec_digest": childDigest("1"), "output_slot": "weights", "weights_transaction_id": childDigest("5"), "tensorfs_receipt_digest": childDigest("4"), "tensorfs_receipt_canonical_bytes": []byte(`{}`), "_format": "cozy.worker.v1.WeightsReceipt/1"})
	must(t, err)
	receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
	weights := records.ModelTransferWeights{RequestID: producer.ID, Attempt: 1, OutputSlot: "weights", ManifestID: childDigest("6"), ManifestLength: 321, InvocationDigest: childDigest("1"), TransactionID: childDigest("5"), ReceiptDigest: receiptDigest, Receipt: receipt,
		Objects: []records.ModelTransferObject{{ObjectID: childDigest("7"), Length: 4096, SourceRef: "blob-1"}, {ObjectID: childDigest("8"), Length: 8192, SourceRef: "blob-2"}}}
	fatal(t, store.RecordModelTransferWeights(weights))
	fatal(t, store.RecordModelTransferWeights(weights))
	changed := weights
	changed.Objects = append([]records.ModelTransferObject(nil), weights.Objects...)
	changed.Objects[1].Length = 1
	if problem := store.RecordModelTransferWeights(changed); problem == nil || problem.ErrName() != "model_transfer.weights_changed" {
		t.Fatalf("a changed object inventory replayed as the same fact: %v", problem)
	}
}
