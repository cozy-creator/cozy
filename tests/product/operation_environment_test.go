package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestOperationNumericsBelongToCalleeAndRemainPinnedThroughLookupReplay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer func() { store.Close() }()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	makeCall := func(label string) records.Request {
		parent := offerChildParent(t, store, recordPrivateTransaction(t, store, label+"-script", ""))
		call := records.Request{ID: "req-" + label, IdemKey: label, Kind: "job", Package: "local/callee", Entrypoint: "compute", Payload: []byte(`{"value":7}`), BodyDigest: childDigest("a"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("b"), ChildTargetDigest: childDigest("c"), ChildReusable: true}
		child, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
		fatal(t, problem)
		return child
	}
	first := makeCall("first")
	if problem := store.BeginOperationLookup(first.ID, childDigest("d")); problem == nil {
		t.Fatal("lookup accepted before numerical qualification")
	}
	context, problem := store.BindOperationContext(first.ID, childDigest("e"))
	fatal(t, problem)
	fatal(t, store.BeginOperationLookup(first.ID, context.Key))
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	replayed, problem := store.BindOperationContext(first.ID, childDigest("e"))
	fatal(t, problem)
	if *replayed != *context {
		t.Fatal("reconnect changed the pending numerical identity")
	}
	if _, problem := store.BindOperationContext(first.ID, childDigest("f")); problem == nil || problem.Name != "operation.numerical_environment_changed" {
		t.Fatal("changed environment replaced the pinned lookup")
	}
	lookup, problem := store.OperationLookup(first.ID)
	fatal(t, problem)
	if lookup.Key != context.Key || lookup.State != "pending" {
		t.Fatal("failed requalification changed the lookup obligation")
	}
	recorded, problem := store.RequestRow(first.ID)
	fatal(t, problem)
	if recorded.ChildTargetDigest != first.ChildTargetDigest {
		t.Fatal("numerical qualification rewrote captured code identity")
	}
	edited := makeCall("edited")
	same, problem := store.BindOperationContext(edited.ID, childDigest("e"))
	fatal(t, problem)
	if same.Key != context.Key {
		t.Fatal("independent script identity contaminated function memoization")
	}
	other := makeCall("other-environment")
	changed, problem := store.BindOperationContext(other.ID, childDigest("f"))
	fatal(t, problem)
	if changed.Key == context.Key {
		t.Fatal("actual numerical environment change reused the old computation key")
	}
}

func TestOldPendingLookupOnlyReconcilesForCancellation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer func() {
		if store != nil {
			store.Close()
		}
	}()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	first := offerChildParent(t, store, recordPrivateTransaction(t, store, "old-source-script", ""))
	source := offerChildParent(t, store, operationHistory(t, store, "old-source", first, true))
	receipt, _, err := canonical.Identity(&pb.WeightsReceipt{OwnerAuthorityScope: "owner", RequestId: source.ID, InvocationSpecDigest: childDigest("1"), OutputSlot: "weights", WeightsTransactionId: childDigest("5"), TensorfsReceiptDigest: childDigest("4"), TensorfsReceiptCanonicalBytes: []byte(`{}`)})
	must(t, err)
	receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
	fatal(t, store.RecordModelTransferWeights(records.ModelTransferWeights{RequestID: source.ID, Attempt: 1, OutputSlot: "weights", ManifestID: childDigest("6"), ManifestLength: 161, InvocationDigest: childDigest("1"), TransactionID: childDigest("5"), ReceiptDigest: receiptDigest, Receipt: receipt}))
	closeChild(t, store, source, "SUCCEEDED", "succeeded")
	second := offerChildParent(t, store, recordPrivateTransaction(t, store, "old-consumer-script", ""))
	consumer := operationHistory(t, store, "old-consumer", second, true)
	key, problem := records.OperationKey(consumer)
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	restorePriorCallIndexBounds(t, db)
	for _, table := range []string{"request_child_arguments", "attempt_serving_placements"} {
		_, err = db.Exec(`DROP TABLE ` + table)
		must(t, err)
	}
	_, err = db.Exec(`INSERT INTO request_operation_lookups(request_id,computation_digest,state) VALUES(?,?,'pending')`, consumer.ID, key)
	must(t, err)
	_, err = db.Exec(`DROP TABLE IF EXISTS successful_work_releases; PRAGMA user_version=34`)
	must(t, err)
	db.Close()
	store, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	attempt, problem := store.AttemptRow(source.ID, 1)
	fatal(t, problem)
	cached := records.CachedOperation{Key: key, SourceRequestID: source.ID, SourceAttempt: 1, InvocationDigest: attempt.InvocationDigest, OutcomeID: attempt.TerminalID, OutcomeDigest: attempt.TerminalDigest, OutcomeBody: attempt.TerminalBody}
	hold := records.WeightsRetention{RequestID: consumer.ID, Kind: "result", Slot: "weights/weights", ProducerRequestID: source.ID, ProducerAttempt: 1, ProducerOutputSlot: "weights", RetentionID: childDigest("d"), InstanceID: "private-worker", WorkerBootID: "private-boot", State: "held"}
	cached.Retentions = []records.WeightsRetention{hold}
	if problem := store.AdoptCachedOperation(consumer.ID, cached); problem == nil || problem.Name != "operation.context_absent" {
		t.Fatal("unqualified historical lookup served a new result")
	}
	if _, problem := store.BindOperationContext(consumer.ID, childDigest("f")); problem == nil || problem.Name != "operation.key_changed" {
		t.Fatal("pending legacy key was rewritten under a new environment")
	}
	fatal(t, store.RequestRetainedCancellation(consumer.ID, "upgrade cancellation"))
	fatal(t, store.AdoptCachedOperation(consumer.ID, cached))
	fatal(t, store.AdoptCachedOperation(consumer.ID, cached))
	row, problem := store.RequestRow(consumer.ID)
	fatal(t, problem)
	if row.State != "canceling" || row.ReusedFrom != "" || row.ChildTargetDigest != consumer.ChildTargetDigest {
		t.Fatal("legacy cancellation changed execution provenance")
	}
	context, problem := store.OperationContext(consumer.ID)
	fatal(t, problem)
	if context != nil {
		t.Fatal("legacy cancellation invented a numerical identity")
	}
	finished, problem := store.ReleaseRetainedWork(consumer.ID)
	fatal(t, problem)
	if finished {
		t.Fatal("legacy cancellation dropped an unacknowledged native hold")
	}
	fatal(t, store.BeginWeightsRetentionRelease(consumer.ID, false))
	fatal(t, store.CompleteWeightsRetentionRelease(hold.RetentionID))
	finished, problem = store.ReleaseRetainedWork(consumer.ID)
	fatal(t, problem)
	if !finished {
		t.Fatal("reconciled legacy cancellation remained pending")
	}
}
