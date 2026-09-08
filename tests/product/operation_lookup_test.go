package producttest

import (
	"database/sql"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func operationHistory(t *testing.T, store *records.Store, label string, parent records.Request, native ...bool) records.Request {
	t.Helper()
	request := records.Request{ID: "req-" + label, IdemKey: label, Kind: "job", Package: "local/scalar", Entrypoint: "compute", Worker: parent.Worker, Payload: []byte(`{"value":7}`), BodyDigest: childDigest("a"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("b"), ChildTargetDigest: childDigest("c"), ChildReusable: true}
	if len(native) > 0 && native[0] {
		request.ChildArtifacts = true
		request.WeightsOutputs = `[{"output_id":"weights"}]`
	}
	child, _, problem := store.SubmitChild(request, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	return child
}

func TestOperationArtifactScopeRequiresNativeCacheAdoption(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	first := offerChildParent(t, store, recordPrivateTransaction(t, store, "first", "workspace"))
	source := offerChildParent(t, store, operationHistory(t, store, "native-source", first, true))
	native := childDigest("4")
	receipt, _, err := canonical.Identity(&pb.WeightsReceipt{OwnerAuthorityScope: "owner", RequestId: source.ID, InvocationSpecDigest: childDigest("1"), OutputSlot: "weights", WeightsTransactionId: childDigest("5"), TensorfsReceiptDigest: native, TensorfsReceiptCanonicalBytes: []byte(`{}`)})
	must(t, err)
	receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
	fatal(t, store.RecordModelTransferWeights(records.ModelTransferWeights{RequestID: source.ID, Attempt: 1, OutputSlot: "weights", ManifestID: childDigest("6"), ManifestLength: 161, InvocationDigest: childDigest("1"), TransactionID: childDigest("5"), ReceiptDigest: receiptDigest, Receipt: receipt}))
	closeChild(t, store, source, "SUCCEEDED", "succeeded")
	second := offerChildParent(t, store, recordPrivateTransaction(t, store, "fresh", "workspace"))
	artifact := records.ModelArtifact{ProducerRequestID: source.ID, OutputSlot: "weights", Manifest: records.ArtifactObjectRef{Digest: childDigest("6"), Length: 161}, TensorFSReceiptDigest: native}
	payload, err := json.Marshal(map[string]any{"source": artifact})
	must(t, err)
	next := records.Request{ID: "req-model-consumer", IdemKey: "model-consumer", Kind: "job", Package: "local/next", Entrypoint: "compute", Worker: "workspace", Payload: payload, BodyDigest: childDigest("8"), ParentRequestID: second.ID, ParentCallIndex: 1, ChildIntentDigest: childDigest("9"), ChildTargetDigest: childDigest("a"), Models: []records.ModelRef{{Slot: "source", Manifest: artifact.Manifest.Digest, ManifestLength: 161}}}
	if _, _, problem := store.SubmitChild(next, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("same worker allowed a forged cross-run artifact grant")
	}
	consumer := operationHistory(t, store, "cached-native", second, true)
	key, problem := records.OperationKey(consumer)
	fatal(t, problem)
	fatal(t, store.BeginOperationLookup(consumer.ID, key))
	last, problem := store.AttemptRow(source.ID, 1)
	fatal(t, problem)
	hold := records.WeightsRetention{RequestID: consumer.ID, Kind: "result", Slot: "weights/weights", ProducerRequestID: source.ID, ProducerAttempt: 1, ProducerOutputSlot: "weights", RetentionID: childDigest("d"), InstanceID: "private-worker", WorkerBootID: "private-boot", State: "held"}
	fatal(t, store.AdoptCachedOperation(consumer.ID, records.CachedOperation{Key: key, SourceRequestID: source.ID, SourceAttempt: 1, InvocationDigest: last.InvocationDigest, OutcomeID: last.TerminalID, OutcomeDigest: last.TerminalDigest, OutcomeBody: last.TerminalBody, Retentions: []records.WeightsRetention{hold}}))
	_, _, problem = store.SubmitChild(next, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	owned, problem := store.ArtifactHasCustody(source.ID, 1, "weights", second.ReuseScope)
	fatal(t, problem)
	if !owned {
		t.Fatal("authenticated cache hit did not grant the new run independent artifact custody")
	}
	fatal(t, store.RequestRetainedCancellation(consumer.ID, "test"))
	done, problem := store.ReleaseRetainedWork(consumer.ID)
	fatal(t, problem)
	if done {
		t.Fatal("cancellation dropped a still-held cache result")
	}
	fatal(t, store.BeginWeightsRetentionRelease(consumer.ID, false))
	fatal(t, store.CompleteWeightsRetentionRelease(hold.RetentionID))
	done, problem = store.ReleaseRetainedWork(consumer.ID)
	fatal(t, problem)
	if !done {
		t.Fatal("released native result did not finish cancellation")
	}
	third := offerChildParent(t, store, recordPrivateTransaction(t, store, "cancel-race", "workspace"))
	stopped := operationHistory(t, store, "lookup-reply-lost", third, true)
	fatal(t, store.BeginOperationLookup(stopped.ID, key))
	fatal(t, store.RequestRetainedCancellation(stopped.ID, "while Host lookup reply was lost"))
	hold.RequestID, hold.RetentionID = stopped.ID, childDigest("e")
	fatal(t, store.AdoptCachedOperation(stopped.ID, records.CachedOperation{Key: key, SourceRequestID: source.ID, SourceAttempt: 1, InvocationDigest: last.InvocationDigest, OutcomeID: last.TerminalID, OutcomeDigest: last.TerminalDigest, OutcomeBody: last.TerminalBody, Retentions: []records.WeightsRetention{hold}}))
	stoppedRow, problem := store.RequestRow(stopped.ID)
	fatal(t, problem)
	if stoppedRow.State != "canceling" || stoppedRow.ReusedFrom != "" {
		t.Fatal("late cache receipt overrode cancellation")
	}
	lookup, problem := store.OperationLookup(stopped.ID)
	fatal(t, problem)
	if lookup.State != "hit" {
		t.Fatal("cancellation failed to reconcile the lost lookup response")
	}
	done, problem = store.ReleaseRetainedWork(stopped.ID)
	fatal(t, problem)
	if done {
		t.Fatal("late native ownership escaped the cancellation release barrier")
	}
	fatal(t, store.BeginWeightsRetentionRelease(stopped.ID, false))
	fatal(t, store.CompleteWeightsRetentionRelease(hold.RetentionID))
	done, problem = store.ReleaseRetainedWork(stopped.ID)
	fatal(t, problem)
	if !done {
		t.Fatal("reconciled and released cache hit left cancellation unfinished")
	}
}

func TestOperationLookupAdoptsAcrossFreshRunHistory(t *testing.T) {
	for _, workspace := range []string{"", "workspace"} {
		name := "local"
		if workspace != "" {
			name = "rental"
		}
		t.Run(name, func(t *testing.T) { proveOperationLookupAcrossFreshRunHistory(t, workspace) })
	}
}

func proveOperationLookupAcrossFreshRunHistory(t *testing.T, workspace string) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	first := offerChildParent(t, store, recordPrivateTransaction(t, store, "first", workspace))
	source := operationHistory(t, store, "original-op", first)
	closeChild(t, store, source, "SUCCEEDED", "succeeded")
	second := offerChildParent(t, store, recordPrivateTransaction(t, store, "fresh", workspace))
	consumer := operationHistory(t, store, "fresh-op", second)
	if consumer.ReuseScope == source.ReuseScope || second.RetryOf != "" {
		t.Fatal("fixture did not create an unrelated run")
	}
	key, problem := records.OperationKey(consumer)
	fatal(t, problem)
	last, problem := store.AttemptRow(source.ID, 1)
	fatal(t, problem)
	cached := records.CachedOperation{Key: key, SourceRequestID: source.ID, SourceAttempt: 1, InvocationDigest: last.InvocationDigest, OutcomeID: last.TerminalID, OutcomeDigest: last.TerminalDigest, OutcomeBody: last.TerminalBody}
	if problem := store.AdoptCachedOperation(consumer.ID, cached); problem == nil {
		t.Fatal("cache observation without a pending owner lookup was accepted")
	}
	fatal(t, store.BeginOperationLookup(consumer.ID, key))
	changed := cached
	changed.OutcomeID = "forged"
	if problem := store.AdoptCachedOperation(consumer.ID, changed); problem == nil {
		t.Fatal("cache hit changed original terminal provenance")
	}
	fatal(t, store.AdoptCachedOperation(consumer.ID, cached))
	row, problem := store.RequestRow(consumer.ID)
	fatal(t, problem)
	if row.State != "succeeded" || row.Ordinal != 0 || row.ReusedFrom != source.ID || row.ReuseScope != second.ReuseScope {
		t.Fatalf("workspace adoption rewrote run ownership: %+v", row)
	}
	attempts, problem := store.Attempts(consumer.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("cache hit invented an execution receipt")
	}
	lookup, problem := store.OperationLookup(consumer.ID)
	fatal(t, problem)
	if lookup.State != "hit" || lookup.Key != key {
		t.Fatal("cache hit lost its durable RPC receipt")
	}
	third := offerChildParent(t, store, recordPrivateTransaction(t, store, "pause-hit", workspace))
	paused := operationHistory(t, store, "paused-hit", third)
	fatal(t, store.BeginOperationLookup(paused.ID, key))
	_, problem = store.RequestPause(paused.ID, "lookup pending")
	fatal(t, problem)
	fatal(t, store.AdoptCachedOperation(paused.ID, cached))
	done, problem := store.CompleteRequestPause(paused.ID)
	fatal(t, problem)
	if !done {
		t.Fatal("observed cache HIT did not complete pause")
	}
	_, problem = store.ResumeRequest(paused.ID, "resume owned result")
	fatal(t, problem)
	// No Host lookup or cache-index survival is needed after ownership is ours.
	fatal(t, store.CompleteReusedChild(paused.ID))
	resumed, problem := store.RequestRow(paused.ID)
	fatal(t, problem)
	if resumed.State != "succeeded" || resumed.Ordinal != 0 || resumed.ReusedFrom != source.ID {
		t.Fatal("paused cached result was executed again on resume")
	}
}

func TestOperationLookupBlocksPauseAndCancelUntilReconciled(t *testing.T) {
	for _, action := range []string{"pause", "cancel"} {
		t.Run(action, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "creator.sqlite")
			store, problem := records.Open(path)
			fatal(t, problem)
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
			parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "parent", "workspace"))
			child := operationHistory(t, store, "pending-op", parent)
			key, problem := records.OperationKey(child)
			fatal(t, problem)
			fatal(t, store.BeginOperationLookup(child.ID, key))
			if action == "pause" {
				_, problem = store.RequestPause(child.ID, "test")
				fatal(t, problem)
				finished, problem := store.CompleteRequestPause(child.ID)
				fatal(t, problem)
				if finished {
					t.Fatal("pause claimed a pending native cache effect was quiet")
				}
			} else {
				fatal(t, store.RequestRetainedCancellation(child.ID, "test"))
				finished, problem := store.ReleaseRetainedWork(child.ID)
				fatal(t, problem)
				if finished {
					t.Fatal("cancellation overtook unobserved native ownership")
				}
			}
			store.Close()
			store, problem = records.Open(path)
			fatal(t, problem)
			defer store.Close()
			lookup, problem := store.OperationLookup(child.ID)
			fatal(t, problem)
			if lookup.Key != key || lookup.State != "pending" {
				t.Fatal("restart lost exact pending lookup")
			}
			fatal(t, store.CompleteOperationMiss(child.ID, key))
			if action == "pause" {
				done, problem := store.CompleteRequestPause(child.ID)
				fatal(t, problem)
				if !done {
					t.Fatal("reconciled miss did not finish pause")
				}
			} else {
				done, problem := store.ReleaseRetainedWork(child.ID)
				fatal(t, problem)
				if !done {
					t.Fatal("reconciled miss did not finish cancellation")
				}
			}
		})
	}
}

func TestOperationLookupSchemaUpgradePreservesRequests(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	old := recordPrivateTransaction(t, store, "schema28", "")
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	for _, table := range []string{"native_artifact_retentions", "native_calls", "request_operation_lookups"} {
		_, err = db.Exec(`DROP TABLE ` + table)
		must(t, err)
	}
	_, err = db.Exec(`PRAGMA user_version=28`)
	must(t, err)
	must(t, db.Close())
	store, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestRow(old.ID)
	fatal(t, problem)
	if row.BodyDigest != old.BodyDigest || row.ReuseScope != old.ReuseScope {
		t.Fatal("cache RPC migration changed prior work")
	}
	lookup, problem := store.OperationLookup(old.ID)
	fatal(t, problem)
	if lookup != nil {
		t.Fatal("migration invented a cache lookup")
	}
}
