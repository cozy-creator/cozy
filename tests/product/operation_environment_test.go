package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
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
	if _, problem := store.BindOperationContext(first.ID, childDigest("f")); problem == nil || problem.Name != "operation.key_changed" {
		t.Fatal("changed environment replaced the in-flight lookup")
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
	// An unoffered call whose lookup missed moves with its worker: a changed Runtime or
	// machine rebinds it to the environment it will run in instead of parking it.
	moved := makeCall("moved")
	bound, problem := store.BindOperationContext(moved.ID, childDigest("e"))
	fatal(t, problem)
	fatal(t, store.BeginOperationLookup(moved.ID, bound.Key))
	fatal(t, store.CompleteOperationMiss(moved.ID, bound.Key))
	rebound, problem := store.BindOperationContext(moved.ID, childDigest("f"))
	fatal(t, problem)
	if rebound.Key != changed.Key || rebound.NumericalEnvironment != childDigest("f") {
		t.Fatalf("a missed lookup did not rebind to the new environment: %+v", rebound)
	}
	fatal(t, store.BeginOperationLookup(moved.ID, rebound.Key))
}
