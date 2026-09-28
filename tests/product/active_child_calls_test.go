package producttest

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestActiveCallLimitSharesPackageAndNativeWorkButAllowsLongHistory(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "active-parent", ""))
	makeCall := func(index int64) records.Request {
		id := fmt.Sprintf("req-active-%d", index)
		return records.Request{ID: id, IdemKey: id, Kind: "job", Package: "local/op", Entrypoint: "run", Payload: []byte(`{}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID, ParentCallIndex: index, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3")}
	}
	var first records.Request
	for index := int64(0); index < 32; index++ {
		call, fresh, problem := store.SubmitChild(makeCall(index), 1, childDigest("1"), "private-boot", nil)
		fatal(t, problem)
		if !fresh {
			t.Fatal("new index reused a call")
		}
		if index == 0 {
			first = call
		}
	}
	if _, _, problem := store.SubmitChild(makeCall(32), 1, childDigest("1"), "private-boot", nil); problem == nil || problem.ErrName() != "child.active_limit" {
		t.Fatalf("33rd active call: %v", problem)
	}
	native := records.NativeCall{ID: "native-active", ParentRequestID: parent.ID, CallIndex: 33, Kind: "source", Operation: "download_huggingface", IntentDigest: childDigest("4"), Request: []byte(`{}`)}
	if _, _, problem := store.AcceptNativeCall(native, 1, childDigest("1"), "private-boot"); problem == nil || problem.ErrName() != "child.active_limit" {
		t.Fatalf("native call bypassed shared limit: %v", problem)
	}
	replay, fresh, problem := store.SubmitChild(makeCall(0), 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if fresh || replay.ID != first.ID {
		t.Fatal("full active set blocked same-index recovery")
	}
	first = offerChildParent(t, store, first)
	closeChild(t, store, first, "SUCCEEDED", "succeeded")
	_, fresh, problem = store.AcceptNativeCall(native, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if !fresh {
		t.Fatal("completed package call did not free one active slot")
	}
	fatal(t, store.StopNativeCall(native.ID, "failed", "fixture complete"))
	for index := int64(34); index < 80; index++ {
		call, _, problem := store.SubmitChild(makeCall(index), 1, childDigest("1"), "private-boot", nil)
		fatal(t, problem)
		call = offerChildParent(t, store, call)
		closeChild(t, store, call, "SUCCEEDED", "succeeded")
	}
	last, problem := store.ChildAt(parent.ID, 79)
	fatal(t, problem)
	if last == nil || last.State != "succeeded" {
		t.Fatal("durable history lost its later accepted index")
	}
}
