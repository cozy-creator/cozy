package producttest

import (
	"bytes"
	"github.com/cozy-creator/cozy/internal/records"
	"path/filepath"
	"testing"
)

func TestNativeAndPackageCallsShareOneParentIndexAndFreezeBeforeEffects(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "native-parent", ""))
	call := records.NativeCall{ID: "source-native", ParentRequestID: parent.ID, CallIndex: 0, Kind: "source", Operation: "download_civitai", IntentDigest: childDigest("4"), Request: []byte(`{"version":42}`)}
	first, fresh, problem := store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if !fresh || first.State != "accepted" {
		t.Fatal("native admission not recorded")
	}
	_, fresh, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if fresh {
		t.Fatal("replay created another call")
	}
	changed := call
	changed.Kind = "effect"
	if _, _, problem = store.AcceptNativeCall(changed, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("source index accepted effect reinterpretation")
	}
	child := records.Request{ID: "req-collision", IdemKey: "collision", BodyDigest: childDigest("3"), Package: "local/operation", Entrypoint: "run", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5")}
	if _, _, problem = store.SubmitChild(child, 1, childDigest("1"), "private-boot", nil); problem == nil {
		t.Fatal("package call stole native index")
	}
	child.ParentCallIndex = 1
	_, _, problem = store.SubmitChild(child, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	call.ID = "source-collision"
	call.CallIndex = 1
	if _, _, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("native call stole package index")
	}
	if store.StartNativeCall(first.ID) == nil {
		t.Fatal("execution started before frozen intent")
	}
	frozen := []byte(`{"checkpoint":"exact"}`)
	fatal(t, store.FreezeNativeCall(first.ID, frozen))
	fatal(t, store.StartNativeCall(first.ID))
	fatal(t, store.StartNativeCall(first.ID))
	if store.FreezeNativeCall(first.ID, []byte(`{"checkpoint":"changed"}`)) == nil {
		t.Fatal("executing intent changed")
	}
	result, receipt := []byte(`{"result":"done"}`), []byte(`{"native":"receipt"}`)
	fatal(t, store.CompleteNativeCallAt(first.ID, result, receipt, "private-worker", "private-boot"))
	fatal(t, store.StopNativeCall(first.ID, "canceled", "late_cancel"))
	observed, problem := store.NativeCall(parent.ID, 0)
	fatal(t, problem)
	if observed.State != "succeeded" || !bytes.Equal(observed.Result, result) || observed.InstanceID != "private-worker" || observed.WorkerBootID != "private-boot" {
		t.Fatalf("native terminal changed: %+v", observed)
	}
	if store.CompleteNativeCall(first.ID, []byte(`{"result":"other"}`), receipt) == nil {
		t.Fatal("contradictory completion accepted")
	}
}
