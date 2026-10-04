package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestApplicationOperationKeysPreserveLargeSeedTypeAndOrder(t *testing.T) {
	payload := `{"number":1.0,"ordered":[2,1],"seed":18446744073709551615}`
	request := records.Request{ChildTargetDigest: childDigest("a"), Payload: []byte(payload)}
	first, problem := records.OperationKey(request)
	fatal(t, problem)
	request.Payload = []byte(`{ "seed":18446744073709551615, "number":1e0, "ordered":[2, 1] }`)
	reattached, problem := records.OperationKey(request)
	fatal(t, problem)
	if first != reattached {
		t.Fatal("formatting changed application computation")
	}
	for _, change := range []string{
		strings.Replace(payload, "18446744073709551615", "18446744073709551614", 1),
		strings.Replace(payload, "1.0", "1", 1),
		strings.Replace(payload, "[2,1]", "[1,2]", 1),
	} {
		request.Payload = []byte(change)
		changed, problem := records.OperationKey(request)
		fatal(t, problem)
		if first == changed {
			t.Fatalf("different authored computation reused the operation: %s", change)
		}
	}
}

func TestNativeApplicationFactsRetainLargeNumbersThroughDurableCompletion(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "application-native", ""))
	body := []byte(`{"number":1.0,"seed":18446744073709551615}`)
	call := records.NativeCall{ID: "application-native", ParentRequestID: parent.ID, Kind: "source", Operation: "source_files", IntentDigest: childDigest("4"), Request: body}
	_, _, problem = store.AcceptNativeCall(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	fatal(t, store.FreezeNativeCall(call.ID, body))
	fatal(t, store.StartNativeCall(call.ID))
	fatal(t, store.CompleteNativeCallAt(call.ID, body, []byte("native receipt"), "private-worker", "private-boot"))
	completed, problem := store.NativeCall(parent.ID, 0)
	fatal(t, problem)
	if completed.State != "succeeded" || !bytes.Equal(completed.Request, body) || !bytes.Equal(completed.Frozen, body) || !bytes.Equal(completed.Result, body) {
		t.Fatalf("durable application facts changed: %+v", completed)
	}
	if problem := store.CompleteNativeCallAt(call.ID, bytes.Replace(body, []byte("1.0"), []byte("1"), 1), []byte("native receipt"), "private-worker", "private-boot"); problem == nil {
		t.Fatal("terminal application number kind was rewritten")
	}
}
