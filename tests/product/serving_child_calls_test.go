package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestServingChildRetainsOriginalCallWithoutMemoizingInference(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "serving-parent", ""))
	call := records.Request{ID: "req-serving-one", IdemKey: "serving-one", Kind: "serving", Package: "local/renderer", Entrypoint: "generate", Payload: []byte(`{"prompt":"same"}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3")}
	arguments := []byte(`{"models":{},"payload":{"prompt":"same"}}`)
	first, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments)
	fatal(t, problem)
	if !fresh || first.IsJob() || first.ChildReusable || first.ReusedFrom != "" || !first.RetainWork {
		t.Fatalf("not a retained, nonmemoized serving child: %+v", first)
	}
	replay, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments)
	fatal(t, problem)
	if fresh || replay.ID != first.ID {
		t.Fatal("accepted call-index replay duplicated inference")
	}
	call.ID, call.IdemKey, call.ParentCallIndex = "req-serving-repeat", "serving-repeat", 1
	repeat, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments)
	fatal(t, problem)
	if !fresh || repeat.ID == first.ID || repeat.ReusedFrom != "" {
		t.Fatal("repeat reused reference inference")
	}
	call.ID, call.IdemKey, call.ParentCallIndex, call.ChildReusable = "req-serving-bad", "serving-bad", 2, true
	if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments); problem == nil {
		t.Fatal("memoized serving admission was accepted")
	}
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestRow(first.ID)
	fatal(t, problem)
	actual, problem := store.ChildArguments(*row)
	fatal(t, problem)
	if string(actual) != string(arguments) || string(row.Payload) != `{"prompt":"same"}` {
		t.Fatal("restart lost call preimage or changed inference payload")
	}
}

func TestServingCallEnvelopeRefusesUnknownOrUncanonicalArguments(t *testing.T) {
	for _, body := range []string{`{"models":{}}`, `{"models":{},"payload":{},"extra":1}`, `{"models":null,"payload":{}}`, `{"models":{},"payload":null}`, `{"models":{}, "payload":{}}`, `{"models":{},"models":{},"payload":{}}`} {
		if _, _, problem := records.ServingCallArguments([]byte(body)); problem == nil {
			t.Fatalf("accepted %s", body)
		}
	}
}

func TestServingArgumentsSchemaUpgradePreservesPrivateParent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	prior := recordPrivateTransaction(t, store, "schema33-parent", "")
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`DROP TABLE request_child_arguments`)
	must(t, err)
	_, err = db.Exec(`PRAGMA user_version=33`)
	must(t, err)
	db.Close()
	store, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer store.Close()
	after, problem := store.RequestRow(prior.ID)
	fatal(t, problem)
	if after == nil || after.BodyDigest != prior.BodyDigest || string(after.Payload) != string(prior.Payload) {
		t.Fatal("schema34 changed its existing parent")
	}
}
