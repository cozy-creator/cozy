package producttest

import (
	"database/sql"
	"fmt"
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
	state, problem := store.RequestPause(first.ID, "parent interrupted")
	fatal(t, problem)
	if state != "pausing" {
		t.Fatal("serving child did not retain its pause intent")
	}
	paused, problem := store.CompleteRequestPause(first.ID)
	fatal(t, problem)
	if !paused {
		t.Fatal("undispatched serving child did not become paused")
	}
	resumed, problem := store.ResumeRequest(first.ID, "parent resumed")
	fatal(t, problem)
	if !resumed {
		t.Fatal("serving child was not queued for fresh inference")
	}
	call.ID, call.IdemKey, call.ParentCallIndex = "req-serving-repeat", "serving-repeat", 1
	repeat, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments)
	fatal(t, problem)
	if !fresh || repeat.ID == first.ID || repeat.ReusedFrom != "" {
		t.Fatal("repeat reused reference inference")
	}
	call.ID, call.IdemKey, call.ParentCallIndex, call.ChildReusable = "req-serving-bad", "serving-bad", 1, true
	if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments); problem == nil {
		t.Fatal("accepted inference was changed into a memoized measurement")
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
	for _, version := range []int{33, 34} {
		t.Run(fmt.Sprint(version), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "creator.sqlite")
			store, problem := records.Open(path)
			fatal(t, problem)
			prior := recordPrivateTransaction(t, store, "schema33-parent", "")
			store.Close()
			db, err := sql.Open("sqlite", path)
			must(t, err)
			restorePriorCallIndexBounds(t, db)
			_, err = db.Exec(`DROP TABLE attempt_serving_placements`)
			must(t, err)
			_, err = db.Exec(`DROP TABLE request_child_arguments`)
			must(t, err)
			_, err = db.Exec(`DROP TABLE successful_work_releases`)
			must(t, err)
			_, err = db.Exec(fmt.Sprintf(`PRAGMA user_version=%d`, version))
			must(t, err)
			db.Close()
			store, problem = records.OpenForDaemon(path, "")
			fatal(t, problem)
			defer store.Close()
			after, problem := store.RequestRow(prior.ID)
			fatal(t, problem)
			if after == nil || after.BodyDigest != prior.BodyDigest || string(after.Payload) != string(prior.Payload) {
				t.Fatal("schema35 changed its existing parent")
			}
		})
	}
}

// Null is the captured caller asking Creator to resolve the callee's default.
// Only native artifact arguments acquire output retention obligations.
func TestServingChildDownloadedDefaultPreservesArguments(t *testing.T) {
	for _, checkpoint := range []bool{false, true} {
		t.Run(fmt.Sprintf("checkpoint=%t", checkpoint), func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
			parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "default-parent", ""))
			model := records.ModelRef{Package: "local/renderer", Slot: "model", BindingPath: "generate.models.model", Model: "proof/base", Manifest: childDigest("4"), ManifestLength: 161, Release: "1.0.0", Lane: "bf16"}
			if checkpoint {
				model.Release, model.Lane, model.HubCheckpoint = "", "", true
			}
			call := records.Request{ID: "default-child", IdemKey: "default-child", Kind: "serving", Package: "local/renderer", Entrypoint: "generate", Payload: []byte(`{"prompt":"same","seed":1234}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3"), Models: []records.ModelRef{model}}
			arguments := []byte(`{"models":{"model":null},"payload":{"prompt":"same","seed":1234}}`)
			accepted, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments)
			fatal(t, problem)
			if !fresh || len(accepted.Models) != 1 || accepted.Models[0].Manifest != model.Manifest {
				t.Fatal("downloaded default changed during admission")
			}
			retained, problem := store.WeightsRetentions(accepted.ID)
			fatal(t, problem)
			if len(retained) != 0 {
				t.Fatal("Hub default invented native custody")
			}
			actual, problem := store.ChildArguments(accepted)
			fatal(t, problem)
			if string(actual) != string(arguments) {
				t.Fatal("model selection changed caller arguments")
			}

			call.ID, call.IdemKey, call.ParentCallIndex = "wrong-child", "wrong-child", 1
			wrong := []byte(`{"models":{"other":null},"payload":{"prompt":"same","seed":1234}}`)
			if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", wrong); problem == nil {
				t.Fatal("unknown slot silently selected the declared default")
			}
			call.Models[0].Release, call.Models[0].Lane, call.Models[0].HubCheckpoint = "", "", false
			if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", arguments); problem == nil {
				t.Fatal("native input without custody was admitted as a default")
			}
		})
	}
}
