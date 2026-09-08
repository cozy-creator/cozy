package producttest

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func effectCancelFixture(t *testing.T) (*records.Store, records.NativeCall, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "effect-cancel", ""))
	call, _, problem := store.AcceptNativeCall(records.NativeCall{ID: "effect-cancel", ParentRequestID: parent.ID, Kind: "effect", Operation: "publish_release", IntentDigest: childDigest("4"), Request: []byte(`{}`)}, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	return store, call, path
}

func cancelEffect(t *testing.T, store *records.Store, call records.NativeCall) *records.NativeCall {
	t.Helper()
	row, problem := store.RequestNativeEffectCancel(call.ParentRequestID, call.CallIndex, 1, childDigest("1"), "private-boot", call.IntentDigest)
	fatal(t, problem)
	if row == nil {
		t.Fatal("accepted effect disappeared")
	}
	parent, problem := store.RequestRow(call.ParentRequestID)
	fatal(t, problem)
	if parent.State != "dispatching" {
		t.Fatal("canceling one await stopped the parent")
	}
	return row
}

func TestAwaitedEffectCancellationFencesWritesAndPreservesPausedIntent(t *testing.T) {
	for _, phase := range []string{"accepted", "frozen", "executing", "paused"} {
		t.Run(phase, func(t *testing.T) {
			store, call, _ := effectCancelFixture(t)
			frozen := []byte(`{"baseline":1}`)
			if phase != "accepted" {
				fatal(t, store.FreezeNativeCall(call.ID, frozen))
			}
			if phase == "executing" {
				fatal(t, store.StartNativeEffectWrite(call.ID))
			}
			for _, bad := range []struct {
				attempt               int64
				spec, session, intent string
			}{
				{2, childDigest("1"), "private-boot", call.IntentDigest},
				{1, childDigest("2"), "private-boot", call.IntentDigest},
				{1, childDigest("1"), "another-boot", call.IntentDigest},
				{1, childDigest("1"), "private-boot", childDigest("5")},
				{1, strings.ToUpper(childDigest("1")), "private-boot", call.IntentDigest},
			} {
				if _, problem := store.RequestNativeEffectCancel(call.ParentRequestID, call.CallIndex, bad.attempt, bad.spec, bad.session, bad.intent); problem == nil {
					t.Fatal("unowned cancellation accepted")
				}
			}
			if phase == "paused" {
				_, problem := store.RequestPause(call.ParentRequestID, "preserve work")
				fatal(t, problem)
				row, problem := store.RequestNativeEffectCancel(call.ParentRequestID, 0, 1, childDigest("1"), "private-boot", call.IntentDigest)
				fatal(t, problem)
				if row.CancelRequested || row.State != "frozen" {
					t.Fatalf("pause canceled the effect: %+v", row)
				}
				if problem = store.StartNativeEffectWrite(call.ID); problem == nil || problem.Code != exit.Unavailable {
					t.Fatal("paused parent issued write", problem)
				}
				return
			}
			row := cancelEffect(t, store, call)
			want := "canceled"
			if phase == "executing" {
				want = "executing"
			}
			if !row.CancelRequested || row.State != want || (phase != "accepted" && !bytes.Equal(row.Frozen, frozen)) {
				t.Fatalf("cancellation erased frozen history: %+v", row)
			}
			if problem := store.StartNativeEffectWrite(call.ID); problem == nil || problem.Code != exit.Canceled {
				t.Fatal("canceled await issued write", problem)
			}
		})
	}
}

func TestCanceledEffectReconcilesLostReleaseReplyWithoutAnotherCAS(t *testing.T) {
	for _, outcome := range []string{"committed", "superseded", "not_committed"} {
		t.Run(outcome, func(t *testing.T) {
			store, call, _ := effectCancelFixture(t)
			revision, writes := int64(1), 0
			lanes := map[string]string{"bf16": childDigest("1")}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if r.Method == http.MethodGet {
					entries := []map[string]string{}
					for name, id := range lanes {
						entries = append(entries, map[string]string{"lane": name, "manifest_id": id})
					}
					json.NewEncoder(w).Encode(map[string]any{"org": "alice", "name": "model", "releases": []any{map[string]any{"release": "v1", "revision": revision, "lanes": entries}}})
					return
				}
				writes++
				var request struct {
					Expected int64             `json:"expected_revision"`
					Set      map[string]string `json:"set_lanes"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				if request.Expected != 1 {
					t.Error("CAS lost original revision")
				}
				if outcome != "not_committed" {
					revision = 2
					lanes["fp8"] = request.Set["fp8"]
				}
				if outcome == "superseded" {
					revision = 3
					lanes["bf16"] = childDigest("3")
				}
				row, problem := store.RequestNativeEffectCancel(call.ParentRequestID, 0, 1, childDigest("1"), "private-boot", call.IntentDigest)
				if problem != nil || row == nil || !row.CancelRequested || row.State != "executing" {
					t.Error("in-flight cancel lost uncertainty", problem)
				}
				conn, _, err := w.(http.Hijacker).Hijack()
				if err != nil {
					t.Error(err)
					return
				}
				conn.Close()
			}))
			defer server.Close()
			client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("test")}, "effect-cancel-test")
			intent, problem := publication.PrepareRelease(context.Background(), client, publication.ReleaseRequest{Destination: "alice/model", Release: "v1", Lanes: map[string]string{"fp8": childDigest("2")}})
			fatal(t, problem)
			frozen, problem := publication.Canonical(intent)
			fatal(t, problem)
			fatal(t, store.FreezeNativeCall(call.ID, frozen))
			before := func() *exit.Error { return store.StartNativeEffectWrite(call.ID) }
			if _, problem = publication.ApplyRelease(context.Background(), client, intent, false, before); problem == nil {
				t.Fatal("lost reply was acknowledged")
			}
			observed, problem := store.NativeCall(call.ParentRequestID, 0)
			fatal(t, problem)
			if !observed.CancelRequested || observed.State != "executing" || !bytes.Equal(observed.Frozen, frozen) {
				t.Fatalf("lost reply erased cancellation/intent: %+v", observed)
			}
			receipt, problem := publication.ApplyRelease(context.Background(), client, intent, true, before)
			switch outcome {
			case "committed":
				fatal(t, problem)
				if receipt.Observation != "observed_convergence" || receipt.Revision != 2 || receipt.Lanes["bf16"] != childDigest("1") {
					t.Fatalf("incorrect late result: %+v", receipt)
				}
				result, problem := publication.Canonical(receipt)
				fatal(t, problem)
				fatal(t, store.CompleteNativeCall(call.ID, result, nil))
			case "superseded":
				if problem == nil || problem.ErrName() != "publication.outcome_unknown" {
					t.Fatal("ambiguous late state claimed rollback", problem)
				}
			case "not_committed":
				if problem == nil || problem.Code != exit.Canceled {
					t.Fatal("canceled unchanged baseline attempted another CAS", problem)
				}
			}
			if writes != 1 {
				t.Fatalf("cancellation allowed %d writes", writes)
			}
			parent, problem := store.RequestRow(call.ParentRequestID)
			fatal(t, problem)
			if parent.State != "dispatching" {
				t.Fatal("publication stopped the parent")
			}
		})
	}
}

func TestCanceledGrantReplyCannotStartNativePUTOrFinalize(t *testing.T) {
	store, call, _ := effectCancelFixture(t)
	manifest := []byte(`{"entries":[]}`)
	digest, _ := canonical.Spell(canonical.Digest(manifest))
	object := hub.Object{ID: digest, Length: int64(len(manifest))}
	intent := publication.UploadIntent{Request: publication.UploadRequest{Destination: "alice/model", Artifact: records.ModelArtifact{ProducerRequestID: "source", OutputSlot: "model", Manifest: records.ArtifactObjectRef{Digest: digest, Length: object.Length}, TensorFSReceiptDigest: childDigest("1")}}, Objects: []hub.Object{object}}
	frozen, problem := publication.Canonical(intent)
	fatal(t, problem)
	fatal(t, store.FreezeNativeCall(call.ID, frozen))
	puts, finalizes, grants := 0, 0, 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodGet:
			http.Error(w, `{"error":{"code":"not_found","message":"absent"}}`, 404)
		case r.Method == http.MethodPut:
			json.NewEncoder(w).Encode(hub.OpenPublicationResponse{Publication: hub.Session{Operation: call.ID, State: "open", Objects: []hub.Transfer{{ObjectID: digest, Length: object.Length, State: "claimed"}}}})
		case strings.HasSuffix(r.URL.Path, "/grants"):
			grants++
			_, problem := store.RequestNativeEffectCancel(call.ParentRequestID, 0, 1, childDigest("1"), "private-boot", call.IntentDigest)
			if problem != nil {
				t.Error(problem)
			}
			json.NewEncoder(w).Encode(hub.GrantResponse{ServerTimeUnix: 100, Grants: []hub.Grant{{ObjectID: digest, Length: object.Length, URL: "http://object.invalid/object", ExpiresAtUnix: 200}}})
		case strings.HasSuffix(r.URL.Path, "/finalize"):
			finalizes++
		default:
			t.Error("unexpected request", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("test")}, "effect-cancel-test")
	_, problem = publication.UploadCheckpoint(context.Background(), client, call.ID, intent, func() *exit.Error { return store.StartNativeEffectWrite(call.ID) }, func(context.Context, hub.Grant, int64) *exit.Error { puts++; return nil })
	if problem == nil || problem.Code != exit.Canceled || grants != 1 || puts != 0 || finalizes != 0 {
		t.Fatalf("canceled grant permitted writes: problem%v grants%d puts%d finalizes%d", problem, grants, puts, finalizes)
	}
}

func TestNativeEffectCancellationMigrationPreservesExecutingIntent(t *testing.T) {
	store, call, path := effectCancelFixture(t)
	frozen := []byte(`{"baseline":1}`)
	fatal(t, store.FreezeNativeCall(call.ID, frozen))
	fatal(t, store.StartNativeEffectWrite(call.ID))
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	// Rebuild the exact previous schema, retaining its actual executing row.
	var ddl string
	must(t, db.QueryRow(`SELECT sql FROM sqlite_master WHERE name='native_calls'`).Scan(&ddl))
	_, err = db.Exec(`ALTER TABLE native_calls RENAME TO prior32`)
	must(t, err)
	ddl = strings.Replace(ddl, " cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),\n", "", 1)
	_, err = db.Exec(ddl)
	must(t, err)
	const columns = "id,parent_request_id,call_index,kind,operation,intent_digest,request,frozen,state,result,native_receipt,safe_code,worker,instance_id,worker_boot_id,parent_attempt"
	_, err = db.Exec(`INSERT INTO native_calls(` + columns + `) SELECT ` + columns + ` FROM prior32`)
	must(t, err)
	_, err = db.Exec(`DROP TABLE prior32`)
	must(t, err)
	revertRentalsBeforeWidth(t, db)
	for _, statement := range []string{`DROP TABLE attempt_serving_placements`, `DROP TABLE request_child_arguments`, `DROP TABLE byte_outputs`, `ALTER TABLE requests DROP COLUMN capture`, `ALTER TABLE native_artifact_retentions DROP COLUMN artifact_kind`, `ALTER TABLE native_artifact_retentions DROP COLUMN producer_attempt`, `ALTER TABLE native_artifact_retentions DROP COLUMN producer_output_id`, `ALTER TABLE native_artifact_retentions DROP COLUMN content_bytes`} {
		_, err = db.Exec(statement)
		must(t, err)
	}
	_, err = db.Exec(`PRAGMA user_version=31`)
	must(t, err)
	must(t, db.Close())
	migrated, problem := records.OpenForDaemon(path, filepath.Join(t.TempDir(), "triage"))
	fatal(t, problem)
	defer migrated.Close()
	row, problem := migrated.NativeCall(call.ParentRequestID, 0)
	fatal(t, problem)
	if row.State != "executing" || row.CancelRequested || !bytes.Equal(row.Frozen, frozen) || row.IntentDigest != call.IntentDigest {
		t.Fatalf("migration changed accepted history: %+v", row)
	}
	cancelEffect(t, migrated, call)
}
