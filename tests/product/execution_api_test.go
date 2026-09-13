package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestExecutionCaptureAPIReconcilesAndScopesRoots(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	authority, grant := privateAdmissionOwner(t)
	owner, problem := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store, PrivateExecution: authority})
	fatal(t, problem)
	defer owner.Close(0)
	credential := secret.New("execution-api-test")
	imports := 0
	server := api.New(api.Options{Orchestrator: owner, Cfg: config.Config{Home: layout.Root}, Addr: "127.0.0.1:8818",
		Creds: api.Credentials{CLI: credential}, ExecutionGrantDigest: grant,
		ExecutionCapture: func(ctx context.Context, key string, raw []byte) (orchestrator.Submission, *exit.Error) {
			imports++
			digest, _ := canonical.Spell(canonical.Digest(raw))
			return orchestrator.Submission{RequestID: "job-00112233445566778899aabb", ExecutionGrantDigest: grant,
				IdemKey: key, BodyDigest: digest, Kind: "job", Package: "local/script", Entrypoint: "main", Payload: []byte(`{}`),
				Worker: "private-execution-worker", RequestedRental: "private-execution-worker", Rental: true, RetainWork: true}, nil
		}})
	// Decoder/file custody are exercised separately with the fixed Creator binary.
	// This callback isolates HTTP admission, lost-reply replay and generation scope.
	raw := []byte(`{"capture":"already validated"}`)
	row, fresh, problem := server.AdmitExecutionCapture(context.Background(), "first", raw)
	fatal(t, problem)
	if !fresh || row.ExecutionGrantDigest != grant || imports != 1 {
		t.Fatal("root was not recorded in its generation")
	}
	handler, problem := server.Handler()
	fatal(t, problem)
	call := func(method, path string, body []byte, key string) *httptest.ResponseRecorder {
		request := httptest.NewRequest(method, "http://127.0.0.1:8818"+path, bytes.NewReader(body))
		request.RemoteAddr = "127.0.0.1:1234"
		api.Authorize(request, credential)
		if key != "" {
			request.Header.Set("Idempotency-Key", key)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	replay := call(http.MethodPost, "/v1/local/execution-captures", raw, "first")
	if replay.Code != http.StatusOK || imports != 1 || !strings.Contains(replay.Body.String(), `"idempotent_replay":true`) {
		t.Fatalf("lost reply repeated import/admission: %d %s", replay.Code, replay.Body.String())
	}
	if response := call(http.MethodPost, "/v1/local/execution-captures", []byte(`{"capture":"changed"}`), "first"); response.Code != http.StatusConflict || imports != 1 {
		t.Fatal("same key accepted a changed capture")
	}
	scope := call(http.MethodGet, "/v1/local/execution-roots/"+row.ID, nil, "")
	if scope.Code != http.StatusOK || !strings.Contains(scope.Body.String(), `"execution_grant_digest":"`+grant+`"`) {
		t.Fatal("scoped root receipt is absent")
	}
	_, _, problem = store.Submit(records.Request{ID: "other-root", IdemKey: "other", BodyDigest: childDigest("d"), Package: "local/other", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), State: "succeeded"})
	fatal(t, problem)
	if response := call(http.MethodGet, "/v1/local/execution-roots/other-root", nil, ""); response.Code != http.StatusNotFound {
		t.Fatal("untagged root crossed generation scope")
	}
	busy := func() bool {
		response := call(http.MethodGet, "/v1/local/execution-activity", nil, "")
		var value struct {
			Busy bool `json:"busy"`
		}
		if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &value) != nil {
			t.Fatal("activity unavailable")
		}
		return value.Busy
	}
	if !busy() {
		t.Fatal("pending root was idle")
	}
	fatal(t, store.SettleRequest(row.ID, "paused"))
	if !busy() {
		t.Fatal("paused retained root was idle")
	}
	fatal(t, store.SettleRequest(row.ID, "succeeded"))
	fatal(t, store.AppendEvent(row.ID, "request.completed", 0, map[string]any{"result": nil}))
	fatal(t, store.SettleRequest("other-root", "succeeded"))
	if !busy() {
		t.Fatal("uncollected scalar outcome allowed idle release")
	}
	collection, problem := store.ExecutionCollection(row.ID, grant)
	fatal(t, problem)
	if collection == nil || collection.Collected {
		t.Fatal("status lookup acknowledged collection")
	}
	if _, problem := store.CollectExecution(row.ID, childDigest("f"), collection.TerminalEventID, 0); problem == nil {
		t.Fatal("foreign grant collected a root")
	}
	if _, problem := store.CollectExecution(row.ID, grant, collection.TerminalEventID, 1); problem == nil {
		t.Fatal("wrong attempt collected a root")
	}
	body, err := json.Marshal(map[string]any{"terminal_event_id": collection.TerminalEventID, "attempt": 0})
	must(t, err)
	collected := call(http.MethodPost, "/v1/local/execution-roots/"+row.ID+"/collect", body, "")
	if collected.Code != http.StatusOK {
		t.Fatalf("explicit collection failed: %d %s", collected.Code, collected.Body.String())
	}
	firstReceipt, problem := store.ExecutionCollection(row.ID, grant)
	fatal(t, problem)
	if firstReceipt == nil || !firstReceipt.Collected {
		t.Fatal("explicit collection was not durable")
	}
	collected = call(http.MethodPost, "/v1/local/execution-roots/"+row.ID+"/collect", body, "")
	if collected.Code != http.StatusOK {
		t.Fatal("collection replay failed")
	}
	replayed, problem := store.ExecutionCollection(row.ID, grant)
	fatal(t, problem)
	if replayed.CollectionEventID != firstReceipt.CollectionEventID {
		t.Fatal("collection replay appended another receipt")
	}
	reopened, problem := records.Open(layout.DB)
	fatal(t, problem)
	persisted, problem := reopened.ExecutionCollection(row.ID, grant)
	fatal(t, problem)
	reopened.Close()
	if persisted == nil || !persisted.Collected || persisted.CollectionEventID != firstReceipt.CollectionEventID {
		t.Fatal("collection receipt did not survive reopening records")
	}
	if busy() {
		t.Fatal("collected scalar roots kept an idle coordinator busy")
	}
	fatal(t, store.SettleRequest(row.ID, "blocked"))
	fatal(t, store.AppendEvent(row.ID, "request.blocked", 0, map[string]any{"error": "retained work"}))
	blocked, problem := store.ExecutionCollection(row.ID, grant)
	fatal(t, problem)
	if blocked == nil || blocked.Collected {
		t.Fatal("old collection covered a changed outcome")
	}
	_, problem = store.CollectExecution(row.ID, grant, blocked.TerminalEventID, 0)
	fatal(t, problem)
	if !busy() {
		t.Fatal("collection released blocked retained work")
	}
}
