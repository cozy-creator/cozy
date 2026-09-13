package producttest

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestServingStatusExposesItsRecordedParent(t *testing.T) {
	owner := hostOwner(t, "serving-ancestry")
	fatal(t, owner.store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/script", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, owner.store, recordPrivateTransaction(t, owner.store, "ancestry-parent", ""))
	child := records.Request{ID: "req-serving-child", IdemKey: "serving-child", Kind: "serving", Package: "local/renderer", Entrypoint: "generate",
		Payload: []byte(`{}`), BodyDigest: childDigest("1"), ParentRequestID: parent.ID,
		ParentCallIndex: 0, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3")}
	_, _, problem := owner.store.SubmitChild(child, 1, childDigest("1"), "private-boot", []byte(`{"models":{},"payload":{}}`))
	fatal(t, problem)
	credential := secret.New("serving-ancestry-credential")
	server := api.New(api.Options{Orchestrator: owner.c, Cfg: config.Config{Home: owner.root}, Addr: "127.0.0.1:8818", Creds: api.Credentials{CLI: credential}})
	handler, problem := server.Handler()
	fatal(t, problem)
	request := httptest.NewRequest(http.MethodGet, "http://127.0.0.1:8818/v1/requests/"+child.ID, nil)
	request.RemoteAddr = "127.0.0.1:1234"
	api.Authorize(request, credential)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"parent_request_id":"`+parent.ID+`"`) ||
		!strings.Contains(response.Body.String(), `"parent_call_index":0`) {
		t.Fatalf("serving status omitted immutable ancestry: %d %s", response.Code, response.Body.String())
	}
}
