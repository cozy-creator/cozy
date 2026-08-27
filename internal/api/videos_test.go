package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	"github.com/cozy-creator/cozy-creator-v2/internal/video"
)

type fakeVideoComposer struct {
	request video.ComposeRequest
}

func (f *fakeVideoComposer) Compose(request video.ComposeRequest) (video.Composition, *exit.Error) {
	f.request = request
	return video.Composition{SourceDigest: workflowTestDigest,
		CreativePlanDigest: workflowTestDigest, ShotCount: 2}, nil
}

func TestVideoCompositionRouteIsStrictAndCLIOnly(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	owner, problem := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store})
	if problem != nil {
		t.Fatal(problem)
	}
	composer := &fakeVideoComposer{}
	credential := secret.New("video-api-token")
	browser := secret.New("video-browser-token")
	server := New(Options{Orchestrator: owner, Videos: composer,
		Creds: Credentials{CLI: credential, Browser: browser}, Addr: "127.0.0.1:2699"})
	handler, problem := server.Handler()
	if problem != nil {
		t.Fatal(problem)
	}
	body, _ := json.Marshal(VideoComposeRequest{Source: []byte(validAPIVideoSource),
		BaseDir: "/private/project", H3Endpoint: "cozy/h3",
		AssemblyEndpoint: "cozy/assembly"})
	call := func(token *secret.Value, body []byte) *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodPost, "/v1/local/video-compositions",
			bytes.NewReader(body))
		request.RemoteAddr, request.Host = "127.0.0.1:12345", "127.0.0.1:2699"
		if token != nil {
			Authorize(request, *token)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	if response := call(nil, body); response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized=%d %s", response.Code, response.Body.String())
	}
	if response := call(&browser, body); response.Code != http.StatusForbidden {
		t.Fatalf("browser=%d %s", response.Code, response.Body.String())
	}
	if response := call(&credential, append(body[:len(body)-1], []byte(`,"unknown":true}`)...)); response.Code != http.StatusBadRequest {
		t.Fatalf("unknown=%d %s", response.Code, response.Body.String())
	}
	response := call(&credential, body)
	if response.Code != http.StatusOK || string(composer.request.Source) != validAPIVideoSource ||
		composer.request.BaseDir != "/private/project" {
		t.Fatalf("response=%d %s request=%#v", response.Code, response.Body.String(), composer.request)
	}
}

const validAPIVideoSource = "format: cozy.video/1\nshots: []\nassembly: {}\n"
