package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
	workflowpkg "github.com/cozy-creator/cozy-creator-v2/internal/workflow"
)

const workflowTestDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

type workflowResolver struct{ placement orchestrator.DesiredPlacement }

func (r workflowResolver) ResolvePlacement(endpoint string) (orchestrator.DesiredPlacement, *exit.Error) {
	if endpoint != r.placement.Endpoint {
		return orchestrator.DesiredPlacement{}, exit.New(exit.NotFound, "unknown endpoint")
	}
	return r.placement, nil
}
func (r workflowResolver) Resolve(endpoint string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	placement, problem := r.ResolvePlacement(endpoint)
	return orchestrator.WorkerLaunchSpec{Placement: placement}, problem
}
func (r workflowResolver) ResolveInstall(id string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	if id != r.placement.InstallID {
		return orchestrator.WorkerLaunchSpec{}, exit.New(exit.NotFound, "unknown install")
	}
	return orchestrator.WorkerLaunchSpec{Placement: r.placement}, nil
}
func (r workflowResolver) Entrypoint(id, name string) (*launch.Entrypoint, *exit.Error) {
	if id != r.placement.InstallID || name != "run" {
		return nil, exit.New(exit.NotFound, "unknown entrypoint")
	}
	asset := launch.Field{Name: "first_frame", Type: []byte(`{"asset":"video"}`), Wire: "optional"}
	asset.AssetBound.MaxBytes = 128 << 20
	return &launch.Entrypoint{Name: "run", Request: launch.Struct{Fields: []launch.Field{
		{Name: "prompt", Type: []byte(`"str"`), Wire: "optional"}, asset,
	}}}, nil
}
func (r workflowResolver) Jobs(string) ([]launch.JobFacts, *exit.Error) { return nil, nil }
func (r workflowResolver) List() []string                               { return []string{r.placement.Endpoint} }

func workflowAPI(t *testing.T) (http.Handler, *workflowpkg.Engine, secret.Value) {
	t.Helper()
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(store.Close)
	_, problem = store.Activate(records.EndpointInstall{
		ID: "install-api", Endpoint: "org/ep", Major: 1, Version: "1.0.0",
		SourceKind: "dir", SourceRef: ".", SourceDigest: workflowTestDigest,
		Dir: ".", Python: "python", UV: "uv", LockDigest: workflowTestDigest,
		Platform: "test", LinkMode: "copy", Closure: "none", Descriptor: workflowTestDigest,
	})
	if problem != nil {
		t.Fatal(problem)
	}
	binding := &orchestrator.Binding{Entrypoint: "run", Outputs: []string{"video"},
		RuntimePlan: &orchestrator.BindingPlanSubject{SubjectID: workflowTestDigest,
			Digest: workflowTestDigest, Kind: "plan", Length: 1}}
	resolver := workflowResolver{placement: orchestrator.DesiredPlacement{
		Endpoint: "org/ep", ReleaseID: "org/ep@1.0.0", InstallID: "install-api",
		DescriptorDigest: workflowTestDigest, Bindings: []*orchestrator.Binding{binding}}}
	owner, problem := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store})
	if problem != nil {
		t.Fatal(problem)
	}
	engine, problem := workflowpkg.Open(workflowpkg.Options{
		Store: store, Owner: owner, Resolver: resolver, Layout: layout,
	})
	if problem != nil {
		t.Fatal(problem)
	}
	credential := secret.New("workflow-api-token")
	server := New(Options{Orchestrator: owner, Endpoints: resolver, Workflows: engine,
		Creds: Credentials{CLI: credential}, Addr: "127.0.0.1:2699"})
	handler, problem := server.Handler()
	if problem != nil {
		t.Fatal(problem)
	}
	return handler, engine, credential
}

func workflowPlan(prompt string) json.RawMessage {
	payload, _ := json.Marshal(map[string]string{"prompt": prompt})
	plan := map[string]any{
		"format": "cozy.workflow.Plan/1", "creative_plan_digest": workflowTestDigest,
		"steps": []any{map[string]any{
			"endpoint": "org/ep", "endpoint_release_id": "org/ep@1.0.0",
			"entrypoint": "run", "entrypoint_binding_plan_id": workflowTestDigest,
			"payload_b64": base64.StdEncoding.EncodeToString(payload),
			"outputs":     []string{"video"},
		}},
	}
	data, _ := json.Marshal(plan)
	return data
}

func TestWorkflowRoutesIdempotencyAndCancellation(t *testing.T) {
	handler, engine, credential := workflowAPI(t)
	call := func(method, path, key string, body any) *httptest.ResponseRecorder {
		t.Helper()
		data, _ := json.Marshal(body)
		req := httptest.NewRequest(method, path, bytes.NewReader(data))
		req.RemoteAddr, req.Host = "127.0.0.1:12345", "127.0.0.1:2699"
		if key != "" {
			req.Header.Set("Idempotency-Key", key)
		}
		Authorize(req, credential)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		return response
	}
	submission := WorkflowSubmission{Plan: workflowPlan("same")}
	if missing := call(http.MethodPost, "/v1/local/workflows", "", submission); missing.Code != http.StatusBadRequest {
		t.Fatalf("missing key = %d %s", missing.Code, missing.Body.String())
	}
	fresh := call(http.MethodPost, "/v1/local/workflows", "workflow-key", submission)
	if fresh.Code != http.StatusAccepted {
		t.Fatalf("fresh = %d %s", fresh.Code, fresh.Body.String())
	}
	var handle WorkflowHandle
	if err := json.Unmarshal(fresh.Body.Bytes(), &handle); err != nil || handle.WorkflowID == "" {
		t.Fatalf("handle = %#v err=%v", handle, err)
	}
	replay := call(http.MethodPost, "/v1/local/workflows", "workflow-key", submission)
	if replay.Code != http.StatusOK {
		t.Fatalf("replay = %d %s", replay.Code, replay.Body.String())
	}
	changed := call(http.MethodPost, "/v1/local/workflows", "workflow-key",
		WorkflowSubmission{Plan: workflowPlan("changed")})
	if changed.Code != http.StatusConflict {
		t.Fatalf("changed = %d %s", changed.Code, changed.Body.String())
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	cancel := call(http.MethodPost, "/v1/local/workflows/"+handle.WorkflowID+"/cancel", "", nil)
	if cancel.Code != http.StatusAccepted {
		t.Fatalf("cancel = %d %s", cancel.Code, cancel.Body.String())
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	status := call(http.MethodGet, "/v1/local/workflows/"+handle.WorkflowID, "", nil)
	if status.Code != http.StatusOK {
		t.Fatalf("status = %d %s", status.Code, status.Body.String())
	}
	var state WorkflowState
	if err := json.Unmarshal(status.Body.Bytes(), &state); err != nil || state.Status != "canceled" ||
		len(state.Steps) != 1 || state.Steps[0].ChildRequest == "" {
		t.Fatalf("state = %#v err=%v", state, err)
	}
}
