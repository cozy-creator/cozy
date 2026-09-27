package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/secret"
)

// The API receives these already-resolved package facts. Neither a CLI visibility
// filter nor a worker refusal can substitute for admission before queue insertion.
type internalAPIResolver struct {
	api.Resolver
	pkg      string
	resolved int
}

func (r *internalAPIResolver) entry(name, kind string) *launch.Entrypoint {
	r.resolved++
	return &launch.Entrypoint{Name: name, Kind: kind, Internal: true,
		Request: launch.Struct{Fields: []launch.Field{}}, Result: launch.Struct{Fields: []launch.Field{}}}
}

func (r *internalAPIResolver) RefreshEditable(string) (string, bool, bool, *exit.Error) {
	return "internal-api-install", false, false, nil
}

func (r *internalAPIResolver) ResolveInstall(id string, _ []orchestrator.ModelRef) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	return orchestrator.WorkerLaunchSpec{Placement: orchestrator.DesiredPlacement{Package: r.pkg,
		Release: "1.0.0", InstallID: id, Entrypoints: []orchestrator.Entrypoint{{Name: "segment", Digest: childDigest("9")}}}}, nil
}

func (r *internalAPIResolver) Entrypoint(_ string, name string) (*launch.Entrypoint, bool, *exit.Error) {
	return r.entry(name, "entrypoint"), false, nil
}

func (r *internalAPIResolver) ResolveRemoteRelease(hub, pkg, release, name string, _ []orchestrator.ModelRef) (orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error) {
	return orchestrator.LogicalPackage{Package: pkg, Release: release, Function: name, PlanID: childDigest("9")}, r.entry(name, "entrypoint"), nil
}

func (r *internalAPIResolver) ResolveRemoteJob(hub, pkg, release, name string, _ []orchestrator.ModelRef, _ bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	return orchestrator.LogicalJob{Package: pkg, Release: release, Function: name, DescriptorID: childDigest("9")}, r.entry(name, "job"), nil
}

func (r *internalAPIResolver) JobsInstall(string) ([]launch.JobFacts, *exit.Error) {
	entry := r.entry("segment", "job")
	return []launch.JobFacts{{Name: entry.Name, Internal: entry.Internal, DescriptorID: childDigest("9"),
		Request: entry.Request, Result: entry.Result}}, nil
}

func TestInternalCallablesRefuseDirectHTTPBeforeRecording(t *testing.T) {
	for _, arm := range []struct {
		name, path, pkg string
		rental          bool
	}{
		{"remote-serving", "/v1/requests", "alice/internal", true},
		{"remote-job", "/v1/local/jobs", "alice/internal", true},
		{"local-serving", "/v1/requests", "alice/internal", false},
		{"local-job", "/v1/local/jobs", "alice/internal", false},
		{"private-job", "/v1/local/jobs", "local/internal", true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			o := hostOwner(t, "internal-api-"+arm.name)
			resolver := &internalAPIResolver{pkg: arm.pkg}
			const bearer = "internal-api-fixture"
			handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg,
				Creds: api.Credentials{CLI: secret.New(bearer)}, Addr: "127.0.0.1:11111",
				Web: http.NotFoundHandler(), Packages: resolver, MachineExecutions: publishedRouteObserver{}}).Handler()
			fatal(t, problem)
			body := map[string]any{"package": arm.pkg, "release": "1.0.0", "function": "segment",
				"input": map[string]any{}, "rental": arm.rental}
			if !arm.rental || strings.HasPrefix(arm.pkg, "local/") {
				body["install_id"] = "internal-api-install"
			}
			raw, err := json.Marshal(body)
			must(t, err)
			request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111"+arm.path, bytes.NewReader(raw))
			request.RemoteAddr = "127.0.0.1:12345"
			request.Header.Set("Authorization", "Bearer "+bearer)
			request.Header.Set("Content-Type", "application/json")
			request.Header.Set("Idempotency-Key", arm.name)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusNotFound || !strings.Contains(response.Body.String(), `"code":"callable_internal"`) {
				t.Fatalf("internal root admission = %d %s", response.Code, response.Body.String())
			}
			if resolver.resolved != 1 {
				t.Fatalf("refusal did not use exactly one resolved callable: %d", resolver.resolved)
			}
			rows, problem := o.store.Requests("", "", 10)
			fatal(t, problem)
			if len(rows) != 0 {
				t.Fatalf("internal root reached the durable queue: %+v", rows)
			}
		})
	}
}
