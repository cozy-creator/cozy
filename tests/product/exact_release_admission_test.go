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

type exactReleaseResolver struct {
	api.Resolver
	reads int
}

func (r *exactReleaseResolver) ResolveRemoteRelease(_, _, _, _ string, _ []orchestrator.ModelRef) (orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error) {
	r.reads++
	return orchestrator.LogicalPackage{}, nil, exit.Named(exit.Unavailable, "proof.source_lookup", "explicit release lookup observed")
}

func (r *exactReleaseResolver) ResolveRemoteJob(_, _, _, _ string, _ []orchestrator.ModelRef, _ bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error) {
	r.reads++
	return orchestrator.LogicalJob{}, nil, exit.Named(exit.Unavailable, "proof.source_lookup", "explicit release lookup observed")
}

// The API runs a published package as one exact release, which its client sends (#1157: the
// installed one, the newest kept under the catalog revision, or the one the machine names).
// A submission naming none is refused before any lookup and records nothing; one naming its
// release is resolved as that release.
func TestPublishedAdmissionRequiresAnExactRelease(t *testing.T) {
	for kind, path := range map[string]string{"entrypoints": "/v1/requests", "jobs": "/v1/local/jobs"} {
		t.Run(kind, func(t *testing.T) {
			o := hostOwner(t, "exact-release-"+kind)
			resolver := &exactReleaseResolver{}
			const bearer = "exact-release-fixture"
			handler, problem := api.New(api.Options{Orchestrator: o.c, Cfg: o.cfg,
				Creds: api.Credentials{CLI: secret.New(bearer)}, Addr: "127.0.0.1:11111",
				Web: http.NotFoundHandler(), Packages: resolver, MachineExecutions: publishedRouteObserver{}}).Handler()
			fatal(t, problem)
			send := func(key string, body map[string]any) *httptest.ResponseRecorder {
				t.Helper()
				raw, err := json.Marshal(body)
				must(t, err)
				request := httptest.NewRequest(http.MethodPost, "http://127.0.0.1:11111"+path, bytes.NewReader(raw))
				request.RemoteAddr = "127.0.0.1:12345"
				request.Header.Set("Authorization", "Bearer "+bearer)
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Idempotency-Key", key)
				response := httptest.NewRecorder()
				handler.ServeHTTP(response, request)
				return response
			}
			body := map[string]any{"package": "proof/render", "function": "generate",
				"input": map[string]any{"prompt": "A walking cat"}, "rental": true}
			response := send("bare", body)
			if response.Code < 400 || !strings.Contains(response.Body.String(), "one exact") || resolver.reads != 0 {
				t.Fatalf("an unversioned submission was not refused before any lookup: %d %s (lookups %d)", response.Code, response.Body, resolver.reads)
			}
			if recorded, problem := o.store.RequestByIdempotencyKey("bare"); problem != nil || recorded != nil {
				t.Fatalf("the refused submission entered the queue: %+v %v", recorded, problem)
			}
			body["release"] = "1.0.0"
			response = send("explicit", body)
			if response.Code < 400 || !strings.Contains(response.Body.String(), "proof.source_lookup") || resolver.reads != 1 {
				t.Fatalf("an exact release was not resolved as itself: %d %s (lookups %d)", response.Code, response.Body, resolver.reads)
			}
		})
	}
}
