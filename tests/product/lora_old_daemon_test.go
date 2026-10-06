package producttest

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestModelOverrideClientRequiresOnlyItsOperationCapability(t *testing.T) {
	for _, response := range []string{`{}`, `{"model_overrides":false}`, `{"model_overrides":true,"future_hint":"ignored"}`} {
		t.Run(response, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			fatal(t, problem)
			must(t, os.WriteFile(layout.Daemon, nil, 0600))
			_, problem = api.Mint(layout)
			fatal(t, problem)
			var reads, submitted atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/v1/capabilities":
					reads.Add(1)
					_, _ = w.Write([]byte(response))
				case "/v1/requests":
					submitted.Add(1)
					_, _ = w.Write([]byte(`{"request_id":"accepted","status":"queued"}`))
				default:
					t.Errorf("unexpected request %s", r.URL.Path)
				}
			}))
			defer server.Close()
			client, problem := localclient.Open(config.Config{Home: root}, daemon.State{Addr: strings.TrimPrefix(server.URL, "http://")})
			fatal(t, problem)
			_, problem = client.Submit(api.Submission{Package: "proof/parent", Function: "compose", Models: []records.ModelRef{{
				Choice: true, Package: "proof/parent", Slot: "child.models.model", Model: "proof/base",
			}}}, "child")
			supported := strings.Contains(response, `"model_overrides":true`)
			if supported {
				fatal(t, problem)
			} else if problem == nil || problem.ErrName() != "daemon.model_overrides_unavailable" || submitted.Load() != 0 {
				t.Fatalf("absent capability submitted the child override: %v", problem)
			}
			_, problem = client.Submit(api.Submission{Package: "proof/parent", Function: "compose", Models: []records.ModelRef{{
				Choice: true, Package: "proof/parent", Slot: "compose.models.model", Model: "proof/base",
			}}}, "base")
			fatal(t, problem)
			want := int32(1)
			if supported {
				want++
			}
			if reads.Load() != 1 || submitted.Load() != want {
				t.Fatalf("ordinary base override required new capability: %d reads, %d submissions", reads.Load(), submitted.Load())
			}
		})
	}
}
