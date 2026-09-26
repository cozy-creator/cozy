package producttest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

// The worker deliberately never acknowledges the already queued run. A detached
// public CLI command must return its durable ID instead of polling acceptance.
func TestDetachedMachineJobReturnsQueuedRun(t *testing.T) {
	for _, jsonMode := range []bool{false, true} {
		name := "human"
		if jsonMode {
			name = "json"
		}
		t.Run(name, func(t *testing.T) {
			check := func(problem *exit.Error) {
				t.Helper()
				if problem != nil {
					t.Fatal(problem)
				}
			}
			iface := []byte(`{"format":"cozy.package.interface/1","application":"proof:app","entrypoints":[],"jobs":[{"name":"compute","models":[],"publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[]}]}`)
			catalog := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/v1/packages/proof/queued":
					_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "queued"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
				case "/v1/packages/proof/queued/releases/1.0.0":
					var detail hub.PackageReleaseDetail
					detail.Release.Release = "1.0.0"
					detail.Release.PackageInterfaceLength = int64(len(iface))
					detail.PackageInterface = iface
					_ = json.NewEncoder(w).Encode(detail)
				default:
					t.Errorf("unexpected catalog request: %s", r.URL)
					http.Error(w, "unexpected request", 500)
				}
			}))
			defer catalog.Close()
			var reads, submissions atomic.Int32
			local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == "GET" && r.URL.Path == "/":
				case r.Method == "POST" && r.URL.Path == "/v1/local/jobs":
					submissions.Add(1)
					if r.Header.Get("Authorization") == "" {
						t.Error("missing daemon authentication")
					}
					_ = json.NewEncoder(w).Encode(api.JobHandle{Number: 42, JobID: "req-queued", Package: "proof/queued", Function: "compute", Status: "queued", MachineExecution: true})
				case r.Method == "GET" && r.URL.Path == "/v1/local/jobs/req-queued":
					if reads.Add(1) > 1 {
						http.Error(w, "detached command polled machine acceptance", 500)
						return
					}
					_ = json.NewEncoder(w).Encode(api.JobState{Number: 42, JobID: "req-queued", Package: "proof/queued", Function: "compute", Status: "queued", Stage: "uploading captured package", MachineExecution: &api.MachineExecutionView{Machine: "cooler", Accepted: false}})
				default:
					t.Errorf("unexpected daemon request: %s %s", r.Method, r.URL)
					http.Error(w, "unexpected request", 500)
				}
			}))
			defer local.Close()
			layout, problem := home.Open(t.TempDir())
			check(problem)
			held, problem := daemon.Hold(layout, strings.TrimPrefix(local.URL, "http://"), "")
			check(problem)
			defer held.Release()
			_, problem = api.Mint(layout)
			check(problem)
			var out, diagnostic bytes.Buffer
			err := (&cli.RunExecuteCmd{Target: "proof/queued/compute", RentalOnly: true}).Run(&cli.Runtime{Cfg: config.Config{Home: layout.Root, HubURL: catalog.URL}, Out: &out, Err: &diagnostic, Mode: output.Mode{JSON: jsonMode}})
			if err != nil && err != (*exit.Error)(nil) {
				t.Fatalf("queued command did not return: %v; stdout=%s stderr=%s", err, &out, &diagnostic)
			}
			if reads.Load() != 1 || submissions.Load() != 1 {
				t.Fatalf("submissions=%d reads=%d", submissions.Load(), reads.Load())
			}
			for _, want := range []string{"42", "cooler", "queued", "uploading captured package"} {
				if !strings.Contains(out.String(), want) {
					t.Errorf("missing %q from %s", want, &out)
				}
			}
			if jsonMode {
				if !strings.Contains(out.String(), `"machine_accepted":false`) || diagnostic.Len() != 0 {
					t.Errorf("incorrect JSON receipt: stdout=%s stderr=%s", &out, &diagnostic)
				}
			} else if !strings.Contains(diagnostic.String(), "Queued run 42") || strings.Contains(diagnostic.String(), "Invoking") {
				t.Errorf("durable acceptance hidden by diagnostic: %s", &diagnostic)
			}
		})
	}
}
