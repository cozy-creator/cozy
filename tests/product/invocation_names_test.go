package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
)

// Real CLI subprocesses read a published package and submit through their normal HTTP
// transport. The daemon fixture observes submission without starting machine work.
func invocationNamesRoot(t *testing.T, raw []byte) (string, <-chan api.JobSubmission) {
	t.Helper()
	submitted := make(chan api.JobSubmission, 8)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == "GET" && r.URL.Path == "/":
		case r.Method == "GET" && r.URL.Path == "/v1/packages/proof/names":
			_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "names"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
		case r.Method == "GET" && r.URL.Path == "/v1/packages/proof/names/releases/1.0.0":
			var detail hub.PackageReleaseDetail
			detail.PackageInterface = raw
			detail.Release.Release = "1.0.0"
			detail.Release.PackageInterfaceLength = int64(len(raw))
			_ = json.NewEncoder(w).Encode(detail)
		case r.Method == "POST" && r.URL.Path == "/v1/local/jobs":
			var submission api.JobSubmission
			if err := json.NewDecoder(r.Body).Decode(&submission); err != nil {
				t.Error(err)
			}
			submitted <- submission
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":{"code":"proof.submitted","message":"submission recorded by test"}}`))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	t.Cleanup(held.Release)
	_, problem = api.Mint(layout)
	fatal(t, problem)
	must(t, os.WriteFile(filepath.Join(layout.Root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	return layout.Root, submitted
}

func TestInvocationSeparatorsPreserveRegisteredNames(t *testing.T) {
	for _, kind := range []string{"entrypoints", "jobs"} {
		for _, registered := range []string{"long_form_cuts", "long-form-cuts", "long_form-cuts"} {
			t.Run(kind+"/"+registered, func(t *testing.T) {
				raw := []byte(`{"format":"cozy.package.interface/1","application":"proof:app","` + kind + `":[{"name":"` + registered + `","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
				root, _ := invocationNamesRoot(t, raw)
				var described string
				for _, spelling := range []string{"long_form_cuts", "long-form-cuts", "long_form-cuts", "long-form_cuts"} {
					code, out := runCozy(t, root, "run", "proof/names/"+spelling, "--rental-only", "--describe")
					if code != 0 || !strings.Contains(out, "proof/names/"+registered) {
						t.Fatalf("%s lost registered name [exit %d]: %s", spelling, code, out)
					}
					if described != "" && described != out {
						t.Fatalf("%s selected a different contract: %s", spelling, out)
					}
					described = out
				}
			})
		}
	}
}

func TestInvocationSeparatorsSubmitRegisteredJobName(t *testing.T) {
	for _, registered := range []string{"long_form", "long-form"} {
		t.Run(registered, func(t *testing.T) {
			raw := []byte(`{"format":"cozy.package.interface/1","application":"proof:app","jobs":[{"name":"` + registered + `","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
			root, submitted := invocationNamesRoot(t, raw)
			for _, spelling := range []string{"long_form", "long-form"} {
				code, out := runCozy(t, root, "run", "proof/names/"+spelling, "--rental-only", "--json")
				if code == 0 || !strings.Contains(out, `"code":"proof.submitted"`) {
					t.Fatalf("%s did not reach submission [exit %d]: %s", spelling, code, out)
				}
				select {
				case submission := <-submitted:
					if submission.Package != "proof/names" || submission.Function != registered || submission.Release != "" || len(submission.PackageInterface) == 0 {
						t.Fatalf("submission lost its authored target: %+v", submission)
					}
				default:
					t.Fatal("missing job submission")
				}
			}
		})
	}
}

func TestInvocationSeparatorsKeepRefusalsAndAmbiguity(t *testing.T) {
	raw := []byte(`{"format":"cozy.package.interface/1","application":"proof:app","jobs":[{"name":"internal_job","internal":true,"publishes":false,"request":{"fields":[]},"result":{"fields":[]}},{"name":"future_job","internal":"future","publishes":false,"request":{"fields":[]},"result":{"fields":[]}},{"name":"long_form_cuts","publishes":false,"request":{"fields":[]},"result":{"fields":[]}},{"name":"long-form-cuts","publishes":false,"request":{"fields":[]},"result":{"fields":[]}}]}`)
	root, _ := invocationNamesRoot(t, raw)
	surface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	unavailable := surface.Unavailable["future_job"]
	if unavailable == nil {
		t.Fatal("fixture did not preserve an unavailable callable")
	}
	for _, spelling := range []string{"future_job", "future-job"} {
		code, out := runCozy(t, root, "run", "proof/names/"+spelling, "--rental-only", "--json")
		if code == 0 || !strings.Contains(out, `"code":"`+unavailable.ErrName()+`"`) || !strings.Contains(out, unavailable.Message) {
			t.Fatalf("%s lost its unavailable refusal [exit %d]: %s", spelling, code, out)
		}
	}
	for _, spelling := range []string{"internal_job", "internal-job"} {
		code, out := runCozy(t, root, "run", "proof/names/"+spelling, "--rental-only", "--json")
		if code == 0 || !strings.Contains(out, `"code":"callable_internal"`) {
			t.Fatalf("%s escaped its internal boundary [exit %d]: %s", spelling, code, out)
		}
	}
	code, out := runCozy(t, root, "run", "proof/names/missing-job", "--rental-only", "--json")
	if code == 0 || !strings.Contains(out, `"code":"not_found"`) || !strings.Contains(out, "missing-job") {
		t.Fatalf("unknown function lost its original spelling [exit %d]: %s", code, out)
	}
	for _, spelling := range []string{"long_form-cuts", "long-form_cuts"} {
		code, out := runCozy(t, root, "run", "proof/names/"+spelling, "--rental-only", "--json")
		if code != 2 || !strings.Contains(out, "long-form-cuts, long_form_cuts") {
			t.Fatalf("%s silently selected an ambiguous function [exit %d]: %s", spelling, code, out)
		}
	}
	for _, exact := range []string{"long_form_cuts", "long-form-cuts"} {
		code, out := runCozy(t, root, "run", "proof/names/"+exact, "--rental-only", "--describe")
		if code != 0 || !strings.Contains(out, "proof/names/"+exact) {
			t.Fatalf("registered exact function %s changed [exit %d]: %s", exact, code, out)
		}
	}
}
