package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestInvocationSeparatorsPreserveRegisteredNames(t *testing.T) {
	for _, kind := range []string{"entrypoint", "job"} {
		for _, registered := range []string{"long_form_cuts", "long-form-cuts", "long_form-cuts"} {
			t.Run(kind+"/"+registered, func(t *testing.T) {
				callable := launch.Entrypoint{Name: registered, Kind: kind}
				surface := &launch.PackageInterface{}
				if kind == "job" {
					surface.Jobs = []launch.Entrypoint{callable}
				} else {
					surface.Entrypoints = []launch.Entrypoint{callable}
				}
				var described string
				for _, spelling := range []string{"long_form_cuts", "long-form-cuts", "long_form-cuts", "long-form_cuts"} {
					var out bytes.Buffer
					ctx := &Context{Inv: &Invocation{Bools: map[string]bool{"--describe": true}}, Out: &out}
					if problem := runTarget(ctx, Target{Package: "proof/h3", Function: spelling}, surface); problem != nil {
						t.Fatalf("%s: %v", spelling, problem)
					}
					if !strings.Contains(out.String(), "proof/h3/"+registered) {
						t.Fatalf("%s lost registered name: %s", spelling, &out)
					}
					if described != "" && described != out.String() {
						t.Fatalf("%s selected a different contract: %s", spelling, &out)
					}
					described = out.String()
				}
			})
		}
	}
}

func TestInvocationSeparatorsSubmitRegisteredJobName(t *testing.T) {
	for _, registered := range []string{"long_form", "long-form"} {
		t.Run(registered, func(t *testing.T) {
			var submitted []api.JobSubmission
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/local/jobs" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				} else {
					var submission api.JobSubmission
					if err := json.NewDecoder(r.Body).Decode(&submission); err != nil {
						t.Error(err)
					}
					submitted = append(submitted, submission)
				}
				// Observe the real CLI-to-daemon submission without starting machine work.
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(`{"error":{"code":"proof.submitted","message":"submission recorded by test"}}`))
			}))
			defer server.Close()
			cfg := config.Config{Home: t.TempDir()}
			controller := localapi.At(cfg, strings.TrimPrefix(server.URL, "http://"), secret.Value{})
			surface := &launch.PackageInterface{Jobs: []launch.Entrypoint{{Name: registered, Kind: "job"}}}
			for _, spelling := range []string{"long_form", "long-form"} {
				var out bytes.Buffer
				ctx := &Context{Cfg: cfg, foregroundClient: controller, Out: &out, Err: &out,
					Inv: &Invocation{Args: []string{"proof/h3/" + spelling}}}
				problem := runTarget(ctx, Target{Package: "proof/h3", Release: "1.0.0", Function: spelling}, surface)
				if problem == nil || problem.ErrName() != "proof.submitted" {
					t.Fatalf("%s did not reach submission: %v", spelling, problem)
				}
			}
			if len(submitted) != 2 {
				t.Fatalf("want two submissions, got %d", len(submitted))
			}
			for _, submission := range submitted {
				if submission.Package != "proof/h3" || submission.Function != registered || submission.Release != "1.0.0" {
					t.Fatalf("submission lost its authored target: %+v", submission)
				}
			}
		})
	}
}

func TestInvocationSeparatorsKeepRefusalsAndAmbiguity(t *testing.T) {
	unavailable := exit.Named(exit.Structural, "callable_unavailable", "future_job needs a newer runtime")
	surface := &launch.PackageInterface{
		Jobs: []launch.Entrypoint{
			{Name: "internal_job", Kind: "job", Internal: true},
			{Name: "long_form_cuts", Kind: "job"},
			{Name: "long-form-cuts", Kind: "job"},
		},
		Unavailable: map[string]*exit.Error{"future_job": unavailable},
	}
	for _, spelling := range []string{"future_job", "future-job"} {
		if problem := runTarget(nil, Target{Package: "proof/h3", Function: spelling}, surface); problem != unavailable {
			t.Fatalf("%s lost its unavailable refusal: %v", spelling, problem)
		}
	}
	for _, spelling := range []string{"internal_job", "internal-job"} {
		if problem := runTarget(nil, Target{Package: "proof/h3", Function: spelling}, surface); problem == nil || problem.ErrName() != "callable_internal" {
			t.Fatalf("%s escaped its internal boundary: %v", spelling, problem)
		}
	}
	if problem := runTarget(nil, Target{Package: "proof/h3", Function: "missing-job"}, surface); problem == nil || problem.Code != exit.NotFound || !strings.Contains(problem.Message, "missing-job") {
		t.Fatalf("unknown function lost its original spelling: %v", problem)
	}
	for _, spelling := range []string{"long_form-cuts", "long-form_cuts"} {
		problem := runTarget(nil, Target{Package: "proof/h3", Function: spelling}, surface)
		if problem == nil || problem.Code != exit.Usage || !strings.Contains(problem.Remedy, "long-form-cuts, long_form_cuts") {
			t.Fatalf("%s silently selected an ambiguous function: %v", spelling, problem)
		}
	}
	for _, exact := range []string{"long_form_cuts", "long-form-cuts"} {
		name, problem := invocationFunctionName(exact, surface)
		if problem != nil || name != exact {
			t.Fatalf("registered exact function %s changed: %s %v", exact, name, problem)
		}
	}
}
