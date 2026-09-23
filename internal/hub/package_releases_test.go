package hub

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestPackageReleaseFinalizeWaitsForQueuedCommit(t *testing.T) {
	var statusCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		write := func(status int, value PackageReleaseCommit) {
			raw, err := json.Marshal(value)
			if err != nil {
				t.Fatalf("encode response: %v", err)
			}
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", fmt.Sprint(len(raw)))
			w.WriteHeader(status)
			_, _ = w.Write(raw)
		}
		if r.Header.Get("Authorization") != "Bearer proof" {
			t.Fatalf("missing finalize credential")
		}
		switch r.Method + " " + r.URL.Path {
		case "POST /v1/packages/paul/demo/publish/1.0.0/finalize":
			var body struct {
				PublicationID string `json:"publication_id"`
				PythonVersion string `json:"python_version"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode finalize: %v", err)
			}
			if body.PublicationID != "pub-1" || body.PythonVersion != "3.12.12" {
				t.Fatalf("wrong finalize body: %+v", body)
			}
			write(http.StatusAccepted, PackageReleaseCommit{PublicationID: "pub-1", State: "queued", StatusURL: "/v1/packages/paul/demo/publish/1.0.0/status"})
		case "GET /v1/packages/paul/demo/publish/1.0.0/status":
			if statusCalls.Add(1) == 1 {
				write(http.StatusOK, PackageReleaseCommit{PublicationID: "pub-1", State: "verifying", StatusURL: "/v1/packages/paul/demo/publish/1.0.0/status"})
				return
			}
			write(http.StatusOK, PackageReleaseCommit{PublicationID: "pub-1", State: "committed"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := New(config.Config{HubURL: server.URL, HubToken: secret.New("proof")}, "package-release-test")
	ref := Ref{Org: "paul", Name: "demo"}
	initial, problem := client.CommitPackageRelease(t.Context(), ref, "1.0.0", "pub-1", nil, "publish demo", "3.12.12")
	if problem != nil {
		t.Fatalf("finalize: %s", problem)
	}
	if initial.State != "queued" || initial.StatusURL == "" {
		t.Fatalf("unexpected accepted result: %+v", initial)
	}
	done, problem := client.WaitPackageRelease(t.Context(), ref, "1.0.0", initial, nil)
	if problem != nil {
		t.Fatalf("wait: %s", problem)
	}
	if done.State != "committed" || statusCalls.Load() != 2 {
		t.Fatalf("unexpected terminal result: %+v, status calls=%d", done, statusCalls.Load())
	}
}

func TestPackageReleaseStatusRendersTypedFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(PackageReleaseCommit{
			PublicationID: "pub-1", State: "failed",
			Error: &PackageReleaseFailure{Code: "package_release.registry_fetch_failed", Message: "registry unavailable", Remedy: "retry later"},
		})
	}))
	defer server.Close()
	client := New(config.Config{HubURL: server.URL, HubToken: secret.New("proof")}, "package-release-test")
	status, problem := client.PackageReleaseStatus(t.Context(), Ref{Org: "paul", Name: "demo"}, "1.0.0")
	if problem != nil {
		t.Fatalf("status: %s", problem)
	}
	if status.State != "failed" {
		t.Fatalf("unexpected status: %+v", status)
	}
	problem = status.failure()
	if problem == nil || problem.ErrName() != "package_release.registry_fetch_failed" || problem.Remedy != "retry later" {
		t.Fatalf("failure was not preserved: %#v", problem)
	}
}
