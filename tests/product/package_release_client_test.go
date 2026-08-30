package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestPackageReleaseClientContract(t *testing.T) {
	ref := hub.Ref{Org: "proof", Name: "package"}
	releasePath := "/v1/packages/proof/package/releases/1.0.0"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer proof-token" || r.Header.Get("X-Tensorhub-Reason") != "proof publish" {
			t.Errorf("package mutation omitted authentication or audit reason")
		}
		fixture := ""
		switch {
		case r.Method == http.MethodPost && r.URL.Path == releasePath:
			fixture = "begin.json"
			assertEmptyObject(t, r.Body)
		case r.Method == http.MethodPost && r.URL.Path == releasePath+"/uploads":
			fixture = "uploads.json"
			var body struct {
				Paths            []string `json:"paths"`
				DependencyWheels []string `json:"dependency_wheels"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil ||
				!reflect.DeepEqual(body.Paths, []string{"package.toml"}) ||
				!reflect.DeepEqual(body.DependencyWheels, []string{"proof_dependency-1.0.0-py3-none-any.whl"}) {
				t.Errorf("upload registration changed: %+v err=%v", body, err)
			}
		case r.Method == http.MethodPut && r.URL.Path == releasePath:
			fixture = "finalize.json"
			assertEmptyObject(t, r.Body)
		default:
			http.NotFound(w, r)
			return
		}
		raw, err := os.ReadFile(filepath.Join("testdata", "package-release", fixture))
		if err != nil {
			t.Error(err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(raw)
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("proof-token")}, "cozy-test")

	begin, problem := client.BeginPackageRelease(context.Background(), ref, "1.0.0", "proof publish")
	if problem != nil || begin.State != "pending" {
		t.Fatalf("begin response changed: %+v problem=%v", begin, problem)
	}
	uploads, problem := client.PackageReleaseUploads(context.Background(), ref, "1.0.0",
		[]string{"package.toml"}, []string{"proof_dependency-1.0.0-py3-none-any.whl"}, "proof publish")
	if problem != nil || len(uploads.Uploads) != 2 || uploads.Uploads[0].Kind != "source" ||
		uploads.Uploads[1].Kind != "dependency_wheel" {
		t.Fatalf("uploads response changed: %+v problem=%v", uploads, problem)
	}
	finalized, problem := client.FinalizePackageRelease(context.Background(), ref, "1.0.0", "proof publish")
	if problem != nil || finalized.QualificationState != "qualified" ||
		len(finalized.PackageExecutions) != 1 || finalized.PackageExecutions[0].Function != "generate" ||
		len(finalized.Profiles) != 1 ||
		finalized.Profiles[0].BaseRealizationDigest != "index.docker.io/tensorhub/worker@sha256:"+strings.Repeat("a", 64) ||
		finalized.Profiles[0].PackageEnvironmentSpec.Length != 501 ||
		finalized.Profiles[0].ResolutionLock.Length != 502 ||
		finalized.Profiles[0].ResolvedWheelSet.Length != 503 {
		t.Fatalf("finalize response changed: %+v problem=%v", finalized, problem)
	}
}

func TestPackageReleaseClientRejectsUnknownResponseFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"project_wheel_upload":{"already_uploaded":true,"required_headers":{},"url":""},"state":"committed","renamed_field":true}`)
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("proof-token")}, "cozy-test")
	_, problem := client.BeginPackageRelease(context.Background(), hub.Ref{Org: "proof", Name: "package"}, "1.0.0", "proof")
	if problem == nil || problem.Name != "hub.unreadable_answer" {
		t.Fatalf("unknown package response field did not become version-skew refusal: %v", problem)
	}
}

func TestPackagePublishPendingWireFlowBoundsUploads(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "wire-package", "1.0.0", nil, "", true)
	for i := range 48 {
		name := filepath.Join(project, "wire_package", fmt.Sprintf("source_%02d.py", i))
		must(t, os.WriteFile(name, []byte("VALUE = 1\n"), 0o644))
	}

	var active, peak, began, registered, finalized atomic.Int64
	releasePath := "/v1/packages/proof/wire-package/releases/1.0.0"
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/storage/"):
			at := active.Add(1)
			defer active.Add(-1)
			for {
				seen := peak.Load()
				if at <= seen || peak.CompareAndSwap(seen, at) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && r.URL.Path == releasePath:
			began.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"state": "pending",
				"project_wheel_upload": map[string]any{
					"already_uploaded": false, "required_headers": map[string]string{},
					"url": server.URL + "/storage/project",
				},
			})
		case r.Method == http.MethodPost && r.URL.Path == releasePath+"/uploads":
			registered.Add(1)
			var body struct {
				Paths            []string `json:"paths"`
				DependencyWheels []string `json:"dependency_wheels"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.Paths) < 50 || len(body.DependencyWheels) != 0 {
				t.Errorf("Creator did not register the complete source set: paths=%d dependencies=%v err=%v",
					len(body.Paths), body.DependencyWheels, err)
			}
			uploads := make([]map[string]any, 0, len(body.Paths))
			for i, path := range body.Paths {
				uploads = append(uploads, map[string]any{
					"already_uploaded": false, "kind": "source", "path": path,
					"required_headers": map[string]string{},
					"url":              server.URL + "/storage/" + strconv.Itoa(i),
				})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"uploads": uploads})
		case r.Method == http.MethodPut && r.URL.Path == releasePath:
			finalized.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"compatible_profiles": []string{"cpu-py311-torch213"},
				"package_executions": []map[string]any{{
					"digest": "sha256:" + strings.Repeat("4", 64), "function": "generate",
					"profile": "cpu-py311-torch213", "state": "qualified",
				}},
				"profiles": []map[string]any{{
					"base_realization_kind": "base-worker-image", "candidate_id": "pqc-proof",
					"profile": "cpu-py311-torch213", "state": "qualified",
				}},
				"qualification_state": "qualified", "requirements": []string{}, "requires_python": ">=3.11",
			})
		default:
			http.Error(w, "unexpected route", http.StatusNotFound)
		}
	}))
	defer server.Close()

	code, out := runCozyDir(t, t.TempDir(), project,
		[]string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}, "package", "publish")
	if code != 0 || !strings.Contains(out, "eligible_worker_profiles:") ||
		!strings.Contains(out, "generate@cpu-py311-torch213=qualified") ||
		!strings.Contains(out, "Building package wheel and local dependencies...") ||
		!strings.Contains(out, "Registering 52 source files and 0 dependency wheels...") ||
		!strings.Contains(out, "Uploading files: 0/53") ||
		!strings.Contains(out, "Uploading files: 53/53") ||
		!strings.Contains(out, "Finalizing release and evaluating worker profiles...") ||
		strings.Contains(out, "qualification:") {
		t.Fatalf("pending package wire flow failed [exit %d]\n%s", code, out)
	}
	if began.Load() != 1 || registered.Load() != 1 || finalized.Load() != 1 {
		t.Fatalf("package route calls changed: begin=%d uploads=%d finalize=%d",
			began.Load(), registered.Load(), finalized.Load())
	}
	if got := peak.Load(); got > 16 || got < 2 {
		t.Fatalf("package upload concurrency=%d, want 2..16", got)
	}
}

func assertEmptyObject(t *testing.T, body io.Reader) {
	t.Helper()
	var value map[string]any
	if err := json.NewDecoder(body).Decode(&value); err != nil || len(value) != 0 {
		t.Errorf("request body is not one empty object: %v err=%v", value, err)
	}
}
