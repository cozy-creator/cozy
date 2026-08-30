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
	"github.com/cozy-creator/cozy/internal/transfer"
)

func TestPackageReleaseClientContract(t *testing.T) {
	ref := hub.Ref{Org: "proof", Name: "package"}
	releasePath := "/v1/packages/proof/package/releases/1.0.0"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer proof-token" || r.Header.Get("X-Tensorhub-Reason") != "proof publish" {
			t.Errorf("package mutation omitted authentication or audit reason")
		}
		switch {
		case r.Method == http.MethodPost && r.URL.Path == releasePath:
			var body struct {
				Paths            []string `json:"paths"`
				DependencyWheels []string `json:"dependency_wheels"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil ||
				!reflect.DeepEqual(body.Paths, []string{"package.toml"}) ||
				!reflect.DeepEqual(body.DependencyWheels, []string{"proof_dependency-1.0.0-py3-none-any.whl"}) {
				t.Errorf("package declaration changed: %+v err=%v", body, err)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"state": "pending", "uploads": []map[string]any{
				{"kind": "project_wheel", "path": "project.whl", "url": "https://storage.invalid/project", "required_headers": map[string]string{}, "already_uploaded": false},
				{"kind": "source", "path": "package.toml", "url": "https://storage.invalid/source", "required_headers": map[string]string{}, "already_uploaded": false},
				{"kind": "dependency_wheel", "path": "proof_dependency-1.0.0-py3-none-any.whl", "url": "https://storage.invalid/dependency", "required_headers": map[string]string{}, "already_uploaded": false},
			}})
			return
		case r.Method == http.MethodPut && r.URL.Path == releasePath:
			assertEmptyObject(t, r.Body)
			_ = json.NewEncoder(w).Encode(map[string]any{"state": "committed", "qualification_state": "qualified", "package_release": map[string]any{
				"canonical_bytes": []byte(`{"format":"cozy.package.release/1"}`),
				"digest":          "sha256:" + strings.Repeat("a", 64), "length": 35,
			}})
			return
		default:
			http.NotFound(w, r)
			return
		}
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("proof-token")}, "cozy-test")

	draft, problem := client.DeclarePackageRelease(context.Background(), ref, "1.0.0",
		[]string{"package.toml"}, []string{"proof_dependency-1.0.0-py3-none-any.whl"}, "proof publish")
	if problem != nil || draft.State != "pending" || len(draft.Uploads) != 3 ||
		draft.Uploads[0].Kind != "project_wheel" || draft.Uploads[2].Kind != "dependency_wheel" {
		t.Fatalf("declaration response changed: %+v problem=%v", draft, problem)
	}
	committed, problem := client.CommitPackageRelease(context.Background(), ref, "1.0.0", "proof publish")
	if problem != nil || committed.State != "committed" || committed.PackageRelease.Length != 35 {
		t.Fatalf("commit response changed: %+v problem=%v", committed, problem)
	}
}

func TestPackageReleaseYankClientContract(t *testing.T) {
	ref := hub.Ref{Org: "proof", Name: "package"}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete || r.URL.Path != "/v1/packages/proof/package/releases/1.2.3" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer proof-token" ||
			r.Header.Get("X-Tensorhub-Reason") != "cozy package yank proof/package@1.2.3" {
			t.Errorf("package yank omitted authentication or derived audit text")
		}
		assertEmptyObject(t, r.Body)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"state": "yanked", "release": "1.2.3", "changed": true,
			"yanked_at": "2026-08-30T12:34:56Z",
		})
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL,
		HubToken: secret.New("proof-token")}, "cozy-test")
	yanked, problem := client.YankPackageRelease(context.Background(), ref, "1.2.3",
		"cozy package yank proof/package@1.2.3")
	if problem != nil || yanked.State != "yanked" || yanked.Release != "1.2.3" ||
		!yanked.Changed || yanked.YankedAt != "2026-08-30T12:34:56Z" {
		t.Fatalf("package yank response changed: %+v problem=%v", yanked, problem)
	}
}

func TestPackageReleaseClientRejectsUnknownResponseFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"state":"committed","uploads":[],"renamed_field":true}`)
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("proof-token")}, "cozy-test")
	_, problem := client.DeclarePackageRelease(context.Background(), hub.Ref{Org: "proof", Name: "package"}, "1.0.0", nil, nil, "proof")
	if problem == nil || problem.Name != "hub.unreadable_answer" {
		t.Fatalf("unknown package response field did not become version-skew refusal: %v", problem)
	}
}

func TestPackageInstallPlanContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/packages/proof/package":
			_, _ = io.WriteString(w, `{"package":{"org":"proof","name":"package","created_at":"2026-08-30T00:00:00Z"},"releases":[{"release":"1.2.3","cut_at":"2026-08-30T00:00:00Z"}]}`)
		case "/v1/packages/proof/package/releases/1.2.3/selection":
			if r.Method != http.MethodPost {
				t.Errorf("package selection used %s", r.Method)
			}
			var body struct {
				Capability      hub.PackageInstallTarget `json:"capability"`
				ModelSelections []hub.ModelSelection     `json:"model_selections"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.ModelSelections == nil ||
				len(body.ModelSelections) != 0 {
				t.Errorf("package selection body changed: %+v err=%v", body, err)
			}
			switch body.Capability.Accelerator {
			case "cpu":
				if body.Capability.OS != "linux" || body.Capability.Arch != "x86" ||
					body.Capability.DriverCUDA != "" || body.Capability.ComputeCapability != "" {
					t.Errorf("CPU package install target changed: %+v", body.Capability)
				}
			case "nvidia":
				if body.Capability.OS != "linux" || body.Capability.Arch != "x86" ||
					body.Capability.DriverCUDA != "13.2" || body.Capability.ComputeCapability != "8.9" {
					t.Errorf("NVIDIA package install target changed: %+v", body.Capability)
				}
			default:
				t.Errorf("package install target omitted accelerator: %+v", body.Capability)
			}
			_, _ = io.WriteString(w, `{"profile":"cpu-test","placement_set":{"canonical_bytes":"e30=","digest":"sha256:`+strings.Repeat("2", 64)+`","length":2},"package_release":{"canonical_bytes":"e30=","digest":"sha256:`+strings.Repeat("3", 64)+`","length":2},"package_descriptor":{"canonical_bytes":"e30=","digest":"sha256:`+strings.Repeat("4", 64)+`","length":2},"qualification":{"canonical_bytes":"e30=","digest":"sha256:`+strings.Repeat("5", 64)+`","length":2},"downloads":[{"digest":"sha256:`+strings.Repeat("1", 64)+`","kind":"project_wheel","length":4,"path":"proof.whl","url":"https://storage.invalid/proof.whl"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL}, "cozy-test")
	ref := hub.Ref{Org: "proof", Name: "package"}
	card, problem := client.PackageCard(context.Background(), ref)
	if problem != nil || len(card.Releases) != 1 || card.Releases[0].Release != "1.2.3" {
		t.Fatalf("package card changed: %+v problem=%v", card, problem)
	}
	plan, problem := client.PackageInstallPlan(context.Background(), ref, "1.2.3", hub.PackageInstallTarget{
		Accelerator: "cpu", OS: "linux", Arch: "x86",
	})
	if problem != nil || plan.Profile != "cpu-test" ||
		len(plan.Downloads) != 1 || plan.Downloads[0].Path != "proof.whl" {
		t.Fatalf("package install plan changed: %+v problem=%v", plan, problem)
	}
	_, problem = client.PackageInstallPlan(context.Background(), ref, "1.2.3", hub.PackageInstallTarget{
		Accelerator: "nvidia", OS: "linux", Arch: "x86",
		DriverCUDA: "13.2", ComputeCapability: "8.9",
	})
	if problem != nil {
		t.Fatalf("NVIDIA package install target was refused: %v", problem)
	}
}

func TestPackagePublishPendingWireFlowBoundsUploads(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "wire-package", "1.0.0", nil, "", true)
	for i := range 48 {
		name := filepath.Join(project, "wire_package", fmt.Sprintf("source_%02d.py", i))
		must(t, os.WriteFile(name, []byte("VALUE = 1\n"), 0o644))
	}

	var active, peak, began, finalized atomic.Int64
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
			var body struct {
				Paths            []string `json:"paths"`
				DependencyWheels []string `json:"dependency_wheels"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.Paths) < 50 || len(body.DependencyWheels) != 0 {
				t.Errorf("Creator did not declare the complete source set: paths=%d dependencies=%v err=%v",
					len(body.Paths), body.DependencyWheels, err)
			}
			uploads := []map[string]any{{
				"already_uploaded": false, "kind": "project_wheel",
				"path":             "project.whl",
				"required_headers": map[string]string{}, "url": server.URL + "/storage/project",
			}}
			for i, path := range body.Paths {
				uploads = append(uploads, map[string]any{
					"already_uploaded": false, "kind": "source", "path": path,
					"required_headers": map[string]string{},
					"url":              server.URL + "/storage/" + strconv.Itoa(i),
				})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"state": "pending", "uploads": uploads})
		case r.Method == http.MethodPut && r.URL.Path == releasePath:
			finalized.Add(1)
			releaseBytes := []byte(`{"format":"cozy.package.release/1"}`)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"state": "committed", "qualification_state": "qualified", "package_release": map[string]any{
					"canonical_bytes": releaseBytes, "digest": "sha256:" + strings.Repeat("4", 64),
					"length": len(releaseBytes),
				},
			})
		default:
			http.Error(w, "unexpected route", http.StatusNotFound)
		}
	}))
	defer server.Close()

	code, out := runCozyDir(t, t.TempDir(), project,
		[]string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}, "package", "publish")
	if code != 0 || !strings.Contains(out, "status:  published") ||
		strings.Contains(out, "package_release_digest:") || strings.Contains(out, "qualification:") ||
		!strings.Contains(out, "Building package wheel and local dependencies...") ||
		!strings.Contains(out, "Declaring 52 source files and 0 dependency wheels...") ||
		!strings.Contains(out, "Uploading files: 0/53") ||
		!strings.Contains(out, "Uploading files: 53/53") ||
		!strings.Contains(out, "Committing exact package release...") {
		t.Fatalf("pending package wire flow failed [exit %d]\n%s", code, out)
	}
	if began.Load() != 1 || finalized.Load() != 1 {
		t.Fatalf("package route calls changed: declare=%d commit=%d", began.Load(), finalized.Load())
	}
	if got := peak.Load(); got > 16 || got < 2 {
		t.Fatalf("package upload concurrency=%d, want 2..16", got)
	}
}

func TestPackageUploadCountsBytesWhenSuccessResponseIsLost(t *testing.T) {
	payload := []byte("exact package bytes")
	path := filepath.Join(t.TempDir(), "package.whl")
	must(t, os.WriteFile(path, payload, 0o600))
	var attempts atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if attempts.Add(1) == 1 {
			_, _ = io.Copy(io.Discard, r.Body)
			panic(http.ErrAbortHandler) // bytes landed; only the success response was lost
		}
		w.WriteHeader(http.StatusPreconditionFailed)
	}))
	defer server.Close()

	moved, problem := transfer.UploadPresigned(context.Background(), "project_wheel", path,
		server.URL, map[string]string{"if-none-match": "*"})
	if problem != nil || moved != int64(len(payload)) || attempts.Load() != 2 {
		t.Fatalf("lost-response accounting: moved=%d attempts=%d problem=%v",
			moved, attempts.Load(), problem)
	}
}

func assertEmptyObject(t *testing.T, body io.Reader) {
	t.Helper()
	var value map[string]any
	if err := json.NewDecoder(body).Decode(&value); err != nil || len(value) != 0 {
		t.Errorf("request body is not one empty object: %v err=%v", value, err)
	}
}
