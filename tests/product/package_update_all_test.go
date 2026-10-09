package producttest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

func updateAllInstall(t *testing.T, layout home.Layout, store *records.Store, n int, pkg, version, source, origin string) records.PackageInstall {
	t.Helper()
	prior := cleanupTestInstall(layout, fmt.Sprintf("%016x", n), version)
	prior.Package, prior.SourceRef, prior.SourceKind = pkg, pkg+"@"+version, source
	prior.Hub = origin
	must(t, os.MkdirAll(prior.Dir, 0700))
	must(t, os.WriteFile(filepath.Join(prior.Dir, "prior.py"), []byte("# retained working install\n"), 0600))
	_, problem := store.Activate(prior)
	fatal(t, problem)
	return prior
}

func TestPackageUpdateAllContinuesFailuresAndPreservesSelections(t *testing.T) {
	plan, wheel := reportingRelease(t)
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	defer server.Close()
	priors := map[string]records.PackageInstall{}
	for i, entry := range []struct{ pkg, version, source string }{
		{"a-failed/install-reporting", "1.0.0", "tensorhub"},
		{"current/install-reporting", "1.0.1", "tensorhub"},
		{"development/install-reporting", "2.0.0rc1", "tensorhub"},
		{"local/editable", "1.0.0", "local"},
		{"newer/install-reporting", "2.0.0", "tensorhub"},
		{"private/dependency", "1.0.0", "wheel"},
		{"z-updated/install-reporting", "1.0.0", "tensorhub"},
	} {
		priors[entry.pkg] = updateAllInstall(t, layout, store, i+1, entry.pkg, entry.version, entry.source, server.URL)
	}
	var downloads atomic.Int32
	mux.HandleFunc("GET /v1/packages/{org}/install-reporting", func(w http.ResponseWriter, r *http.Request) {
		if r.PathValue("org") == "development" {
			t.Error("development pin attempted a registry update")
		}
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: r.PathValue("org"), Name: "install-reporting"},
			Releases: []hub.ReleaseSummary{{Release: "1.0.1"}, {Release: "9.0.0", Yanked: true}}})
	})
	mux.HandleFunc("POST /v1/packages/{org}/install-reporting/download", func(w http.ResponseWriter, r *http.Request) {
		downloads.Add(1)
		if selected := r.URL.Query().Get("release"); selected != "1.0.1" {
			t.Errorf("update did not pin the selected release: %s", selected)
		}
		selectedPlan := plan
		if r.PathValue("org") == "a-failed" {
			// A wheel whose bytes disagree with the declared hash must fail the
			// ordinary staged installer without replacing its previous selection.
			selectedPlan.Downloads = append([]hub.PackageInstallDownload(nil), plan.Downloads...)
			selectedPlan.Downloads[0].Digest = "sha256:" + strings.Repeat("f", 64)
		}
		_ = json.NewEncoder(w).Encode(selectedPlan)
	})
	mux.HandleFunc("GET /v1/index/{org}/simple/install-reporting/", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `<a href="/project.whl/%s#sha256=%x">%s</a>`, plan.Downloads[0].Path, sha256.Sum256(wheel), plan.Downloads[0].Path)
	})
	mux.HandleFunc("GET /v1/index/{org}/simple/{dependency}/", http.NotFound)
	mux.HandleFunc("GET /project.whl/{name}", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) })
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("bulk code update made an unexpected request: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	code, out := runCozy(t, root, "package", "update-all", "--json")
	var result struct {
		Packages                          []map[string]string
		Updated, Current, Failed, Skipped int
	}
	if code != 1 || json.Unmarshal([]byte(out), &result) != nil {
		t.Fatalf("bulk result must be one JSON document with exit 1: %d %s", code, out)
	}
	if result.Updated != 1 || result.Current != 1 || result.Failed != 1 || result.Skipped != 4 || len(result.Packages) != 7 || downloads.Load() != 2 {
		t.Fatalf("bulk update summary: %s; downloads=%d", out, downloads.Load())
	}
	for _, row := range result.Packages {
		_, active, problem := store.ActivePackage(server.URL, row["package"])
		fatal(t, problem)
		prior := priors[row["package"]]
		if row["status"] == "updated" {
			if active == nil || active.Version != "1.0.1" || active.ID == prior.ID {
				t.Fatalf("ordinary update did not activate a new install: %+v", active)
			}
		} else if active == nil || active.ID != prior.ID {
			t.Fatalf("failed/current/skipped package selection changed: %+v", row)
		}
		if row["status"] == "failed" {
			if row["error_code"] == "" || !strings.Contains(row["detail"], "cause:") {
				t.Fatal("failed package lost its reason")
			}
			raw, err := os.ReadFile(filepath.Join(prior.Dir, "prior.py"))
			must(t, err)
			if string(raw) != "# retained working install\n" {
				t.Fatal("failed update damaged prior bytes")
			}
		}
	}
	fatal(t, store.Unpin(server.URL, "a-failed/install-reporting", 1))
	code, out = runCozy(t, root, "package", "update-all", "--json")
	if code != 0 || json.Unmarshal([]byte(out), &result) != nil || result.Updated != 0 || result.Current != 2 || result.Failed != 0 || downloads.Load() != 2 {
		t.Fatalf("already current packages were installed again: %d %s", code, out)
	}
}

func TestPackageUpdateAllEmptyAndSkippedOutput(t *testing.T) {
	root := t.TempDir()
	code, out := runCozy(t, root, "package", "update-all", "--json")
	if code != 0 || !strings.Contains(out, `"packages":[]`) || !strings.Contains(out, `"updated":0`) {
		t.Fatalf("empty update: %d %s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	for i := 0; i < 22; i++ {
		updateAllInstall(t, layout, store, i+1, fmt.Sprintf("local/project-%02d", i), "1.0.0", "local", "")
	}
	store.Close()
	code, out = runCozy(t, root, "package", "update-all", "--json")
	var result struct{ Packages []map[string]string }
	if code != 0 || json.Unmarshal([]byte(out), &result) != nil || len(result.Packages) != 22 || strings.Contains(out, `"omitted"`) {
		t.Fatalf("bulk update omitted package results: %d %s", code, out)
	}
	code, out = runCozy(t, root, "package", "update-all")
	if code != 0 || !strings.Contains(out, "STATUS") || !strings.Contains(out, "skipped") || !strings.Contains(out, "local/project-21") {
		t.Fatalf("human bulk table: %d %s", code, out)
	}
}

func TestPackageUpdateAllPreservesConcurrentLocalSelection(t *testing.T) {
	plan, _ := reportingRelease(t)
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	const pkg = "proof/install-reporting"
	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	defer server.Close()
	prior := updateAllInstall(t, layout, store, 1, pkg, "1.0.0", "tensorhub", server.URL)
	replacement := cleanupTestInstall(layout, "dddddddddddddddd", "2.0.0")
	replacement.Package, replacement.SourceKind, replacement.Hub = pkg, "local", server.URL
	must(t, os.MkdirAll(replacement.Dir, 0700))
	mux.HandleFunc("GET /v1/packages/proof/install-reporting", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "install-reporting"}, Releases: []hub.ReleaseSummary{{Release: "1.0.1"}}})
	})
	mux.HandleFunc("POST /v1/packages/proof/install-reporting/download", func(w http.ResponseWriter, _ *http.Request) {
		// Another installer switches this package after bulk inventory selection.
		if _, problem := store.Activate(replacement); problem != nil {
			t.Errorf("concurrent pin: %v", problem)
			w.WriteHeader(500)
			return
		}
		_ = json.NewEncoder(w).Encode(plan)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("changed selection reached installation: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(500)
	})
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	code, out := runCozy(t, root, "package", "update-all", "--json")
	if code != 1 || !strings.Contains(out, `"error_code":"package.update_changed"`) {
		t.Fatalf("concurrent selection was not refused: %d %s", code, out)
	}
	_, active, problem := store.ActivePackage(server.URL, pkg)
	fatal(t, problem)
	if active == nil || active.ID != replacement.ID || active.SourceKind != "local" {
		t.Fatalf("concurrent local choice was overwritten: %+v", active)
	}
	if _, err := os.Stat(filepath.Join(prior.Dir, "prior.py")); err != nil {
		t.Fatalf("bulk update removed previous bytes: %v", err)
	}
}
