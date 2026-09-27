package producttest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

// Package installation is code/environment-only. Model weights are never
// fetched as a side effect; callers use explicit rental preparation or the
// exact inference request instead.
func TestPublishedInstallDoesNotPrefetchModels(t *testing.T) {
	plan, wheel := reportingRelease(t)
	for _, mode := range []string{"new", "superseded", "retained", "unsafe-prior", "before-install"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			prior := cleanupTestInstall(layout, "cccccccccccccccc", "1.0.0")
			prior.Package, prior.SourceRef = "proof/install-reporting", "proof/install-reporting@1.0.0"
			if mode == "unsafe-prior" {
				prior.Dir = filepath.Join(root, "outside-install-root")
			}
			if mode != "new" {
				writeReadOnlyPackage(t, filepath.Join(prior.Dir, "venv", "lib", "python3.12", "site-packages", "prior"))
				_, problem = store.Activate(prior)
				fatal(t, problem)
				if mode == "retained" {
					_, _, problem = store.Submit(records.Request{ID: "req-keep-prior", IdemKey: "keep-prior",
						BodyDigest: "sha256:" + strings.Repeat("1", 64), Package: prior.Package,
						Entrypoint: "compute", Payload: []byte("{}"), InstallID: prior.ID})
					fatal(t, problem)
				}
			}
			store.Close()
			t.Cleanup(func() {
				restoreDirectoryWrites(filepath.Join(prior.Dir, "venv", "lib", "python3.12", "site-packages", "prior"))
			})

			var modelRequests atomic.Int32
			mux := http.NewServeMux()
			mux.HandleFunc("POST /v1/packages/proof/install-reporting/download", func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(plan)
			})
			mux.HandleFunc("GET /v1/index/proof/simple/install-reporting/", func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `<a href="/project.whl/%s#sha256=%x">%s</a>`, plan.Downloads[0].Path, sha256.Sum256(wheel), plan.Downloads[0].Path)
			})
			mux.HandleFunc("GET /project.whl/{name}", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) })
			mux.HandleFunc("GET /v1/packages/proof/install-reporting/bindings", func(w http.ResponseWriter, _ *http.Request) {
				modelRequests.Add(1)
				http.Error(w, "package install must not request model bindings", http.StatusInternalServerError)
			})
			server := httptest.NewServer(mux)
			defer server.Close()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
			if mode == "before-install" {
				writer, problem := install.Lock(layout)
				fatal(t, problem)
				defer writer.Unlock()
			}
			code, stdout, stderr := runCozyStreams(t, root, "package", "install", "proof/install-reporting", "--version=1.0.1", "--json", "--full")
			if mode == "before-install" {
				if code == 0 || !strings.Contains(stdout, `"code":"conflict"`) {
					t.Fatalf("pre-install writer refusal lost: %d %s %s", code, stdout, stderr)
				}
			} else {
				if modelRequests.Load() != 0 {
					t.Fatal("package installation requested model bindings")
				}
				if code != 0 || !strings.Contains(stdout, `"status":"installed"`) {
					t.Fatalf("committed install reported failure: %d %s %s", code, stdout, stderr)
				}
				if mode == "unsafe-prior" && !strings.Contains(stdout, "cleanup of the prior version was deferred") {
					t.Fatalf("cleanup refusal lost its warning: %s", stdout)
				}
				if mode == "retained" {
					writer, problem := install.Lock(layout)
					fatal(t, problem)
					defer writer.Unlock()
					// With a competing writer held, replaying the exact install
					// must remain read-only and must not force reclamation of its prior.
					code, replay, stderr := runCozyStreams(t, root, "package", "install", "proof/install-reporting",
						"--version=1.0.1", "--json", "--full")
					if code != 0 || !strings.Contains(replay, `"status":"already installed"`) || strings.Contains(replay, `"reclaimed"`) {
						t.Fatalf("idempotent install needs no writer or cleanup: %d %s %s", code, replay, stderr)
					}
				}
			}
			store, problem = records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			_, active, problem := store.ActivePackage(prior.Package)
			fatal(t, problem)
			want := "1.0.1"
			if mode == "before-install" {
				want = "1.0.0"
			}
			if active == nil || active.Version != want {
				t.Fatalf("active install = %+v, want %s", active, want)
			}
			_, err := os.Stat(prior.Dir)
			if mode == "superseded" && !os.IsNotExist(err) {
				t.Fatalf("unreferenced prior install was not reclaimed: %v", err)
			}
			if (mode == "retained" || mode == "unsafe-prior" || mode == "before-install") && err != nil {
				t.Fatalf("referenced or active prior install was removed: %v", err)
			}
		})
	}
}

func reportingRelease(t *testing.T) (hub.PackageDownloadPlan, []byte) {
	return reportingReleaseWith(t, "")
}

// reportingReleaseWith builds the real published fixture with extra pyproject text.
func reportingReleaseWith(t *testing.T, extra string) (hub.PackageDownloadPlan, []byte) {
	t.Helper()
	project := t.TempDir()
	pyproject := `[project]
name = "install-reporting"
version = "1.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>=0.16.8"]
[project.entry-points."cozy.application"]
default = "reporting:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["reporting.py"]
` + extra
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(pyproject), 0600))
	must(t, os.WriteFile(filepath.Join(project, "reporting.py"), []byte("from cozy_runtime.author import App\napp = App()\n"), 0600))
	for _, args := range [][]string{{"lock"}, {"build", "--wheel", "--out-dir", "dist"}} {
		cmd := exec.Command("uv", args...)
		cmd.Dir, cmd.Env = project, childEnv(t, project)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("build real published fixture: %v\n%s", err, out)
		}
	}
	filename := "install_reporting-1.0.1-py3-none-any.whl"
	wheel, err := os.ReadFile(filepath.Join(project, "dist", filename))
	must(t, err)
	lock, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	exact := func(raw []byte) hub.ExactDocument {
		return hub.ExactDocument{CanonicalBytes: raw, Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(raw)), Length: int64(len(raw))}
	}
	return hub.PackageDownloadPlan{Release: "1.0.1", PackageConfig: exact([]byte("[application]\nobject = \"reporting:app\"\n")),
		PackageInterface: exact([]byte(`{"application":"reporting:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[]}`)),
		Pyproject:        exact([]byte(pyproject)), UVLock: exact(lock), Downloads: []hub.PackageInstallDownload{{
			Kind: "project_wheel", Path: filename, Distribution: "install-reporting", Version: "1.0.1",
			Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(wheel)), Length: int64(len(wheel)),
			Tags: []string{"py3-none-any"}, ImportRoots: []string{"reporting"},
		}}}, wheel
}

// A release whose pyproject restricts versions with uv constraint-dependencies installs:
// the lock already applied them, and they are never unpinned rows of the hash-pinned
// install.
func TestPublishedInstallWithUVConstraintDependencies(t *testing.T) {
	plan, wheel := reportingReleaseWith(t, "[tool.uv]\nconstraint-dependencies = [\"packaging<99\", \"numpy<3\"]\n")
	root := t.TempDir()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/packages/proof/install-reporting/download", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(plan)
	})
	mux.HandleFunc("GET /v1/index/proof/simple/install-reporting/", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `<a href="/project.whl/%s#sha256=%x">%s</a>`, plan.Downloads[0].Path, sha256.Sum256(wheel), plan.Downloads[0].Path)
	})
	mux.HandleFunc("GET /project.whl/{name}", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) })
	server := httptest.NewServer(mux)
	defer server.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	code, stdout, stderr := runCozyStreams(t, root, "package", "install", "proof/install-reporting", "--version=1.0.1", "--json")
	if code != 0 || !strings.Contains(stdout, `"status":"installed"`) {
		t.Fatalf("a release with uv constraint-dependencies did not install: %d %s %s", code, stdout, stderr)
	}
}
