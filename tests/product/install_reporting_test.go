package producttest

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

func reportingRelease(t *testing.T) (hub.PackageDownloadPlan, []byte) {
	return reportingReleaseWith(t, "")
}

// reportingReleaseWith builds the real published fixture with extra pyproject text.
func reportingReleaseWith(t *testing.T, extra string) (hub.PackageDownloadPlan, []byte) {
	return reportingReleaseWithDependencies(t, `"cozy-runtime>=0.16.8"`, extra)
}

// reportingReleaseWithDependencies builds the real published fixture with the given
// [project] dependency list items and extra pyproject text.
func reportingReleaseWithDependencies(t *testing.T, dependencies, extra string) (hub.PackageDownloadPlan, []byte) {
	t.Helper()
	project := t.TempDir()
	pyproject := `[project]
name = "install-reporting"
version = "1.0.1"
requires-python = ">=3.12,<3.13"
dependencies = [` + dependencies + `]
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

// A universal lock forks a distribution by environment marker, so its export names one
// distribution twice under different markers. The release installs, and the row whose
// marker matches this host is the one installed.
func TestPublishedInstallSelectsTheForkMatchingThisHost(t *testing.T) {
	plan, wheel := reportingReleaseWithDependencies(t,
		`"cozy-runtime>=0.16.8", "idna==3.6; sys_platform == 'win32'", "idna==3.7; sys_platform != 'win32'"`, "")
	if !strings.Contains(string(plan.UVLock.CanonicalBytes), "resolution-markers") {
		t.Fatalf("the fixture lock did not fork:\n%s", plan.UVLock.CanonicalBytes)
	}
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
		t.Fatalf("a release whose lock forks by marker did not install: %d %s %s", code, stdout, stderr)
	}
	var installed []string
	must(t, filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err == nil && entry.IsDir() && strings.HasPrefix(entry.Name(), "idna-") && strings.HasSuffix(entry.Name(), ".dist-info") {
			installed = append(installed, entry.Name())
		}
		return nil
	}))
	if len(installed) == 0 || slices.ContainsFunc(installed, func(name string) bool { return name != "idna-3.7.dist-info" }) {
		t.Fatalf("installed idna %v, want only the fork matching this host (3.7)", installed)
	}
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

// The org index carries an older release of a package the lock pins from PyPI. uv takes a name
// from the first index that has it, so an install resolving the closure through both indexes
// refused it (minimax-h3 1.26.0's diffusers==0.40.0 on both Hubs). Each locked package is
// taken from the source its uv.lock records.
func TestPublishedInstallTakesEachPackageFromItsLockedSource(t *testing.T) {
	plan, wheel := reportingReleaseWithDependencies(t, `"cozy-runtime>=0.16.8", "idna==3.7"`, "")
	root := t.TempDir()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/packages/proof/install-reporting/download", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(plan)
	})
	mux.HandleFunc("GET /v1/index/proof/simple/install-reporting/", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `<a href="/project.whl/%s#sha256=%x">%s</a>`, plan.Downloads[0].Path, sha256.Sum256(wheel), plan.Downloads[0].Path)
	})
	mux.HandleFunc("GET /project.whl/{name}", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) })
	// The org's own idna, older than the lock's: an index-resolved install stops here.
	mux.HandleFunc("GET /v1/index/proof/simple/idna/", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `<a href="/project.whl/idna-3.6-py3-none-any.whl#sha256=%s">idna-3.6-py3-none-any.whl</a>`, strings.Repeat("0", 64))
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	code, stdout, stderr := runCozyStreams(t, root, "package", "install", "proof/install-reporting", "--version=1.0.1", "--json")
	if code != 0 || !strings.Contains(stdout, `"status":"installed"`) {
		t.Fatalf("a locked package the org index also names was not taken from its locked source: %d %s %s", code, stdout, stderr)
	}
	var installed []string
	must(t, filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err == nil && entry.IsDir() && strings.HasPrefix(entry.Name(), "idna-") && strings.HasSuffix(entry.Name(), ".dist-info") {
			installed = append(installed, entry.Name())
		}
		return nil
	}))
	if !slices.Equal(installed, []string{"idna-3.7.dist-info"}) {
		t.Fatalf("installed idna %v, want the locked 3.7", installed)
	}
}
