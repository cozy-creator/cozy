package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// The public install command reaches a real standalone agent and Runtime over the
// rental transport. Only the Hub and provider are loopback fixtures. Preparation
// installs packages with uv; it runs no inference and downloads no model weights. It never
// imports the package: the interface comes from the Hub's release detail, for releases
// with and without embedded metadata.
func TestPublishedInstallUsesEmbeddedInterfaceWithoutImport(t *testing.T) {
	h, root, _, store := parityMachines(t)
	imports := filepath.Join(t.TempDir(), "imports")
	objects := map[string][]byte{}
	files := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, found := objects[r.URL.Path]
		if !found {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(body)
	}))
	t.Cleanup(files.Close)
	releases := map[string][]byte{}
	plans := map[string]hub.PackageDownloadPlan{}
	for _, version := range []string{"1.0.0", "1.0.1", "1.0.2"} {
		project := t.TempDir()
		must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(fmt.Sprintf(`[project]
name = "install-interface-probe"
version = %q
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>=0.18.90"]
[project.entry-points."cozy.application"]
default = "install_probe:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["install_probe.py"]
`, version)), 0600))
		must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"install_probe:app\"\n"), 0600))
		must(t, os.WriteFile(filepath.Join(project, "install_probe.py"), []byte(fmt.Sprintf(`import time
from pathlib import Path
from cozy_runtime.author import App
import msgspec
time.sleep(4)
with Path(%q).open("a") as stream:
    stream.write(%q + "\n")
app = App()
class Request(msgspec.Struct):
    value: int = 1
class Result(msgspec.Struct):
    value: int
@app.job
def main(payload: Request) -> Result:
    return Result(payload.value)
`, imports, version)), 0600))
		if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
			t.Fatalf("locking fixture: %v\n%s", err, out)
		}
		pack, problem := packagepublish.PrepareFrom(project)
		fatal(t, problem)
		t.Cleanup(pack.Close)
		fatal(t, pack.Build(t.Context()))
		path := pack.Wheel
		if version != "1.0.2" {
			// Older published artifacts have no embedded metadata. Keep those bytes.
			dist := t.TempDir()
			if out, err := exec.Command("uv", "build", "--wheel", "--project", project, "-o", dist).CombinedOutput(); err != nil {
				t.Fatalf("building legacy fixture: %v\n%s", err, out)
			}
			path = filepath.Join(dist, "install_interface_probe-"+version+"-py3-none-any.whl")
		}
		body, err := os.ReadFile(path)
		must(t, err)
		objects["/"+filepath.Base(path)] = body
		digest := sha256.Sum256(body)
		closure, err := exec.Command("uv", "export", "--project", project, "--frozen", "--no-dev", "--no-emit-project", "--no-header").Output()
		must(t, err)
		locked := string(closure) + "install-interface-probe @ " + files.URL + "/" + filepath.Base(path) + " --hash=sha256:" + hex.EncodeToString(digest[:]) + "\n"
		iface, err := os.ReadFile(pack.PackageInterface)
		must(t, err)
		detail, err := json.Marshal(map[string]any{"release": map[string]string{"release": version}, "package_interface": json.RawMessage(iface), "requires_python": ">=3.12,<3.13"})
		must(t, err)
		prefix := "/v1/packages/proof/install-interface-probe/releases/" + version
		releases[prefix], releases[prefix+"/locked-requirements"] = detail, []byte(locked)
		plans[version] = hub.PackageDownloadPlan{Release: version, Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: filepath.Base(path)}}}
	}
	// The machine installs each release by name, reading its card and lock at its Hub (th-245).
	var machineReads sync.Map
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, found := releases[r.URL.Path]; found && r.Method == http.MethodGet {
			machineReads.Store(r.URL.Path, true)
			_, _ = w.Write(body)
			return
		}
		doors.ServeHTTP(w, r)
	})
	account := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, found := releases[r.URL.Path]; found && r.Method == http.MethodGet {
			_, _ = w.Write(body)
			return
		}
		if r.Method == "POST" && r.URL.Path == "/v1/packages/proof/install-interface-probe/download" {
			plan, found := plans[r.URL.Query().Get("release")]
			if !found {
				http.NotFound(w, r)
				return
			}
			_ = json.NewEncoder(w).Encode(plan)
			return
		}
		account.ServeHTTP(w, r)
	})
	for _, version := range []string{"1.0.0", "1.0.1", "1.0.2", "1.0.2"} {
		started := time.Now()
		code, out := runCozy(t, root, "package", "install", "proof/install-interface-probe", "--version", version, "--rental=tessa", "--json")
		var accepted struct{ ID string }
		if code != 0 || json.Unmarshal([]byte(out), &accepted) != nil || accepted.ID == "" {
			t.Fatalf("install %s: %d %s", version, code, out)
		}
		eventually(t, root, "accepted installation settles", func() bool {
			row, problem := store.RentalInstall(accepted.ID)
			fatal(t, problem)
			if row != nil && row.State == "failed" {
				t.Fatalf("installation failed: %s %s", row.ErrorCode, row.Error)
			}
			return row != nil && row.State == "succeeded"
		})
		t.Logf("ordinary published install %s: %s", version, time.Since(started))
		if _, read := machineReads.Load("/v1/packages/proof/install-interface-probe/releases/" + version + "/locked-requirements"); !read {
			t.Fatalf("the machine did not install %s by name at its Hub", version)
		}
		if body, err := os.ReadFile(imports); !os.IsNotExist(err) {
			t.Fatalf("installing %s imported the package: %q %v", version, body, err)
		}
	}
}
