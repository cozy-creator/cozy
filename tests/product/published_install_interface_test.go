package producttest

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
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
	// 1.0.3 resolves at the Hub, but its machine finds no such release there.
	plans["1.0.3"] = hub.PackageDownloadPlan{Release: "1.0.3", Downloads: plans["1.0.2"].Downloads}
	logged := func(line string) bool {
		log, err := os.ReadFile(filepath.Join(root, "daemon.log"))
		must(t, err)
		return regexp.MustCompile(`(?m)^\S+Z ` + line + `$`).Match(log)
	}
	// Interrupting the command detaches it from 1.0.0's installation, which goes on.
	interrupted := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "package", "install", "proof/install-interface-probe", "--version", "1.0.0", "--rental=tessa")
	interrupted.Env = childEnv(t, root)
	stderr, err := interrupted.StderrPipe()
	must(t, err)
	var stdout bytes.Buffer
	interrupted.Stdout = &stdout
	must(t, interrupted.Start())
	t.Cleanup(func() { _ = interrupted.Process.Kill() })
	progress := bufio.NewReader(stderr)
	for line := ""; !strings.HasPrefix(line, "tessa: "); {
		if line, err = progress.ReadString('\n'); err != nil {
			t.Fatalf("the install did not report its progress: %v", err)
		}
	}
	must(t, interrupted.Process.Signal(os.Interrupt))
	rest, _ := io.ReadAll(progress)
	t.Logf("interrupted install:\n%s%s", rest, &stdout)
	detached := regexp.MustCompile(`detached from installation (\S+); it continues on tessa`).FindSubmatch(rest)
	if err := interrupted.Wait(); err != nil || detached == nil || !regexp.MustCompile(`queued|installing`).Match(stdout.Bytes()) {
		t.Fatalf("the interrupt did not detach: %v\n%s\n%s", err, rest, &stdout)
	}
	eventually(t, root, "the detached installation settles", func() bool {
		row, problem := store.RentalInstall(string(detached[1]))
		fatal(t, problem)
		return row != nil && row.State == "succeeded"
	})
	// Each later install follows its installation to its outcome.
	for _, version := range []string{"1.0.1", "1.0.2", "1.0.2"} {
		started := time.Now()
		code, out := runCozy(t, root, "package", "install", "proof/install-interface-probe", "--version", version, "--rental=tessa", "--json")
		var installed struct{ ID, Status, Target string }
		if code != 0 || json.Unmarshal([]byte(out), &installed) != nil || installed.Status != "succeeded" || installed.Target != "proof/install-interface-probe@"+version {
			t.Fatalf("install %s: %d %s", version, code, out)
		}
		t.Logf("ordinary published install %s: %s", version, time.Since(started))
		if !logged(regexp.QuoteMeta(installed.ID + " on " + parityRental + " succeeded: proof/install-interface-probe@" + version)) {
			t.Fatalf("the daemon did not log %s's outcome:\n%s", version, tail(filepath.Join(root, "daemon.log")))
		}
	}
	for _, version := range []string{"1.0.0", "1.0.1", "1.0.2"} {
		if _, read := machineReads.Load("/v1/packages/proof/install-interface-probe/releases/" + version + "/locked-requirements"); !read {
			t.Fatalf("the machine did not install %s by name at its Hub", version)
		}
	}
	if body, err := os.ReadFile(imports); !os.IsNotExist(err) {
		t.Fatalf("installing imported the package: %q %v", body, err)
	}
	// What the rental's machine holds is listed by its name.
	code, out := runCozy(t, root, "package", "list", "--rental=tessa", "--json")
	if code != 0 || !strings.Contains(out, `"package":"proof/install-interface-probe","release":"1.0.2"`) {
		t.Fatalf("package list --rental: %d %s", code, out)
	}
	// A failed installation ends the command with the machine's reason, and the daemon logs it.
	code, out = runCozy(t, root, "package", "install", "proof/install-interface-probe", "--version", "1.0.3", "--rental=tessa", "--json")
	t.Logf("failed install [exit %d]: %s", code, out)
	if code == 0 || !strings.Contains(out, "tessa: ") {
		t.Fatalf("a failed install did not end with its reason: %d %s", code, out)
	}
	if !logged(`rental-install-\S+ on ` + parityRental + ` failed: proof/install-interface-probe@1\.0\.3: .+`) {
		t.Fatalf("the daemon did not log the failure:\n%s", tail(filepath.Join(root, "daemon.log")))
	}
}
