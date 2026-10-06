package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/calcifer"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRemoteSourceChildReusesItsOwnRetainedEnvironment(t *testing.T) {
	cfg, problem := config.Load()
	fatal(t, problem)
	root := t.TempDir()
	parent, child := filepath.Join(root, "parent"), filepath.Join(root, "child")
	write := func(directory, name, body string) {
		t.Helper()
		must(t, os.MkdirAll(directory, 0700))
		must(t, os.WriteFile(filepath.Join(directory, name), []byte(body), 0600))
	}
	metadata := func(name, module, dependencies string) string {
		return fmt.Sprintf(`[project]
name = %q
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = [%s]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = [%q]
`, name, dependencies, module+".py")
	}
	write(child, "pyproject.toml", metadata("source-child", "source_child", `"cozy-runtime>=0.18.32"`))
	write(child, "package.toml", "[application]\nobject='source_child:app'\n")
	write(child, "source_child.py", `from cozy_runtime.author import App, Context, invocable
import msgspec
class Result(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    value: int
@invocable()
async def compute(ctx: Context, *, value: int) -> Result:
    return Result(value)
app = App()
app.job(compute)
`)
	write(parent, "pyproject.toml", metadata("source-parent", "source_parent", `"source-child"`)+"\n[tool.uv.sources]\nsource-child={path='../child',editable=true}\n")
	write(parent, "package.toml", "[application]\nobject='source_parent:app'\n")
	write(parent, "source_parent.py", `from cozy_runtime.author import App
from source_child import Result, compute
import msgspec
class Input(msgspec.Struct):
    value: int = 1
app = App()
@app.job
async def main(payload: Input) -> Result:
    return await compute(value=payload.value)
`)
	for _, directory := range []string{child, parent} {
		write(directory, ".python-version", "3.12\n")
		command := exec.Command("uv", "lock", "--python", "3.12")
		command.Dir, command.Env = directory, cfg.Tool()
		if out, err := command.CombinedOutput(); err != nil {
			t.Fatalf("lock: %s: %s", err, out)
		}
	}
	layout, problem := home.Open(filepath.Join(root, "records"))
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	_, problem = install.Run(layout, store, install.Request{Ref: install.Ref{Package: "local/source-parent"}, Local: &install.LocalSource{Tree: parent, Package: "local/source-parent", Release: "1.0.0"}})
	fatal(t, problem)
	var mu sync.Mutex
	var queued []records.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			return
		}
		if r.Method == "POST" && r.URL.Path == "/v1/local/jobs" {
			var submission api.JobSubmission
			if err := json.NewDecoder(r.Body).Decode(&submission); err != nil {
				t.Error(err)
				http.Error(w, "decode", 500)
				return
			}
			id := records.NewID("job")
			row, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: strings.Repeat("1", 64), Package: submission.Package, Entrypoint: submission.Function, Kind: "job", InstallID: submission.InstallID, Payload: submission.Input})
			if problem != nil {
				t.Error(problem)
				http.Error(w, "submit", 500)
				return
			}
			mu.Lock()
			queued = append(queued, row)
			number := len(queued)
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(api.JobHandle{Number: int64(number), JobID: id, Package: row.Package, Function: row.Entrypoint, Status: "queued", MachineExecution: true})
			return
		}
		if r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/v1/local/jobs/") {
			id := strings.TrimPrefix(r.URL.Path, "/v1/local/jobs/")
			row, problem := store.RequestRow(id)
			if problem != nil || row == nil {
				http.Error(w, "missing", 404)
				return
			}
			_ = json.NewEncoder(w).Encode(api.JobState{JobID: id, Package: row.Package, Function: row.Entrypoint, Status: "queued", Stage: "captured source queued", MachineExecution: &api.MachineExecutionView{Machine: "proof"}})
			return
		}
		t.Errorf("unexpected execution/network request: %s", r.URL)
		http.Error(w, "unexpected", 500)
	}))
	defer server.Close()
	held, problem := calcifer.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)
	var firstRoot, firstChild *records.PackageInstall
	queue := func(remote string) (*records.PackageInstall, *records.PackageInstall) {
		t.Helper()
		code, out := runCozy(t, layout.Root, "run", "local/source-parent/main", remote, "--json", "--full")
		if code != 0 || !strings.Contains(out, `"queued"`) {
			t.Fatalf("durable queued CLI [%d]: %s", code, out)
		}
		mu.Lock()
		row := queued[len(queued)-1]
		mu.Unlock()
		if !strings.Contains(out, row.ID) {
			t.Fatalf("CLI did not return its durable queued ID: %s", out)
		}
		rootInstall, problem := store.Install(row.InstallID)
		fatal(t, problem)
		bindings, problem := store.ChildBindings(row.InstallID)
		fatal(t, problem)
		for _, binding := range bindings {
			if binding.Module == "source_child" {
				childInstall, problem := store.Install(binding.ChildInstallID)
				fatal(t, problem)
				return rootInstall, childInstall
			}
		}
		t.Fatal("source child lost its accepted binding")
		return nil, nil
	}
	for run := 0; run < 3; run++ {
		began := time.Now()
		remote := "--rental-only"
		if run == 2 {
			remote = "--rent-new"
		}
		rootInstall, childInstall := queue(remote)
		if strings.Contains(childInstall.Closure, "source-parent==") {
			t.Fatal("child inherited its caller's dependency closure")
		}
		if run == 0 {
			firstRoot, firstChild = rootInstall, childInstall
			if _, err := os.Stat(filepath.Join(firstChild.Dir, "venv", "pyvenv.cfg")); err != nil {
				t.Fatal("initial child did not retain its own preparation environment", err)
			}
			// A completed caller's GC keeps the newest local environment for its
			// package/version; the child's recorded environment remains usable for
			// metadata reads under the next capture's writer lock.
			mu.Lock()
			first := queued[len(queued)-1].ID
			mu.Unlock()
			fatal(t, store.SettleRequest(first, "canceled"))
			_, problem = install.Reclaim(layout, store, rootInstall.ID)
			fatal(t, problem)
			if prior, problem := store.Install(rootInstall.ID); problem != nil || prior == nil {
				t.Fatalf("newest local caller environment was not retained: %v", problem)
			}
			continue
		}
		// An unchanged tree runs the snapshot it already has: nothing is captured or rebuilt.
		if rootInstall.ID != firstRoot.ID || childInstall.ID != firstChild.ID {
			t.Fatalf("an unchanged tree was captured again: root %s (was %s), child %s (was %s)",
				rootInstall.ID, firstRoot.ID, childInstall.ID, firstChild.ID)
		}
		t.Logf("repeat public CLI queued durable run in %s on its existing snapshot", time.Since(began))
	}
	// A changed caller is captured anew, and its unchanged child's environment is reused.
	source, err := os.ReadFile(filepath.Join(parent, "source_parent.py"))
	must(t, err)
	write(parent, "source_parent.py", strings.Replace(string(source), "value: int = 1", "value: int = 2", 1))
	rootInstall, childInstall := queue("--rental-only")
	if rootInstall.ID == firstRoot.ID {
		t.Fatal("a changed caller reused its previous snapshot")
	}
	if childInstall.Closure != firstChild.Closure {
		t.Fatal("child environment reuse changed its selected dependencies")
	}
	for _, installation := range []*records.PackageInstall{rootInstall, childInstall} {
		if installation.ID != firstChild.ID {
			if _, err := os.Stat(filepath.Join(installation.Dir, "venv")); !os.IsNotExist(err) {
				t.Fatalf("remote run rebuilt local environment for %s: %v", installation.Package, err)
			}
		}
	}
}
