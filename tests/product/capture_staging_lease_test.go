package producttest

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// A real PEP517 build hook holds capture at the same point where H3's longer
// metadata/build work was interrupted by daemon startup. No inference is simulated.
func TestRemoteCaptureSurvivesDaemonStartupWhileBuildingAppWheelClosure(t *testing.T) {
	root, project := t.TempDir(), t.TempDir()
	library := filepath.Join(project, "library")
	caller := filepath.Join(project, "caller")
	for _, path := range []string{library, caller} {
		must(t, os.MkdirAll(path, 0700))
	}
	write := func(dir, name, body string) {
		t.Helper()
		must(t, os.WriteFile(filepath.Join(dir, name), []byte(body), 0600))
	}
	metadata := func(name, module string) string {
		return fmt.Sprintf(`[project]
name=%q
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s", "msgspec"]
[project.entry-points."cozy.application"]
default=%q
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=[%q]
`, name, hostruntime.PackageFloor, module+":app", module+".py")
	}
	write(library, "pyproject.toml", metadata("capture-library-proof", "capture_library"))
	write(library, "capture_library.py", `import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    value: int = 7
class Result(msgspec.Struct):
    value: int
def helper(value):
    return value + 1
@invocable(memoize=True)
async def leaf(ctx: Context, *, value: int) -> Result:
    return Result(helper(value))
app.job(leaf)
`)
	wheels := filepath.Join(project, "wheels")
	build := exec.Command("uv", "build", "--wheel", "--out-dir", wheels, library)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build dependency: %v %s", err, out)
	}
	wheel := filepath.Join(wheels, "capture_library_proof-0.1.0-py3-none-any.whl")
	declared := strings.Replace(metadata("capture-parent-proof", "capture_parent"), `"msgspec"]`, `"msgspec", "capture-library-proof"]`, 1)
	declared += fmt.Sprintf("\n[tool.uv.sources]\ncapture-library-proof={path=%q}\n[tool.hatch.build.hooks.custom]\npath='build_hook.py'\n", wheel)
	write(caller, "pyproject.toml", declared)
	write(caller, "package.toml", "[application]\nobject='capture_parent:app'\n")
	body := `import msgspec
from cozy_runtime.author import App
from capture_library import helper
app = App()
class Request(msgspec.Struct):
    value: int = 7
class Result(msgspec.Struct):
    value: int
@app.job
def verify(payload: Request) -> Result:
    return Result(helper(payload.value))
`
	write(caller, "capture_parent.py", body)
	hold, entered := filepath.Join(project, "hold"), filepath.Join(project, "entered")
	write(caller, "build_hook.py", fmt.Sprintf(`from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from pathlib import Path
import time
class CustomHook(BuildHookInterface):
    def initialize(self, version, build_data):
        hold = Path(%q)
        if hold.exists():
            Path(%q).touch()
            while hold.exists():
                time.sleep(0.01)
`, hold, entered))
	if code, out := runCozy(t, root, "run", caller+"/verify", "--describe", "--json"); code != 0 {
		t.Fatalf("local describe: %d %s", code, out)
	}
	// Different root code needs a new capture, while unchanged dependency metadata
	// must reuse the prior local metadata environment without creating another venv.
	write(caller, "capture_parent.py", strings.Replace(body, "helper(payload.value))", "helper(payload.value) + 10)", 1))
	must(t, os.WriteFile(hold, []byte("build in progress"), 0600))
	command := exec.Command(cozyBin, "run", caller+"/verify", "--rental-only", "--describe", "--json")
	command.Env = childEnv(t, root, "CUDA_VISIBLE_DEVICES=")
	setProcessGroup(command)
	var output bytes.Buffer
	command.Stdout, command.Stderr = &output, &output
	must(t, command.Start())
	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	t.Cleanup(func() { _ = os.Remove(hold); _ = killGroup(command.Process.Pid) })
	deadline := time.NewTimer(45 * time.Second)
	defer deadline.Stop()
	for {
		if _, err := os.Stat(entered); err == nil {
			break
		}
		select {
		case err := <-done:
			t.Fatalf("capture ended before build barrier: %v %s", err, output.String())
		case <-deadline.C:
			t.Fatal("capture did not reach build barrier")
		case <-time.After(10 * time.Millisecond):
		}
	}
	layout := home.Paths(root)
	var held []string
	entries, err := os.ReadDir(layout.Tmp)
	must(t, err)
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "invocation-source-") || strings.HasPrefix(entry.Name(), "child-interfaces-") {
			held = append(held, filepath.Join(layout.Tmp, entry.Name()))
		}
	}
	if len(held) != 2 {
		t.Fatalf("expected both active capture roots, got %v", held)
	}
	startDaemonProcess(t, root)
	for _, path := range held {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("daemon removed live capture %s: %v", path, err)
		}
	}
	must(t, os.Remove(hold))
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("capture after sweep: %v %s", err, output.String())
		}
	case <-time.After(45 * time.Second):
		t.Fatal("capture did not finish after release")
	}
	for _, path := range held {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("capture did not release its own temporary bytes: %s: %v", path, err)
		}
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	installs, problem := store.SourceEnvironments("local/capture-parent-proof", "0.1.0")
	fatal(t, problem)
	var local, remote bool
	for _, inst := range installs {
		if _, err := os.Stat(filepath.Join(inst.Dir, "venv", "pyvenv.cfg")); err == nil {
			local = true
		} else if os.IsNotExist(err) {
			remote = true
		}
	}
	if !local || !remote {
		t.Fatalf("did not retain prior environment and remote capture: local=%t remote=%t", local, remote)
	}
	if _, err := os.Stat(filepath.Join(caller, "uv.lock")); !os.IsNotExist(err) {
		t.Fatalf("capture changed author lock: %v", err)
	}
}

func TestProjectMetadataRefusalNamesTheLostPreparedFile(t *testing.T) {
	project, _, _ := directoryProofProject(t)
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version=1\n"), 0600))
	prepared, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	defer prepared.Close()
	path := filepath.Join(project, "pyproject.toml")
	must(t, os.Remove(path))
	_, readError := os.ReadFile(path)
	problem = prepared.Build(context.Background())
	if problem == nil || problem.ErrName() != "project_metadata_unreadable" || !strings.Contains(problem.Message, path) || !strings.Contains(problem.Message, readError.Error()) {
		t.Fatalf("lost source metadata refusal omits its path/cause: %v", problem)
	}
}
