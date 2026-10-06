package producttest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Real CPU tensor jobs exercise immutable dependency capture without pretending
// that a CPU runner qualifies CUDA serving. Both independent scripts run via CLI.
func TestEditedScriptsReuseImmutableDependencies(t *testing.T) {
	integration(t)
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact Runtime wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	root, err := os.MkdirTemp("", "cozy-dependency-reuse-")
	must(t, err)
	control := filepath.Join(t.TempDir(), "control")
	python := filepath.Join(control, "bin", "python")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", python, wheel, "numpy>=1.26"}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("CPU SDK: %v %s", err, out)
		}
	}
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("immutable dependency evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := t.TempDir()
	library := filepath.Join(project, "library")
	must(t, os.MkdirAll(library, 0700))
	module := `import msgspec
from cozy_runtime.author import App, Context, invocable
app=App()
class Result(msgspec.Struct):
    value: int
@invocable
async def double(ctx: Context, *, value: int) -> Result:
    import numpy as np
    return Result(int(np.asarray([value, value]).sum()))
app.job(double)
`
	must(t, os.WriteFile(filepath.Join(library, "numerical_tools.py"), []byte(module), 0600))
	metadata := fmt.Sprintf(`[project]
name="numerical-tools"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s", "numpy>=1.26"]
[project.entry-points."cozy.application"]
default="numerical_tools:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["numerical_tools.py"]
[tool.uv.sources]
cozy-runtime={path=%s}
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(library, "package.toml"), []byte("[application]\nobject=\"numerical_tools:app\"\n"), 0600))
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s","numerical-tools>=1.0.0"]
# [tool.uv.sources]
# cozy-runtime={path=%s}
# numerical-tools={path="./library",editable=true}
# ///
from numerical_tools import double
async def main(ctx):
    first=await double(value=2)
    second=await double(value=3)
    assert first.value==4 and second.value==6
`, version, strconv.Quote(wheel))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	computations := map[int]string{}
	for i := range 2 {
		script := filepath.Join(project, fmt.Sprintf("caller%d.py", i))
		must(t, os.WriteFile(script, []byte(body+fmt.Sprintf("    ctx.log('independent caller %d')\n", i)), 0600))
		if status, out := runCozyPath(t, root, path, "run", script, "--await", "--json"); status != 0 {
			t.Fatalf("caller %d [%d]: %s", i, status, out)
		}
		calls := machineChildren(t, root, store, strconv.Itoa(i+1))
		if len(calls) != 2 {
			t.Fatalf("expected two Runtime-owned numerical calls: %+v", calls)
		}
		for j, call := range calls {
			if call.Executions != 1 || call.State != "succeeded" {
				t.Fatalf("numerical job did not execute: %+v", call)
			}
			assertMachineChildScalar(t, call, 4+j*2)
			if computations[j] == "" {
				computations[j] = call.Computation
			} else if computations[j] != call.Computation {
				t.Fatal("edited caller changed an unchanged library call's identity")
			}
		}
	}
	// The machine installs each captured script as an ordinary uv installation; the public
	// NumPy wheel's bytes come from uv's cache, so every installation links one inode.
	installations := machineInstallations(root)
	libraries, err := filepath.Glob(filepath.Join(installations, "*", "env", "lib", "python3.12", "site-packages", "numpy", "_core", "_multiarray_umath*.so"))
	must(t, err)
	if len(libraries) < 2 {
		t.Fatalf("expected an installation per captured script, found %v", libraries)
	}
	first, err := os.Stat(libraries[0])
	must(t, err)
	for _, path := range libraries {
		current, err := os.Stat(path)
		must(t, err)
		if !os.SameFile(first, current) {
			t.Fatal("edited capture copied immutable NumPy bytes again")
		}
	}
	// Editable author source has a different owner from uv's public wheel cache.
	// Its frozen executable may never alias that mutable source inode.
	source, err := os.Stat(filepath.Join(library, "numerical_tools.py"))
	must(t, err)
	capturedModules, err := filepath.Glob(filepath.Join(installations, "*", "source", "numerical_tools.py"))
	must(t, err)
	installed, err := filepath.Glob(filepath.Join(installations, "*", "env", "lib", "python3.12", "site-packages", "numerical_tools.py"))
	must(t, err)
	capturedModules = append(capturedModules, installed...)
	if len(capturedModules) == 0 {
		t.Fatal("editable numerical library has no frozen executable")
	}
	for _, path := range capturedModules {
		captured, err := os.Stat(path)
		must(t, err)
		if os.SameFile(source, captured) {
			t.Fatal("frozen executable aliases mutable editable source")
		}
	}
	t.Logf("%d installations share uv's %d-byte NumPy inode; editable source remains separate", len(libraries), first.Size())
	compositionDown(t, root, path)
	must(t, removeAllForce(control))
	must(t, removeAllForce(filepath.Join(root, "local-packages")))
	for _, library := range libraries {
		generation := library
		for range 6 {
			generation = filepath.Dir(generation)
		}
		retainedPython := filepath.Join(generation, "bin", "python")
		if out, err := exec.Command(retainedPython, "-I", "-c", "import numpy as np; assert np.asarray([2,3]).sum()==5").CombinedOutput(); err != nil {
			t.Fatalf("retained dependency after source removal: %v %s", err, out)
		}
	}
}
