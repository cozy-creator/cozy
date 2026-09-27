package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/records"
)

// The public base library has no project App entrypoint. The actual CLI must
// capture its fixed builtin, execute real native quantization, and give a fresh
// edited caller custody of the memoized result before its independent readback.
func TestRuntimeBuiltinQuantizeSurvivesCallerEdit(t *testing.T) {
	integration(t)
	version := runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	runtimeInstall, runtimeSource := "cozy-runtime>="+hostruntime.ToolFloor, ""
	if *privateChildRuntimeWheel != "" {
		wheel, err := filepath.Abs(*privateChildRuntimeWheel)
		must(t, err)
		runtimeInstall = wheel
		runtimeSource = "cozy-runtime = {path = " + strconv.Quote(wheel) + "}\n"
	}
	control := filepath.Join(t.TempDir(), "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin/python"), runtimeInstall}} {
		if output, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, output)
		}
	}
	root, err := os.MkdirTemp("", "cozy-builtin-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("retained builtin proof home", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := t.TempDir()
	library := filepath.Join(project, "library")
	must(t, os.Mkdir(library, 0700))
	metadata := fmt.Sprintf(`[project]
name="native-quantize-fixture"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[project.entry-points."cozy.application"]
default="runtime_quantize_source:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["runtime_quantize_source.py"]
[tool.uv.sources]
%s`, version, runtimeSource)
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	source, err := os.ReadFile(filepath.Join("testdata", "runtime_quantize_source.py"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(library, "runtime_quantize_source.py"), source, 0600))
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s", "native-quantize-fixture>=1.0.0"]
# [tool.uv.sources]
# native-quantize-fixture={path="./library", editable=true}
%s# ///
from cozy_runtime.derive.operations import QuantizationPlan, quantize
from runtime_quantize_source import produce, inspect
async def main(ctx):
    source = await produce()
    result = await quantize(source=source, plan=QuantizationPlan(components=("body",)), encoding="fp8-rowwise/1")
    checked = await inspect(original=source, candidate=result)
    assert checked.nonzero
`, version, strings.ReplaceAll(runtimeSource, "cozy-runtime =", "# cozy-runtime ="))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	unrelated := filepath.Join(project, "unrelated.py")
	coreScript := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s"]
# [tool.uv.sources]
%s# ///
async def main(ctx):
    ctx.log("This script does not quantize")
`, version, strings.ReplaceAll(runtimeSource, "cozy-runtime =", "# cozy-runtime ="))
	must(t, os.WriteFile(unrelated, []byte(coreScript), 0600))
	if status, output := runCozyPath(t, root, path, "run", unrelated, "--await", "--json"); status != 0 {
		t.Fatalf("unrelated script requires a numerical dependency [%d]: %s", status, output)
	}
	plain, problem := store.RequestByReference("1")
	fatal(t, problem)
	if plain == nil || plain.State != "succeeded" {
		t.Fatal("unrelated script did not complete")
	}
	bindings, problem := store.ChildBindings(plain.InstallID)
	fatal(t, problem)
	if len(bindings) != 0 {
		t.Fatal("unrelated script eagerly prepared a numerical operation")
	}
	assertBaseUnchanged := func() {
		t.Helper()
		if _, err := os.Stat(filepath.Join(control, "lib", "python3.12", "site-packages", "numpy")); !os.IsNotExist(err) {
			t.Fatal("operation preparation changed the NumPy-free base SDK")
		}
	}
	assertBaseUnchanged()
	builtinRoot := filepath.Join(root, "runtime", "environments", "runtime-builtins")
	if _, err := os.Stat(filepath.Join(builtinRoot, "numerical")); !os.IsNotExist(err) {
		t.Fatal("unrelated script eagerly prepared a numerical Runtime environment")
	}
	var first []machineChildProof
	for i := range 2 {
		script := filepath.Join(project, fmt.Sprintf("caller%d.py", i))
		current := body
		if i == 1 {
			current += "    ctx.log(\"The caller changed; the operation did not\")\n"
		}
		must(t, os.WriteFile(script, []byte(current), 0600))
		status, output := runCozyPath(t, root, path, "run", script, "--await", "--json")
		if status != 0 {
			t.Fatalf("builtin caller %d failed [%d]: %s", i, status, output)
		}
		reference := ""
		for _, line := range strings.Split(output, "\n") {
			var answer struct {
				Job string `json:"job"`
			}
			if json.Unmarshal([]byte(line), &answer) == nil && answer.Job != "" {
				reference = answer.Job
			}
		}
		if reference == "" {
			t.Fatal("CLI completion omitted its run reference")
		}
		parent, problem := store.RequestByReference(reference)
		fatal(t, problem)
		if parent == nil || parent.State != "succeeded" {
			t.Fatalf("missing completed caller: %+v", parent)
		}
		children := machineChildren(t, root, store, reference)
		if len(children) != 3 {
			t.Fatalf("expected Runtime source, builtin and fresh reader: %+v", children)
		}
		for _, child := range children {
			if child.State != "succeeded" {
				t.Fatalf("Runtime child failed: %+v", child)
			}
		}
		if children[2].Executions != 1 {
			t.Fatal("native readback was skipped")
		}
		for index, child := range children {
			t.Logf("Runtime caller %d child %d: request=%s computation=%s executions=%d state=%s", i, index, child.Request, child.Computation, child.Executions, child.State)
		}
		if i == 0 {
			first = children
			if first[0].Executions != 1 || first[1].Executions != 1 {
				t.Fatalf("Runtime did not execute source and builtin: %+v", first)
			}
		} else {
			for index := range 2 {
				if children[index].Executions != 0 || first[index].Computation == "" || first[index].Computation != children[index].Computation {
					t.Fatalf("caller edit invalidated Runtime computation: first=%+v edited=%+v", first, children)
				}
			}
		}
	}

	assertBaseUnchanged()
	// Runtime owns the on-demand numerical environment, while the caller SDK
	// remains NumPy-free. Inspect its installed carrier metadata independently.
	interpreters, err := filepath.Glob(filepath.Join(builtinRoot, "numerical", "contents", "*", "bin", "python"))
	must(t, err)
	if len(interpreters) != 1 {
		t.Fatalf("expected one Runtime numerical environment, got %v", interpreters)
	}
	out, err := exec.Command(interpreters[0], "-I", "-c", `import importlib.metadata as m
import numpy
from cozy_runtime.derive.operations import app
from cozy_runtime.author import describe
assert describe(app)
requirements = m.requires("cozy-runtime-operations")
assert [item for item in requirements if item.startswith("numpy")] == ["numpy>=1.26"], requirements
print("numpy==" + m.version("numpy"))
`).CombinedOutput()
	t.Logf("Runtime numerical callee: %s", out)
	if err != nil || !strings.Contains(string(out), "numpy==") {
		t.Fatalf("Runtime quantizer lost its numerical dependency or authored range: %v\n%s", err, out)
	}
}
