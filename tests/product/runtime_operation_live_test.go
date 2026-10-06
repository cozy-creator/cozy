package producttest

import (
	"database/sql"
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
	unpressuredMachine(t, root)
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

	// The Runtime's operations are a callee App of the caller's own environment: quantize ran
	// as package runtime/operations in its caller's generation (its numerical dependencies are
	// the Runtime's own), not in an environment of its own.
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "machine/root/var/lib/cozy/rust-machine/execution/executions.sqlite3")+"?mode=ro")
	must(t, err)
	defer db.Close()
	rows, err := db.Query("SELECT id, invocation FROM executions ORDER BY id")
	must(t, err)
	defer rows.Close()
	generations := map[string]string{}
	var builtin []string
	for rows.Next() {
		var id string
		var raw []byte
		must(t, rows.Scan(&id, &raw))
		var invocation struct{ Package, Generation, Parent string }
		must(t, json.Unmarshal(raw, &invocation))
		generations[id] = invocation.Generation
		if invocation.Package == "runtime/operations" && invocation.Generation != "" {
			builtin = append(builtin, invocation.Parent+" "+invocation.Generation)
		}
	}
	must(t, rows.Err())
	if len(builtin) != 1 {
		t.Fatalf("quantize did not run once as runtime/operations: %v", builtin)
	}
	parent, generation, _ := strings.Cut(builtin[0], " ")
	if generations[parent] != generation {
		t.Fatalf("quantize ran outside its caller's environment: %s, its caller %s", generation, generations[parent])
	}
}
