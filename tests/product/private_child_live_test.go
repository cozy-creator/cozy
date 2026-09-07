package producttest

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

var privateChildRuntimeWheel = flag.String("child-runtime-wheel", "", "exact Runtime40 wheel for actual private child composition")

// This uses the actual Creator binary, installed interface wheels, independent
// package executors and typed broker. No control peer or executor is simulated.
func TestPrivateChildCompositionReusesAAfterParentAndLibraryEdits(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires the paired Runtime40 wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	control := filepath.Join(t.TempDir(), "control")
	run := func(args ...string) {
		t.Helper()
		command := exec.Command("uv", args...)
		if out, err := command.CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	run("venv", control, "--python", "3.12")
	run("pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel)
	path := filepath.Join(control, "bin") + string(os.PathListSeparator) + os.Getenv("PATH")
	root, err := os.MkdirTemp("", "cozy-calls-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozyPath(t, root, path, "down", "--all"); _ = os.RemoveAll(root) })
	project := t.TempDir()
	for _, name := range []string{"source", "candidate"} {
		lib := filepath.Join(project, name)
		must(t, os.MkdirAll(lib, 0o700))
		module := "private_" + name
		metadata := fmt.Sprintf(`[project]
name = "private-%s"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime==0.2.31"]
[tool.uv.sources]
cozy-runtime = {path = %s}
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = [%q]
[project.entry-points."cozy.application"]
default = %q
`, name, strconv.Quote(wheel), module+".py", module+":app")
		must(t, os.WriteFile(filepath.Join(lib, "pyproject.toml"), []byte(metadata), 0o600))
		must(t, os.WriteFile(filepath.Join(lib, "package.toml"), []byte(fmt.Sprintf("[application]\nobject=%q\n", module+":app")), 0o600))
		body := `from importlib.metadata import distributions
if any((d.metadata.get("Name") or "").startswith("cozy-script-") for d in distributions()):
    raise RuntimeError("implementation imported into parent")
import msgspec
from cozy_runtime.author import App, Context, invocable
class Result(msgspec.Struct, frozen=True):
    value: int
@invocable(reusable=True)
`
		if name == "source" {
			body += "async def compute(ctx: Context, *, value: int) -> Result:\n    return Result(value + 100)\n"
		} else {
			body += "async def compute(ctx: Context, *, value: int, factor: int) -> Result:\n    if factor == 0:\n        raise ValueError('candidate quality gate failed')\n    return Result(value * factor)\n"
		}
		body += "app = App()\napp.job(compute)\n"
		must(t, os.WriteFile(filepath.Join(lib, module+".py"), []byte(body), 0o600))
		run("lock", "--project", lib)
	}
	script := filepath.Join(project, "recipe.py")
	code := `# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==0.2.31", "private-source==0.1.0", "private-candidate==0.1.0"]
# [tool.uv.sources]
# cozy-runtime = {path = ` + strconv.Quote(wheel) + `}
# private-source = {path = "./source"}
# private-candidate = {path = "./candidate"}
# ///
import msgspec
from cozy_runtime.author import App, Context
from private_source import compute as source
from private_candidate import compute as candidate
class Request(msgspec.Struct):
    value: int = 7
class Result(msgspec.Struct):
    value: int
app = App()
@app.job
async def main(request: Request, ctx: Context) -> Result:
    original = await source(ctx, value=request.value)
    improved = await candidate(ctx, value=original.value, factor=0)
    return Result(improved.value)
`
	must(t, os.WriteFile(script, []byte(code), 0o600))
	status, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status == 0 || !strings.Contains(out, "candidate quality gate failed") {
		t.Fatalf("first parent did not fail through child B [%d]: %s", status, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	first, problem := store.RequestByReference("1")
	fatal(t, problem)
	children, problem := store.Children(first.ID)
	fatal(t, problem)
	if first.State != "blocked" || len(children) != 2 || children[0].State != "succeeded" || children[1].State != "blocked" {
		t.Fatalf("first parent/children lost their state: %+v %+v", first, children)
	}
	originalA, originalB := children[0], children[1]
	originalBInstall, problem := store.Install(originalB.InstallID)
	fatal(t, problem)
	code = strings.Replace(code, "factor=0", "factor=2", 1)
	must(t, os.WriteFile(script, []byte(code), 0o600))
	status, out = runCozyPath(t, root, path, "run", script, "--retry", "1", "--await", "--json")
	if status != 0 || !strings.Contains(out, `"value":214`) {
		t.Fatalf("edited parent did not complete [%d]: %s", status, out)
	}
	second, problem := store.RequestByReference("4")
	fatal(t, problem)
	children, problem = store.Children(second.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].ReusedFrom != originalA.ID || children[0].Ordinal != 0 || children[1].Ordinal != 1 || children[1].ChildTargetDigest != originalB.ChildTargetDigest || children[1].ChildIntentDigest == originalB.ChildIntentDigest {
		t.Fatalf("parent edit did not reuse only A: %+v", children)
	}
	secondB := children[1]
	implementation := filepath.Join(project, "candidate", "private_candidate.py")
	raw, err := os.ReadFile(implementation)
	must(t, err)
	must(t, os.WriteFile(implementation, []byte(strings.Replace(string(raw), "return Result(value * factor)", "return Result(value * factor + 1)", 1)), 0o600))
	status, out = runCozyPath(t, root, path, "run", script, "--retry", "1", "--await", "--json")
	if status != 0 || !strings.Contains(out, `"value":215`) {
		t.Fatalf("same-version library edit did not execute [%d]: %s", status, out)
	}
	third, problem := store.RequestByReference("7")
	fatal(t, problem)
	children, problem = store.Children(third.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].ReusedFrom != originalA.ID || children[0].Ordinal != 0 || children[1].Ordinal != 1 || children[1].ChildTargetDigest == secondB.ChildTargetDigest || children[1].ChildIntentDigest != secondB.ChildIntentDigest {
		t.Fatalf("library edit did not invalidate exactly B: %+v", children)
	}
	old, problem := store.RequestRow(first.ID)
	fatal(t, problem)
	if old.State != "blocked" || old.BodyDigest != first.BodyDigest {
		t.Fatal("new parent rewrote the original transaction")
	}
	retained, err := os.ReadFile(filepath.Join(originalBInstall.SourceRef, "private_candidate.py"))
	must(t, err)
	if strings.Contains(string(retained), "value * factor + 1") {
		t.Fatal("library edit modified the original captured implementation")
	}
}
