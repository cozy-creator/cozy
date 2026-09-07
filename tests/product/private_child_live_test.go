package producttest

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

var privateChildRuntimeWheel = flag.String("child-runtime-wheel", "", "exact Runtime40 wheel for actual private child composition")

// This uses the actual Creator binary, installed interface wheels, independent
// package executors and typed broker. No control peer or executor is simulated.
func TestPrivateChildCompositionReusesLocalWorkspace(t *testing.T) {
	runtimeVersion := "0.4.0"
	runtimeInstall := "cozy-runtime==" + runtimeVersion
	runtimeSource := ""
	if *privateChildRuntimeWheel != "" {
		wheel, err := filepath.Abs(*privateChildRuntimeWheel)
		must(t, err)
		parts := strings.Split(filepath.Base(wheel), "-")
		if len(parts) < 3 || parts[0] != "cozy_runtime" {
			t.Fatal("child fixture requires a named Runtime wheel")
		}
		runtimeVersion = parts[1]
		runtimeInstall = wheel
		runtimeSource = "cozy-runtime = {path = " + strconv.Quote(wheel) + "}\n"
	}
	control := filepath.Join(t.TempDir(), "control")
	run := func(args ...string) {
		t.Helper()
		command := exec.Command("uv", args...)
		if out, err := command.CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	run("venv", control, "--python", "3.12")
	run("pip", "install", "--python", filepath.Join(control, "bin", "python"), runtimeInstall)
	root, err := os.MkdirTemp("", "cozy-calls-")
	must(t, err)
	defer tracePrivateChildWait(t, root)()
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
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
dependencies = ["cozy-runtime==%s"]
[tool.uv.sources]
%s
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = [%q]
[project.entry-points."cozy.application"]
default = %q
`, name, runtimeVersion, runtimeSource, module+".py", module+":app")
		must(t, os.WriteFile(filepath.Join(lib, "pyproject.toml"), []byte(metadata), 0o600))
		must(t, os.WriteFile(filepath.Join(lib, "package.toml"), []byte(fmt.Sprintf("[application]\nobject=%q\n", module+":app")), 0o600))
		body := `from importlib.metadata import distributions
if any((d.metadata.get("Name") or "").startswith("cozy-script-") for d in distributions()):
    raise RuntimeError("implementation imported into parent")
import msgspec
from cozy_runtime.author import App, Context, invocable
class Result(msgspec.Struct, frozen=True):
    value: int
@invocable(memoize=True)
`
		if name == "source" {
			body += "async def compute(ctx: Context, *, value: int) -> Result:\n    return Result(value + 100)\n"
		} else {
			body += "async def compute(ctx: Context, *, value: int, factor: int) -> Result:\n    if factor == 0:\n        raise ValueError('candidate quality gate failed')\n    return Result(value * factor)\n"
		}
		body += "app = App()\napp.job(compute)\n"
		must(t, os.WriteFile(filepath.Join(lib, module+".py"), []byte(body), 0o600))
	}
	script := filepath.Join(project, "recipe.py")
	runtimeScriptSource := ""
	if runtimeSource != "" {
		runtimeScriptSource = "# " + runtimeSource
	}
	code := `# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==` + runtimeVersion + `", "private-source==0.1.0", "private-candidate==0.1.0"]
# [tool.uv.sources]
` + runtimeScriptSource + `# private-source = {path = "./source"}
# private-candidate = {path = "./candidate"}
# ///
from private_source import compute as source
from private_candidate import compute as candidate
async def main():
    original = await source(value=7)
    await candidate(value=original.value, factor=0)
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
	if status != 0 {
		t.Fatalf("edited parent did not complete [%d]: %s", status, out)
	}
	second, problem := store.RequestByReference("4")
	fatal(t, problem)
	children, problem = store.Children(second.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].ReusedFrom != originalA.ID || children[0].Ordinal != 0 || children[0].ChildTargetDigest != originalA.ChildTargetDigest || children[1].Ordinal != 1 || children[1].ChildTargetDigest != originalB.ChildTargetDigest || children[1].ChildIntentDigest == originalB.ChildIntentDigest {
		t.Fatalf("local composition did not reuse A and execute changed B: %+v", children)
	}
	secondB := children[1]
	assertChildScalar(t, store, secondB.ID, 214)
	implementation := filepath.Join(project, "candidate", "private_candidate.py")
	raw, err := os.ReadFile(implementation)
	must(t, err)
	must(t, os.WriteFile(implementation, []byte(strings.Replace(string(raw), "return Result(value * factor)", "return Result(value * factor + 1)", 1)), 0o600))
	status, out = runCozyPath(t, root, path, "run", script, "--retry", "1", "--await", "--json")
	if status != 0 {
		t.Fatalf("same-version library edit did not execute [%d]: %s", status, out)
	}
	third, problem := store.RequestByReference("7")
	fatal(t, problem)
	children, problem = store.Children(third.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].ReusedFrom != originalA.ID || children[0].Ordinal != 0 || children[0].ChildTargetDigest != originalA.ChildTargetDigest || children[1].Ordinal != 1 || children[1].ChildTargetDigest == secondB.ChildTargetDigest || children[1].ChildIntentDigest != secondB.ChildIntentDigest {
		t.Fatalf("library edit did not invalidate exactly B: %+v", children)
	}
	assertChildScalar(t, store, children[1].ID, 215)
	thirdB := children[1]
	status, out = runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status != 0 {
		t.Fatalf("fresh scalar run did not reuse its local workspace [%d]: %s", status, out)
	}
	fresh, problem := store.RequestByReference("10")
	fatal(t, problem)
	cached, problem := store.Children(fresh.ID)
	fatal(t, problem)
	if fresh.RetryOf != "" || len(cached) != 2 || cached[0].Ordinal != 0 || cached[1].Ordinal != 0 || cached[0].ReusedFrom != originalA.ID || cached[1].ReusedFrom != thirdB.ID {
		t.Fatalf("fresh scalar run computed instead of acquiring cached results: %+v", cached)
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
	for _, name := range []string{"source", "candidate"} {
		if _, err := os.Stat(filepath.Join(project, name, "uv.lock")); !os.IsNotExist(err) {
			t.Fatal("private child intake modified the author's lock files")
		}
	}
}

func assertChildScalar(t *testing.T, store *records.Store, id string, value int) {
	t.Helper()
	attempts, problem := store.Attempts(id)
	fatal(t, problem)
	if len(attempts) != 1 {
		t.Fatalf("child result has %d executions", len(attempts))
	}
	doc, err := canonical.Read(attempts[0].TerminalBody, &pb.AttemptOutcomeBody{})
	must(t, err)
	raw, err := base64.StdEncoding.DecodeString(doc.Sub("result").Str("inline_result"))
	must(t, err)
	var result struct {
		Value int `json:"value"`
	}
	must(t, json.Unmarshal(raw, &result))
	if result.Value != value {
		t.Fatalf("child returned %d, want %d", result.Value, value)
	}
}
