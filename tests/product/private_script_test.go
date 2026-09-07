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

var privateScriptRuntimeWheel = flag.String("script-runtime-wheel", "", "Exact Runtime wheel used for private script product proofs")

// A real Python library is captured with a script, even when its version and
// pyproject stay unchanged. The corrected run must use the edited library while
// preserving the failed run's source and execution history.
func TestPrivateScriptCapturesEditableDependencyAndRetries(t *testing.T) {
	root, err := os.MkdirTemp("", "cozy-script-proof-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	project := t.TempDir()
	lib := filepath.Join(project, "algorithm")
	must(t, os.MkdirAll(lib, 0o700))
	must(t, os.WriteFile(filepath.Join(lib, "pyproject.toml"), []byte(`[project]
name = "private-script-algorithm"
version = "0.0.1"
requires-python = ">=3.12"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["algorithm.py"]
`), 0o600))
	module := filepath.Join(lib, "algorithm.py")
	must(t, os.WriteFile(module, []byte("def compute(value):\n    raise ValueError('candidate failed quality gate')\n"), 0o600))
	script := filepath.Join(project, "experiment.py")
	runtimeSource := ""
	if wheel := *privateScriptRuntimeWheel; wheel != "" {
		runtimeSource = "# cozy-runtime = {path = " + strconv.Quote(wheel) + "}\n"
	}
	code := `# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime", "private-script-algorithm"] # //cozy:allow script dependency metadata, not a runtime invocation
# [tool.uv.sources]
# private-script-algorithm = {path = "./algorithm", editable = true}
` + runtimeSource + `# ///
import msgspec
from cozy_runtime.author import App
from algorithm import compute
app = App()
class Request(msgspec.Struct):
    value: int = 7
class Result(msgspec.Struct):
    value: int
def helper(value: int) -> int:
    return compute(value)
@app.job
def main(payload: Request) -> Result:
    return Result(helper(payload.value))
`
	must(t, os.WriteFile(script, []byte(code), 0o600))
	if status, out := runCozy(t, root, "run", script, "--describe", "--json"); status != 0 {
		t.Fatalf("single job script with local library refused before execution [%d]: %s", status, out)
	}
	status, out := runCozy(t, root, "run", script, "--await", "--json")
	if status == 0 || !strings.Contains(out, "candidate failed quality gate") {
		t.Fatalf("first candidate did not fail through the real job [%d]: %s", status, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	first, problem := store.RequestByReference("1")
	fatal(t, problem)
	if first == nil || first.State != "blocked" || !first.RetainWork {
		t.Fatalf("failed run lost retention: %+v", first)
	}
	original, problem := store.Install(first.InstallID)
	fatal(t, problem)
	if original == nil {
		t.Fatal("failed run lost immutable code")
	}
	must(t, os.WriteFile(module, []byte("def compute(value):\n    return value + 100\n"), 0o600))
	status, out = runCozy(t, root, "run", script, "--retry", "1", "--await", "--json")
	if status != 0 || !strings.Contains(out, `"value":107`) {
		t.Fatalf("edited dependency was not used on retry [%d]: %s", status, out)
	}
	second, problem := store.RequestByReference("2")
	fatal(t, problem)
	if second == nil || second.RetryOf != first.ID || second.ID == first.ID || second.InstallID == first.InstallID {
		t.Fatalf("corrected run replaced original history: first=%+v second=%+v", first, second)
	}
	current, problem := store.Install(second.InstallID)
	fatal(t, problem)
	// Successful weightless installs can already be reclaimed. The old failed
	// install remains owned and proves the dependency snapshot is independent.
	_ = current
	python := filepath.Join(original.Dir, "venv", "bin", "python")
	probe := exec.Command(python, "-c", "from algorithm import compute; compute(7)")
	probe.Dir = original.SourceRef
	result, err := probe.CombinedOutput()
	if err == nil || !strings.Contains(string(result), "candidate failed quality gate") {
		t.Fatalf("editing local library mutated prior environment: %v %s", err, result)
	}
	unchanged, problem := store.RequestRow(first.ID)
	fatal(t, problem)
	if unchanged.State != "blocked" || unchanged.BodyDigest != first.BodyDigest {
		t.Fatal("retry rewrote failed request")
	}
	t.Logf("%s failed; %s uses edited library; original source/environment preserved", first.ID, second.ID)
}

func TestPrivateScriptRejectsMultipleEntrypoints(t *testing.T) {
	root, err := os.MkdirTemp("", "cozy-script-count-")
	must(t, err)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all"); _ = os.RemoveAll(root) })
	p := filepath.Join(t.TempDir(), "two.py")
	metadata := "# /// script\n# requires-python = \">=3.12,<3.13\"\n# dependencies = [\"cozy-runtime\"]\n"
	if wheel := *privateScriptRuntimeWheel; wheel != "" {
		metadata += fmt.Sprintf("# [tool.uv.sources]\n# cozy-runtime = {path = %q}\n", wheel)
	}
	metadata += "# ///\n"
	must(t, os.WriteFile(p, []byte(metadata+`import msgspec
from cozy_runtime.author import App
app = App()
class Request(msgspec.Struct):
    value: int = 1
class Result(msgspec.Struct):
    value: int
@app.job
def first(payload: Request) -> Result:
    return Result(payload.value)
@app.job
def second(payload: Request) -> Result:
    return Result(payload.value)
`), 0o600))
	status, out := runCozy(t, root, "run", p, "--json")
	if status == 0 || !strings.Contains(out, "script_entrypoint_count") {
		t.Fatalf("ambiguous script accepted [%d]: %s", status, out)
	}
	if rows := listInvocations(t, root); len(rows) != 0 {
		t.Fatalf("ambiguous script created work: %+v", rows)
	}
}
