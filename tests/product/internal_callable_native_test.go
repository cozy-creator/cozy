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

	"github.com/cozy-creator/cozy/internal/records"
)

// This is an ordinary local CLI invocation with real Runtime child execution.
// Supply the candidate wheel while its author surface is awaiting publication.
func TestInternalCallableNativeParentExecutesItsOwnChild(t *testing.T) {
	wheel := *privateScriptRuntimeWheel
	if wheel == "" {
		t.Skip("requires the Runtime internal-callable candidate wheel")
	}
	root, err := os.MkdirTemp("", "cozy-ic-")
	must(t, err)
	path := ""
	for _, value := range childEnv(t, root) {
		if strings.HasPrefix(value, "PATH=") {
			path = strings.TrimPrefix(value, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("internal callable proof retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := filepath.Join(t.TempDir(), "internal-proof")
	must(t, os.MkdirAll(project, 0700))
	metadata := fmt.Sprintf(`[project]
name="internal-proof"
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[project.entry-points."cozy.application"]
default="internal_proof:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["internal_proof.py"]
[tool.uv.sources]
cozy-runtime={path=%s}
`, runtimeFixtureVersion(t, wheel), strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"internal_proof:app\"\n"), 0600))
	code := `import msgspec
from cozy_runtime.author import App, Context, invocable

app = App()
class Value(msgspec.Struct):
    value: int

class Request(msgspec.Struct):
    pass

@invocable
async def segment(ctx: Context, *, value: int) -> Value:
    ctx.raise_if_cancelled()
    return Value(value + 1)

app.entrypoint(internal=True)(segment)

@invocable
async def internal_job(ctx: Context, *, value: int) -> Value:
    ctx.raise_if_cancelled()
    return Value(value + 1)

app.job(internal=True)(internal_job)

@app.job
async def long_form(ctx: Context, payload: Request) -> Value:
    first = await segment(value=40)
    return await internal_job(value=first.value)
`
	must(t, os.WriteFile(filepath.Join(project, "internal_proof.py"), []byte(code), 0600))
	lock := exec.Command("uv", "lock", "--project", project)
	lock.Env = childEnv(t, root, "PATH="+path)
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("fixture lock: %v %s", err, out)
	}
	if status, out := runCozyPath(t, root, path, "package", "install", project, "--editable", "--json"); status != 0 {
		t.Fatalf("install: %d %s", status, out)
	}
	if status, out := runCozyPath(t, root, path, "run", "local/internal-proof", "--json", "--full"); status != 0 || strings.Contains(out, `"segment"`) || strings.Contains(out, "internal_job") || !strings.Contains(out, "long_form") {
		t.Fatalf("list: %d %s", status, out)
	}
	for _, name := range []string{"segment", "internal_job"} {
		if status, out := runCozyPath(t, root, path, "run", "local/internal-proof/"+name, "value=40", "--await", "--json"); status == 0 || !strings.Contains(out, `"code":"callable_internal"`) {
			t.Fatalf("root refusal: %d %s", status, out)
		}
	}
	status, out := runCozyPath(t, root, path, "run", "local/internal-proof/long_form", "--await", "--json", "--full")
	var result struct {
		Status string `json:"status"`
		Result struct {
			Value int `json:"value"`
		} `json:"result"`
	}
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		var candidate struct {
			Status string `json:"status"`
			Result struct {
				Value int `json:"value"`
			} `json:"result"`
		}
		if json.Unmarshal([]byte(line), &candidate) == nil && candidate.Status == "completed" {
			result.Status = candidate.Status
			result.Result.Value = candidate.Result.Value
		}
	}
	if status != 0 || result.Status != "completed" || result.Result.Value != 42 {
		t.Fatalf("native parent/children: %d %s", status, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByReference("1")
	fatal(t, problem)
	if row == nil || row.State != "succeeded" || row.Entrypoint != "long_form" {
		t.Fatalf("unexpected public request: %+v", row)
	}
	if extra, problem := store.RequestByReference("2"); problem != nil || extra != nil {
		t.Fatalf("refused internal root was recorded: %+v %v", extra, problem)
	}
	t.Logf("ordinary CLI parent %s completed with both serving/job internal children; direct roots refused", row.ID)
}
