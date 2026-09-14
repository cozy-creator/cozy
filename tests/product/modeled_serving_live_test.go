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

	"github.com/cozy-creator/cozy/internal/records"
)

// Match H3's durable job -> same-package @invocable serving segment, then
// invoke the same segment through its generated external caller as well.
func TestModeledServingSelfAndImportedCallsLoadTheSameNativeModel(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact Runtime56 and CUDA")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	control := filepath.Join(t.TempDir(), "control")
	python := filepath.Join(control, "bin", "python")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", python, wheel, "torch==2.13.0", "numpy>=1.26"}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("modeled SDK: %v %s", err, out)
		}
	}
	requireCapturedModelDefaults(t, python)
	if out, err := exec.Command(python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 77)").CombinedOutput(); err != nil {
		if status, ok := err.(*exec.ExitError); ok && status.ExitCode() == 77 {
			t.Skip("actual serving needs CUDA")
		}
		t.Fatalf("CUDA probe: %v %s", err, out)
	}
	root, err := os.MkdirTemp("", "cozy-modeled-serving-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, entry := range childEnv(t, root) {
		if strings.HasPrefix(entry, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(entry, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("modeled serving evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	defer tracePrivateChildWait(t, root)()
	project := t.TempDir()
	library := filepath.Join(project, "model_tools")
	must(t, os.MkdirAll(library, 0700))
	seedRaw, err := exec.Command(python, filepath.Join("testdata", "local_serving_preparation", "seed.py"), filepath.Join(project, "catalog")).CombinedOutput()
	if err != nil {
		t.Fatalf("native model: %v %s", err, seedRaw)
	}
	var seed servingSeed
	must(t, json.Unmarshal(seedRaw, &seed))
	catalog, downloads := capturedDefaultCatalog(t, seed, python)
	defer catalog.Close()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte("tensorhub_url: "+catalog.URL+"\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	module, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "model_tools.py"))
	must(t, err)
	module = append(module, []byte(`
@invocable(memoize=False,defaults={"model":[{"gpu":"*","lane":"proof/ordered@1.0.0/bf16"}]})
async def segment(ctx:Context,*,payload:Request,model:OrderedModel,tel:Telemetry)->Result:
    return model.measure(payload.seed,payload.steps,tel)
app.entrypoint(segment)

@invocable(memoize=False)
async def long_form(ctx:Context)->Result:
    first=await segment(payload=Request(seed=1234,steps=2))
    second=await segment(payload=Request(seed=1234,steps=2))
    assert first.value==second.value and first.cuda_rng==second.cuda_rng
    assert first.device=="cuda"
    return second
app.job(long_form)
`)...)
	must(t, os.WriteFile(filepath.Join(library, "model_tools.py"), module, 0600))
	metadata := fmt.Sprintf(`[project]
name="model-tools"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s","torch==2.13.0","numpy>=1.26"]
[project.entry-points."cozy.application"]
default="model_tools:app"
[tool.uv.sources]
cozy-runtime={path=%s}
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["model_tools.py"]
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(library, "package.toml"), []byte("[application]\nobject=\"model_tools:app\"\n"), 0600))
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime==%s","model-tools==0.0.1"]
# [tool.uv.sources]
# cozy-runtime={path=%s}
# model-tools={path="./model_tools"}
# ///
from model_tools import long_form,segment,Request
async def main()->int:
    own=await long_form()
    imported=await segment(payload=Request(seed=1234,steps=2))
    assert own.value==imported.value and own.cuda_rng==imported.cuda_rng
    assert imported.device=="cuda"
    return 3
`, version, strconv.Quote(wheel))
	script := filepath.Join(project, "use.py")
	must(t, os.WriteFile(script, []byte(body), 0600))
	status, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status != 0 || !strings.Contains(out, `"value":3`) {
		t.Fatalf("modeled self/imported calls [%d]: %s\n%s", status, out, productWorkerLogs(root))
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	if calls := machineChildren(t, root, store, "1"); len(calls) != 2 {
		t.Fatalf("wrong outer call inventory: %+v", calls)
	}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro&_pragma=busy_timeout(5000)")
	must(t, err)
	defer db.Close()
	var calls, executions, pins int
	must(t, db.QueryRow(`SELECT count(*),sum((SELECT count(*) FROM attempts a WHERE a.owner=c.owner AND a.request=c.child_request)) FROM execution_calls c JOIN executions e ON e.owner=c.owner AND e.request=c.child_request WHERE json_extract(CAST(c.intent AS TEXT),'$.export')='segment' AND e.state='succeeded'`).Scan(&calls, &executions))
	must(t, db.QueryRow(`SELECT count(*) FROM execution_checkpoint_inputs WHERE input_id='model:model' AND repository='proof/ordered' AND state='released'`).Scan(&pins))
	if calls != 3 || executions != 3 || pins != 3 || downloads.Load() == 0 {
		t.Fatalf("actual Model.load/custody proof: calls%d executions%d pins%d downloads%d", calls, executions, pins, downloads.Load())
	}
	t.Log("three independently executed self/imported serving calls loaded the same native model; all catalog pins released")
}
