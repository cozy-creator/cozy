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
)

// Only the provider is a fixture; every submission, preparation, managed call,
// model load and result transfer runs through the ordinary CLI and actual Host.
func TestModeledServingConsecutiveCallsActualHost(t *testing.T) {
	if *privateChildRuntimeWheel == "" || *childHostRuntimeBin == "" || *childHostHome == "" {
		t.Skip("requires an explicit owned CUDA Host and coherent Runtime wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	python := filepath.Join(*childHostRuntimeBin, "python")
	requireCapturedModelDefaults(t, python)
	project := filepath.Join(*childHostHome, "modeled-project")
	library := filepath.Join(project, "model_tools")
	must(t, os.MkdirAll(library, 0700))
	layout, store, host, path, _ := startActualChildHost(t)
	module, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "model_tools.py"))
	must(t, err)
	// A unique marker changes only on actual model construction, so object-id reuse
	// after an unload cannot make cold execution appear warm.
	module = []byte(strings.Replace(string(module), "self.pipe = loader.construct(OrderedPipeline, factory=build_pipeline)", "self.load_identity = __import__('uuid').uuid4().hex\n        self.pipe = loader.construct(OrderedPipeline, factory=build_pipeline)", 1))
	module = append(module, []byte(`
import os

class SegmentProof(msgspec.Struct):
    value: float
    cuda_rng: str
    device: str
    executor_pid: int
    load_identity: str

@invocable(memoize=False,defaults={"model":[{"gpu":"*","lane":"proof/ordered@1.0.0/bf16"}]})
async def segment(ctx:Context,*,payload:Request,model:OrderedModel,tel:Telemetry)->SegmentProof:
    result=model.measure(payload.seed,payload.steps,tel)
    return SegmentProof(result.value,result.cuda_rng,result.device,os.getpid(),model.load_identity)
app.entrypoint(segment)

@invocable(memoize=False)
async def long_form(ctx:Context)->SegmentProof:
    weights=await produce()
    first=await segment(payload=Request(seed=1234,steps=2),model=weights)
    second=await segment(payload=Request(seed=1234,steps=2),model=weights)
    assert first.value==second.value and first.cuda_rng==second.cuda_rng
    assert first.device==second.device=="cuda"
    return second
app.job(long_form)
`)...)
	must(t, os.WriteFile(filepath.Join(library, "model_tools.py"), module, 0600))
	metadata := fmt.Sprintf(`[project]
name="model-tools"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s","torch>=2.13.0","numpy>=1.26"]
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
# dependencies=["cozy-runtime>=%s","model-tools>=0.0.1"]
# [tool.uv.sources]
# cozy-runtime={path=%s}
# model-tools={path="./model_tools"}
# ///
from model_tools import long_form,segment,Request,produce
async def main()->int:
    weights=await produce()
    own=await long_form()
    imported=await segment(payload=Request(seed=1234,steps=2),model=weights)
    assert own.value==imported.value and own.cuda_rng==imported.cuda_rng
    assert imported.device=="cuda"
    return 3
`, version, strconv.Quote(wheel))
	script := filepath.Join(project, "use.py")
	must(t, os.WriteFile(script, []byte(body), 0600))
	status, out := runCozyPath(t, layout.Root, path, "run", script, "--rental", "child-host", "--await", "--json", "--idempotency-key", "modeled-serving-host")
	if status != 0 || !strings.Contains(out, `"value":3`) {
		t.Fatalf("modeled actual Host self/imported calls [%d]: %s", status, out)
	}
	request, problem := store.RequestByIdempotencyKey("modeled-serving-host")
	fatal(t, problem)
	if request == nil || request.State != "succeeded" {
		t.Fatalf("modeled root did not succeed: %+v", request)
	}
	children, problem := store.Children(request.ID)
	fatal(t, problem)
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(children) != 0 || len(attempts) != 0 {
		t.Fatal("Creator owns machine attempts or children")
	}
	proof, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json,sqlite3
from cozy_runtime.internal.placement_materialization import LOCAL_TENSORFS_ROOT
c=sqlite3.connect('file:'+str(LOCAL_TENSORFS_ROOT/'.cozy-workspace/journal.sqlite3')+'?mode=ro',uri=True)
rows=c.execute("select c.child_request,c.result,e.state from execution_calls c join executions e on e.owner=c.owner and e.request=c.child_request where json_extract(cast(c.intent as text),'$.export')='segment' order by c.created_ms").fetchall()
results=[dict(request=r,result=json.loads(b),state=s) for r,b,s in rows]
source_calls=c.execute("select count(*) from execution_calls where json_extract(cast(intent as text),'$.export')='produce'").fetchone()[0]
source_executions=c.execute("select count(*) from attempts a where exists(select 1 from execution_calls c where c.owner=a.owner and c.child_request=a.request and json_extract(cast(c.intent as text),'$.export')='produce')").fetchone()[0]
print(json.dumps(dict(segments=results,source_calls=source_calls,source_executions=source_executions)))`).CombinedOutput()
	if err != nil {
		t.Fatalf("actual model result evidence: %v %s", err, proof)
	}
	must(t, os.WriteFile(filepath.Join(layout.Root, "modeled-serving-proof.json"), proof, 0600))
	var observed struct {
		Segments []struct {
			State  string `json:"state"`
			Result struct {
				PID  int    `json:"executor_pid"`
				Load string `json:"load_identity"`
			} `json:"result"`
		} `json:"segments"`
		SourceCalls      int `json:"source_calls"`
		SourceExecutions int `json:"source_executions"`
	}
	must(t, json.Unmarshal(proof, &observed))
	if len(observed.Segments) != 3 || observed.SourceCalls != 2 || observed.SourceExecutions != 1 {
		t.Fatalf("missing actual serving/custody proof: %s", proof)
	}
	for _, segment := range observed.Segments {
		if segment.State != "succeeded" || segment.Result.PID == 0 || segment.Result.Load == "" {
			t.Fatalf("incomplete actual serving: %s", proof)
		}
	}
	t.Logf("actual Host executed self/imported serving calls with no Creator children: %s", proof)
	if observed.Segments[0].Result != observed.Segments[1].Result {
		t.Fatalf("consecutive self calls reloaded their model or changed executor: %s", proof)
	}
}
