package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The only synthetic service is provider discovery. Submission and every call
// use the ordinary CLI and the image's real Host, Runtime and executors.
func TestPrivateParentSurvivesAnotherScriptPreparation(t *testing.T) {
	privateParentSurvivesAnotherScriptPreparation(t, false)
}
func TestPrivateModeledChildPreservesCPUParent(t *testing.T) {
	privateParentSurvivesAnotherScriptPreparation(t, true)
}
func privateParentSurvivesAnotherScriptPreparation(t *testing.T, modeled bool) {
	layout, store, host, path, _ := startActualChildHost(t)
	t.Cleanup(func() {
		compositionDown(t, layout.Root, path)
		label, err := exec.Command("docker", "inspect", "--format", `{{index .Config.Labels "com.cozy.task"}}`, host.Container).Output()
		if err != nil || strings.TrimSpace(string(label)) != "proto051-disconnected-owner" {
			t.Error("refused cleanup of unowned container")
			return
		}
		for _, args := range [][]string{{"stop", host.Container}, {"rm", host.Container}} {
			if out, err := exec.Command("docker", args...).CombinedOutput(); err != nil {
				t.Errorf("owned cleanup: %v %s", err, out)
			}
		}
	})
	version, err := exec.Command("docker", "exec", host.Container, "cozy-runtime", "--json", "version").Output() //cozy:allow read-only image provenance; all function execution uses Creator CLI
	must(t, err)
	must(t, os.WriteFile(filepath.Join(layout.Root, "host-runtime-version.json"), version, 0600))
	var observed struct {
		Distribution string `json:"distribution"`
		Commit       string `json:"commit"`
	}
	must(t, json.Unmarshal(version, &observed))
	if observed.Distribution != "0.16.10" || !strings.HasPrefix(observed.Commit, "6c5a72d5") {
		t.Fatalf("wrong qualified Runtime: %+v", observed)
	}
	project := filepath.Join(layout.Root, "isolation-project")
	library := filepath.Join(project, "library")
	must(t, os.MkdirAll(library, 0700))
	metadata := `[project]
name="isolation-step"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==0.16.10"]
[project.entry-points."cozy.application"]
default="isolation_step:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["isolation_step.py"]
`
	if modeled {
		metadata = strings.Replace(metadata, `"cozy-runtime==0.16.10"]`, `"cozy-runtime==0.16.10", "torch==2.13.0", "tensorfs"]`, 1)

	}
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	source := `import asyncio
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Context, invocable
class Result(msgspec.Struct, frozen=True):
    value: int
class Request(msgspec.Struct, forbid_unknown_fields=True):
    value: int
@invocable(memoize=True)
async def advance(ctx: Context, *, value: int, hold: bool = False) -> Result:
    if hold:
        root = Path("/tmp/cozy-cl253")
        root.mkdir(exist_ok=True)
        (root / "entered").write_text(str(value))
        while not (root / "continue").exists():
            ctx.raise_if_cancelled()
            await asyncio.sleep(0.05)
    return Result(value + 1)
app = App()
app.job(advance)
@app.entrypoint
def serve(ctx: Context, payload: Request) -> Result:
    return Result(payload.value * 2)
`
	if modeled {
		source += `
import struct
import tensorfs
from cozy_runtime.author import Config, Loader, Model, ModelArtifact, WeightsConfig, WeightsOutput, WeightsPart, WeightsSink, WeightsTarget, WeightsTensor, uses_components
class Pipeline:
    def __init__(self):
        import torch
        self.components = {"zeta": torch.nn.Linear(2, 2, bias=False, dtype=torch.float32), "alpha": torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)}
def factory(config: Config) -> Pipeline:
    return Pipeline()
class TinyModel(Model[Pipeline]):
    pipe: Pipeline
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=factory)
    @uses_components("alpha", "zeta")
    def measure(self, value: int) -> Result:
        import torch
        with torch.inference_mode():
            vector = torch.full((1, 2), float(value), device=self.pipe.components["zeta"].weight.device)
            answer = self.pipe.components["alpha"](self.pipe.components["zeta"](vector))
            return Result(int(answer.sum().item()))
@app.entrypoint
def generate(ctx: Context, payload: Request, model: TinyModel) -> Result:
    return model.measure(payload.value)
@invocable(memoize=True)
async def produce(ctx: Context, *, artifacts: WeightsSink) -> ModelArtifact:
    encoding = dict(tensorfs.seed_digests())["plain/1"]
    tensor = WeightsTensor(logical_dtype="f32", shape=(2, 2), encoding=encoding, parts={"value": WeightsPart("f32", (2, 2))})
    with artifacts.open("weights", sources={}, targets={name: WeightsTarget(add={"weight": tensor}) for name in ("alpha", "zeta")}, configs={"pipeline": WeightsConfig(data=b"{}")}, order=(("alpha", "weight"), ("zeta", "weight"))) as writer:
        for name, scale in (("alpha", 2.0), ("zeta", 1.0)):
            writer.add_part(name, "weight", "value", struct.pack("<4f", scale, 0, 0, scale))
        writer.add_config("pipeline", b"{}")
        return writer.commit().artifact
app.job(produce, weights=(WeightsOutput("weights", max_new_bytes=4096),))
`
	}
	must(t, os.WriteFile(filepath.Join(library, "isolation_step.py"), []byte(source), 0600))
	header := `# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime==0.16.10", "isolation-step==1.0.0"]
# [tool.uv.sources]
# isolation-step={path="./library", editable=true}
# ///
from isolation_step import advance, serve
`
	if modeled {
		header = strings.Replace(header, `"isolation-step==1.0.0"]`, `"isolation-step==1.0.0", "torch==2.13.0"]`, 1)

		header += "from isolation_step import produce, generate\n"
	}
	firstScript := filepath.Join(project, "first.py")
	secondScript := filepath.Join(project, "second.py")
	firstBody := `async def main(ctx):
    result = await advance(value=41, hold=True)
    result = await advance(value=result.value)
    result = await serve(value=result.value)
    assert result.value == 86
    ctx.log("first parent and serving child completed")
`
	if modeled {
		firstBody = strings.Replace(firstBody, "result = await serve(value=result.value)\n    assert result.value == 86", "model = await produce()\n    result = await generate(value=result.value, model=model)\n    assert result.value == 172", 1)
	}
	must(t, os.WriteFile(firstScript, []byte(header+firstBody), 0600))
	must(t, os.WriteFile(secondScript, []byte(header+`async def main(ctx):
    result = await advance(value=9)
    assert result.value == 10
    ctx.log("second independent parent completed")
`), 0600))
	submit := func(script, receipt string) {
		t.Helper()
		status, out := runCozyPath(t, layout.Root, path, "run", script, "--rental-only", "--json")
		must(t, os.WriteFile(filepath.Join(layout.Root, receipt), []byte(out), 0600))
		if status != 0 {
			t.Fatalf("ordinary script [%d]: %s", status, out)
		}
	}
	submit(firstScript, "first-submit.json")
	parent, problem := store.RequestByReference("1")
	fatal(t, problem)
	if parent == nil {
		t.Fatal("missing first request")
	}
	for deadline := time.Now().Add(3 * time.Minute); ; {
		if data, err := exec.Command("docker", "exec", host.Container, "cat", "/tmp/cozy-cl253/entered").Output(); err == nil && string(data) == "41" {
			break
		}
		row, problem := store.RequestRow(parent.ID)
		fatal(t, problem)
		if row != nil && (row.State == "blocked" || row.State == "failed") {
			t.Fatalf("first parent failed before barrier: %+v", row)
		}
		if time.Now().After(deadline) {
			t.Fatal("bounded scalar child did not reach its barrier")
		}
		time.Sleep(100 * time.Millisecond)
	}
	children, problem := store.Children(parent.ID)
	fatal(t, problem)
	if len(children) != 1 || children[0].Ordinal != 1 {
		t.Fatalf("unexpected active child: %+v", children)
	}
	submit(secondScript, "second-submit.json")
	second, problem := store.RequestByReference("3")
	fatal(t, problem)
	if second == nil {
		t.Fatal("missing second request")
	}
	for deadline := time.Now().Add(20 * time.Second); ; {
		events, problem := store.EventsAfter(second.ID, 0, 100)
		fatal(t, problem)
		waiting := false
		for _, event := range events {
			b, _ := json.Marshal(event.Payload)
			if strings.Contains(string(b), "active request") || strings.Contains(string(b), "earlier request") {
				waiting = true
			}
		}
		if waiting {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("second root did not report waiting behind active work")
		}
		time.Sleep(50 * time.Millisecond)
	}
	active, problem := store.RequestRow(children[0].ID)
	fatal(t, problem)
	attempts, problem := store.Attempts(second.ID)
	fatal(t, problem)
	if active == nil || active.Ordinal != 1 || active.State != "dispatching" || len(attempts) != 0 {
		t.Fatalf("second preparation disturbed child or ran: %+v %+v", active, attempts)
	}
	out, err := exec.Command("docker", "exec", host.Container, "touch", "/tmp/cozy-cl253/continue").CombinedOutput()
	if err != nil {
		t.Fatalf("release own barrier: %v %s", err, out)
	}
	for _, entry := range []struct{ id, receipt string }{{parent.ID, "first-result.json"}, {second.ID, "second-result.json"}} {
		status, out := runCozyPath(t, layout.Root, path, "run", "watch", entry.id, "--json")
		must(t, os.WriteFile(filepath.Join(layout.Root, entry.receipt), []byte(out), 0600))
		if status != 0 {
			t.Fatalf("private composition [%d]: %s", status, out)
		}
	}
	first, problem := store.RequestRow(parent.ID)
	fatal(t, problem)
	completed, problem := store.Children(parent.ID)
	fatal(t, problem)
	count := 3
	if modeled {
		count = 4
	}
	if first.Ordinal != 1 || first.State != "succeeded" || len(completed) != count {
		t.Fatalf("parent changed during serving switch: %+v %+v", first, completed)
	}
	for _, child := range completed {
		if child.Ordinal != 1 || child.State != "succeeded" {
			t.Fatalf("child retried or failed: %+v", child)
		}
	}
	proof, _ := json.MarshalIndent(map[string]any{"parent": first, "children": completed, "second": second.ID, "host": host.Container}, "", "  ")
	must(t, os.WriteFile(filepath.Join(layout.Root, "execution-proof.json"), proof, 0600))
}
