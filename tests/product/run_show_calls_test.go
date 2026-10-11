package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// shownRun is `cozy run show --json`, read as a client reads it.
type shownRun struct {
	Events []struct {
		Type    string          `json:"type"`
		Payload json.RawMessage `json:"payload"`
	} `json:"events"`
	Calls       []shownCall `json:"calls"`
	ExecutionMS *int64      `json:"execution_ms"`
	Stages      []struct {
		Name        string  `json:"name"`
		StartUnixMS int64   `json:"start_unix_ms"`
		MS          float64 `json:"ms"`
		Detail      string  `json:"detail"`
	} `json:"stages"`
}

type shownCall struct {
	Number   int    `json:"number"`
	Request  string `json:"request"`
	Function string `json:"function"`
	Label    string `json:"label"`
	Status   string `json:"status"`
	GPUs     []struct {
		GPU int `json:"gpu"`
		PID int `json:"pid"`
	} `json:"gpus"`
	Stages []struct {
		Name string `json:"name"`
		Kind string `json:"kind"`
	} `json:"stages"`
	Steps []struct {
		Name   string       `json:"name"`
		Count  int          `json:"count"`
		Series [][2]float64 `json:"series"`
	} `json:"steps"`
}

// Each child call a run makes on this computer's machine is its own record in `cozy run
// show`: the real Runtime journals the call's function, the author's label, its timed
// stages and every step. (A weightless call holds no GPU; the flood test covers grants.)
func TestRunShowListsEachChildCallOfALocalRun(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor this computer's machine runs")
	}
	root, err := os.MkdirTemp("", "czc")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			log, _ := os.ReadFile(filepath.Join(root, "machine", "host.log"))
			t.Logf("call evidence retained at %s\nHost log:\n%.4000s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	provisionMachine(t, root)
	if code, out := runCozy(t, root, "package", "install", callsProject(t)); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	const key = "child-calls"
	code, out := runCozy(t, root, "run", "local/child-calls/film", "--await", "--json", "--idempotency-key", key)
	if code != 0 || !strings.Contains(out, `"segments":3`) {
		t.Fatalf("film [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)

	code, human := runCozy(t, root, "run", "show", request.ID)
	t.Logf("cozy run show:\n%s", human)
	for index := range 3 {
		row := regexp.MustCompile(fmt.Sprintf(`(?m)^%d +Segment %d of 3 +render +succeeded +\S+ +\+\S+ +\S+ +denoise 4× `, index+1, index+1))
		if code != 0 || !row.MatchString(human) {
			t.Fatalf("run show [%d] lacks segment %d's call row:\n%s", code, index+1, human)
		}
	}
	code, out = runCozy(t, root, "run", "show", request.ID, "--json")
	var shown shownRun
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || len(shown.Calls) != 4 || shown.Calls[0].Number != 0 {
		t.Fatalf("run show --json [%d]:\n%.2000s", code, out)
	}
	// The machine measures the run's own running time; each step of its preparation is one row
	// that says when it began and how long it took.
	if shown.ExecutionMS == nil || *shown.ExecutionMS <= 0 {
		t.Fatalf("run show --json has no execution time:\n%.2000s", out)
	}
	steps := map[string]bool{}
	for _, stage := range shown.Stages {
		if stage.Name != "machine preparation" {
			continue
		}
		if steps[stage.Detail] || stage.StartUnixMS <= 0 || stage.MS <= 0 {
			t.Fatalf("machine preparation step %q is repeated or unmeasured:\n%.3000s", stage.Detail, out)
		}
		steps[stage.Detail] = true
	}
	if len(steps) == 0 {
		t.Fatalf("run show --json lists no machine preparation:\n%.3000s", out)
	}
	for _, call := range shown.Calls[1:] {
		kinds := map[string]string{}
		for _, stage := range call.Stages {
			kinds[stage.Name] = stage.Kind
		}
		if call.Function != "render" || call.Status != "succeeded" || kinds["condition"] != "inference" ||
			kinds["decode"] != "inference" || len(call.Steps) != 1 || call.Steps[0].Count != 4 ||
			len(call.Steps[0].Series) != 4 {
			t.Fatalf("call %d is not the real Runtime's record of its segment: %+v", call.Number, call)
		}
	}
	code, one := runCozy(t, root, "run", "show", request.ID, "--call", "2")
	t.Logf("cozy run show --call 2:\n%s", one)
	if code != 0 || !strings.Contains(one, "call 2 of 3  Segment 2 of 3  render  succeeded") ||
		!strings.Contains(one, "steps denoise: 4 in") {
		t.Fatalf("run show --call 2 [%d]:\n%s", code, one)
	}
}

// callsProject is a film: a root job that renders three segments, each a child call under
// its author label that times a stage, four denoising steps and a decode.
func callsProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "child-calls")
	must(t, os.MkdirAll(project, 0o700))
	runtime := "cozy-runtime>=" + runtimeFloor
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="child-calls"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["`+runtime+`", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="child_calls:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["child_calls.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"child_calls:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "child_calls.py"), []byte(`import asyncio

import msgspec
from cozy_runtime.author import App, Context, Telemetry, invocable


class FilmRequest(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Rendered(msgspec.Struct):
    index: int


class Film(msgspec.Struct):
    segments: int


app = App()


@invocable
async def render(ctx: Context, tel: Telemetry, *, index: int) -> Rendered:
    with tel.stage("condition"):
        await asyncio.sleep(0.01)
    on_step = tel.step_callback(4, stage="denoise")
    for step in range(4):
        await asyncio.sleep(0.005)
        on_step(step)
    with tel.stage("decode"):
        await asyncio.sleep(0.01)
    return Rendered(index)


app.entrypoint(render)


@app.job
async def film(ctx: Context, payload: FilmRequest, tel: Telemetry) -> Film:
    for index in range(3):
        with tel.scope(f"Segment {index + 1} of 3"):
            await render(index=index)
    return Film(3)
`), 0o600))
	lock := exec.Command("uv", "lock", "--project", project)
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("locking the child-calls package: %v\n%s", err, out)
	}
	return project
}
