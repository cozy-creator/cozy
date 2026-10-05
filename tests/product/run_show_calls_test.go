package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

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

// longForm journals what Runtime records for a long_form root: a prefetch narrating every
// position of its download (4,065 such rows filled run 1510's 4,096-row record and hid its
// nine segments), then each segment's input check, GPU grant and release, and the call
// record Runtime journals when the segment settles. Its worker was launched on cards 4-7, so
// each grant's envelope ordinals 0-3 are GPUs 4-7, and its one process ran on GPU 4.
func (m *runtimeMachine) longForm(segments, positions int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	log := func(payload map[string]any) []byte {
		body, _ := json.Marshal(map[string]any{"type": "log", "payload": payload})
		return body
	}
	for position := range positions {
		m.record("log", log(map[string]any{"name": "model-prefetch", "value": "info", "fields": map[string]any{
			"position": position, "total": positions, "stage": "motion_segment_turbo / Downloading model weights"}}))
	}
	now := time.Now().UnixMilli()
	for index := range segments {
		child := fmt.Sprintf("call-%040x", index)
		m.record("log", log(map[string]any{"name": "Checking model inputs", "value": "info", "at_unix_ms": now,
			"fields": map[string]any{"child_request": child, "phase": "Checking model inputs", "completed": true, "elapsed_ms": 101.5}}))
		cards := []map[string]any{{"gpu": 4, "uuid": "GPU-4"}, {"gpu": 5, "uuid": "GPU-5"},
			{"gpu": 6, "uuid": "GPU-6"}, {"gpu": 7, "uuid": "GPU-7"}}
		grant, _ := json.Marshal(map[string]any{"key": child + "#1", "ordinals": []int{0, 1, 2, 3}, "gpus": cards})
		m.record("gpu.grant", grant)
		attention := map[string]any{"observed": "sageattention"}
		release, _ := json.Marshal(map[string]any{"key": child + "#1", "ordinals": []int{0, 1, 2, 3}, "cause": "exited",
			"ranks": []map[string]any{{"rank": 0, "ordinal": 4, "pid": 4000 + index, "uuid": "GPU-4", "arch": "sm_120",
				"attention": attention}},
			"gpus": []map[string]any{{"gpu": 4, "pid": 4000 + index, "uuid": "GPU-4", "arch": "sm_120",
				"attention": attention}}})
		m.record("gpu.release", release)
		series := [][2]float64{}
		for step := range 8 {
			series = append(series, [2]float64{float64(now + int64(step)*8600), 8600})
		}
		record, _ := json.Marshal(map[string]any{"request": child, "parent": m.state.RequestId, "index": index, "attempt": 1,
			"module": "h3", "export": "motion_segment_turbo", "label": fmt.Sprintf("Segment %d of %d", index+1, segments),
			"status": "succeeded", "error": "", "called_unix_ms": now,
			"stages": map[string]any{
				"conditioning": map[string]any{"count": 1, "total_ms": 3100.0, "started_unix_ms": now},
				"decode_video": map[string]any{"count": 1, "total_ms": 9200.0, "started_unix_ms": now + 72000}},
			"steps": map[string]any{"denoise": map[string]any{"count": 8, "total_ms": 68800.0, "first_ms": 8600.0,
				"min_ms": 8600.0, "max_ms": 8600.0, "started_unix_ms": now + 3100, "series": series}}})
		m.record("call", record)
	}
	// Its film uploads as a checkpoint: an effect call, recorded on the root like a child's.
	upload, _ := json.Marshal(map[string]any{"request": fmt.Sprintf("call-%040x", segments), "parent": m.state.RequestId,
		"index": segments, "attempt": 1, "module": "cozy_runtime.author.publication", "export": "upload_checkpoint",
		"label": "Upload checkpoint to proof/film", "status": "succeeded", "error": "", "called_unix_ms": now,
		"stages": map[string]any{
			"Uploading checkpoint":  map[string]any{"count": 1, "total_ms": 2000.0, "started_unix_ms": now, "bytes": 64 << 20},
			"Publishing checkpoint": map[string]any{"count": 1, "total_ms": 1500.0, "started_unix_ms": now + 2000}},
		"steps": map[string]any{}})
	m.record("call", upload)
}

// A long_form's record keeps every segment call however much its prefetch narrates: the
// narration is progress, read as its latest sample, and never counts against the grants,
// phases, call records and terminal. `cozy run show` lists each call; `--call` shows one.
func TestRunShowKeepsEveryCallPastAProgressFlood(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &runtimeMachine{triage: bundle}
	pod := &fakePod{machine: machine, deviceCount: 4,
		mediaRequest: func(w http.ResponseWriter, r *http.Request) bool {
			if r.Method != http.MethodGet || r.URL.Path != "/v1/triage/trb-rented" {
				return false
			}
			_, _ = w.Write(bundle)
			return true
		}}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	const key = "long-form-flood"
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json",
		"--idempotency-key", key); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	const segments, positions = 9, 5000
	machine.longForm(segments, positions)
	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "the collected result", func() bool {
		row, problem := store.RequestByIdempotencyKey(key)
		return problem == nil && row.State == "succeeded"
	})

	code, human := runCozy(t, root, "run", "show", "1")
	t.Logf("cozy run show 1:\n%s", human)
	for index := range segments {
		row := regexp.MustCompile(fmt.Sprintf(`(?m)^%d +Segment %d of %d +motion_segment_turbo +succeeded +4-7 .* denoise 8× 8.6s +- +sageattention$`,
			index+1, index+1, segments))
		if code != 0 || !strings.Contains(human, "calls (10)") || !row.MatchString(human) {
			t.Fatalf("run show [%d] lacks segment %d's call row:\n%s", code, index+1, human)
		}
	}
	code, out := runCozy(t, root, "run", "show", "1", "--json")
	var shown shownRun
	// Call 0 is the run's own execution, then its segments, then its upload.
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || len(shown.Calls) != segments+2 ||
		shown.Calls[0].Number != 0 || shown.Calls[0].Label != "this run" || shown.Calls[0].Function != "generate" {
		t.Fatalf("run show --json [%d] does not carry its execution and %d calls:\n%.2000s", code, segments+1, out)
	}
	upload := shown.Calls[segments+1]
	if upload.Number != segments+1 || upload.Function != "upload_checkpoint" || len(upload.Stages) != 2 ||
		upload.Stages[0].Name != "Uploading checkpoint" || upload.Stages[0].Kind != "transfer" {
		t.Fatalf("the checkpoint upload is not a call with its transfer: %+v", upload)
	}
	if !regexp.MustCompile(`(?m)^10 +Upload checkpoint to proof/film +upload_checkpoint +succeeded +- `).MatchString(human) ||
		!regexp.MustCompile(`(?m)^0 +this run +generate +succeeded `).MatchString(human) {
		t.Fatalf("run show lacks the run's own row or its upload's:\n%s", human)
	}
	var narrated []string
	for _, event := range shown.Events {
		if strings.Contains(string(event.Payload), "model-prefetch") {
			narrated = append(narrated, string(event.Payload))
		}
	}
	if len(narrated) != 1 || !strings.Contains(narrated[0], fmt.Sprintf(`"position":%d`, positions-1)) {
		t.Fatalf("the record holds %d prefetch samples, not its one latest: %.500v", len(narrated), narrated)
	}
	for _, call := range shown.Calls[1 : segments+1] {
		if call.Status != "succeeded" || fmt.Sprint(call.GPUs) != fmt.Sprintf("[{4 %d} {5 0} {6 0} {7 0}]", 4000+call.Number-1) ||
			len(call.Steps) != 1 || call.Steps[0].Count != 8 || len(call.Steps[0].Series) != 8 || len(call.Stages) != 4 {
			t.Fatalf("call %d lost its record: %+v", call.Number, call)
		}
	}

	code, one := runCozy(t, root, "run", "show", "1", "--call", "segment 3 of 9")
	t.Logf("cozy run show 1 --call 'segment 3 of 9':\n%s", one)
	for _, want := range []string{"call 3 of 10  Segment 3 of 9  motion_segment_turbo  succeeded",
		"Checking model inputs  phase", "decode_video", "GPUs 4-7", "GPU  ARCH",
		"steps denoise: 8 in 1m8.8s; first 8.6s, then mean 8.6s", "GPUs (1)"} {
		if code != 0 || !strings.Contains(one, want) {
			t.Fatalf("run show --call [%d] lacks %q:\n%s", code, want, one)
		}
	}
	code, one = runCozy(t, root, "run", "show", "1", "--call", "10")
	if code != 0 || !regexp.MustCompile(`(?m)^Uploading checkpoint +transfer +\+\S+ +2\.0s +64\.0MiB, 32\.0MiB/s$`).MatchString(one) ||
		!regexp.MustCompile(`(?m)^Publishing checkpoint +phase +\+\S+ +1\.5s`).MatchString(one) {
		t.Fatalf("run show --call 10 [%d] lacks the upload's bytes and rate:\n%s", code, one)
	}
	if code, out := runCozy(t, root, "run", "show", "1", "--call", "11"); code == 0 || !strings.Contains(out, "has 10 call(s)") {
		t.Fatalf("an absent call number was not refused [%d]:\n%s", code, out)
	}
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
	if code, out := runCozy(t, root, "package", "install", callsProject(t), "--editable"); code != 0 {
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
