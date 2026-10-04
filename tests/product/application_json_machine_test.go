package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// Typed Python observes the input's number kind and exact unsigned seed. Its managed
// child returns those values through the real machine/result/controller boundary.
func TestApplicationJSONTypedMachineJobPreserves64BitValuesAndNumberKind(t *testing.T) {
	if *machineHostBinary == "" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires -machine-host and -script-runtime-wheel")
	}
	root, err := os.MkdirTemp("", "cz-json-")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := t.TempDir()
	wheel, err := filepath.Abs(*privateScriptRuntimeWheel)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(fmt.Sprintf(`[project]
name="application-json-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[tool.uv.sources]
cozy-runtime={path=%q}
[project.entry-points."cozy.application"]
default="json_proof:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["json_proof.py"]
`, runtimeFixtureVersion(t, wheel), wheel)), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"json_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "json_proof.py"), []byte(`import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    seed: int
    number: int | float
    ordered: list[int]
class Value(msgspec.Struct):
    seed: int
    number: int | float
    ordered: list[int]
    kind: str
@invocable(memoize=True)
async def child(ctx: Context, *, seed: int, number: int | float, ordered: list[int]) -> Value:
    return Value(seed, number, ordered, type(number).__name__)
app.job(internal=True)(child)
@app.job
async def echo(ctx: Context, payload: Request) -> Value:
    return await child(seed=payload.seed, number=payload.number, ordered=payload.ordered)
`), 0o600))
	if out, err := exec.Command("/usr/bin/nice", "-n", "19", "uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v %s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [%d]\n%s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	check := func(raw json.RawMessage, seed uint64, kind string, ordered []uint64) {
		t.Helper()
		var got struct {
			Seed    uint64
			Number  json.Number
			Ordered []uint64
			Kind    string
		}
		must(t, json.Unmarshal(raw, &got))
		if got.Seed != seed || got.Kind != kind || !slices.Equal(got.Ordered, ordered) || kind == "float" && got.Number.String() != "1.0" || kind == "int" && got.Number.String() != "1" {
			t.Fatalf("typed result changed semantics: %s", raw)
		}
	}
	for i, arm := range []struct {
		seed         uint64
		number, kind string
		ordered      []uint64
	}{
		{9007199254740993, "1.0", "float", []uint64{9007199254740993, 9007199254740992}},
		{18446744073709551615, "1", "int", []uint64{18446744073709551615, 0}},
		{9007199254740993, "1", "int", []uint64{9007199254740992, 9007199254740993}},
		{0, "1.0", "float", []uint64{0, 18446744073709551615}},
	} {
		in := filepath.Join(root, fmt.Sprintf("input-%d.json", i))
		order, _ := json.Marshal(arm.ordered)
		must(t, os.WriteFile(in, []byte(fmt.Sprintf(`{"seed":%d,"number":%s,"ordered":%s}`, arm.seed, arm.number, order)), 0o600))
		code, out := runCozy(t, root, "run", "local/application-json-proof/echo", "--input", in, fmt.Sprintf("seed=%d", arm.seed), "--await", "--json")
		if code != 0 {
			t.Fatalf("ordinary typed CLI job [%d]\n%s", code, out)
		}
		var result struct{ Result json.RawMessage }
		must(t, json.Unmarshal([]byte(lastJSONLine(out)), &result))
		check(result.Result, arm.seed, arm.kind, arm.ordered)
		row, problem := store.RequestByReference(fmt.Sprint(i + 1))
		fatal(t, problem)
		if row == nil {
			t.Fatal("typed root missing")
		}
		var payload struct {
			Seed   uint64
			Number json.Number
		}
		must(t, json.Unmarshal(row.Payload, &payload))
		if payload.Seed != arm.seed || payload.Number.String() != arm.number {
			t.Fatalf("accepted root changed: %s", row.Payload)
		}
	}
	// Use the ordinary authenticated HTTP client independently of CLI argument parsing.
	code, stdout, stderr := runCozyStreams(t, root, "run", "local/application-json-proof/echo", "seed=18446744073709551615", "number:=1.0", "ordered:=[2,1]", "--await")
	if code != 0 {
		t.Fatalf("ordinary default-format typed CLI job [%d]\n%s\n%s", code, stdout, stderr)
	}
	for _, exact := range []string{`"seed":18446744073709551615`, `"number":1.0`, `"ordered":[2,1]`, `"kind":"float"`} {
		if !strings.Contains(stdout, exact) {
			t.Fatalf("ordinary default output changed typed result: %s\n%s", stdout, stderr)
		}
	}
	cfg := config.Config{Home: root, HubURL: config.DefaultHubURL}
	client, problem := localapi.Open(cfg, daemon.Probe(cfg))
	fatal(t, problem)
	sub := api.JobSubmission{Package: "local/application-json-proof", Function: "echo", Input: json.RawMessage(`{"seed":18446744073709551615,"number":1.0,"ordered":[2,1]}`)}
	h, problem := client.SubmitJob(sub, "json-api-semantic-replay")
	fatal(t, problem)
	var state api.JobState
	landed(t, "typed API job to complete", func() bool {
		state, problem = client.Job(h.JobID)
		fatal(t, problem)
		if state.Status == "failed" || state.Status == "canceled" {
			t.Fatalf("typed API root ended %s: %s %s", state.Status, state.ErrorCode, state.Error)
		}
		return state.Status == "completed" || state.Status == "succeeded"
	})
	result, err := json.Marshal(state.Result)
	must(t, err)
	check(result, 18446744073709551615, "float", []uint64{2, 1})
	sub.Input = json.RawMessage("{ \"ordered\": [2, 1], \"number\": 1e0, \"seed\": 18446744073709551615 }")
	replay, problem := client.SubmitJob(sub, "json-api-semantic-replay")
	fatal(t, problem)
	if !replay.Replay || replay.JobID != h.JobID {
		t.Fatalf("equivalent API request did not reattach: %+v", replay)
	}
	for _, changed := range []string{`{"seed":18446744073709551614,"number":1.0,"ordered":[2,1]}`, `{"seed":18446744073709551615,"number":1,"ordered":[2,1]}`, `{"seed":18446744073709551615,"number":1.0,"ordered":[1,2]}`} {
		sub.Input = json.RawMessage(changed)
		if _, problem := client.SubmitJob(sub, "json-api-semantic-replay"); problem == nil || problem.Code != exit.Conflict {
			t.Fatalf("changed API intent did not conflict: %s (%v)", changed, problem)
		}
	}
}
