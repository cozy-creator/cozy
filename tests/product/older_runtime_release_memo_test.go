package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A published release whose bounds exclude the machine's own Runtime pair runs on the older
// pair it names (the machine's SDK fallback), and its memoized calls are still reused: they
// key by the files they read, as that Runtime does, instead of being refused.
func TestAReleaseOnAnOlderRuntimeReusesItsMemos(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	project := filepath.Join(t.TempDir(), "machine-parity")
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="machine-parity"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=0.19,<0.20", "tensorfs>=0.4,<0.5", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="machine_parity:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["machine_parity.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"machine_parity:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "machine_parity.py"), []byte(`from pathlib import Path

import msgspec
from cozy_runtime.author import App, Context, invocable

app = App()


class Probe(msgspec.Struct):
    value: int
    counter: str


class Measured(msgspec.Struct):
    square: int


@invocable(memoize=True)
async def measure(ctx: Context, *, payload: Probe) -> Measured:
    counter = Path(payload.counter)
    counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
    return Measured(square=payload.value * payload.value)


class Survey(msgspec.Struct):
    values: list[int]
    counter: str


class Surveyed(msgspec.Struct):
    squares: list[int]


async def survey(ctx: Context, payload: Survey) -> Surveyed:
    return Surveyed(squares=[(await measure(payload=Probe(v, payload.counter))).square for v in payload.values])


app.entrypoint(measure)
app.job(survey)
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the release: %v\n%s", err, out)
	}
	publishParityRelease(t, h, root, project)

	counter := filepath.Join(t.TempDir(), "measured")
	input := filepath.Join(t.TempDir(), "survey.json")
	must(t, os.WriteFile(input, []byte(`{"values":[2,3,2],"counter":"`+counter+`"}`), 0o600))
	for _, round := range []string{"first", "second"} {
		code, out := runCozy(t, root, "run", parityPublished+"/survey", "--input", input, "--await", "--json")
		if code != 0 || !strings.Contains(out, `"squares":[4,9,4]`) {
			t.Fatalf("the %s survey [exit %d]\n%s", round, code, out)
		}
		measured, _ := os.ReadFile(counter)
		t.Logf("after the %s survey, measure ran %s times", round, measured)
		if string(measured) != "2" {
			t.Fatalf("after the %s survey measure ran %q times, want 2: its memo was not reused", round, measured)
		}
	}
	code, out := runCozy(t, root, "run", "show", "2", "--json")
	var shown struct {
		Calls []struct {
			Function          string `json:"function"`
			Memoized          bool   `json:"memoized"`
			ComputationDigest string `json:"computation_digest"`
		} `json:"calls"`
	}
	reused := 0
	if code == 0 && json.Unmarshal([]byte(out), &shown) == nil {
		for _, call := range shown.Calls {
			if call.Function == "measure" && call.Memoized {
				reused++
				t.Logf("reused call: computation_digest %q (empty: keyed by files)", call.ComputationDigest)
			}
		}
	}
	if reused != 3 {
		t.Fatalf("the second survey's 3 calls are not shown as reused (%d) [exit %d]\n%.3000s", reused, code, out)
	}
	generations, _ := filepath.Glob(filepath.Join(root, "machine", "root", "var", "lib", "cozy", "rust-machine", "generations", "*", "generation.json"))
	for _, path := range generations {
		raw, err := os.ReadFile(path)
		must(t, err)
		var generation struct {
			SDKFallback string `json:"sdk_fallback"`
		}
		_ = json.Unmarshal(raw, &generation)
		var record map[string]any
		_ = json.Unmarshal(raw, &record)
		if record["package"] != "machine-parity" {
			continue
		}
		t.Logf("generation: sdk=%v sdk_fallback=%q dependencies=%v", record["sdk"], generation.SDKFallback, record["dependencies"])
		if generation.SDKFallback == "" || !strings.Contains(string(raw), `"cozy-runtime","version":"0.19.0"`) && !strings.Contains(string(raw), `"name":"cozy-runtime","version":"0.19.0"`) {
			t.Fatalf("the release did not run on its own older Runtime: %s", raw)
		}
		return
	}
	t.Fatalf("no generation of the release on this computer's machine: %v", generations)
}
