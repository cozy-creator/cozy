package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// restartProof is a package whose one call appends a line to a file (its effect), reports a
// step a second for `seconds`, then saves a report of how many effects the file holds.
const restartProof = `"""One call with one observable effect, for machine-restart proofs."""
from __future__ import annotations

import time
from typing import Annotated

import msgspec

from cozy_runtime.author import App, AssetBound, Context, FileAsset, Outputs, Telemetry

app = App()


class Work(msgspec.Struct):
    effect: str
    seconds: int = 0


class Done(msgspec.Struct):
    effects: int
    report: Annotated[FileAsset, AssetBound(media_types=("text/plain",))]


@app.entrypoint
def work(payload: Work, ctx: Context, out: Outputs, tel: Telemetry) -> Done:
    with open(payload.effect, "a", encoding="utf-8") as record:
        record.write("ran\n")
    on_step = tel.step_callback(max(payload.seconds, 1), stage="work")
    for index in range(payload.seconds):
        ctx.raise_if_cancelled()
        time.sleep(1)
        on_step(index)
    with open(payload.effect, encoding="utf-8") as record:
        effects = len(record.readlines())
    report = out.save_bytes(f"effects={effects}\n".encode(), media_type="text/plain")
    return Done(effects, report)
`

// A machine killed outright (SIGKILL, its executors with it) and started again never runs
// started work twice: the call that was running ends FAILED with the reason, its effect made
// once; a call that had finished keeps its result and output, which this client, attached to
// neither while it happened, collects from the restarted machine.
func TestAMachineKilledMidRunFailsItWithoutReplayAndKeepsFinishedRuns(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host=<cozy-machine>")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czk")
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
	project := filepath.Join(t.TempDir(), "restart_proof")
	must(t, os.MkdirAll(filepath.Join(project, "restart_proof"), 0o755))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"

[project]
name = "machine-restart-proof"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["cozy-runtime>=0.18.67", "msgspec>=0.19,<1"]

[project.entry-points."cozy.application"]
default = "restart_proof:app"

[tool.uv.build-backend]
module-name = "restart_proof"
module-root = ""
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = \"restart_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "restart_proof", "__init__.py"), []byte(restartProof), 0o600))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}

	type machine struct {
		Running bool              `json:"running"`
		PID     int               `json:"pid"`
		Live    []json.RawMessage `json:"live_runs"`
	}
	show := func() machine {
		t.Helper()
		_, out := runCozy(t, root, "machine", "show", "--json")
		var shown machine
		_ = json.Unmarshal([]byte(lastJSONLine(out)), &shown)
		return shown
	}
	type run struct {
		Status string `json:"status"`
		Output []struct {
			Name, Status, Path string
		} `json:"output"`
	}
	shown := func(number string) (run, string) {
		t.Helper()
		_, out := runCozy(t, root, "run", "show", number, "--json")
		var state run
		_ = json.Unmarshal([]byte(lastJSONLine(out)), &state)
		return state, out
	}
	effects := func(path string) int {
		raw, _ := os.ReadFile(path)
		return strings.Count(string(raw), "ran\n")
	}
	submit := func(effect string, seconds string) {
		t.Helper()
		if code, out := runCozy(t, root, "run", "local/machine-restart-proof/work", "effect="+effect, "seconds="+seconds, "--json"); code != 0 {
			t.Fatalf("submit [exit %d]\n%s", code, out)
		}
	}

	// Run 1 finishes and run 2 starts while no client is attached: the daemon is down.
	finished, running := filepath.Join(root, "finished.effect"), filepath.Join(root, "running.effect")
	submit(finished, "20")
	landed(t, "the first call to start", func() bool { return effects(finished) == 1 })
	submit(running, "3600")
	landed(t, "the machine to hold both runs", func() bool { return len(show().Live) == 2 })
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	if held := len(show().Live); held != 2 {
		t.Fatalf("the first run ended before the daemon went down (%d live): its collection would prove nothing", held)
	}
	landed(t, "the first run to end and the second to be mid-call", func() bool {
		return effects(running) == 1 && len(show().Live) == 1
	})

	before := show()
	if !before.Running || before.PID <= 0 {
		t.Fatalf("the machine is not running before the kill: %+v", before)
	}
	must(t, syscall.Kill(before.PID, syscall.SIGKILL))
	landed(t, "the killed machine to be gone", func() bool { return !show().Running })
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start after the kill [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "up"); code != 0 {
		t.Fatalf("up [exit %d]\n%s", code, out)
	}

	// The call that was running: FAILED with the reason, never run again.
	landed(t, "the interrupted run to settle", func() bool {
		state, _ := shown("2")
		return state.Status != "in_progress" && state.Status != "queued" && state.Status != ""
	})
	interrupted, out := shown("2")
	if interrupted.Status != "failed" || !strings.Contains(out, "will not be replayed") {
		t.Fatalf("the interrupted run did not fail with the no-replay reason:\n%s", out)
	}
	// The call that had finished: its result and output, collected from the restarted machine.
	landed(t, "the finished run to be collected", func() bool {
		state, _ := shown("1")
		return state.Status == "completed" && len(state.Output) == 1 && state.Output[0].Status == "completed"
	})
	kept, out := shown("1")
	if !strings.Contains(out, `"effects":1`) || kept.Output[0].Name != "report" {
		t.Fatalf("the finished run lost its result or output:\n%s", out)
	}
	if report, err := os.ReadFile(kept.Output[0].Path); err != nil || string(report) != "effects=1\n" {
		t.Fatalf("the finished run's output reads %q, %v", report, err)
	}
	// Nothing runs again: the machine holds no live run and each effect was made once.
	landed(t, "the restarted machine to hold no live run", func() bool { return len(show().Live) == 0 })
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(500 * time.Millisecond) {
		if effects(running) != 1 || effects(finished) != 1 {
			t.Fatalf("started work ran again: %d and %d effects", effects(finished), effects(running))
		}
	}
	if after := show(); !after.Running || after.PID == before.PID {
		t.Fatalf("the machine after the restart: %+v (killed pid %d)", after, before.PID)
	}
}
