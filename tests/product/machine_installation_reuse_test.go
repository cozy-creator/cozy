package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Exercise the actual command, Host, Runtime, installed environments and journal.
// The source fixture returns a fresh random token so installation reuse cannot be
// confused with reusing the result of an earlier execution.
func TestMachineExecutionActualHostInstallationReuse(t *testing.T) {
	if *machineExecutionScript == "" {
		t.Skip("requires a current captured script and actual Host cohort")
	}
	layout, store, host, path, restartHost := startActualChildHost(t)
	raw, err := os.ReadFile(*machineExecutionScript)
	must(t, err)
	script := filepath.Join(layout.Root, "installation-reuse.py")
	must(t, os.WriteFile(script, raw, 0600))
	tokens := map[string]bool{}
	runKey := strconv.FormatInt(time.Now().UnixNano(), 10)
	type proof struct {
		Request               string `json:"request"`
		Uploaded, Reused      int
		ElapsedMS, ReportedMS int64
	}
	var rows []proof
	run := func(label string, reused bool) {
		t.Helper()
		key := "installation-" + runKey + "-" + label
		began := time.Now()
		code, out := runCozyPath(t, layout.Root, path, "run", script, "--rental", "child-host", "--await", "--json", "--full", "--idempotency-key", key)
		elapsed := time.Since(began).Milliseconds()
		if code != 0 {
			t.Fatalf("%s [%d]: %s", label, code, out)
		}
		var result struct {
			WallMS int64 `json:"wall_ms"`
			Result struct {
				Value string `json:"value"`
			} `json:"result"`
		}
		for _, line := range strings.Split(out, "\n") {
			var candidate struct {
				WallMS int64 `json:"wall_ms"`
				Result struct {
					Value string `json:"value"`
				} `json:"result"`
			}
			if json.Unmarshal([]byte(line), &candidate) == nil && candidate.WallMS > 0 {
				result = candidate
			}
		}
		if result.Result.Value == "" || tokens[result.Result.Value] {
			t.Fatalf("%s did not execute fresh code: %s", label, out)
		}
		tokens[result.Result.Value] = true
		if result.WallMS < 0 || result.WallMS > elapsed {
			t.Fatalf("recorded request timer exceeds the observing command: real=%dms reported=%dms", elapsed, result.WallMS)
		}
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if request == nil || request.State != "succeeded" {
			t.Fatalf("%s not succeeded: %+v", label, request)
		}
		events, problem := store.EventsAfter(request.ID, 0, 1000)
		fatal(t, problem)
		terminalAt, problem := store.TerminalEventAt(request.ID)
		fatal(t, problem)
		started, err := time.Parse(time.RFC3339Nano, request.CreatedAt)
		must(t, err)
		ended, err := time.Parse(time.RFC3339Nano, terminalAt)
		must(t, err)
		if result.WallMS != ended.Sub(started).Milliseconds() {
			t.Fatalf("initial awaited machine wall differs from durable timestamps: %s", out)
		}
		row := proof{Request: request.ID, ElapsedMS: elapsed, ReportedMS: result.WallMS}
		for _, event := range events {
			switch event.Type {
			case "machine.package_uploaded":
				row.Uploaded++
			case "machine.package_reused":
				row.Reused++
			}
		}
		if reused && (row.Uploaded != 0 || row.Reused == 0) {
			t.Fatalf("%s installed again: %+v", label, row)
		}
		if !reused && row.Uploaded == 0 {
			t.Fatalf("%s did not install its missing revision: %+v", label, row)
		}
		rows = append(rows, row)
	}
	run("cold", false)
	run("warm", true)
	code, out := runCozyPath(t, layout.Root, path, "down", "--json")
	if code != 0 {
		t.Fatalf("client down [%d]: %s", code, out)
	}
	restartHost()
	run("restart", true)
	// No invocation is running. Remove only this fixture's installed generations,
	// preserving them under a different name for diagnosis rather than deleting them.
	diagnosis, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `from pathlib import Path
import json
from cozy_runtime.internal.placement_materialization import INSTALL_ROOT
root=Path(INSTALL_ROOT)/'contents'
assert root.is_dir()
target=root.with_name('contents-before-repair')
assert not target.exists()
root.rename(target)
root.mkdir()
print(json.dumps({'retained':str(target)}))`).CombinedOutput()
	if err != nil {
		t.Fatalf("retain fixture installation before repair: %v %s", err, diagnosis)
	}
	run("missing", false)
	run("repaired", true)
	must(t, os.WriteFile(script, append(raw, []byte("\n# changed caller capture\n")...), 0600))
	run("changed", false)
	observed, err := json.MarshalIndent(rows, "", "  ")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(layout.Root, "installation-reuse-proof.json"), observed, 0600))
	t.Logf("actual CLI installation reuse: %s", observed)
}
