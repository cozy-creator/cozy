package producttest

import (
	"flag"
	"os"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A development worker's immutable image inventory is deliberately not the live
// SDK authority. Its ordinary protected-base intake must enforce the controller's
// additional minimum even when the user's own compatible bound allows old code.
func TestCapturedScriptRefusesOldLiveWorkerRuntime(t *testing.T) {
	if !*childHostUpdatable {
		t.Skip("requires an explicitly updatable old task worker")
	}
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
				t.Errorf("own cleanup: %v %s", err, out)
			}
		}
	})
	script := filepath.Join(layout.Root, "old-worker.py")
	must(t, os.WriteFile(script, []byte(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=0.16.0,<1"]
# ///
async def main(ctx):
    ctx.log("must not execute on an unsafe worker")
`), 0600))
	status, out := runCozyPath(t, layout.Root, path, "run", script, "--rental-only", "--await", "--json")
	must(t, os.WriteFile(filepath.Join(layout.Root, "old-worker-refusal.json"), []byte(out), 0600))
	if status == 0 || !strings.Contains(out, "package_environment_dependency_base_conflict") || !strings.Contains(out, "0.16.8") || !strings.Contains(out, ">=0.16.10") {
		t.Fatalf("old live Runtime did not refuse its exact execution requirement [%d]: %s", status, out)
	}
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	if request == nil || request.Ordinal != 0 || request.State != "blocked" {
		t.Fatalf("old worker executed before refusal: %+v", request)
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("an execution attempt crossed the unsupported Runtime boundary")
	}
}

var childHostPriorCLI = flag.String("child-host-prior-cli", "", "immutable prior Creator CLI for captured-revision cutover control")

func TestOldImmutableCaptureRefusesResumeWithoutRewritingWork(t *testing.T) {
	if *childHostPriorCLI == "" {
		t.Skip("requires an explicit immutable prior Creator CLI")
	}
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
				t.Errorf("own cleanup: %v %s", err, out)
			}
		}
	})
	prior := func(args ...string) string {
		t.Helper()
		cmd := exec.Command(*childHostPriorCLI, args...)
		cmd.Env = childEnv(t, layout.Root, "PATH="+path)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("prior ordinary CLI: %v %s", err, out)
		}
		return string(out)
	}
	script := filepath.Join(layout.Root, "legacy.py")
	must(t, os.WriteFile(script, []byte(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=0.16.0,<1"]
# ///
import asyncio
from pathlib import Path
async def main(ctx):
    Path("/tmp/cozy-legacy-retained-work").write_text("completed preparation")
    while True:
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.05)
`), 0600))
	must(t, os.WriteFile(filepath.Join(layout.Root, "legacy-submit.json"), []byte(prior("run", script, "--rental-only", "--json")), 0600))
	for deadline := time.Now().Add(2 * time.Minute); ; {
		data, err := exec.Command("docker", "exec", host.Container, "cat", "/tmp/cozy-legacy-retained-work").Output()
		if err == nil && string(data) == "completed preparation" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("old script did not establish retained work")
		}
		time.Sleep(100 * time.Millisecond)
	}
	must(t, os.WriteFile(filepath.Join(layout.Root, "legacy-pause.json"), []byte(prior("run", "pause", "1", "--json")), 0600))
	waitUntil(t, "prior request pauses", func() bool { r, p := store.RequestByReference("1"); return p == nil && r != nil && r.State == "paused" })
	before, problem := store.RequestByReference("1")
	fatal(t, problem)
	snapshot := daemon.Probe(config.Config{Home: layout.Root})
	if !snapshot.Up || snapshot.PID <= 0 {
		t.Fatal("prior task controller identity is absent")
	}
	must(t, syscall.Kill(snapshot.PID, syscall.SIGTERM))
	waitUntil(t, "prior controller closes normally", func() bool { return !daemon.Probe(config.Config{Home: layout.Root}).Up })
	// Current Creator observes the untouched old wheel and refuses a new attempt.
	status, out := runCozyPath(t, layout.Root, path, "run", "resume", "1", "--json")
	must(t, os.WriteFile(filepath.Join(layout.Root, "legacy-resume.json"), []byte(out), 0600))
	if status == 0 {
		status, out = runCozyPath(t, layout.Root, path, "run", "watch", "1", "--json")
	}
	must(t, os.WriteFile(filepath.Join(layout.Root, "legacy-refusal.json"), []byte(out), 0600))
	if status == 0 || !strings.Contains(out, "capture_runtime_floor_unproven") {
		t.Fatalf("old immutable capture resumed without a proven minimum [%d]: %s", status, out)
	}
	after, problem := store.RequestByReference("1")
	fatal(t, problem)
	attempts, problem := store.Attempts(before.ID)
	fatal(t, problem)
	if after.Ordinal != before.Ordinal || len(attempts) != 1 || after.LocalInstallationID != before.LocalInstallationID || after.InstallID != before.InstallID || !after.RetainWork {
		t.Fatalf("refusal changed immutable work: before=%+v after=%+v attempts=%d", before, after, len(attempts))
	}
	data, err := exec.Command("docker", "exec", host.Container, "cat", "/tmp/cozy-legacy-retained-work").Output()
	must(t, err)
	if string(data) != "completed preparation" {
		t.Fatal("refusal lost completed intermediary work")
	}
}
