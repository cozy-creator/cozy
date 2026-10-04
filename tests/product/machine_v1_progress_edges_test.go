package producttest

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

var cpuProgressBurst = flag.String("cpu-progress-burst", "", "cozy-machine tests/fixtures/cpu_progress_burst: genuine CPU progress burst")

// Ordinary CLI submits an authored CPU callable to the real Rust machine, receives SDK
// step callbacks through the managed executor, and reattaches to the actual stored record.
// This qualifies the consumer pipeline, not model inference or GPU throughput.
func TestMachineV1ProgressStageEndpointSurvivesOrdinaryCLIAndRestart(t *testing.T) {
	if *machineHostBinary == "" || *cpuProgressBurst == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-progress-burst=<cozy-machine>/tests/fixtures/cpu_progress_burst")
	}
	t.Setenv("CUDA_VISIBLE_DEVICES", "")
	root, err := os.MkdirTemp(os.TempDir(), "czprogress")
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
	project := filepath.Join(t.TempDir(), "cpu_progress_burst")
	must(t, os.CopyFS(project, os.DirFS(*cpuProgressBurst)))
	if *machineRuntimeWheel != "" {
		wheel, err := filepath.Abs(*machineRuntimeWheel)
		must(t, err)
		manifest := filepath.Join(project, "pyproject.toml")
		content, err := os.ReadFile(manifest)
		must(t, err)
		content = append(content, []byte(fmt.Sprintf("\n[tool.uv.sources]\ncozy-runtime = {path = %q}\n", wheel))...)
		must(t, os.WriteFile(manifest, content, 0o600))
	}
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [%d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/cozy-machine-cpu-progress-burst/burst", "--await", "--json"); code != 0 || !strings.Contains(out, `"steps":30`) {
		t.Fatalf("genuine CPU burst did not complete [%d]\n%s", code, out)
	}
	for _, restart := range []bool{false, true} {
		if restart {
			if code, out := runCozy(t, root, "down"); code != 0 {
				t.Fatalf("owned controller down [%d]\n%s", code, out)
			}
		}
		code, stdout, events := runCozyStreams(t, root, "run", "watch", "1", "--json")
		if code != 0 || !strings.Contains(events, `"stage":"denoise"`) || !strings.Contains(events, `"position":30`) || !strings.Contains(events, `"stage":"decoding"`) {
			t.Fatalf("ordinary watch lost genuine endpoint afterRestart=%v [%d]\n%s\n%s", restart, code, stdout, events)
		}
	}
	_, _ = runCozy(t, root, "down")
	store, problem := records.Open(home.Paths(root).DB)
	fatal(t, problem)
	defer store.Close()
	events, problem := store.EventsAfter("", 0, 1000)
	fatal(t, problem)
	samples := 0
	for _, event := range events {
		if event.Type == "machine.progress" {
			samples++
		}
	}
	if samples == 0 || samples > 10 {
		t.Fatalf("expected bounded transition-flushed samples, got %d", samples)
	}
}
