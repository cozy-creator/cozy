package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
)

// Retired lifecycle metadata is a refusal census, even if a stale activity file
// says idle. It cannot authorize adoption or signal the old process.
func TestLegacyAgentAdoptionPreservesRetiredProcess(t *testing.T) {
	source, err := exec.LookPath("sleep")
	must(t, err)
	dir := t.TempDir()
	host := machines.NewHost(dir, "", nil)
	binary := filepath.Join(host.Root(), "usr/local/bin/pod-supervisor")
	must(t, os.MkdirAll(filepath.Dir(binary), 0755))
	raw, err := os.ReadFile(source)
	must(t, err)
	must(t, os.WriteFile(binary, raw, 0755))
	command := exec.Command(binary, "3600")
	must(t, command.Start())
	t.Cleanup(func() { _ = command.Process.Kill(); _ = command.Wait() })
	record, _ := json.Marshal(map[string]any{"pid": command.Process.Pid, "worker_id": "machine-still-working"})
	must(t, os.WriteFile(filepath.Join(dir, "host.json"), record, 0600))
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), []byte(`{"host":{"module":"github.com/cozy-creator/cozy","name":"cozy"}}`), 0600))
	activity := filepath.Join(host.Root(), "run/cozy/bootstrap/worker-activity")
	must(t, os.MkdirAll(filepath.Dir(activity), 0755))
	for _, state := range []string{`{"active_work":true,"holding":["accepted run 7"]}`, `{"active_work":false,"holding":[]}`} {
		must(t, os.WriteFile(activity, []byte(state), 0600))
		for _, idle := range []bool{false, true} {
			changed, problem := host.Adopt(t.Context(), idle)
			if changed || problem == nil || problem.ErrName() != "machine.legacy_process_running" {
				t.Fatalf("retired namespace authorized adoption: changed=%v problem=%v", changed, problem)
			}
		}
	}
	if err := command.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatalf("retired process was terminated: %v", err)
	}
	after, err := os.ReadFile(binary)
	must(t, err)
	if !bytes.Equal(after, raw) {
		t.Fatal("refusal replaced the retired executable")
	}
	if _, err := os.Stat(filepath.Join(dir, "agent.json")); !os.IsNotExist(err) {
		t.Fatal("refusal created current launch authority")
	}
}

// Old ownership history is not authority over the new one-agent Runtime launch.
func TestLegacyControlHistoryDoesNotBlockStandaloneAgent(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the standalone agent the machine runs")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czg")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		_ = removeAllForce(root)
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel, Pinned: true}
	_, problem = machines.NewHost(layout.Machine, "", nil).Install(context.Background(), source, uv)
	fatal(t, problem)
	virtualInventory(t, filepath.Join(layout.Machine, "root"))
	if code, out := runCozy(t, root, "package", "install", parityProjectOn(t, source), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json"); code != 0 {
		t.Fatalf("the first run [exit %d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]: %s", code, out)
	}
	path := filepath.Join(layout.Machine, "root/run/cozy/worker/ownership.json")
	must(t, os.MkdirAll(filepath.Dir(path), 0755))
	must(t, os.WriteFile(path, []byte(`{"worker_boot_id":"another-boot","record_owner_epoch":999999}`), 0600))
	if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":42`) {
		t.Fatalf("legacy control history prevented an independent supervisor launch [exit %d]: %s", code, out)
	}
}
