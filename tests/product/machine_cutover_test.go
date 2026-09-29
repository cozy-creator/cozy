package producttest

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
)

var legacyHostBinary = flag.String("legacy-host", "", "a tensorhub pod-supervisor: this computer's machine Host before one binary")

// This computer's machine, installed when its Host was the tensorhub pod-supervisor, becomes a
// cozy machine: once it is idle the old Host is stopped by its PID and this cozy runs it. Its
// runs, installs and TensorFS store stay; its runs are still listed, and a new run works.
func TestTheMachineCutsOverToTheCozyHost(t *testing.T) {
	if *legacyHostBinary == "" {
		t.Skip("requires -legacy-host: the tensorhub pod-supervisor a machine ran before")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czc")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		_ = removeAllForce(root)
	})
	if code, out := runCozy(t, root, "machine", "install", "--host", *legacyHostBinary); code == 0 || !strings.Contains(out, "is not a cozy binary") {
		t.Fatalf("machine install accepted the tensorhub Host [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	// The tensorhub Host, pinned while it makes the machine's first run as an older cozy would.
	source := machines.Source{Host: *legacyHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel, Pinned: true}
	_, problem = machines.NewHost(layout.Machine, "", nil).Install(context.Background(), source, uv)
	fatal(t, problem)
	virtualInventory(t, filepath.Join(layout.Machine, "root"))
	if code, out := runCozy(t, root, "package", "install", parityProjectOn(t, source), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	host := func() (int, string) {
		var record struct{ PID int }
		raw, err := os.ReadFile(filepath.Join(layout.Machine, "host.json"))
		must(t, err)
		must(t, json.Unmarshal(raw, &record))
		exe, _ := os.Readlink("/proc/" + strconv.Itoa(record.PID) + "/exe")
		return record.PID, exe
	}
	run := func(key string) string {
		t.Helper()
		code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json", "--idempotency-key", key)
		if code != 0 || !strings.Contains(out, `"value":42`) {
			t.Fatalf("the %s run [exit %d]: %s", key, code, out)
		}
		return out
	}

	run("before")
	before, exe := host()
	if module := machines.HostModule(exe); module == machines.CozyModule {
		t.Fatalf("the machine did not start on the tensorhub Host: %s is %s", exe, module)
	}
	// What an older cozy recorded: the tensorhub Host by a path, no pin.
	path := filepath.Join(layout.Machine, "installed.json")
	var installed map[string]any
	raw, err := os.ReadFile(path)
	must(t, err)
	must(t, json.Unmarshal(raw, &installed))
	delete(installed, "host_pinned")
	delete(installed["host"].(map[string]any), "module")
	raw, err = json.Marshal(installed)
	must(t, err)
	must(t, os.WriteFile(path, raw, 0o600))

	run("after")
	after, exe := host()
	cozy, err := filepath.EvalSymlinks(filepath.Join(layout.Machine, "root/usr/local/bin/cozy"))
	must(t, err)
	if after == before || machines.HostModule(exe) != machines.CozyModule || exe != cozy {
		t.Fatalf("the idle machine did not cut over to the cozy Host: pid %d then %d, running %s", before, after, exe)
	}
	if _, err := os.Stat("/proc/" + strconv.Itoa(before)); err == nil {
		t.Fatalf("the tensorhub Host %d still runs", before)
	}
	code, out := runCozy(t, root, "run", "list", "--json")
	var listed struct{ Completed int }
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || listed.Completed != 2 {
		t.Fatalf("the machine's runs are not all listed after the cutover [exit %d]: %s", code, out)
	}
}

// A machine whose Runtime cannot start (here a control history from another boot, as a Host
// cutover once left it) fails the run with the Runtime's reason instead of waiting forever.
// Once that is repaired, the next run wakes the same Host's Runtime.
func TestARunFailsWhenTheMachinesRuntimeCannotStart(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: a cozy build the machine runs as its Host")
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
	var history map[string]any
	original, err := os.ReadFile(path)
	must(t, err)
	must(t, json.Unmarshal(original, &history))
	history["worker_boot_id"] = strings.Repeat("A", 43)
	raw, err := json.Marshal(history)
	must(t, err)
	must(t, os.WriteFile(path, raw, 0o600))

	code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json")
	if code == 0 || !strings.Contains(out, "machine.runtime_gone") || !strings.Contains(out, "worker_ownership_invalid") {
		t.Fatalf("a run on a machine whose Runtime cannot start did not fail with its reason [exit %d]: %s", code, out)
	}
	must(t, os.WriteFile(path, original, 0o600))
	if code, out := runCozy(t, root, "run", parityPackage+"/add", "value=41", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":42`) {
		t.Fatalf("the next run did not wake the repaired machine [exit %d]: %s", code, out)
	}
}
