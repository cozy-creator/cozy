package producttest

import (
	"encoding/pem"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// A run named for a rental never starts or contacts this computer's machine: not to package a
// local directory, not to describe it, not under a daemon that predates cozy.machine.v1 (the
// run then goes from the command itself). This computer's machine is a sentinel that records
// any launch; the rental is a real machine a provider booted.
func TestARentalRunNeverStartsThisComputersMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the rental's machine")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the rental's machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czn")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	witness := filepath.Join(root, "local-machine-launched")
	stubMachine(t, root, "#!/bin/sh\necho \"$@\" >> "+witness+"\nexit 3\n")
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, token, provider := providerHost(t, h, layout, source, uv)
	h.provider = provider
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "cpu", State: "ready",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID},
		string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})), secret.New(token), identity))
	store.Close()
	untouched := func(t *testing.T, what string) {
		t.Helper()
		if raw, err := os.ReadFile(witness); err == nil {
			t.Fatalf("%s started this computer's machine: %s", what, strings.TrimSpace(string(raw)))
		}
	}
	// Packaging a local directory happens in this command: a static read, no machine.
	if code, out := runCozy(t, root, "package", "install", parityProject(t)); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	untouched(t, "installing a local directory")
	runs := func(t *testing.T, daemon string, value, want string) {
		t.Helper()
		code, out := runCozy(t, root, "run", parityPackage+"/echo", "value="+value, "--rental=tessa", "--await", "--json")
		untouched(t, "a local package's run on a rental under "+daemon)
		if code != 0 || !strings.Contains(out, want) {
			t.Fatalf("the local package did not run on the rental under %s [exit %d]\n%s", daemon, code, out)
		}
		// A package this client never installed, spelled as the command line allows (it was
		// once described by this computer's machine): refused or run, never started here.
		_, _ = runCozy(t, root, "run", "./readbench_pkg/blob", "--rental=tessa", "--await", "--json")
		untouched(t, "an uninstalled package's run on a rental under "+daemon)
	}
	runs(t, "the current daemon", "41", `"value":82`)

	// Under a running daemon that predates cozy.machine.v1, the command runs it itself.
	if *olderCozy == "" {
		return
	}
	if code, stopped := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, stopped)
	}
	up := exec.Command(*olderCozy, "up")
	up.Env = childEnv(t, root)
	if raw, err := up.CombinedOutput(); err != nil {
		t.Fatalf("the older cozy did not start its daemon: %v\n%s", err, raw)
	}
	runs(t, "an older daemon", "5", `"value":10`)
}
