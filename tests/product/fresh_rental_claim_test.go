package producttest

import (
	"encoding/json"
	"encoding/pem"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// A fresh rental's first calls arrive together, as a client's do. The worker takes one
// control stream at a time, and the daemon's orchestrator already holds it, so every call
// runs on the Claim it carries: the worker records one Control Claim, and no call's Claim
// fences another's or the orchestrator's (hakufu: five Claim cycles on a fresh boot).
func TestFreshRentalConcurrentCallsShareOneControlClaim(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor the rental runs")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czf")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
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
	defer store.Close()
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, token, provider := providerHost(t, h, layout, source, uv)
	const model, count = "Virtual Accelerator", 4
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": model, "accelerator_count": count, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr}
	h.inventories = map[string]json.RawMessage{parityRental: json.RawMessage(`{"format":"tensorhub.image_inventory/1","profile":"python3.12-cpu-linux-x86","python":"3.12"}`)}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "virtual-4", State: "ready",
		AcceleratorModel: model, AcceleratorCount: count, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID},
		string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})), secret.New(token), identity))
	if code, out := runCozy(t, root, "package", "install", parityProject(t), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}

	var wg sync.WaitGroup
	failures := make([]string, 3)
	for i := range failures {
		wg.Add(1)
		go func() {
			defer wg.Done()
			code, out := runCozy(t, root, "run", parityPackage+"/add", fmt.Sprintf("value=%d", i), "--rental=tessa",
				"--await", "--json", "--idempotency-key", fmt.Sprintf("fresh-%d", i))
			if code != 0 || !strings.Contains(out, fmt.Sprintf(`"value":%d`, i+1)) {
				failures[i] = fmt.Sprintf("call %d [exit %d]: %s", i, code, out)
			}
		}()
	}
	wg.Wait()
	log, _ := os.ReadFile(filepath.Join(filepath.Dir(provider), "host.log"))
	for _, failure := range failures {
		if failure != "" {
			t.Fatalf("%s\nrental Host log:\n%s", failure, log)
		}
	}
	var ownership struct {
		Epoch int `json:"control_stream_epoch"`
	}
	raw, err := os.ReadFile(filepath.Join(provider, "run/cozy/worker/ownership.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &ownership))
	if ownership.Epoch != 1 {
		t.Fatalf("the fresh rental took %d Control Claims; its owner holds one stream\nrental Host log:\n%s", ownership.Epoch, log)
	}
}
