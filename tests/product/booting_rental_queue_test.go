package producttest

import (
	"encoding/json"
	"encoding/pem"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

// A run named for a rental still starting is accepted and queued, never refused and never
// told a release from the boot: Calcifer sends the runs once the Hub publishes the ready
// machine, in submission order, and a rental that fails first fails its queued run, naming why.
func TestRunsQueueForARentalStillStarting(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the rental's machine")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the rental's machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czq")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ncalcifer:\n  idle_shutdown_s: 0\n"), 0o600))
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
	t.Cleanup(func() { store.Close() })
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, token, provider := providerHost(t, h, layout, source, uv)
	h.provider = provider
	publishParityRelease(t, h, root, parityProject(t))

	// Rentals this computer is acquiring, as `cozy rental new` leaves them while their pods boot.
	booting := func(key, id, name string) {
		t.Helper()
		_, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: key, Hub: h.server.URL, Reason: "manual", HourlyRateUSDMicros: 1},
			func(string) ([]byte, string, *exit.Error) { return []byte(`{}`), key, nil })
		fatal(t, problem)
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: name, SKU: "cpu", State: "booting",
			AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: h.server.URL}))
		fatal(t, store.AdvanceRentalOperation(key, id, "booting"))
		h.mu.Lock()
		h.rentals[id] = map[string]any{"rental_id": id, "name": name, "machine_name": name, "state": "booting",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 1,
			"creator_public_key": identity.PublicKey()}
		h.mu.Unlock()
	}
	// The machine's media token, minted before the paid ask as `cozy rental new` mints it.
	booting("parity-rental", parityRental, "tessa")
	must(t, os.WriteFile(layout.PendingRentalMediaToken("parity-rental"), secret.FileBody(secret.New(token)), 0o600))

	submit := func(key, value, rental string) string {
		t.Helper()
		code, out := runCozy(t, root, "run", parityPublished+"/add", "value="+value, "--rental="+rental, "--json", "--idempotency-key", key)
		request, problem := store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		if code != 0 || request == nil || strings.Contains(out, "did not say which releases") || strings.Contains(out, "not_ready") {
			t.Fatalf("%s was not queued for its starting rental [exit %d]\n%s", key, code, out)
		}
		return request.ID
	}
	first, second := submit("starting-first", "41", "tessa"), submit("starting-second", "1", "tessa")
	var listed struct{ Invocations []struct{ Status string } }
	if code, out := runCozy(t, root, "run", "list", "--json"); code != 0 || json.Unmarshal([]byte(out), &listed) != nil ||
		len(listed.Invocations) != 2 || listed.Invocations[0].Status != "queued" || listed.Invocations[1].Status != "queued" {
		t.Fatalf("the runs are not queued [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list"); code != 0 || !strings.Contains(out, "tessa") {
		t.Fatalf("the runs do not say they wait for tessa [exit %d]\n%s", code, out)
	}

	h.mu.Lock()
	ready := h.rentals[parityRental]
	ready["state"], ready["worker_address"], ready["media_address"] = "ready", launch.Addr, launch.MediaAddr
	ready["cert_pem"] = string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf}))
	ready["worker_id"], ready["worker_boot_id"] = launch.WorkerID, launch.BootID
	h.mu.Unlock()
	settled := func(id, want string) records.Request {
		t.Helper()
		for deadline := time.Now().Add(5 * time.Minute); ; time.Sleep(time.Second) {
			row, problem := store.RequestRow(id)
			fatal(t, problem)
			if records.Settled(row.State) || time.Now().After(deadline) {
				if row.State != want {
					_, shown := runCozy(t, root, "run", "show", id)
					t.Fatalf("%s ended %s, not %s\n%s", id, row.State, want, shown)
				}
				return *row
			}
		}
	}
	for _, run := range []struct{ id, value string }{{first, `"value":42`}, {second, `"value":2`}} {
		settled(run.id, "succeeded")
		if code, out := runCozy(t, root, "run", "show", run.id, "--json"); code != 0 || !strings.Contains(out, run.value) {
			t.Fatalf("%s did not run on tessa [exit %d]\n%s", run.id, code, out)
		}
	}
	number := func(id string) uint64 {
		link, problem := store.MachineExecution(id)
		fatal(t, problem)
		return records.RunV1State(link).GetNumber()
	}
	if number(first) >= number(second) {
		t.Fatalf("tessa took the second run (#%d) before the first (#%d)", number(second), number(first))
	}

	booting("failing-rental", "pr-failing-0001", "tomo")
	doomed := submit("starting-doomed", "5", "tomo")
	h.mu.Lock()
	h.rentals["pr-failing-0001"]["state"] = "failed"
	h.rentals["pr-failing-0001"]["failure"] = map[string]any{"code": "readiness.provider_lost"}
	h.mu.Unlock()
	settled(doomed, "failed")
	if code, out := runCozy(t, root, "run", "show", doomed, "--json"); code != 0 || !strings.Contains(out, "rental ended before the run was sent") ||
		!strings.Contains(out, "readiness.provider_lost") {
		t.Fatalf("the failed rental's run does not say why [exit %d]\n%s", code, out)
	}
}
