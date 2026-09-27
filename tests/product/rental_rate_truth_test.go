package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// th-120, the client half of the red arm: this host locked the $0.50/h catalog
// quote at rent time, the hub has since reconciled the rental to the $0.72/h
// the provider actually bills (GPU plus storage adders), and `cozy rental`
// must say $0.72 — the listing's fleet reconcile adopts the hub's billed rate
// into the local row, and the burn line runs on it, never the cached quote.
func TestRentalListingAdoptsTheHubBilledRate(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-rate-truth")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	hub := newFakeRentalHub(t, 0)
	port := hub.port()
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"), 0o600))

	hub.add("pr-redarm", "twine")
	hub.setRate("pr-redarm", 720_000)

	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "pr-redarm", MachineName: "twine", SKU: "gpu-cu130",
		AcceleratorModel: "NVIDIA GeForce RTX 3090", HourlyRateUSDMicros: 500_000,
		State: "ready", Hub: hubURL,
	}))

	code, out := runCozy(t, root, "rental", "list")
	if code != 0 || !strings.Contains(out, "Current spend per hour: $0.72") {
		t.Fatalf("the burn line still says the quote [exit %d]:\n%s", code, out)
	}
	for _, args := range [][]string{
		{"rental", "list", "--fields=machine,$/hour"},
		{"rental", "list", "--full", "--fields=machine,$/hour"},
	} {
		code, listed := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(listed, "$/HOUR") || !strings.Contains(listed, "$0.72") || strings.Contains(listed, "$0.50") {
			t.Fatalf("rental row does not show the reconciled billed rate [exit %d]:\n%s", code, listed)
		}
	}
	row, problem := store.RentalRow("pr-redarm")
	fatal(t, problem)
	if row == nil || row.HourlyRateUSDMicros != 720_000 {
		t.Fatalf("the local row did not adopt the billed rate: %+v", row)
	}
	// Settle the synthetic rental while its provider is still available. The
	// ordinary test shutdown must not wait on a Hub this fixture already closed.
	if code, out := runCozy(t, root, "rental", "end", "pr-redarm", "--json"); code != 0 {
		t.Fatalf("fixture rental release failed [exit %d]: %s", code, out)
	}
	hub.close()
}

// A confirmed pre-readiness container exit leaves the current fleet. Its
// bounded structured diagnosis remains recorded and names an acquisition failure.
func TestRentalListingShowsStructuredBootFailure(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-boot-failure")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	hubServer := newFakeRentalHub(t, 0)
	port := hubServer.port()
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"), 0o600))

	hubServer.add("pr-boot-failed", "yuzuriha")
	hubServer.mu.Lock()
	hubServer.rentals["pr-boot-failed"]["state"] = "failed"
	hubServer.rentals["pr-boot-failed"]["detail"] = "container_exited_before_readiness"
	hubServer.rentals["pr-boot-failed"]["failure"] = map[string]any{
		"code":                     "container_exited_before_readiness",
		"base_worker_image_digest": "sha256:" + strings.Repeat("a", 64),
		"provider":                 "runpod", "provider_resource_id": "nuowj42m20y65g",
		"provider_host_id": "host-7", "provider_state": "running", "container_state": "exited",
	}
	hubServer.mu.Unlock()

	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "pr-boot-failed", MachineName: "yuzuriha", SKU: "gpu-cu130",
		AcceleratorModel: "L4", HourlyRateUSDMicros: 100_000,
		State: "booting", Hub: hubURL,
	}))
	store.Close()

	code, human := runCozy(t, root, "rental", "list")
	if code != 0 || strings.Contains(human, "yuzuriha") {
		t.Fatalf("human current fleet retained the failed rental [exit %d]:\n%s", code, human)
	}
	code, raw := runCozy(t, root, "rental", "list", "--full", "--json")
	var listed struct {
		Rentals []rentalListingRow `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(raw), &listed) != nil || len(listed.Rentals) != 0 {
		t.Fatalf("typed current fleet retained the failed rental [exit %d]:\n%s", code, raw)
	}
	persisted, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	stored, problem := persisted.RentalRow("pr-boot-failed")
	fatal(t, problem)
	if stored == nil || stored.State != "failed" || stored.Failure.Code != "container_exited_before_readiness" ||
		stored.Failure.BaseWorkerImageDigest != "sha256:"+strings.Repeat("a", 64) ||
		stored.Failure.Provider != "runpod" || stored.Failure.ProviderResourceID != "nuowj42m20y65g" ||
		stored.Failure.ProviderHostID != "host-7" || stored.Failure.ProviderState != "running" ||
		stored.Failure.ContainerState != "exited" {
		t.Fatalf("local records lost the terminal diagnosis: %+v", stored)
	}
	changed := *stored
	changed.Failure.ProviderResourceID = "another-pod"
	if problem := persisted.RecordRental(changed); problem == nil || problem.ErrName() != "rental.attach_projection_conflict" {
		t.Fatalf("a changed terminal diagnosis was not refused: %v", problem)
	}
	persisted.Close()

	// The same diagnosis also remains the machine-stable error of a rental
	// acquisition, which is the error a managed run records when its buy fails.
	hubServer.setSKUs(map[string]any{
		"name": "l4", "accelerator_model": "NVIDIA L4", "accelerator_count": 1,
		"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86", "compute_capability": "8.9",
		"vram_gb": 24, "minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 100_000,
	})
	hubServer.mu.Lock()
	hubServer.rent = func(request map[string]any) map[string]any {
		return map[string]any{
			"rental_id": "pr-new-boot-failed", "name": request["name"], "state": "failed",
			"requested_accelerator_model": "NVIDIA L4", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000,
			"detail": "container_exited_before_readiness",
			"failure": map[string]any{"code": "container_exited_before_readiness",
				"base_worker_image_digest": "sha256:" + strings.Repeat("a", 64),
				"provider":                 "runpod", "provider_resource_id": "pod-new",
				"provider_host_id": "host-new", "provider_state": "running", "container_state": "exited"},
		}
	}
	hubServer.mu.Unlock()
	code, raw, _ = runCozyStreams(t, root, "rental", "new", "l4", "--json")
	var failed struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal([]byte(raw), &failed) != nil || code == 0 || failed.Error.Code != "container_exited_before_readiness" {
		t.Fatalf("rental acquisition lost the typed boot failure [exit %d]:\n%s", code, raw)
	}
}
