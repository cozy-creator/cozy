package producttest

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalNoCreateAfterAcceptedPendingReturnsFailure(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-no-create")
	stand.setSKUs(map[string]any{
		"name": "l4", "accelerator_model": "NVIDIA L4", "accelerator_count": 1,
		"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86", "compute_capability": "8.9",
		"vram_gb": 24, "minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 100_000,
	})
	creates := 0
	accepted, stop := make(chan struct{}), make(chan struct{})
	defer close(stop)
	stand.mu.Lock()
	stand.rent = func(request map[string]any) map[string]any {
		creates++
		if creates == 1 {
			close(accepted)
		}
		return map[string]any{
			"rental_id": "pr-no-create", "name": request["name"], "state": "pending_acquisition",
			"requested_accelerator_model": "NVIDIA L4", "accelerator_count": 1,
			"hourly_rate_usd_micros": 100_000,
		}
	}
	stand.mu.Unlock()
	go func() {
		select {
		case <-accepted:
			// The POST handler holds stand.mu until its pending response is
			// serialized. Only the subsequent poll can observe this failure.
			stand.setState("pr-no-create", "failed", "provider_create_did_not_happen")
		case <-stop:
		}
	}()

	code, raw, _ := runCozyStreams(t, root, "rental", "new", "l4", "--json", "--timeout=10s")
	var refused struct {
		Error struct {
			Code   string `json:"code"`
			Remedy string `json:"remedy"`
		} `json:"error"`
	}
	if code == 0 || json.Unmarshal([]byte(raw), &refused) != nil || refused.Error.Code != "provider_create_did_not_happen" {
		t.Fatalf("no-create did not return typed failure [exit %d]: %s", code, raw)
	}
	if !strings.Contains(raw, "Please try again later or rent a different GPU") || strings.Contains(raw, "reclaim") {
		t.Fatalf("a proven no-create still asks for reclaim: %s", raw)
	}
	code, raw = runCozy(t, root, "rental", "list", "--full", "--json")
	var inventory struct {
		Rentals []json.RawMessage `json:"rentals"`
		Count   int               `json:"machines_running"`
		Rate    int64             `json:"hourly_spend_usd_micros"`
	}
	if code != 0 || json.Unmarshal([]byte(raw), &inventory) != nil || len(inventory.Rentals) != 0 || inventory.Count != 0 || inventory.Rate != 0 {
		t.Fatalf("no-create remains in active inventory [exit %d]: %s", code, raw)
	}
	stand.mu.Lock()
	defer stand.mu.Unlock()
	if creates != 1 || stand.released["pr-no-create"] != 0 {
		t.Fatalf("failed create was retried/released: creates=%d releases=%d", creates, stand.released["pr-no-create"])
	}
}

func TestRentalNoCreateStartupReconcilesStaleAcquiringRow(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "rental-no-create-startup")
	stand.add("pr-no-create-stale", "ayanojou")
	stand.setState("pr-no-create-stale", "failed", "provider_create_did_not_happen")
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	fatal(t, st.RecordRental(records.Rental{ID: "pr-no-create-stale", MachineName: "ayanojou",
		SKU: "l4", AcceleratorModel: "NVIDIA L4", AcceleratorCount: 1,
		HourlyRateUSDMicros: 100_000, State: "acquiring", Hub: hubURL}))
	live := startDaemonProcess(t, root)
	response := live.call(t, "GET", "/v1/local/rentals", nil)
	var inventory api.RentalInventory
	must(t, json.Unmarshal(response.Body, &inventory))
	if len(inventory.Rentals)+len(inventory.Unrecorded)+len(inventory.Pending) != 0 || inventory.MachinesRunning != 0 || inventory.HourlySpendUSDMicros != 0 {
		t.Fatalf("stale failed acquisition remains active: %s", response.Body)
	}
	row, problem := st.RentalRow("pr-no-create-stale")
	fatal(t, problem)
	if row == nil || row.State != "failed" || row.Failure.Code != "provider_create_did_not_happen" {
		t.Fatalf("startup did not preserve failure history: %+v", row)
	}
	if stand.releases("pr-no-create-stale") != 0 {
		t.Fatal("startup attempted to release a nonexistent pod")
	}
	blocked, problem := st.ClientShutdownObligations()
	fatal(t, problem)
	if len(blocked) != 0 {
		t.Fatalf("definitive no-create still blocks client shutdown: %+v", blocked)
	}
}
