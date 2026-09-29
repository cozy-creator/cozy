//go:build !windows

package producttest

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A pod the hub locks at or below the quoted rate is kept; only a rate above the quote
// the renter agreed to is released before it can bill.
func TestRentalLockedAtOrBelowTheQuoteIsKept(t *testing.T) {
	for _, row := range []struct {
		name   string
		locked int64
		kept   bool
	}{{"below", 2_900_000, true}, {"equal", 3_190_000, true}, {"above", 3_500_000, false}} {
		t.Run(row.name, func(t *testing.T) {
			root, _, stand := rentalEndRoot(t, "rental-rate-"+row.name)
			stand.setSKUs(map[string]any{
				"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "torch2.14.0-cu130-cp312-linux-x86", "compute_capability": "9.0", "vram_gb": 94,
				"minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3_190_000,
			})
			id := "pr-rate-" + row.name
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				return map[string]any{"rental_id": id, "name": request["name"], "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1, "hourly_rate_usd_micros": row.locked}
			}
			stand.mu.Unlock()
			code, out := rentUntilRecorded(t, root, id, "h100-nvl", "--idempotency-key", "rate-"+row.name, "--json")
			if code == 0 {
				t.Fatalf("a rental that never became ready succeeded: %s", out)
			}
			released := stand.releases(id)
			if row.kept && (released != 0 || strings.Contains(out, "rental.hourly_rate_changed")) {
				t.Fatalf("a rental locked at %d was released (%d): %s", row.locked, released, out)
			}
			if !row.kept && (released != 1 || !strings.Contains(out, "rental.hourly_rate_changed")) {
				t.Fatalf("a rental locked above the quote was kept (%d): %s", released, out)
			}
		})
	}
}

// Tensorhub may name a machine differently from the request. The hub's name is adopted
// and the purchase continues; a name another local rental holds leaves the requested one.
func TestRentalRenamedByTheHubIsKept(t *testing.T) {
	for _, row := range []struct {
		name, hubName, want string
		taken               bool
	}{{"adopted", "hub-given", "hub-given", false}, {"taken", "hub-taken", "", true}} {
		t.Run(row.name, func(t *testing.T) {
			root, hubURL, stand := rentalEndRoot(t, "rental-rename-"+row.name)
			stand.setSKUs(map[string]any{
				"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "torch2.14.0-cu130-cp312-linux-x86", "compute_capability": "9.0", "vram_gb": 94,
				"price_usd_micros_per_hour": 3_190_000,
			})
			if row.taken {
				store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
				fatal(t, problem)
				fatal(t, store.RecordRental(records.Rental{ID: "pr-rename-other", MachineName: row.hubName, SKU: "cpu",
					AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL}))
				store.Close()
				stand.add("pr-rename-other", row.hubName)
			}
			var requested string
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				requested, _ = request["name"].(string)
				return map[string]any{"rental_id": "pr-rename-" + row.name, "name": row.hubName, "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1, "hourly_rate_usd_micros": 3_190_000}
			}
			stand.mu.Unlock()
			_, out := rentUntilRecorded(t, root, "pr-rename-"+row.name, "h100-nvl", "--idempotency-key", "rename-"+row.name, "--json")
			if strings.Contains(out, "machine_name_changed") {
				t.Fatalf("a renamed rental was refused: %s", out)
			}
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			rented, problem := store.RentalRow("pr-rename-" + row.name)
			fatal(t, problem)
			want := row.want
			if want == "" {
				want = requested
			}
			if rented == nil || rented.MachineName != want {
				t.Fatalf("rental recorded as %+v, want name %q: %s", rented, want, out)
			}
		})
	}
}

// `--disk-gb=160` on production's CPU flavors fits only the $0.48 flavor, while the
// listing prices the default disk at $0.28. The renter consents to the Hub's quote for
// the exact request, which names that machine and its storage before anything is bought,
// so the rental it locks is kept and costs compute plus storage from its create on; a disk
// no machine holds is refused before anything is bought.
func TestRentalConsentsToTheQuoteForItsDisk(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-quote")
	stand.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"base_worker_profile": "torch2.14.0-cpu-cp312-linux-x86", "price_usd_micros_per_hour": 280_000})
	var quoted []any
	stand.mu.Lock()
	stand.quote = func(request map[string]any) (int, string) {
		quoted = append(quoted, request["container_disk_gb"])
		if request["container_disk_gb"] == float64(500) {
			return http.StatusUnprocessableEntity, `{"error":{"code":"rental.disk_unavailable","message":"a 500 GB container disk: cpu offers allow at most 160 GB"}}`
		}
		return http.StatusOK, `{"name":"cpu","accelerator_count":1,"price_usd_micros_per_hour":480000,"storage_usd_micros_per_hour":22240,"container_disk_gb":160,"vcpu_count":16,"memory_gb":32}`
	}
	stand.rent = func(request map[string]any) map[string]any {
		return map[string]any{"rental_id": "pr-quote", "name": request["name"], "state": "pending_acquisition",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 480_000,
			"compute_usd_micros_per_hour": 480_000, "storage_usd_micros_per_hour": 22_240, "vcpu_count": 16, "memory_gb": 32}
	}
	stand.mu.Unlock()
	_, out := rentUntilRecorded(t, root, "pr-quote", "cpu", "--disk-gb=160", "--idempotency-key", "quote-160")
	if stand.releases("pr-quote") != 0 || strings.Contains(out, "rental.hourly_rate_changed") ||
		!strings.Contains(out, "cpu (16 vCPU, 32 GB) with a 160 GB disk: $0.50/hour (compute $0.48 + storage $0.02) (the default-disk listing is $0.28/hr)") {
		t.Fatalf("the quoted rate was not what the renter consented to: %s", out)
	}
	var listed struct {
		Rentals []map[string]any `json:"rentals"`
	}
	if code, out := runCozy(t, root, "rental", "list", "--json"); code != 0 || json.Unmarshal([]byte(out), &listed) != nil ||
		len(listed.Rentals) != 1 || listed.Rentals[0]["compute_usd_micros_per_hour"] != 480_000.0 ||
		listed.Rentals[0]["storage_usd_micros_per_hour"] != 22_240.0 || listed.Rentals[0]["vcpu_count"] != 16.0 {
		t.Fatalf("the listing does not split the fresh rental's cost [exit %d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "list", "--no-watch"); code != 0 || !strings.Contains(out, "$0.50") ||
		!strings.Contains(out, "Current spend per hour: $0.50") {
		t.Fatalf("the board does not show the fresh rental at compute plus storage [exit %d]: %s", code, out)
	}
	code, out := runCozy(t, root, "rental", "new", "cpu", "--disk-gb=500", "--idempotency-key", "quote-500", "--json")
	stand.mu.Lock()
	rentals := len(stand.rentals)
	stand.mu.Unlock()
	if code == 0 || !strings.Contains(out, "rental.disk_unavailable") || rentals != 1 || len(quoted) != 2 {
		t.Fatalf("an unholdable disk reached a purchase (rentals=%d, quotes=%v): %s", rentals, quoted, out)
	}
}
