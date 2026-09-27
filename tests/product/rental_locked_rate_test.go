//go:build !windows

package producttest

import (
	"strings"
	"testing"
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
			code, out := runCozy(t, root, "rental", "new", "h100-nvl", "--idempotency-key", "rate-"+row.name, "--timeout=2s", "--json")
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
