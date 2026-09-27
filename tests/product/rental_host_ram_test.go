//go:build !windows

package producttest

import (
	"strings"
	"testing"
)

// Host RAM is not a placement input: a GPU product the hub lists without a RAM figure,
// or with zero, is listed and bought like any other.
func TestGPUProductWithoutHostRAMFigureIsBought(t *testing.T) {
	for _, row := range []struct {
		name string
		ram  any
	}{{"absent", nil}, {"zero", 0}} {
		t.Run(row.name, func(t *testing.T) {
			root, _, stand := rentalEndRoot(t, "rental-ram-"+row.name)
			sku := map[string]any{
				"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
				"base_worker_profile": "torch2.14.0-cu130-cp312-linux-x86", "compute_capability": "9.0", "vram_gb": 94,
				"price_usd_micros_per_hour": 3_190_000,
			}
			if row.ram != nil {
				sku["minimum_ram_per_gpu_gb"] = row.ram
			}
			stand.setSKUs(sku)
			asked := 0
			stand.mu.Lock()
			stand.rent = func(request map[string]any) map[string]any {
				asked++
				return map[string]any{"rental_id": "pr-ram-" + row.name, "name": request["name"], "state": "pending_acquisition",
					"requested_accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1, "hourly_rate_usd_micros": 3_190_000}
			}
			stand.mu.Unlock()
			if code, out := runCozy(t, root, "rental", "new", "--json"); code != 0 || !strings.Contains(out, "h100-nvl") {
				t.Fatalf("a GPU product without a RAM figure was not listed: %d %s", code, out)
			}
			code, out := runCozy(t, root, "rental", "new", "h100-nvl", "--idempotency-key", "ram-"+row.name, "--timeout=2s", "--json")
			stand.mu.Lock()
			defer stand.mu.Unlock()
			if asked != 1 || strings.Contains(out, "rental_catalog") {
				t.Fatalf("a GPU product without a RAM figure was not bought (%d asks, exit %d): %s", asked, code, out)
			}
		})
	}
}
