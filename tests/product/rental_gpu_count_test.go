package producttest

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
)

func TestRentGPUCountSelectsOneExactSKU(t *testing.T) {
	for _, test := range []struct {
		name, gpu, count, want string
	}{
		{"family_default", "h100", "", "h100-sxm5-80gb"},
		{"family_four", "h100", "4", "h100-sxm5-80gb-x4"},
		{"qualified_four", "h100-sxm5-80gb-x4", "", "h100-sxm5-80gb-x4"},
		{"qualified_equal", "h100-sxm5-80gb-x4", "4", "h100-sxm5-80gb-x4"},
		{"qualified_conflict", "h100-sxm5-80gb-x4", "2", ""},
		{"unoffered_degree", "h100", "3", ""},
		{"zero", "h100", "0", ""},
		{"negative", "h100", "-1", ""},
		{"cpu", "cpu", "1", ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			root, _, peer := rentalEndRoot(t, "rent-gpus")
			peer.publishListing()
			var skus []map[string]any
			for _, count := range []int{1, 4} {
				name := "h100-sxm5-80gb"
				if count > 1 {
					name += fmt.Sprint("-x", count)
				}
				skus = append(skus, map[string]any{"name": name, "accelerator_model": "NVIDIA H100 80GB HBM3",
					"accelerator_count": count, "price_usd_micros_per_hour": 100_000,
					"compute_capability": "9.0", "vram_gb": 80, "minimum_ram_per_gpu_gb": 64,
					"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86"})
			}
			peer.setSKUs(skus...)
			var mu sync.Mutex
			var asks []map[string]any
			peer.rent = func(request map[string]any) map[string]any {
				mu.Lock()
				defer mu.Unlock()
				asks = append(asks, request)
				// Accepted but unattachable deliberately ends the purchase fixture.
				return map[string]any{"rental_id": "rental-gpus", "name": request["name"],
					"state": "pending_acquisition", "requested_accelerator_model": "NVIDIA H100 80GB HBM3",
					"hourly_rate_usd_micros": 100_000}
			}
			args := []string{"rent", test.gpu, "--json"}
			if test.count != "" {
				args = append(args, "--gpus="+test.count)
			}
			code, out := runCozy(t, root, args...)
			mu.Lock()
			defer mu.Unlock()
			if test.want == "" {
				if code == 0 || len(asks) != 0 {
					t.Fatalf("invalid width bought capacity: asks=%v exit=%d %s", asks, code, out)
				}
				return
			}
			if len(asks) != 1 || asks[0]["sku"] != test.want {
				t.Fatalf("wrong exact SKU: asks=%v exit=%d %s", asks, code, out)
			}
			body, _ := json.Marshal(asks[0])
			if strings.Contains(string(body), "gpu_count") || strings.Contains(string(body), "quantity") {
				t.Fatalf("quantity duplicated the Hub SKU authority: %s", body)
			}
		})
	}
}
