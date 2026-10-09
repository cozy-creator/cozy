//go:build !windows

package producttest

import (
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/config"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestRentalCatalogKeepsEachWidthProviderAndBundledStorage(t *testing.T) {
	for _, unknown := range []bool{false, true} {
		name := "itemized"
		if unknown {
			name = "included_not_itemized"
		}
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				wire := `[{"name":"h100-sxm5-80gb","provider":"runpod","accelerator_model":"NVIDIA H100 80GB HBM3","compute_capability":"9.0","vram_gb":80,"widths":[{"accelerator_count":2,"provider":"runpod","price_usd_micros_per_hour":7980000,"storage_usd_micros_per_hour":41700},{"accelerator_count":4,"provider":"vast","price_usd_micros_per_hour":32216667,"included_storage_usd_micros_per_hour":83333,"storage_usd_micros_per_hour":0}]}]`
				if unknown {
					wire = strings.ReplaceAll(wire, `"included_storage_usd_micros_per_hour":83333`, `"included_storage_usd_micros_per_hour":0`)
				}
				_, _ = w.Write([]byte(wire))
			}))
			defer server.Close()
			root := t.TempDir()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
			code, out := runCozy(t, root, "rental", "new", "--json", "--full")
			var document struct {
				GPUs []map[string]string `json:"gpus"`
			}
			if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.GPUs) != 1 {
				t.Fatalf("catalog failed %d %s", code, out)
			}
			row := document.GPUs[0]
			compute, storage := "2x $7.98/hr, 4x $32.13/hr", "2x $0.04/hr, 4x $0.08/hr"
			if unknown {
				compute, storage = "2x $7.98/hr, 4x not itemized", "2x $0.04/hr, 4x included"
			}
			if row["providers"] != "2x runpod, 4x vast" || row["gpu price"] != compute || row["storage price"] != storage || row["price"] != "2x $8.02/hr, 4x $32.22/hr" {
				t.Fatalf("misleading price breakdown: %+v", row)
			}

		})
	}
}

func TestRentalRefusesHubWithoutEnforceablePriceCeiling(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "old-price-hub")
	defer runCozy(t, root, "down")
	stand.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1, "price_usd_micros_per_hour": 100000})
	var paid atomic.Int32
	stand.quote = func(map[string]any) (int, string) { return http.StatusOK, `{"price_usd_micros_per_hour":100000}` }
	stand.rent = func(map[string]any) map[string]any { paid.Add(1); return map[string]any{} }
	code, out := runCozy(t, root, "rental", "new", "cpu", "--development=false", "--json")
	if code == 0 || !strings.Contains(out, "rental.price_ceiling_unsupported") || paid.Load() != 0 {
		t.Fatalf("unbounded paid request: %d %s, posts%d", code, out, paid.Load())
	}
}

func TestRentalPaidRequestKeepsFreshTotalCeiling(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "price-ceiling")
	defer runCozy(t, root, "down")
	stand.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1, "price_usd_micros_per_hour": 100000, "storage_usd_micros_per_hour": 41700})
	var got atomic.Int64
	stand.rent = func(request map[string]any) map[string]any {
		value, _ := request["maximum_total_hourly_rate_usd_micros"].(float64)
		got.Store(int64(value))
		return map[string]any{"rental_id": "pr-price-bound", "name": request["name"], "state": "pending_acquisition", "requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100000}
	}
	_, out := rentUntilRecorded(t, root, "pr-price-bound", "cpu", "--development=false", "--idempotency-key=price-bound", "--json")
	if got.Load() != 141700 {
		t.Fatalf("paid request changed price ceiling to%d: %s", got.Load(), out)
	}
}
