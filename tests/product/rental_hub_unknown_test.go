package producttest

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A Hub whose database was reset no longer knows a rental this host still records
// (production 2026-09-27: pr-8a0e63fa, failed before readiness on 2026-09-23). That one
// row must stay visible and kept, and must not stop `rental list` or `rental new`.
func TestHubUnknownHostRentalDoesNotBlockOtherRentalVerbs(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "rental-hub-unknown")
	stand.setSKUs(map[string]any{
		"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86", "price_usd_micros_per_hour": 100_000,
	})
	const orphan = "pr-8a0e63fa7e215e0ec17f"
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{
		ID: orphan, MachineName: "kimanjirou", SKU: "a100-sxm4-80gb", AcceleratorModel: "NVIDIA A100-SXM4-80GB",
		AcceleratorCount: 1, HourlyRateUSDMicros: 1_390_000, State: "failed", Hub: hubURL,
		RentedAt: "2026-09-23T23:44:01Z",
		Failure:  records.RentalFailure{Code: "container_exited_before_readiness", Provider: "runpod", ProviderResourceID: "9b4eeyi6yh97ry"},
	}))
	store.Close()

	// The production Hub's typed answer for an id its database does not hold.
	original := stand.server.Config.Handler
	stand.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/rentals/"+orphan {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"rental not found"}}`))
			return
		}
		original.ServeHTTP(w, r)
	})
	creates := 0
	stand.mu.Lock()
	stand.rent = func(request map[string]any) map[string]any {
		creates++
		return map[string]any{
			"rental_id": "pr-hubunknownnextcreate", "name": request["name"], "state": "failed",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000,
			"failure": map[string]any{"code": "provider_create_did_not_happen"},
		}
	}
	stand.mu.Unlock()

	code, raw := runCozy(t, root, "rental", "list", "--json", "--full")
	var inventory struct {
		Rentals []map[string]any `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(raw), &inventory) != nil {
		t.Fatalf("one Hub-unknown row broke rental list [exit %d]\n%s", code, raw)
	}
	found := false
	for _, row := range inventory.Rentals {
		if row["rental_id"] == orphan && row["hub_unknown"] == true {
			found = true
		}
	}
	if !found {
		t.Fatalf("the Hub-unknown row is not listed as such\n%s", raw)
	}
	if code, out := runCozy(t, root, "rental", "list"); code != 0 || !strings.Contains(out, "unknown to Hub") {
		t.Fatalf("the board does not name the Hub-unknown row [exit %d]\n%s", code, out)
	}

	code, raw = runCozy(t, root, "rental", "new", "cpu", "--json", "--timeout=10s")
	if strings.Contains(raw, "rental.hub_record_missing") || !strings.Contains(raw, "provider_create_did_not_happen") {
		t.Fatalf("rental new did not reach acquisition past the Hub-unknown row [exit %d]\n%s", code, raw)
	}
	stand.mu.Lock()
	asked := creates
	stand.mu.Unlock()
	if asked != 1 {
		t.Fatalf("the Hub saw %d create(s), wanted 1", asked)
	}

	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RentalRow(orphan)
	fatal(t, problem)
	if row == nil || stand.releases(orphan) != 0 {
		t.Fatal("a rental the Hub could not confirm released was forgotten or released")
	}
}

// Tensorhub records `failed` only after proving the provider resource absent, so a host
// row holding that verdict settles when the Hub later loses the record; a Hub-unknown row
// the host never saw terminal stays refused (TestRentalEndKeepsRecordedMachineWhenHubCannotConfirmRelease).
func TestRentalEndForgetsHubUnknownRentalTheHubHadFailed(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "rental-hub-unknown-end")
	const orphan = "pr-8a0e63fa7e215e0ec17f"
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{
		ID: orphan, MachineName: "kimanjirou", SKU: "a100-sxm4-80gb", AcceleratorModel: "NVIDIA A100-SXM4-80GB",
		AcceleratorCount: 1, HourlyRateUSDMicros: 1_390_000, State: "failed", Hub: hubURL,
		Failure: records.RentalFailure{Code: "container_exited_before_readiness"},
	}))
	store.Close()
	original := stand.server.Config.Handler
	stand.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/rentals/"+orphan {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"rental not found"}}`))
			return
		}
		original.ServeHTTP(w, r)
	})

	code, out := runCozy(t, root, "rental", "end", orphan, "--json")
	if code != 0 || !strings.Contains(out, `"state":"ended"`) || !strings.Contains(out, `"forgotten":true`) {
		t.Fatalf("a rental the Hub had failed was not settled [exit %d]\n%s", code, out)
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RentalRow(orphan)
	fatal(t, problem)
	if row != nil {
		t.Fatalf("the settled rental is still recorded: %+v", row)
	}
	if code, out := runCozy(t, root, "rental", "list", "--json", "--full"); code != 0 || strings.Contains(out, orphan) {
		t.Fatalf("the settled rental still reaches the board [exit %d]\n%s", code, out)
	}
}
