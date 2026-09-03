package producttest

import (
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
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-rate-truth")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 10.00\n"), 0o600))

	hub := newFakeRentalHub(t, port)
	hub.add("pr-redarm", "twine")
	hub.setRate("pr-redarm", 720_000)

	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{
		ID: "pr-redarm", MachineName: "twine", SKU: "gpu-cu130",
		AcceleratorModel: "NVIDIA GeForce RTX 3090", HourlyRateUSDMicros: 500_000,
		State: "ready", Hub: hubURL,
	}))

	code, out := runCozy(t, root, "rental")
	if code != 0 || !strings.Contains(out, "Current spend per hour: $0.72") {
		t.Fatalf("the burn line still says the quote [exit %d]:\n%s", code, out)
	}
	row, problem := store.RentalRow("pr-redarm")
	fatal(t, problem)
	if row == nil || row.HourlyRateUSDMicros != 720_000 {
		t.Fatalf("the local row did not adopt the billed rate: %+v", row)
	}
	hub.close()
}

// th-126, the selection half: pre-spend quotes speak the total the pod will
// bill, decomposed. Paul's live L4 read $0.49/hr in the ladder while RunPod
// billed ~$0.70 — the 1536 GB image spec adds 1536 × 139 micros/GB/h of
// storage. The ladder must render gpu + storage = total on every rung.
func TestRentalLadderRendersTheTotalDecomposed(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-ladder")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		fmt.Sprintf("tensorhub_url: http://127.0.0.1:%d\n", port)+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 10.00\n"), 0o600))
	hub := newFakeRentalHub(t, port)
	sku := func(name string, count, price, storage int64) map[string]any {
		return map[string]any{"name": name, "accelerator_model": "NVIDIA L4",
			"accelerator_count": count, "base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86",
			"compute_capability": "8.9", "vram_gb": 24, "minimum_ram_per_gpu_gb": 64,
			"price_usd_micros_per_hour": price, "storage_usd_micros_per_hour": storage}
	}
	// Both widths of the card share the one pod-disk adder (1536 GB × 139).
	hub.setSKUs(sku("l4", 1, 490_000, 213_504), sku("l4-x4", 4, 1_960_000, 213_504))

	code, out := runCozy(t, root, "rental", "new")
	if code != 0 {
		t.Fatalf("cozy rental new [exit %d]:\n%s", code, out)
	}
	for _, want := range []string{
		"$0.49/hr", "$0.213504/hr", "$0.703504/hr", // the L4 rung, decomposed
		"$1.96/hr", "$2.173504/hr", // the x4 rung: its own price, the same adder
		"GPU", "STORAGE", "PRICE",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("the ladder does not render %q:\n%s", want, out)
		}
	}
	hub.close()
}
