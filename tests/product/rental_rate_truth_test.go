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
