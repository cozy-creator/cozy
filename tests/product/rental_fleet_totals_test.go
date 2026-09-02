package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A rental the hub terminally failed holds no provider pod — the hub commits
// `failed` only after readback-proven absence — so it is neither a running
// machine nor hourly spend. Live, 2026-09-02: seven failed RTX PRO 4500
// acquisitions held $5.04/hour of a $10 fleet cap while zero pods existed,
// and the cap then refused real capacity.
func TestFailedRentalLeavesTheFleetTotals(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	fatal(t, problem)
	defer store.Close()
	record := func(id, state string, rate int64) {
		t.Helper()
		fatal(t, store.RecordRental(records.Rental{
			ID: id, MachineName: id, SKU: "rtx-pro-4500",
			AcceleratorModel:    "NVIDIA RTX PRO 4500 Blackwell",
			HourlyRateUSDMicros: rate, State: state, Hub: "http://127.0.0.1:1",
		}))
	}
	record("rental-ready", "ready", 720_000)
	record("rental-failed", "failed", 720_000)
	record("rental-released", "released", 70_000)

	count, burn, problem := store.RentalFleetTotals()
	fatal(t, problem)
	if count != 1 || burn != 720_000 {
		t.Fatalf("fleet totals = %d machines at %d micros/hour, want the one ready "+
			"machine at 720000; a proven-absent rental is still counted as spend", count, burn)
	}
}
