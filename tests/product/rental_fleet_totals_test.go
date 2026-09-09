package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// A rental the hub terminally failed holds no provider pod — the hub commits
// `failed` only after readback-proven absence — so it is neither a running
// machine nor hourly spend. Live, 2026-09-02: seven failed RTX PRO 4500
// acquisitions held $5.04/hour of a $10 fleet cap while zero pods existed,
// and the cap then refused real capacity.
func TestFailedRentalLeavesTheFleetTotals(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	record := func(id, state string, rate int64) {
		t.Helper()
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
			ID: id, MachineName: id, SKU: "rtx-pro-4500",
			AcceleratorModel:    "NVIDIA RTX PRO 4500 Blackwell",
			HourlyRateUSDMicros: rate, State: state, Hub: "http://127.0.0.1:1",
		}))
	}
	record("rental-ready", "ready", 720_000)
	record("rental-failed", "failed", 720_000)
	record("rental-released", "released", 70_000)

	count, burn, problem := store.RentalFleetTotals("", nil)
	fatal(t, problem)
	if count != 1 || burn != 720_000 {
		t.Fatalf("fleet totals = %d machines at %d micros/hour, want the one ready "+
			"machine at 720000; a proven-absent rental is still counted as spend", count, burn)
	}
}

// th-126's fleet-cap red arm: burn is billed-truth money (th-120), so the
// admission figure for a new rental is the estimated TOTAL it will bill. A SKU
// whose GPU rate fits the headroom but whose total exceeds it is refused, and
// the refusal decomposes the figure.
func TestFleetCapAdmitsTheEstimatedTotalNotTheGPURate(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	author := func(machineName string) ([]byte, string, *exit.Error) {
		return []byte(`{"name":"` + machineName + `"}`), "sha256:" + strings.Repeat("cd", 32), nil
	}
	// Cap $0.70/h; the L4 quote is $0.49 GPU + $0.213504 storage = $0.703504.
	_, _, problem = store.BeginRentalOperation(records.RentalOperation{
		Key: "op-l4", Hub: "http://127.0.0.1:1", Reason: "cozy rental new l4",
		HourlyRateUSDMicros: 490_000,
	}, 700_000, 213_504, author, nil)

	if problem == nil || problem.ErrName() != "rental.fleet_spend_cap" ||
		!strings.Contains(problem.Error(), "703504 (490000 gpu + 213504 storage)") {
		t.Fatalf("a total above the cap was admitted on its GPU rate alone: %v", problem)
	}
	// The same SKU under a cap that covers the total is admitted.
	_, replay, problem := store.BeginRentalOperation(records.RentalOperation{
		Key: "op-l4-fits", Hub: "http://127.0.0.1:1", Reason: "cozy rental new l4",
		HourlyRateUSDMicros: 490_000,
	}, 710_000, 213_504, author, nil)

	fatal(t, problem)
	if replay {
		t.Fatal("a fresh operation replayed")
	}
}
