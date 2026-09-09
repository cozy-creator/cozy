package producttest

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalCensusDoesNotHideSameIDFromAnotherHub(t *testing.T) {
	root, origin, peer := rentalEndRoot(t, "census-hub-scope")
	peer.publishListing()
	peer.add("rental-collision", "current")
	peer.setRate("rental-collision", 40_000)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: "rental-collision", MachineName: "elsewhere",
		Hub: "http://127.0.0.1:1", State: "ready", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000}))
	code, out := runCozy(t, root, "rental", "list", "--json", "--full")
	var listed struct {
		Count int              `json:"machines_running"`
		Rate  int64            `json:"hourly_spend_usd_micros"`
		Rows  []map[string]any `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || listed.Count != 2 || listed.Rate != 140_000 || len(listed.Rows) != 2 {
		t.Fatalf("same ID in another Hub hid or repriced the current account's rental: exit=%d %s", code, out)
	}
	if listed.Rows[1]["hub"] != origin || listed.Rows[1]["machine"] != "current" {
		t.Fatalf("current Hub observation was not preserved: %s", out)
	}
}

func TestRentalCensusRateControlsTransactionalAdmission(t *testing.T) {
	for _, actual := range []int64{40_000, 140_000} {
		t.Run(fmt.Sprint(actual), func(t *testing.T) {
			root, origin, peer := rentalEndRoot(t, "census-admission")
			peer.publishListing()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
				"tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\n"+
					"rentals:\n  max_hourly_spend_usd: 0.15\n  idle_release_s: 0\n"+
					"daemon:\n  idle_shutdown_s: 0\n"), 0600))
			peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
				"price_usd_micros_per_hour": 100_000, "storage_usd_micros_per_hour": 10_000,
				"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86"})
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			_, _, problem = store.BeginRentalOperation(records.RentalOperation{Key: "old", Hub: origin,
				Reason: "cozy rent cpu", HourlyRateUSDMicros: 100_000}, 1_000_000, 0,
				func(name string) ([]byte, string, *exit.Error) {
					return []byte(fmt.Sprintf(`{"name":%q,"sku":"cpu"}`, name)), "sha256:" + strings.Repeat("a", 64), nil
				}, nil)
			fatal(t, problem)
			fatal(t, store.AdvanceRentalOperation("old", "rental-old", "pending_acquisition"))
			peer.add("rental-old", "known")
			peer.setRate("rental-old", actual)
			var asks atomic.Int64
			peer.rent = func(request map[string]any) map[string]any {
				asks.Add(1)
				// Stop at the same accepted-but-unattachable response as cl-232.
				return map[string]any{"rental_id": "rental-next", "name": request["name"],
					"state": "pending_acquisition", "requested_accelerator_model": "CPU",
					"hourly_rate_usd_micros": 100_000}
			}
			code, out := runCozy(t, root, "rent", "cpu", "--idempotency-key=next", "--json")
			if actual == 40_000 {
				if asks.Load() != 1 || code == 0 || !strings.Contains(out, "accelerator count") {
					t.Fatalf("observed 0.04 + new 0.11 should reach the paid POST under 0.15 cap: asks=%d exit=%d %s", asks.Load(), code, out)
				}
			} else if asks.Load() != 0 || !strings.Contains(out, "rental.fleet_spend_cap") {
				t.Fatalf("observed 0.14 + new 0.11 must refuse before POST: asks=%d exit=%d %s", asks.Load(), code, out)
			}
			operations, problem := store.RentalOperations()
			fatal(t, problem)
			for _, op := range operations {
				if op.Key == "old" && op.HourlyRateUSDMicros != 100_000 {
					t.Fatal("observed billing rewrote the original idempotency quote")
				}
			}
		})
	}
}

func TestRentalCensusIdentityScopeAndUnknownRates(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const origin = "http://127.0.0.1:1"
	for _, item := range []struct {
		key  string
		rate int64
	}{{"known", 100_000}, {"unknown", 30_000}} {
		_, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: item.key,
			Hub: origin, HourlyRateUSDMicros: item.rate}, 1_000_000, 0,
			func(name string) ([]byte, string, *exit.Error) {
				return []byte(fmt.Sprintf(`{"name":%q}`, name)), "sha256:" + strings.Repeat("a", 64), nil
			}, nil)
		fatal(t, problem)
	}
	fatal(t, store.AdvanceRentalOperation("known", "rental-known", "pending_acquisition"))
	if problem := store.AdvanceRentalOperation("unknown", "rental-known", "pending_acquisition"); problem == nil {
		t.Fatal("two operation quotes acquired the same immutable rental identity")
	}
	check := func(hub string, observed map[string]int64, count int, rate int64) {
		t.Helper()
		gotCount, gotRate, problem := store.RentalFleetTotals(hub, observed)
		fatal(t, problem)
		if gotCount != count || gotRate != rate {
			t.Fatalf("fleet=%d/%d, want %d/%d", gotCount, gotRate, count, rate)
		}
	}
	check(origin+"/", map[string]int64{"rental-known": 40_000}, 2, 70_000)
	check(origin, map[string]int64{"rental-known": 0}, 2, 130_000)
	check("http://127.0.0.1:2", map[string]int64{"rental-known": 200_000}, 3, 330_000)
	_, _, problem = store.RentalFleetTotals(origin, map[string]int64{"rental-other": 0})
	if problem == nil || problem.ErrName() != "rental.rate_unknown" {
		t.Fatalf("an unpriced orphan was reported as zero spend: %v", problem)
	}
	row := records.Rental{ID: "rental-known", MachineName: "known", Hub: origin, State: "ready",
		AcceleratorCount: 1, HourlyRateUSDMicros: 40_000}
	fatal(t, store.RecordRental(row))
	check(origin, map[string]int64{"rental-known": 40_000}, 2, 70_000)
	row.State = "failed"
	fatal(t, store.RecordRental(row))
	fatal(t, store.AdvanceRentalOperation("known", "rental-known", "failed"))
	check(origin, nil, 1, 30_000)
}

func TestRentalFleetRateOverflowRefusesAdmission(t *testing.T) {
	for _, source := range []string{"recorded", "unknown_operation", "new_estimate"} {
		t.Run(source, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			const origin = "http://127.0.0.1:1"
			author := func(name string) ([]byte, string, *exit.Error) {
				return []byte(fmt.Sprintf(`{"name":%q}`, name)), "sha256:" + strings.Repeat("a", 64), nil
			}
			if source == "recorded" {
				fatal(t, store.RecordRental(records.Rental{ID: "rental-large", MachineName: "large",
					Hub: origin, State: "ready", AcceleratorCount: 1, HourlyRateUSDMicros: math.MaxInt64}))
			} else {
				storage := int64(0)
				if source == "new_estimate" {
					storage = 1
				}
				_, _, problem = store.BeginRentalOperation(records.RentalOperation{Key: "large", Hub: origin,
					HourlyRateUSDMicros: math.MaxInt64}, math.MaxInt64, storage, author, nil)
				if source == "new_estimate" {
					if problem == nil || problem.ErrName() != "rental.spend_overflow" {
						t.Fatalf("overflowing new estimate was admitted: %v", problem)
					}
					return
				}
				fatal(t, problem)
			}
			_, _, problem = store.RentalFleetTotals(origin, map[string]int64{"rental-other": 1})
			if problem == nil || problem.ErrName() != "rental.spend_overflow" {
				t.Fatalf("overflowing account spend was reported as a number: %v", problem)
			}
			_, _, problem = store.BeginRentalOperation(records.RentalOperation{Key: "next", Hub: origin,
				HourlyRateUSDMicros: 1}, math.MaxInt64, 0, author, map[string]int64{"rental-other": 1})
			if problem == nil || problem.ErrName() != "rental.spend_overflow" {
				t.Fatalf("overflowing account spend reached reservation: %v", problem)
			}
		})
	}
}
