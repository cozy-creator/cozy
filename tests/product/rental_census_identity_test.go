package producttest

import (
	"encoding/json"
	"fmt"
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
	type listing struct {
		Count  int              `json:"machines_running"`
		Rate   int64            `json:"hourly_spend_usd_micros"`
		Rows   []map[string]any `json:"rentals"`
		Others []map[string]any `json:"other_hubs"`
	}
	// A listing narrowed to the current hub holds its own observation and names the other
	// hub's rental rather than listing it.
	code, out := runCozy(t, root, "rental", "list", "--json", "--full", "--tensorhub="+origin)
	var listed listing
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || listed.Count != 1 || listed.Rate != 40_000 ||
		len(listed.Rows) != 1 || listed.Rows[0]["hub"] != origin || listed.Rows[0]["machine"] != "current" ||
		len(listed.Others) != 1 || listed.Others[0]["hub"] != "http://127.0.0.1:1" {
		t.Fatalf("same ID in another Hub hid or repriced the current account's rental: exit=%d %s", code, out)
	}
	// Every hub's listing, the default, holds both, each on its own hub; the unreachable
	// one is unverified.
	code, out = runCozy(t, root, "rental", "list", "--json", "--full")
	listed = listing{}
	if json.Unmarshal([]byte(out), &listed) != nil || len(listed.Rows) != 2 {
		t.Fatalf("every-hub listing lost a same-ID rental: exit=%d %s", code, out)
	}
	byMachine := map[string]map[string]any{}
	for _, row := range listed.Rows {
		byMachine[row["machine"].(string)] = row
	}
	if byMachine["current"]["hub"] != origin || byMachine["current"]["unverified"] == true ||
		byMachine["elsewhere"]["hub"] != "http://127.0.0.1:1" || byMachine["elsewhere"]["unverified"] != true {
		t.Fatalf("same-ID rentals were not kept on their own hubs: %s", out)
	}
}

// cl-233: spend admission is the hub's. Under the hub's owner cap an observed
// $0.04 plus the new $0.11 reaches the paid ask; an observed $0.14 is refused by
// the hub, buys nothing, and leaves only a rejected operation behind.
func TestHubSpendCapRefusalBuysNothing(t *testing.T) {
	for _, actual := range []int64{40_000, 140_000} {
		t.Run(fmt.Sprint(actual), func(t *testing.T) {
			root, origin, peer := rentalEndRoot(t, "census-admission")
			peer.publishListing()
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
				"tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\n"+
					"daemon:\n  idle_shutdown_s: 0\n"), 0600))
			peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
				"price_usd_micros_per_hour": 100_000, "storage_usd_micros_per_hour": 10_000,
				"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86"})
			peer.mu.Lock()
			peer.spendCap = 150_000
			peer.mu.Unlock()
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
			code, out := runCozy(t, root, "rental", "new", "cpu", "--idempotency-key=next", "--json")
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			op, problem := store.RentalOperation("next")
			fatal(t, problem)
			if actual == 40_000 {
				if asks.Load() != 1 || code == 0 || !strings.Contains(out, "accelerator count") {
					t.Fatalf("observed 0.04 + new 0.11 should reach the paid ask under a 0.15 cap: asks=%d exit=%d %s", asks.Load(), code, out)
				}
				return
			}
			if asks.Load() != 0 || code == 0 || !strings.Contains(out, "rental.fleet_spend_cap") {
				t.Fatalf("observed 0.14 + new 0.11 must be refused by the hub: asks=%d exit=%d %s", asks.Load(), code, out)
			}
			// The live rental is known only to the hub; the refusal neither exposes that
			// bookkeeping nor ends the machine on its own.
			if strings.Contains(out, "holds no record") || strings.Contains(out, "recorded only at the hub") {
				t.Fatalf("the ceiling refusal exposes controller bookkeeping: %s", out)
			}
			if peer.releases("rental-old") != 0 {
				t.Fatal("a spend refusal ended a machine on its own")
			}
			if op == nil || op.State != "rejected" {
				t.Fatalf("a refused ask must leave a rejected operation: %+v", op)
			}
		})
	}
}

func TestTwoOperationsCannotShareOneRentalIdentity(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	for _, item := range []struct {
		key  string
		rate int64
	}{{"known", 100_000}, {"unknown", 30_000}} {
		_, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: item.key,
			Hub: "http://127.0.0.1:1", HourlyRateUSDMicros: item.rate},
			func(name string) ([]byte, string, *exit.Error) {
				return []byte(fmt.Sprintf(`{"name":%q}`, name)), "sha256:" + strings.Repeat("a", 64), nil
			})
		fatal(t, problem)
	}
	fatal(t, store.AdvanceRentalOperation("known", "rental-known", "pending_acquisition"))
	if problem := store.AdvanceRentalOperation("unknown", "rental-known", "pending_acquisition"); problem == nil {
		t.Fatal("two operation quotes acquired the same immutable rental identity")
	}
}
