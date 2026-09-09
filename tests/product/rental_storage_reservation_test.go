package producttest

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
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

func storageReservationBody(name string) ([]byte, string, *exit.Error) {
	body, _ := json.Marshal(map[string]string{"name": name, "sku": "cpu"})
	return body, fmt.Sprintf("sha256:%x", sha256.Sum256(body)), nil
}

func TestConcurrentRentalStorageReservationsSurviveRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	first, problem := records.Open(path)
	fatal(t, problem)
	defer first.Close()
	second, problem := records.Open(path)
	fatal(t, problem)
	defer second.Close()
	start := make(chan struct{})
	results := make(chan *exit.Error, 2)
	for i, store := range []*records.Store{first, second} {
		go func() {
			<-start
			_, _, problem := store.BeginRentalOperation(records.RentalOperation{
				Key: fmt.Sprint("storage-", i), Hub: "http://example.invalid", HourlyRateUSDMicros: 100000,
			}, 300000, 100000, storageReservationBody, nil)
			results <- problem
		}()
	}
	close(start)
	admitted := 0
	for range 2 {
		if problem := <-results; problem == nil {
			admitted++
		} else if problem.ErrName() != "rental.fleet_spend_cap" {
			t.Fatal(problem)
		}
	}
	if admitted != 1 {
		t.Fatalf("admitted %d intents at 200000 each under a 300000 cap", admitted)
	}
	first.Close()
	second.Close()
	reopened, problem := records.Open(path)
	fatal(t, problem)
	defer reopened.Close()
	count, reserved, problem := reopened.RentalFleetTotals("", nil)
	fatal(t, problem)
	if count != 1 || reserved != 200000 {
		t.Fatalf("after restart, reserved %d/%d; want one 200000 reservation", count, reserved)
	}
	ops, problem := reopened.RentalOperations()
	fatal(t, problem)
	if len(ops) != 1 || ops[0].HourlyRateUSDMicros != 100000 || ops[0].RentalID != "" || ops[0].EstimatedHourlyRateUSDMicros == nil || *ops[0].EstimatedHourlyRateUSDMicros != 200000 {
		t.Fatalf("the original GPU quote or unknown identity changed: %+v", ops)
	}
	replayed, replay, problem := reopened.BeginRentalOperation(records.RentalOperation{
		Key: ops[0].Key, Hub: ops[0].Hub, HourlyRateUSDMicros: 1,
	}, 1, 0, func(string) ([]byte, string, *exit.Error) {
		t.Error("replay re-authored the paid request")
		return nil, "", nil
	}, nil)
	fatal(t, problem)
	if !replay || replayed.HourlyRateUSDMicros != 100000 || replayed.RequestDigest != ops[0].RequestDigest || !bytes.Equal(replayed.RequestBody, ops[0].RequestBody) {
		t.Fatal("replay changed the original quote or idempotency identity")
	}
	_, reserved, problem = reopened.RentalFleetTotals("", nil)
	fatal(t, problem)
	if reserved != 200000 {
		t.Fatal("replay repriced the durable reservation")
	}
	fatal(t, reopened.AdvanceRentalOperation(ops[0].Key, "rental-priced", "pending_acquisition"))
	count, reserved, problem = reopened.RentalFleetTotals(ops[0].Hub, map[string]int64{"rental-priced": 40000})
	fatal(t, problem)
	if count != 1 || reserved != 40000 {
		t.Fatal("actual billed rate did not replace its total reservation exactly once")
	}
}

func TestRentalCLIRetainsStorageReservationBeforeIdentityArrives(t *testing.T) {
	root, origin, peer := rentalEndRoot(t, "pending-storage")
	peer.publishListing()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 0.30\n  idle_release_s: 0\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0600))
	peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"price_usd_micros_per_hour": 100000, "storage_usd_micros_per_hour": 100000,
		"base_worker_profile": "python3.12-cpu-linux-x86"})
	var asks atomic.Int32
	peer.rent = func(request map[string]any) map[string]any {
		asks.Add(1)
		// Accepted response with no identity: the durable intent must survive it.
		return map[string]any{"name": request["name"], "state": "pending_acquisition"}
	}
	code, out := runCozy(t, root, "rent", "cpu", "--idempotency-key=first", "--json")
	if code == 0 || !strings.Contains(out, "hub.rental_unnamed") || asks.Load() != 1 {
		t.Fatalf("first accepted intent did not retain its unknown identity: %d %s", code, out)
	}
	code, out = runCozy(t, root, "rental", "list", "--json", "--full")
	var board struct {
		Count int   `json:"machines_running"`
		Rate  int64 `json:"hourly_spend_usd_micros"`
		Rows  []struct {
			Rate int64 `json:"hourly_rate_usd_micros"`
		} `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &board) != nil || board.Count != 1 || board.Rate != 200000 || len(board.Rows) != 1 || board.Rows[0].Rate != 200000 {
		t.Fatalf("pending row and aggregate omitted its storage reservation: %d %s", code, out)
	}
	code, out = runCozy(t, root, "rent", "cpu", "--idempotency-key=second", "--json")
	if code == 0 || !strings.Contains(out, "rental.fleet_spend_cap") || asks.Load() != 1 {
		t.Fatalf("second intent reached POST despite 400000 estimated under 300000 cap: %d %s, asks=%d", code, out, asks.Load())
	}
}

func TestTerminalRentalIntentReleasesItsStorageReservation(t *testing.T) {
	for _, state := range []string{"failed", "rejected", "released"} {
		t.Run(state, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			_, _, problem = store.BeginRentalOperation(records.RentalOperation{
				Key: "closed", Hub: "http://example.invalid", HourlyRateUSDMicros: 100000,
			}, 200000, 100000, storageReservationBody, nil)
			fatal(t, problem)
			fatal(t, store.AdvanceRentalOperation("closed", "", state))
			count, burn, problem := store.RentalFleetTotals("", nil)
			fatal(t, problem)
			if count != 0 || burn != 0 {
				t.Fatalf("terminal intent still reserves %d/%d", count, burn)
			}
		})
	}
}

func TestSchema37RentalReservationMigrationPreservesUnknownEstimates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	schema, err := os.ReadFile("testdata/records-schema37-before-rental-reservation.sql")
	must(t, err)
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	_, err = db.Exec(string(schema))
	must(t, err)
	for _, key := range []string{"unknown", "known", "finished"} {
		id, state := "", "pending_acquisition"
		if key == "known" {
			id = "rental-known"
		}
		if key == "finished" {
			state = "released"
		}
		body, digest, problem := storageReservationBody(key)
		fatal(t, problem)
		_, err = db.Exec(`INSERT INTO rental_operations VALUES(?,?,?,?,?,?,?,?,?,?,?)`,
			key, digest, body, "http://example.invalid", "retained intent", 100000, "", id, state,
			"2026-09-08T01:02:03Z", "2026-09-08T02:03:04Z")
		must(t, err)
	}
	before := tableRows(t, db, "rental_operations")
	if reader, problem := records.Open(path); problem == nil || problem.ErrName() != "records_schema_upgrade_required" {
		if reader != nil {
			reader.Close()
		}
		t.Fatalf("ordinary reader migrated the database: %v", problem)
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 37 || tableRows(t, db, "rental_operations") != before {
		t.Fatal("ordinary reader changed retained schema or intents")
	}
	store, problem := records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer store.Close()
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 39 || tableRows(t, db, "rental_operations") != before {
		t.Fatal("migration changed retained quote, identity, state or timestamps")
	}
	ops, problem := store.RentalOperations()
	fatal(t, problem)
	for _, op := range ops {
		if op.EstimatedHourlyRateUSDMicros != nil {
			t.Fatal("migration invented a historical storage estimate")
		}
	}
	_, _, problem = store.RentalFleetTotals("http://example.invalid", map[string]int64{"rental-known": 40000})
	if problem == nil || problem.ErrName() != "rental.reservation_unknown" {
		t.Fatalf("unknown-ID legacy intent stopped reserving unknown exposure: %v", problem)
	}
	fatal(t, store.AdvanceRentalOperation("unknown", "", "rejected"))
	_, _, problem = store.RentalFleetTotals("http://other.invalid", map[string]int64{"rental-known": 40000})
	if problem == nil || problem.ErrName() != "rental.reservation_unknown" {
		t.Fatalf("another Hub replaced the unknown legacy reservation: %v", problem)
	}
	count, rate, problem := store.RentalFleetTotals("http://example.invalid", map[string]int64{"rental-known": 40000})
	fatal(t, problem)
	if count != 1 || rate != 40000 {
		t.Fatal("same-Hub actual rate did not replace the unknown estimate")
	}
	replay, replayed, problem := store.BeginRentalOperation(records.RentalOperation{Key: "known"}, 1, 1, storageReservationBody, nil)
	fatal(t, problem)
	if !replayed || replay.EstimatedHourlyRateUSDMicros != nil || replay.HourlyRateUSDMicros != 100000 {
		t.Fatal("replay repriced an unknown historical estimate")
	}
	store.Close()
	reopened, problem := records.Open(path)
	fatal(t, problem)
	reopened.Close()
}
