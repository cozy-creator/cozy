package producttest

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRecordsSchemaIsExactV5AndStable(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()

	db, err := sql.Open("sqlite", path)
	must(t, err)
	var version, before int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&before))
	if version != 5 {
		t.Fatalf("fresh records version = %d, want exact v5", version)
	}
	for _, column := range []string{
		"rental", "package_revision_digest", "environment_digest", "config_digest",
	} {
		var count int
		must(t, db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('requests') WHERE name=?`,
			column).Scan(&count))
		if count != 1 {
			t.Fatalf("exact requests schema omitted %s", column)
		}
	}
	var deleted int
	must(t, db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('requests') WHERE name='max_cost_usd_micros'`).Scan(&deleted))
	if deleted != 0 {
		t.Fatal("exact requests schema retained max_cost_usd_micros")
	}
	var steps, legacyTables, stepIndex, stepName int
	must(t, db.QueryRow(`SELECT COUNT(*) FROM sqlite_master
		WHERE type='table' AND name='model_production_steps'`).Scan(&steps))
	must(t, db.QueryRow(`SELECT COUNT(*) FROM sqlite_master
		WHERE type='table' AND name='model_production_nodes'`).Scan(&legacyTables))
	must(t, db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('model_production_steps')
		WHERE name='step_index'`).Scan(&stepIndex))
	must(t, db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('model_production_steps')
		WHERE name='step_name'`).Scan(&stepName))
	if steps != 1 || legacyTables != 0 || stepIndex != 1 || stepName != 1 {
		t.Fatalf("production step schema = steps:%d legacy:%d index:%d name:%d",
			steps, legacyTables, stepIndex, stepName)
	}
	must(t, db.Close())

	store, problem = records.Open(path)
	fatal(t, problem)
	store.Close()
	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var after int
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&after))
	if after != before {
		t.Fatalf("exact v5 reopen performed DDL: schema_version %d -> %d", before, after)
	}
}

func TestRecordsRefusesEveryPreV5VersionWithoutMutation(t *testing.T) {
	for _, older := range []int{1, 2, 3, 4} {
		t.Run(fmt.Sprintf("v%d", older), func(t *testing.T) {
			path := t.TempDir() + "/records.db"
			db, err := sql.Open("sqlite", path)
			must(t, err)
			_, err = db.Exec(fmt.Sprintf(
				`CREATE TABLE older_owner(value TEXT); PRAGMA user_version=%d`, older))
			must(t, err)
			var before int
			must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&before))
			must(t, db.Close())

			opened, problem := records.Open(path)
			if opened != nil {
				opened.Close()
				t.Fatal("pre-v5 records schema opened")
			}
			if problem == nil || problem.ErrName() != "records.schema_reset_required" ||
				!strings.Contains(problem.Remedy, "move") {
				t.Fatalf("pre-v5 schema refusal = %#v", problem)
			}
			db, err = sql.Open("sqlite", path)
			must(t, err)
			defer db.Close()
			var version, after int
			must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
			must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&after))
			if version != older || after != before {
				t.Fatalf("refused v%d schema mutated: version=%d schema_version=%d->%d",
					older, version, before, after)
			}
		})
	}
}

func TestRentalOperationReservesFleetRateAcrossStoreConnections(t *testing.T) {
	path := t.TempDir() + "/records.db"
	first, problem := records.Open(path)
	fatal(t, problem)
	defer first.Close()
	second, problem := records.Open(path)
	fatal(t, problem)
	defer second.Close()
	operation := records.RentalOperation{
		Key: "managed-rental-one", RequestDigest: "digest-one", RequestBody: []byte(`{"sku":"gpu"}`),
		Hub: "https://tensorhub.test", Reason: "proof", HourlyRateUSDMicros: 750_000,
	}
	stored, replay, problem := first.BeginRentalOperation(operation, 1_000_000)
	if problem != nil || replay || stored.HourlyRateUSDMicros != 750_000 {
		t.Fatalf("first reservation = %+v replay=%v problem=%v", stored, replay, problem)
	}
	other := operation
	other.Key, other.RequestDigest = "managed-rental-two", "digest-two"
	if _, _, problem := second.BeginRentalOperation(other, 1_000_000); problem == nil ||
		problem.ErrName() != "rental.fleet_spend_cap" {
		t.Fatalf("second connection crossed the fleet ceiling: %v", problem)
	}
	if stored, replay, problem = second.BeginRentalOperation(operation, 0); problem != nil || !replay {
		t.Fatalf("existing paid operation did not resume after cap changed: %+v replay=%v problem=%v",
			stored, replay, problem)
	}
	count, burn, problem := first.RentalFleetTotals()
	if problem != nil || count != 1 || burn != 750_000 {
		t.Fatalf("reserved fleet totals = %d, %d, %v", count, burn, problem)
	}
}

func TestRecordsRefusesShapeDrift(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`ALTER TABLE requests DROP COLUMN environment_digest;
		ALTER TABLE rentals DROP COLUMN wheelhouse_manifest_digest;
		ALTER TABLE requests ADD COLUMN compatibility_alias TEXT;
		PRAGMA user_version=2`)
	must(t, err)
	var before int
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&before))
	must(t, db.Close())

	opened, problem := records.Open(path)
	if opened != nil {
		opened.Close()
		t.Fatal("drifted v2 records schema opened")
	}
	if problem == nil || problem.ErrName() != "records.schema_reset_required" {
		t.Fatalf("drifted v2 refusal = %#v", problem)
	}
	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var version, after, requestEnvironment, rentalBase int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&after))
	must(t, db.QueryRow(`SELECT count(*) FROM pragma_table_info('requests')
		WHERE name='environment_digest'`).Scan(&requestEnvironment))
	must(t, db.QueryRow(`SELECT count(*) FROM pragma_table_info('rentals')
		WHERE name='wheelhouse_manifest_digest'`).Scan(&rentalBase))
	if version != 2 || after != before || requestEnvironment != 0 || rentalBase != 0 {
		t.Fatalf("refused v2 schema mutated: version=%d schema_version=%d->%d request=%d rental=%d",
			version, before, after, requestEnvironment, rentalBase)
	}
}

func TestRentalRecordWaitsForAnotherProcessWriter(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()

	writer, err := sql.Open("sqlite", path+"?_pragma=busy_timeout(5000)")
	must(t, err)
	defer writer.Close()
	_, err = writer.Exec(`CREATE TABLE contention_probe (id INTEGER)`)
	must(t, err)
	ctx := context.Background()
	conn, err := writer.Conn(ctx)
	must(t, err)
	defer conn.Close()
	_, err = conn.ExecContext(ctx, `BEGIN IMMEDIATE`)
	must(t, err)
	_, err = conn.ExecContext(ctx, `INSERT INTO contention_probe VALUES (1)`)
	must(t, err)

	done := make(chan error, 1)
	go func() {
		if problem := store.RecordRental(records.Rental{
			ID: "pr-lock-proof", MachineName: "lock-proof", SKU: "cpu-test",
			HourlyRateUSDMicros: 100_000, State: "booting", Hub: "https://tensorhub.com",
		}); problem != nil {
			done <- problem
			return
		}
		done <- nil
	}()

	time.Sleep(100 * time.Millisecond)
	_, err = conn.ExecContext(ctx, `COMMIT`)
	must(t, err)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
