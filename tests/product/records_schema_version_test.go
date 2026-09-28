package producttest

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	_ "modernc.org/sqlite"
)

// An older numbered schema is refused by name and left exactly as it was: this build
// carries no migrations.
func TestRecordsRefuseAnOlderSchemaUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.OpenForDaemon(path)
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	_, err = db.Exec(`PRAGMA user_version=48`)
	must(t, err)
	for _, open := range []func(string) (*records.Store, *exit.Error){records.Open, records.OpenForDaemon} {
		store, problem := open(path)
		if problem == nil {
			store.Close()
			t.Fatal("a schema-48 database was opened")
		}
		if problem.ErrName() != "records_schema_unsupported" || !strings.Contains(problem.Message, "schema 48") {
			t.Fatalf("refusal = %s: %s", problem.ErrName(), problem.Message)
		}
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 48 {
		t.Fatalf("a refused database was re-stamped to %d", version)
	}
}

// A database a newer Creator wrote opens in compatibility mode: this build reads and writes
// only the tables and columns it knows and changes nothing else. It refuses only when a
// table or column it requires is missing or holds an incompatible type, naming it.
func TestRecordsWrittenByANewerCreatorOpenInCompatibilityMode(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "records.db")
	store, problem := records.OpenForDaemon(path)
	if problem != nil {
		t.Fatalf("initialize records: %v", problem)
	}
	seed := records.Rental{ID: "pr-newer", MachineName: "newer", SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3",
		AcceleratorCount: 1, HourlyRateUSDMicros: 2_490_000, State: "ready", Address: "127.0.0.1:1",
		CertPath: "/unused.pem", Hub: "http://127.0.0.1:1"}
	fatal(t, store.RecordRental(seed))
	store.Close()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range []string{
		`ALTER TABLE rentals ADD COLUMN spot_price_micros INTEGER NOT NULL DEFAULT 0`,
		`UPDATE rentals SET spot_price_micros=42 WHERE id='pr-newer'`,
		`CREATE TABLE rental_telemetry(id TEXT PRIMARY KEY, body BLOB NOT NULL)`,
		`INSERT INTO rental_telemetry VALUES('pr-newer', x'01')`,
		`PRAGMA user_version=50`,
	} {
		if _, err := db.Exec(statement); err != nil {
			db.Close()
			t.Fatalf("%s: %v", statement, err)
		}
	}
	db.Close()

	for _, open := range []func(string) (*records.Store, *exit.Error){records.Open,
		records.OpenForDaemon} {
		store, problem = open(path)
		if problem != nil {
			t.Fatalf("a newer database with every required table was refused: %s", problem.Message)
		}
		row, problem := store.RentalRow("pr-newer")
		fatal(t, problem)
		row.State = "attached"
		fatal(t, store.RecordRental(*row))
		fresh := seed
		fresh.ID, fresh.MachineName = "pr-older-write", "older-write"
		fatal(t, store.RecordRental(fresh))
		row, problem = store.RentalRow("pr-newer")
		fatal(t, problem)
		if row == nil || row.State != "attached" || row.HourlyRateUSDMicros != 2_490_000 {
			t.Fatalf("the rental did not round-trip: %+v", row)
		}
		store.Close()
	}

	db, err = sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var version int
	var price, defaulted int64
	var telemetry []byte
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`SELECT spot_price_micros FROM rentals WHERE id='pr-newer'`).Scan(&price))
	must(t, db.QueryRow(`SELECT spot_price_micros FROM rentals WHERE id='pr-older-write'`).Scan(&defaulted))
	must(t, db.QueryRow(`SELECT body FROM rental_telemetry WHERE id='pr-newer'`).Scan(&telemetry))
	if version != 50 || price != 42 || defaulted != 0 || len(telemetry) != 1 {
		t.Fatalf("the older build changed the newer database: version=%d price=%d defaulted=%d telemetry=%x",
			version, price, defaulted, telemetry)
	}

	for _, statement := range []string{
		`ALTER TABLE rentals DROP COLUMN failure_container_state`,
		`ALTER TABLE rentals DROP COLUMN hourly_rate_usd_micros`,
		`ALTER TABLE rentals ADD COLUMN hourly_rate_usd_micros TEXT NOT NULL DEFAULT ''`,
	} {
		if _, err := db.Exec(statement); err != nil {
			t.Fatalf("%s: %v", statement, err)
		}
	}
	store, problem = records.Open(path)
	if problem == nil {
		store.Close()
		t.Fatal("a newer database lacking a required column was opened")
	}
	if problem.ErrName() != "records_schema_newer" || !strings.Contains(problem.Message, "rentals.failure_container_state") ||
		!strings.Contains(problem.Message, "rentals.hourly_rate_usd_micros (TEXT, this build needs INTEGER)") {
		t.Fatalf("the refusal does not name what is missing: %s: %s", problem.ErrName(), problem.Message)
	}
	var restored int
	must(t, db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('rentals') WHERE name='failure_container_state'`).Scan(&restored))
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if restored != 0 || version != 50 {
		t.Fatalf("a refused newer database was changed: column restored=%d version=%d", restored, version)
	}
}
