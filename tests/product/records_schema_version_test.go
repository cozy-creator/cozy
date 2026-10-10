package producttest

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	_ "modernc.org/sqlite"
)

// An older numbered schema other than 49 is refused by name and left exactly as it was.
func TestRecordsRefuseAnOlderSchemaUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	_, err = db.Exec(`PRAGMA user_version=48`)
	must(t, err)
	store, problem = records.Open(path)
	if problem == nil {
		store.Close()
		t.Fatal("a schema-48 database was opened")
	}
	if problem.ErrName() != "records_schema_unsupported" || !strings.Contains(problem.Message, "schema 48") {
		t.Fatalf("refusal = %s: %s", problem.ErrName(), problem.Message)
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 48 {
		t.Fatalf("a refused database was re-stamped to %d", version)
	}
}

// Schema 49 keyed package pins by name alone. Opening it moves each pin under its install's
// recorded hub (a local capture under none) in place, keeping every row.
func TestRecordsMigrateSchema49PinsUnderTheirHub(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	const origin = "http://127.0.0.1:1"
	for _, inst := range []records.PackageInstall{
		{ID: "published", Package: "proof/shared", Hub: origin, SourceKind: "tensorhub"},
		{ID: "captured", Package: "local/shared", Hub: origin, SourceKind: "local"},
	} {
		inst.Major, inst.Version, inst.SourceRef, inst.Dir = 1, "1.0.0", inst.Package+"@1.0.0", "/unused/"+inst.ID
		_, problem := store.Activate(inst)
		fatal(t, problem)
	}
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	for _, statement := range []string{
		`CREATE TABLE pins_49 (package TEXT NOT NULL, major INTEGER NOT NULL, install_id TEXT NOT NULL REFERENCES installs(id), activated_at TEXT NOT NULL, PRIMARY KEY (package))`,
		`INSERT INTO pins_49 SELECT package, major, install_id, activated_at FROM pins`,
		`DROP TABLE pins`, `ALTER TABLE pins_49 RENAME TO pins`, `PRAGMA user_version=49`,
	} {
		_, err := db.Exec(statement)
		must(t, err)
	}
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	for hub, pkg := range map[string]string{origin: "proof/shared", "": "local/shared"} {
		if pin, inst, problem := store.ActivePackage(origin, pkg); problem != nil || pin == nil || pin.Hub != hub || inst == nil {
			t.Fatalf("%s was not carried under hub %q: %+v %+v %v", pkg, hub, pin, inst, problem)
		}
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 50 {
		t.Fatalf("the migrated database is at schema %d", version)
	}
}

// A database a newer Creator wrote opens in compatibility mode: this build reads and writes
// only the tables and columns it knows and changes nothing else. It refuses only when a
// table or column it requires is missing or holds an incompatible type, naming it.
func TestRecordsWrittenByANewerCreatorOpenInCompatibilityMode(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "records.db")
	store, problem := records.Open(path)
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
		`PRAGMA user_version=51`,
	} {
		if _, err := db.Exec(statement); err != nil {
			db.Close()
			t.Fatalf("%s: %v", statement, err)
		}
	}
	db.Close()

	store, problem = records.Open(path)
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
	if version != 51 || price != 42 || defaulted != 0 || len(telemetry) != 1 {
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
	if restored != 0 || version != 51 {
		t.Fatalf("a refused newer database was changed: column restored=%d version=%d", restored, version)
	}
}
