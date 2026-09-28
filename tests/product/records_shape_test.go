package producttest

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	_ "modernc.org/sqlite"
)

// A records database whose shape drifted from this build's authored DDL still opens. Extra
// tables, columns and indexes and different DDL text are the database's own; a missing
// table or column is added. Nothing is moved aside and no rental row is lost.
func TestRecordsOpenAcceptsADriftedShapeAndKeepsRentals(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	store, problem := records.OpenForDaemon(path)
	fatal(t, problem)
	seed := records.Rental{ID: "pr-drifted", MachineName: "drifted", SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3",
		AcceleratorCount: 1, HourlyRateUSDMicros: 2_490_000, State: "ready", Address: "127.0.0.1:1",
		CertPath: "/unused.pem", Hub: "http://127.0.0.1:1"}
	fatal(t, store.RecordRental(seed))
	store.Close()

	db, err := sql.Open("sqlite", path)
	must(t, err)
	for _, statement := range []string{
		`ALTER TABLE rentals ADD COLUMN future_note TEXT NOT NULL DEFAULT 'kept'`,
		`CREATE INDEX future_rentals_state ON rentals(state)`,
		`CREATE TABLE future_ledger(id TEXT)`,
		`DROP TABLE rental_idle`,
		`ALTER TABLE rentals DROP COLUMN failure_container_state`,
	} {
		if _, err := db.Exec(statement); err != nil {
			db.Close()
			t.Fatalf("%s: %v", statement, err)
		}
	}
	db.Close()

	readOnly, problem := records.OpenReadOnly(path)
	if problem == nil || problem.ErrName() != "records_schema_mismatch" {
		t.Fatalf("a read-only open of a database lacking a table = %v, want records_schema_mismatch", problem)
	}
	if readOnly != nil {
		readOnly.Close()
	}

	store, problem = records.Open(path)
	if problem != nil {
		t.Fatalf("a drifted records database did not open: %s", problem.Message)
	}
	row, problem := store.RentalRow(seed.ID)
	fatal(t, problem)
	if row == nil || row.MachineName != "drifted" || row.State != "ready" || row.HourlyRateUSDMicros != 2_490_000 {
		t.Fatalf("the rental row did not survive: %+v", row)
	}
	if _, problem := store.RentalIdleObservation(*row); problem != nil {
		t.Fatalf("the missing rental_idle table was not restored: %s", problem.Message)
	}
	store.Close()

	readOnly, problem = records.OpenReadOnly(path)
	if problem != nil {
		t.Fatalf("a read-only open of the completed database: %s", problem.Message)
	}
	readOnly.Close()

	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var note string
	must(t, db.QueryRow(`SELECT future_note FROM rentals WHERE id=?`, seed.ID).Scan(&note))
	var extras int
	must(t, db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE name IN ('future_rentals_state','future_ledger')`).Scan(&extras))
	if note != "kept" || extras != 2 {
		t.Fatalf("the database's own column, index or table was removed: note=%q objects=%d", note, extras)
	}
}
