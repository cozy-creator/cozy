package producttest

import (
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
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

// A home written before an index existed gets it on the next command, as a missing column
// does. Without it every open's lifecycle-rename probe scanned all recorded events: on the
// owner's 450k-event database each `cozy run` opened its records three times at ~300 ms.
// An index the home's rows refuse (a unique one over duplicates) is named once and skipped:
// the home worked without it before, and it still runs every command.
func TestACommandRestoresTheIndexesAnOlderHomeLacks(t *testing.T) {
	root := t.TempDir()
	if code, out := runCozy(t, root, "package", "list", "--json"); code != 0 {
		t.Fatalf("package list [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	db, err := sql.Open("sqlite", layout.DB)
	must(t, err)
	defer db.Close()
	indexes := []string{"request_events_before_responses", "request_events_memo"}
	for _, statement := range []string{
		`DROP INDEX request_events_before_responses`,
		`DROP INDEX request_events_memo`,
		`DROP INDEX requests_parent_call`,
		`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at,parent_request_id,parent_call_index)
		 VALUES ('req-dup-1','dup-1','sha256:1','local/dup','main','',x'7b7d','succeeded','2026-09-01T00:00:00Z','req-parent',0),
		        ('req-dup-2','dup-2','sha256:2','local/dup','main','',x'7b7d','succeeded','2026-09-01T00:00:00Z','req-parent',0)`,
	} {
		_, err := db.Exec(statement)
		must(t, err)
	}
	code, stdout, stderr := runCozyStreams(t, root, "package", "list", "--json")
	if code != 0 {
		t.Fatalf("package list on the older home [exit %d]\n%s%s", code, stdout, stderr)
	}
	if strings.Count(stderr, "requests_parent_call") != 1 || !strings.Contains(stderr, "commands run without it") {
		t.Fatalf("the refused unique index was not named once:\n%s", stderr)
	}
	for _, index := range append(indexes, "requests_parent_call") {
		var present int
		must(t, db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?`, index).Scan(&present))
		if want := index != "requests_parent_call"; (present == 1) != want {
			t.Fatalf("index %s present=%d after the command", index, present)
		}
	}
	var plan, detail string
	var id, parent, unused int
	rows, err := db.Query(`EXPLAIN QUERY PLAN SELECT EXISTS(SELECT 1 FROM request_events WHERE type IN ('request.submitted','request.accepted','request.completed','request.failed','request.canceled'))`)
	must(t, err)
	for rows.Next() {
		must(t, rows.Scan(&id, &parent, &unused, &detail))
		plan += detail + "\n"
	}
	must(t, rows.Close())
	if !strings.Contains(plan, "request_events_before_responses") {
		t.Fatalf("the open's lifecycle probe still scans every event:\n%s", plan)
	}
}
