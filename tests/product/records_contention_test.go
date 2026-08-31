package producttest

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRecordsV3MigrationIsAtomicAndMinimal(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()

	db, err := sql.Open("sqlite", path)
	must(t, err)
	stale := []string{
		`PRAGMA user_version=0`,
		`CREATE TABLE managed_profile_installs(id TEXT)`,
		`CREATE TABLE workflow_executions(id TEXT PRIMARY KEY)`,
		`CREATE TABLE workflow_steps(workflow_id TEXT)`,
		`CREATE TABLE video_compositions(id TEXT)`,
		`CREATE TABLE placement_acquisition_observations(id TEXT)`,
		`CREATE TABLE artifact_receipts(id TEXT)`,
		`ALTER TABLE rentals ADD COLUMN released_at TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE rentals ADD COLUMN control_snapshot_digest TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE rentals ADD COLUMN control_snapshot_length INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE rentals ADD COLUMN control_snapshot_bytes BLOB NOT NULL DEFAULT x''`,
		`ALTER TABLE rentals ADD COLUMN artifact_grant_revision INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE worker_processes ADD COLUMN incarnation INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE worker_processes ADD COLUMN readiness_epoch INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE worker_processes ADD COLUMN revision INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE worker_processes ADD COLUMN intake TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE outputs ADD COLUMN visible_at TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE outputs ADD COLUMN reclaimed_at TEXT NOT NULL DEFAULT ''`,
	}
	for _, statement := range stale {
		_, err = db.Exec(statement)
		must(t, err)
	}
	must(t, db.Close())

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			opened, problem := records.Open(path)
			if problem != nil {
				errs <- problem
				return
			}
			opened.Close()
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}

	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var version, schemaVersion int
	var quick string
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&schemaVersion))
	must(t, db.QueryRow(`PRAGMA quick_check`).Scan(&quick))
	if version != 3 || quick != "ok" {
		t.Fatalf("migrated version/quick_check = %d/%q", version, quick)
	}
	retained := map[string]bool{
		"install_generations": true, "pins": true, "worker_processes": true,
		"requests": true, "attempts": true, "outputs": true, "request_events": true,
		"publications": true, "job_checkpoints": true, "artifact_finalizations": true,
		"rental_operations": true, "rentals": true,
	}
	rows, err := db.Query(`SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`)
	must(t, err)
	for rows.Next() {
		var name string
		must(t, rows.Scan(&name))
		if !retained[name] {
			t.Errorf("unowned table %s survived migration", name)
		}
		delete(retained, name)
	}
	must(t, rows.Close())
	for name := range retained {
		t.Errorf("owned table %s is absent after migration", name)
	}
	for _, table := range []string{
		"managed_profile_installs", "workflow_executions", "workflow_steps",
		"video_compositions", "placement_acquisition_observations", "artifact_receipts",
		"rental_relay_refusals", "rental_control_refusals",
	} {
		var count int
		must(t, db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?`, table).Scan(&count))
		if count != 0 {
			t.Errorf("retired table %s survived migration", table)
		}
	}
	for table, columns := range map[string][]string{
		"rentals": {"released_at", "control_snapshot_digest", "control_snapshot_length",
			"control_snapshot_bytes", "artifact_grant_revision"},
		"worker_processes": {"incarnation", "readiness_epoch", "revision", "intake"},
		"outputs":          {"visible_at", "reclaimed_at"},
	} {
		held := tableColumnNames(t, db, table)
		for _, column := range columns {
			if held[column] {
				t.Errorf("retired column %s.%s survived migration", table, column)
			}
		}
	}

	opened, problem := records.Open(path)
	fatal(t, problem)
	opened.Close()
	var reopenedSchemaVersion int
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&reopenedSchemaVersion))
	if reopenedSchemaVersion != schemaVersion {
		t.Fatalf("v3 open performed DDL: schema_version %d -> %d", schemaVersion, reopenedSchemaVersion)
	}
}

func TestRecordsV1AddsDurableRequestBudget(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	recorded, fresh, problem := store.Submit(records.Request{
		ID: "req-v1-budget", IdemKey: "v1-budget", BodyDigest: "sha256:v1-budget",
		Package: "proof/package", Entrypoint: "marco", PlanID: "sha256:plan", Payload: []byte("{}"),
	})
	fatal(t, problem)
	if !fresh || recorded.ID != "req-v1-budget" {
		t.Fatalf("v1 fixture request = %+v fresh=%v", recorded, fresh)
	}
	store.Close()

	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`ALTER TABLE requests DROP COLUMN max_cost_usd_micros`)
	must(t, err)
	_, err = db.Exec(`PRAGMA user_version=1`)
	must(t, err)
	must(t, db.Close())

	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestRow("req-v1-budget")
	fatal(t, problem)
	if row == nil || row.BodyDigest != "sha256:v1-budget" || row.MaxCostUSDMicros != 0 {
		t.Fatalf("v1 request migration = %+v", row)
	}
	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	columns := tableColumnNames(t, db, "requests")
	if version != 3 || !columns["max_cost_usd_micros"] || !columns["package_revision_digest"] ||
		!columns["environment_digest"] || !columns["config_digest"] {
		t.Fatalf("v1->v3 migration = version %d columns=%v", version, columns)
	}
}

func TestRecordsRefusesFutureSchemaWithoutMutation(t *testing.T) {
	path := t.TempDir() + "/records.db"
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`CREATE TABLE future_owner(value TEXT); PRAGMA user_version=4`)
	must(t, err)
	var before int
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&before))
	must(t, db.Close())

	opened, problem := records.Open(path)
	if opened != nil {
		opened.Close()
		t.Fatal("future records schema opened")
	}
	if problem == nil || problem.ErrName() != "records.schema_newer" {
		t.Fatalf("future schema refusal = %#v", problem)
	}

	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var version, after int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&after))
	if version != 4 || after != before {
		t.Fatalf("future schema mutated: version=%d schema_version=%d->%d", version, before, after)
	}
	var names string
	must(t, db.QueryRow(`SELECT group_concat(name, ',') FROM sqlite_master WHERE type='table'`).Scan(&names))
	if names != "future_owner" {
		t.Fatalf("future schema tables changed: %s", names)
	}
}

func tableColumnNames(t *testing.T, db *sql.DB, table string) map[string]bool {
	t.Helper()
	rows, err := db.Query(fmt.Sprintf("PRAGMA table_info(%s)", table))
	must(t, err)
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var id, notNull, primaryKey int
		var name, kind string
		var defaultValue sql.NullString
		must(t, rows.Scan(&id, &name, &kind, &notNull, &defaultValue, &primaryKey))
		out[name] = true
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return out
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
			State: "booting", Hub: "https://tensorhub.com",
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

func TestRentalSchemaHardcutPreservesIdentity(t *testing.T) {
	path := t.TempDir() + "/records.db"
	legacy, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = legacy.Exec(`CREATE TABLE rentals (
		id TEXT PRIMARY KEY,machine_name TEXT NOT NULL,sku TEXT NOT NULL,
		package_ref TEXT NOT NULL,accelerator_model TEXT NOT NULL,address TEXT NOT NULL,
		cert_path TEXT NOT NULL,state TEXT NOT NULL,hub TEXT NOT NULL,rented_at TEXT NOT NULL,
		media_address TEXT NOT NULL,selection_profile TEXT NOT NULL DEFAULT '',
		placement_set_digest TEXT NOT NULL DEFAULT '',placement_set_bytes BLOB NOT NULL DEFAULT x'',
		expected_worker_id TEXT NOT NULL DEFAULT '',expected_worker_boot_id TEXT NOT NULL DEFAULT '')`)
	must(t, err)
	_, err = legacy.Exec(`INSERT INTO rentals VALUES(
		'pr-schema-proof','studio','h200','retired/package/v1/run','NVIDIA H200',
		'127.0.0.1:8443','/tmp/cert','ready','https://tensorhub.test','2026-08-30T00:00:00Z',
		'127.0.0.1:8444','retired-profile','retired-digest',x'00','worker-proof','boot-proof')`)
	must(t, err)
	must(t, legacy.Close())

	store, problem := records.Open(path)
	fatal(t, problem)
	row, problem := store.RentalRow("pr-schema-proof")
	fatal(t, problem)
	if row == nil || row.MachineName != "studio" || row.SKU != "h200" ||
		row.ExpectedWorkerID != "worker-proof" || row.ExpectedWorkerBootID != "boot-proof" {
		t.Fatalf("slim rental lost identity: %+v", row)
	}
	store.Close()

	check, err := sql.Open("sqlite", path)
	must(t, err)
	defer check.Close()
	rows, err := check.Query(`PRAGMA table_info(rentals)`)
	must(t, err)
	columns := map[string]bool{}
	for rows.Next() {
		var ordinal, notNull, primary int
		var name, kind string
		var defaultValue sql.NullString
		must(t, rows.Scan(&ordinal, &name, &kind, &notNull, &defaultValue, &primary))
		columns[name] = true
	}
	must(t, rows.Close())
	want := []string{"id", "machine_name", "sku", "accelerator_model", "address", "cert_path",
		"state", "hub", "rented_at", "media_address", "expected_worker_id", "expected_worker_boot_id"}
	if len(columns) != len(want) {
		t.Fatalf("rental columns = %v, want exactly %v", columns, want)
	}
	for _, name := range want {
		if !columns[name] {
			t.Fatalf("slim rental schema omitted %s: %v", name, columns)
		}
	}
	var retired int
	must(t, check.QueryRow(`SELECT count(*) FROM sqlite_master
		WHERE type='table' AND name IN ('rental_relay_refusals','rental_control_refusals')`).Scan(&retired))
	if retired != 0 {
		t.Fatalf("retired rental relay/control tables remain: %d", retired)
	}
}
