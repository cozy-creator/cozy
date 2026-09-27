package producttest

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"

	"github.com/cozy-creator/cozy/internal/records"
	_ "modernc.org/sqlite"
)

// TestRecordsMigrationFromEleven migrates a REAL schema-11 database — the exact released
// DDL, dumped from that schema's own sqlite_master and checked in beside this test — and
// proves the schema-12 rename, the schema-13 rental column, the schema-14 export table,
// the schema-15 request column spelling, schema-17 machine-word backfill, and schema-20
// package-interface/digest cleanup and schema-21 rental failure fields carried
// their rows. The owner's
// machine holds one of these, so the property under test is not "a fresh database has the
// new names" but "an existing database keeps its installs, pins, workers, requests and
// rentals while the shape changes".
func TestRecordsMigrationFromEleven(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "records.db")
	writeSchemaElevenDatabase(t, path)

	// Two triage files beside the database: one an attempt row references — its bytes
	// must land IN that row — and one orphan the audit found 259 of, which nothing may
	// import and the daemon's retired-shape sweep later deletes with the directory.
	triageDir := filepath.Join(root, "triage")
	if err := os.MkdirAll(triageDir, 0o755); err != nil {
		t.Fatal(err)
	}
	bundle := []byte(`{"terminal":{"traceback":"boom"}}`)
	digest := sha256.Sum256(bundle)
	if err := os.WriteFile(filepath.Join(triageDir, "trb-kept.json"), bundle, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(triageDir, "trb-orphan.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	writeSchemaElevenAttempt(t, path, "sha256:"+hex.EncodeToString(digest[:]),
		int64(len(bundle)), filepath.Join(triageDir, "trb-kept.json"))

	if store, problem := records.Open(path); problem == nil || problem.ErrName() != "records_schema_upgrade_required" {
		if store != nil {
			store.Close()
		}
		t.Fatalf("ordinary Open = %v, want records_schema_upgrade_required", problem)
	}
	before, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	var priorVersion int
	if err := before.QueryRow(`PRAGMA user_version`).Scan(&priorVersion); err != nil || priorVersion != 11 {
		t.Fatalf("ordinary Open changed user_version to %d: %v", priorVersion, err)
	}
	before.Close()
	store, problem := records.OpenForDaemon(path, triageDir)
	if problem != nil {
		t.Fatalf("schema-11 database did not migrate: %v", problem)
	}
	defer store.Close()

	// Schema 22: the referenced bundle's exact bytes now live in the attempt row and are
	// served by the attempt's opaque key; the orphan imported nowhere.
	if subject, kept, problem := store.TriageBundle("att-triaged"); problem != nil ||
		subject != "trb-kept" || string(kept) != string(bundle) {
		t.Fatalf("triage bundle after migration = %q/%d bytes, %v", subject, len(kept), problem)
	}

	// The product's own readers answer, so the rows survived as ROWS and not merely as
	// bytes in a table that happens to still exist.
	installed, problem := store.Installed()
	if problem != nil || len(installed) != 1 || installed[0].ID != "1111111111111111" ||
		installed[0].Package != "cozy/example" || installed[0].Version != "1.0.0" {
		t.Fatalf("installed after migration = %+v, %v", installed, problem)
	}
	pin, inst, problem := store.ActivePackage("cozy/example")
	if problem != nil || pin == nil || pin.InstallID != "1111111111111111" || inst == nil {
		t.Fatalf("pin after migration = %+v, %+v, %v", pin, inst, problem)
	}
	workers, problem := store.LiveWorkers()
	if problem != nil || len(workers) != 1 || workers[0].InstallID != "1111111111111111" {
		t.Fatalf("workers after migration = %+v, %v", workers, problem)
	}

	// And the names themselves: one word, in the two columns that spelled it `generation`.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var version int
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&version); err != nil || version != 49 {
		t.Fatalf("user_version = %d, %v", version, err)
	}
	if columns := columnNames(t, db, "attempts"); !columns["triage_bundle"] || columns["triage_path"] {
		t.Fatalf("attempts columns after migration = %v", columns)
	}
	// Schema 46 names the request's local installation, schema 17 records
	// the machine word, schema 19 deletes the duplicate config digest, and schema 20
	// deletes the obsolete aggregate package revision digest.
	if columns := columnNames(t, db, "requests"); !columns["local_installation_id"] || columns["local_package_digest"] ||
		!columns["local_package_uploaded_boot_id"] || columns["private_package_digest"] ||
		!columns["machine"] || !columns["retain_work"] || !columns["retry_of"] || !columns["reuse_scope"] ||
		!columns["control_revision"] || columns["config_digest"] || columns["package_revision_digest"] {
		t.Fatalf("requests columns after migration = %v", columns)
	}
	var retained int
	if err := db.QueryRow(`SELECT COUNT(*) FROM requests WHERE retain_work!=0 OR retry_of!='' OR reuse_scope!='' OR control_revision!=0`).Scan(&retained); err != nil || retained != 0 {
		t.Fatalf("migration silently opted existing requests into retained work: %d, %v", retained, err)
	}
	// Schema 17 backfill: a row whose rental row survives gets that rental's machine
	// word; a row whose rental is gone is unjoinable history and stays BLANK — the raw
	// pr- id must never stand in for a name (cl-107). A row never bound stays blank too.
	for id, want := range map[string]string{
		"request-1": "", "request-rented": "quiet-heron-0000000000000011", "request-orphan": ""} {
		var machine string
		if err := db.QueryRow(`SELECT machine FROM requests WHERE id=?`, id).
			Scan(&machine); err != nil || machine != want {
			t.Fatalf("request %s machine after migration = %q, %v (want %q)", id, machine, err, want)
		}
	}
	for table, want := range map[string]string{"pins": "install_id", "worker_processes": "install_id"} {
		columns := columnNames(t, db, table)
		if !columns[want] || columns["generation"] || columns["package_revision_digest"] {
			t.Fatalf("%s columns after migration = %v", table, columns)
		}
	}
	if columns := columnNames(t, db, "installs"); columns["package_interface"] || columns["package_descriptor"] {
		t.Fatalf("installs columns after migration = %v", columns)
	}
	if columns := columnNames(t, db, "rentals"); !columns["failure_code"] ||
		!columns["failure_image_digest"] || !columns["failure_provider_resource_id"] ||
		!columns["failure_provider_host_id"] || !columns["failure_container_state"] ||
		!columns["accelerator_count"] {
		t.Fatalf("rental failure columns after migration = %v", columns)
	}
	var tables int
	if err := db.QueryRow(`SELECT COUNT(*) FROM sqlite_master
		WHERE type='table' AND name IN ('installs','install_generations')`).Scan(&tables); err != nil ||
		tables != 1 {
		t.Fatalf("install table count after migration = %d, %v", tables, err)
	}
	var requestInstall string
	if err := db.QueryRow(`SELECT install_id FROM requests WHERE id='request-1'`).
		Scan(&requestInstall); err != nil || requestInstall != "1111111111111111" {
		t.Fatalf("request install after migration = %q, %v", requestInstall, err)
	}
	// A settled export keeps its published paths; the payload hash that used to name the
	// file is gone with the column (schema 14: a file is named by its own content digest).
	if columns := columnNames(t, db, "request_output_exports"); columns["payload_hash"] || !columns["directory"] {
		t.Fatalf("request_output_exports columns after migration = %v", columns)
	}
	if columns := columnNames(t, db, "request_model_transfer_outputs"); columns["evidence"] {
		t.Fatalf("request_model_transfer_outputs still carries checkpoint evidence: %v", columns)
	}
	export, problem := store.OutputExportOf("request-1")
	if problem != nil || export == nil || export.State != "published" ||
		len(export.PublishedPaths) != 1 || export.PublishedPaths[0] != "/tmp/out/abc.png" ||
		len(export.Outputs) != 1 || export.Outputs[0].OutputID != "image" {
		t.Fatalf("output export after migration = %+v, %v", export, problem)
	}
	// A rental that was already ready when the schema moved keeps its row and gets no
	// invented ready_at: its idle clock starts at its next settlement.
	rented, problem := store.RentalRow("rental-1")
	if problem != nil || rented == nil || rented.State != "ready" || rented.ReadyAt != "" ||
		rented.MachineName != "quiet-heron-0000000000000011" {
		t.Fatalf("rental after migration = %+v, %v", rented, problem)
	}
	// Schema 33: the retained rental states its WIDTH. Every rental bought before that
	// schema was one card wide — no wider product could be expressed by the Creator that
	// bought it, let alone attached — so 1 is a fact about this row, not a placeholder.
	if rented.AcceleratorCount != 1 {
		t.Fatalf("rental width after migration = %d, want the one card it was bought as",
			rented.AcceleratorCount)
	}
}

// TestRecordsMigrationFromThirtySeven proves the additive attention-pin column is
// defaulted when a schema-37 daemon database is upgraded. It also protects the
// operator-facing refusal older binaries must give when they see this newer shape:
// upgrade in place, never move the records database aside as if it were corrupt.
func TestRecordsMigrationFromThirtySeven(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "records.db")
	store, problem := records.OpenForDaemon(path, "")
	if problem != nil {
		t.Fatalf("initialize records: %v", problem)
	}
	store.Close()

	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at)
		VALUES('request-schema37','idem-schema37','sha256:body','cozy/example','generate','sha256:plan',x'00','queued','2026-01-01T00:00:00Z')`); err != nil {
		db.Close()
		t.Fatalf("plant schema-37 request: %v", err)
	}
	if _, err := db.Exec(`ALTER TABLE requests DROP COLUMN attention_kernel`); err != nil {
		db.Close()
		t.Fatalf("make schema-37 requests shape: %v", err)
	}
	revertRecordsSchema(t, db, 37)
	if _, err := db.Exec(`DROP TRIGGER machine_execution_no_local_attempt; DROP TABLE machine_executions;
		DROP TABLE IF EXISTS successful_work_releases; DROP TABLE device_memory_measurements; DROP TABLE rental_idle; DROP TABLE rental_runtime_updates; PRAGMA user_version=37`); err != nil {
		db.Close()
		t.Fatalf("stamp schema-37 database: %v", err)
	}
	db.Close()

	if store, problem := records.Open(path); problem == nil || problem.ErrName() != "records_schema_upgrade_required" {
		if store != nil {
			store.Close()
		}
		t.Fatalf("ordinary Open = %v, want records_schema_upgrade_required", problem)
	}
	store, problem = records.OpenForDaemon(path, "")
	if problem != nil {
		t.Fatalf("schema-37 database did not migrate: %v", problem)
	}
	defer store.Close()

	db, err = sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var version int
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&version); err != nil || version != 49 {
		t.Fatalf("user_version = %d, %v", version, err)
	}
	var pin string
	if err := db.QueryRow(`SELECT attention_kernel FROM requests WHERE id='request-schema37'`).Scan(&pin); err != nil || pin != "" {
		t.Fatalf("migrated attention pin = %q, %v", pin, err)
	}
}

// A database a newer Creator wrote opens in compatibility mode: this build reads and writes
// only the tables and columns it knows and changes nothing else. It refuses only when a
// table or column it requires is missing or holds an incompatible type, naming it.
func TestRecordsWrittenByANewerCreatorOpenInCompatibilityMode(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "records.db")
	store, problem := records.OpenForDaemon(path, "")
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
		func(path string) (*records.Store, *exit.Error) { return records.OpenForDaemon(path, "") }} {
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

// writeSchemaElevenAttempt plants one closed, triage-bearing attempt in the released
// shape: the bundle bytes live in a FILE the row names, which is exactly what schema 22
// retires.
func writeSchemaElevenAttempt(t *testing.T, path, digest string, length int64, triagePath string) {
	t.Helper()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,
		session_id,invocation_digest,invocation,state,terminal_status,terminal_cause,
		triage_subject,triage_digest,triage_length,triage_path,dispatched_at,closed_at)
		VALUES('request-orphan',1,'att-triaged','worker-1','session-1','sha256:ff',x'00',
		'closed','FAILED','handler_error','trb-kept',?,?,?,
		'2026-01-01T00:00:01Z','2026-01-01T00:00:02Z')`,
		digest, length, triagePath); err != nil {
		t.Fatalf("cannot plant the schema-11 attempt: %v", err)
	}
}

// writeSchemaElevenDatabase builds the database the migration must accept: the released
// schema-11 shape, stamped with its own user_version, carrying one install and the rows
// that reference it. Open refuses any database that is not that exact shape, so a fixture
// that drifted would fail here rather than migrate something else.
func writeSchemaElevenDatabase(t *testing.T, path string) {
	t.Helper()
	ddl, err := os.ReadFile(filepath.Join("testdata", "records", "schema-11.sql"))
	if err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, statement := range []string{string(ddl), `
		INSERT INTO install_generations VALUES('1111111111111111','cozy/example',1,'1.0.0',
		  'tensorhub','cozy/example@1.0.0','sha256:aa',1,'/tmp/installs/1111111111111111',
		  '/usr/bin/python3','','','uv 0.5.0','sha256:bb','linux-x86_64','',7,'example==1.0.0',
		  'sha256:cc','',4096,2048,'2026-01-01T00:00:00Z')`, `
		INSERT INTO pins VALUES('cozy/example',1,'1111111111111111','2026-01-01T00:00:00Z')`, `
		INSERT INTO worker_processes(instance_id,package,generation,package_revision_digest,
		  worker_id,devices,pid,birth,state,opened_at)
		VALUES('worker-1','cozy/example','1111111111111111','sha256:dd','local','|cpu|',
		  4242,'birth-1','registered','2026-01-01T00:00:00Z')`, `
		INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,
		  created_at,install_id)
		VALUES('request-1','idem-1','sha256:ee','cozy/example','predict','plan-1',x'00',
		  'queued','2026-01-01T00:00:00Z','1111111111111111')`, `
		INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,
		  created_at,worker,rental)
		VALUES('request-rented','idem-2','sha256:ee','cozy/example','predict','plan-2',x'00',
		  'succeeded','2026-01-01T00:00:00Z','rental-1',1)`, `
		INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,
		  created_at,worker,rental)
		VALUES('request-orphan','idem-3','sha256:ee','cozy/example','predict','plan-3',x'00',
		  'failed','2026-01-01T00:00:00Z','rental-released-and-gone',1)`, `
		INSERT INTO rentals(id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,address,
		  cert_path,state,hub,rented_at)
		VALUES('rental-1','quiet-heron-0000000000000011','cpu','CPU',100000,'127.0.0.1:1',
		  '/tmp/rental-1.pem','ready','https://hub.invalid','2026-01-01T00:00:00Z')`, `
		INSERT INTO request_output_exports(request_id,directory,payload_hash,outputs,state,
		  published_paths,updated_at)
		VALUES('request-1','/tmp/out','abc',
		  '[{"output_id":"image","media_type":"image/png","filename":"abc.png"}]',
		  'published','["/tmp/out/abc.png"]','2026-01-01T00:00:00Z')`, `
		PRAGMA user_version=11`} {
		if _, err := db.Exec(statement); err != nil {
			t.Fatalf("cannot build the schema-11 fixture: %v", err)
		}
	}
}

func columnNames(t *testing.T, db *sql.DB, table string) map[string]bool {
	t.Helper()
	rows, err := db.Query(`SELECT name FROM pragma_table_info(?)`, table)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			t.Fatal(err)
		}
		out[name] = true
	}
	return out
}
