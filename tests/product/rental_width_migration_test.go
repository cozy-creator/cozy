package producttest

import (
	"database/sql"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func retainedWidthDatabase(t *testing.T, width bool) (string, *sql.DB) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "retained.db")
	raw, err := os.ReadFile("testdata/records-schema33-before-rental-width.sql")
	must(t, err)
	ddl := string(raw)
	if width {
		ddl = strings.Replace(ddl, "  accelerator_model TEXT NOT NULL,\n", "  accelerator_model TEXT NOT NULL,\n  accelerator_count INTEGER NOT NULL DEFAULT 0,\n", 1)
	}
	db, err := sql.Open("sqlite", path)
	must(t, err)
	db.SetMaxOpenConns(1)
	_, err = db.Exec(ddl)
	must(t, err)
	for _, statement := range []string{
		`INSERT INTO rentals(id,machine_name,accelerator_model,hourly_rate_usd_micros,address,cert_path,state,hub,rented_at,ready_at) VALUES('rental','shelly','CPU',74170,'private.example','retained-cert','ready','hub','2026-09-08','2026-09-08')`,
		`INSERT INTO requests(id,idem_key,body_digest,package,entrypoint,plan_id,payload,state,created_at,worker,machine,retain_work,child_reusable) VALUES('request','key','sha256:body','local/client','main','plan',x'7b7d','blocked','2026-09-08','rental','shelly',1,1)`,
		`INSERT INTO worker_processes(instance_id,package,worker_id,devices,pid,birth,state,opened_at) VALUES('worker','local/client','private','[]',123,'birth','running','2026-09-08')`,
		`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,session_id,invocation_digest,invocation,state,dispatched_at,terminal_body) VALUES('request',1,'attempt','worker','session','sha256:invocation',x'7b7d','closed','2026-09-08',x'7b7d')`,
		`INSERT INTO byte_outputs VALUES('request',1,'facts','sha256:bytes',100,'application/json','native-root','sha256:receipt','sha256:manifest',200,100)`,
		`INSERT INTO request_operation_lookups VALUES('request','sha256:computation','hit')`,
		`INSERT INTO native_artifact_retentions(artifact_kind,producer_attempt,producer_output_id,content_bytes,consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,owner_worker,retention_id,state) VALUES('tree',1,'facts',100,'consumer','request','result','facts','request','sha256:manifest',200,'sha256:receipt','transaction','request','rental','retention','held')`,
	} {
		_, err = db.Exec(statement)
		must(t, err)
	}
	return path, db
}

func tableRows(t *testing.T, db *sql.DB, table string) string {
	t.Helper()
	rows, err := db.Query(`SELECT * FROM ` + table)
	must(t, err)
	defer rows.Close()
	columns, err := rows.Columns()
	must(t, err)
	result := []map[string]any{}
	for rows.Next() {
		values, pointers := make([]any, len(columns)), make([]any, len(columns))
		for i := range values {
			pointers[i] = &values[i]
		}
		must(t, rows.Scan(pointers...))
		record := map[string]any{}
		for i, name := range columns {
			if name != "accelerator_count" && name != "native_service_id" && name != "requested_rental" {
				record[name] = values[i]
			}
		}
		result = append(result, record)
	}
	must(t, rows.Err())
	raw, err := json.Marshal(result)
	must(t, err)
	return string(raw)
}

func TestBothReleasedSchema33RentalShapesMigrateWithoutLosingWork(t *testing.T) {
	for _, width := range []bool{false, true} {
		name := "before-width"
		if width {
			name = "recorded-width"
		}
		t.Run(name, func(t *testing.T) {
			path, db := retainedWidthDatabase(t, width)
			if width {
				_, err := db.Exec(`UPDATE rentals SET accelerator_count=8`)
				must(t, err)
			}
			before := map[string]string{}
			for _, table := range []string{"rentals", "requests", "attempts", "byte_outputs", "worker_processes", "request_operation_lookups", "native_artifact_retentions"} {
				before[table] = tableRows(t, db, table)
			}
			if old, problem := records.Open(path); problem == nil || problem.ErrName() != "records_schema_upgrade_required" {
				if old != nil {
					old.Close()
				}
				t.Fatalf("ordinary reader must request daemon migration: %v", problem)
			}
			var version int
			must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
			if version != 33 {
				t.Fatalf("ordinary Open mutated schema: %d", version)
			}
			store, problem := records.OpenForDaemon(path, "")
			fatal(t, problem)
			row, problem := store.RentalRow("rental")
			fatal(t, problem)
			expected := 1
			if width {
				expected = 8
			}
			if row.AcceleratorCount != expected {
				t.Fatalf("width=%d, want %d", row.AcceleratorCount, expected)
			}
			store.Close()
			for table, expected := range before {
				if got := tableRows(t, db, table); got != expected {
					t.Fatalf("%s rows changed: %s != %s", table, got, expected)
				}
			}
			must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
			if version != 37 {
				t.Fatalf("migration version=%d", version)
			}
			violations, err := db.Query(`PRAGMA foreign_key_check`)
			must(t, err)
			if violations.Next() {
				t.Fatal("migration broke foreign keys")
			}
			violations.Close()
			db.Close()
			reopened, problem := records.Open(path)
			fatal(t, problem)
			reopened.Close()
		})
	}
}

func TestSchema33WidthMigrationPreservesExplicitZeroAndRejectsUnknownShape(t *testing.T) {
	path, db := retainedWidthDatabase(t, true)
	store, problem := records.OpenForDaemon(path, "")
	fatal(t, problem)
	rental, problem := store.RentalRow("rental")
	fatal(t, problem)
	if rental.AcceleratorCount != 0 {
		t.Fatal("migration invented a count for an existing zero")
	}
	store.Close()
	db.Close()
	path, db = retainedWidthDatabase(t, false)
	defer db.Close()
	_, err := db.Exec(`ALTER TABLE rentals ADD COLUMN surprise TEXT`)
	must(t, err)
	before := tableRows(t, db, "rentals")
	if store, problem = records.OpenForDaemon(path, ""); problem == nil {
		store.Close()
		t.Fatal("unrecognized schema drift was accepted")
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 33 || tableRows(t, db, "rentals") != before {
		t.Fatal("refused migration changed retained database")
	}
}

func TestRetainedWidthMigrationRollsBackBeforePublishingInvalidForeignKeys(t *testing.T) {
	path, db := retainedWidthDatabase(t, false)
	defer db.Close()
	_, err := db.Exec(`UPDATE byte_outputs SET attempt=99`)
	must(t, err)
	before := tableRows(t, db, "byte_outputs")
	store, problem := records.OpenForDaemon(path, "")
	if store != nil {
		store.Close()
	}
	if problem == nil || problem.ErrName() != "records.migration_foreign_key_failed" {
		t.Fatalf("invalid retained foreign key must refuse before commit: %v", problem)
	}
	var version int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	if version != 33 || columnNames(t, db, "rentals")["accelerator_count"] || tableRows(t, db, "byte_outputs") != before {
		t.Fatal("failed migration published a partial schema or changed retained bytes")
	}
}
