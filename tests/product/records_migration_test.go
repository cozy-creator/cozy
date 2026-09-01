package producttest

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	_ "modernc.org/sqlite"
)

// TestRecordsMigrationFromEleven migrates a REAL schema-11 database — the exact released
// DDL, dumped from that schema's own sqlite_master and checked in beside this test — and
// proves the schema-12 rename carried its rows. The owner's machine holds one of these, so
// the property under test is not "a fresh database has the new names" but "an existing
// database keeps its installs, pins, workers and requests while the names change".
func TestRecordsMigrationFromEleven(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.db")
	writeSchemaElevenDatabase(t, path)

	store, problem := records.Open(path)
	if problem != nil {
		t.Fatalf("schema-11 database did not migrate: %v", problem)
	}
	defer store.Close()

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
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&version); err != nil || version != 12 {
		t.Fatalf("user_version = %d, %v", version, err)
	}
	for table, want := range map[string]string{"pins": "install_id", "worker_processes": "install_id"} {
		columns := columnNames(t, db, table)
		if !columns[want] || columns["generation"] {
			t.Fatalf("%s columns after migration = %v", table, columns)
		}
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
