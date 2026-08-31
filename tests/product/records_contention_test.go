package producttest

import (
	"context"
	"database/sql"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRecordsSchemaIsExactV1AndStable(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()

	db, err := sql.Open("sqlite", path)
	must(t, err)
	var version, before int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&before))
	if version != 1 {
		t.Fatalf("fresh records version = %d, want exact v1", version)
	}
	for _, column := range []string{
		"max_cost_usd_micros", "package_revision_digest", "environment_digest", "config_digest",
	} {
		var count int
		must(t, db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('requests') WHERE name=?`,
			column).Scan(&count))
		if count != 1 {
			t.Fatalf("exact requests schema omitted %s", column)
		}
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
		t.Fatalf("exact v1 reopen performed DDL: schema_version %d -> %d", before, after)
	}
}

func TestRecordsRefusesOtherVersionWithoutMutation(t *testing.T) {
	path := t.TempDir() + "/records.db"
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`CREATE TABLE older_owner(value TEXT); PRAGMA user_version=2`)
	must(t, err)
	var before int
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&before))
	must(t, db.Close())

	opened, problem := records.Open(path)
	if opened != nil {
		opened.Close()
		t.Fatal("noncurrent records schema opened")
	}
	if problem == nil || problem.ErrName() != "records.schema_reset_required" ||
		!strings.Contains(problem.Remedy, "move") {
		t.Fatalf("noncurrent schema refusal = %#v", problem)
	}
	db, err = sql.Open("sqlite", path)
	must(t, err)
	defer db.Close()
	var version, after int
	must(t, db.QueryRow(`PRAGMA user_version`).Scan(&version))
	must(t, db.QueryRow(`PRAGMA schema_version`).Scan(&after))
	if version != 2 || after != before {
		t.Fatalf("refused schema mutated: version=%d schema_version=%d->%d", version, before, after)
	}
}

func TestRecordsRefusesV1ShapeDrift(t *testing.T) {
	path := t.TempDir() + "/records.db"
	store, problem := records.Open(path)
	fatal(t, problem)
	store.Close()
	db, err := sql.Open("sqlite", path)
	must(t, err)
	_, err = db.Exec(`ALTER TABLE requests ADD COLUMN compatibility_alias TEXT`)
	must(t, err)
	must(t, db.Close())

	opened, problem := records.Open(path)
	if opened != nil {
		opened.Close()
		t.Fatal("drifted v1 records schema opened")
	}
	if problem == nil || problem.ErrName() != "records.schema_reset_required" {
		t.Fatalf("drifted v1 refusal = %#v", problem)
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
