package producttest

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

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
