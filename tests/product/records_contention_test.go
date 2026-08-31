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
