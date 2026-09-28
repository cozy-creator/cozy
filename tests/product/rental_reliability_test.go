package producttest

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalObservationWaitsForWriterWithoutAnotherPaidOperation(t *testing.T) {
	fullRun(t, "outwaits SQLite's five-second busy handler")
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	var authored int
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "accepted-once", Hub: "http://hub.example", HourlyRateUSDMicros: 1},
		func(machine string) ([]byte, string, *exit.Error) { authored++; return replacementAuthor(machine) })
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, "pr-observed", "acquiring"))
	writer, err := sql.Open("sqlite", path+"?_txlock=immediate")
	must(t, err)
	defer writer.Close()
	tx, err := writer.Begin()
	must(t, err)
	defer tx.Rollback()
	row := records.Rental{ID: "pr-observed", MachineName: "otter", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, State: "ready", Hub: op.Hub}
	done := make(chan *exit.Error, 1)
	go func() { done <- store.RecordRentalContext(context.Background(), row) }()
	// Exceed the old SQLite busy handler. The Hub fact is still being recorded,
	// rather than making the caller replay a paid acquisition after five seconds.
	select {
	case problem := <-done:
		t.Fatalf("observation stopped while the writer held the lock: %v", problem)
	case <-time.After(5500 * time.Millisecond):
	}
	must(t, tx.Commit())
	select {
	case problem := <-done:
		fatal(t, problem)
	case <-time.After(5 * time.Second):
		t.Fatal("observation did not resume after writer commit")
	}
	recorded, problem := store.RentalRow(row.ID)
	fatal(t, problem)
	if recorded == nil || recorded.State != "ready" {
		t.Fatalf("accepted rental was not recorded: %+v", recorded)
	}
	saved, problem := store.RentalOperation(op.Key)
	fatal(t, problem)
	if authored != 1 || saved.RentalID != row.ID || saved.RequestDigest != op.RequestDigest || string(saved.RequestBody) != string(op.RequestBody) {
		t.Fatal("recording an observation changed the durable paid operation")
	}
	assertRentalObservationRestoredBusyWait(t, store, writer)
}

func TestRentalObservationCancelsWhileWriterStillHoldsLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer store.Close()
	writer, err := sql.Open("sqlite", path+"?_txlock=immediate")
	must(t, err)
	defer writer.Close()
	tx, err := writer.Begin()
	must(t, err)
	defer tx.Rollback()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan *exit.Error, 1)
	go func() { done <- store.RecordRentalContext(ctx, records.Rental{ID: "pr-canceled-observation"}) }()
	select {
	case problem := <-done:
		t.Fatalf("observation did not wait for writer: %v", problem)
	case <-time.After(150 * time.Millisecond):
	}
	cancel()
	select {
	case problem := <-done:
		if problem == nil || problem.Code != exit.Canceled || !strings.Contains(problem.Message, "context canceled") {
			t.Fatalf("wait did not preserve cancellation: %v", problem)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancellation waited for the SQLite writer to release")
	}
	// Still holding the writer proves cancellation ended only this observation.
	must(t, tx.Commit())
	row, problem := store.RentalRow("pr-canceled-observation")
	fatal(t, problem)
	if row != nil {
		t.Fatal("canceled observation wrote a rental")
	}
	assertRentalObservationRestoredBusyWait(t, store, writer)
	// A validation failure must return, not enter a generic error retry loop.
	problem = store.RecordRental(records.Rental{ID: "pr-invalid"})
	if problem == nil || problem.ErrName() != "rental.accelerator_count_missing" {
		t.Fatalf("non-lock error changed: %v", problem)
	}
	assertRentalObservationRestoredBusyWait(t, store, writer)
}

func TestPreparationFailureOnlySettlesCapturedSelection(t *testing.T) {
	for _, change := range []string{"rental", "ordinal", "control_revision", "unchanged"} {
		t.Run(change, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "creator.sqlite")
			store, problem := records.Open(path)
			fatal(t, problem)
			defer store.Close()
			replacementRequest(t, store, "request", "rental-old")
			expected, problem := store.RequestRow("request")
			fatal(t, problem)
			switch change {
			case "rental":
				changed, problem := store.UnpinRentalWork("request", "rental-old")
				fatal(t, problem)
				if !changed {
					t.Fatal("fixture did not unpin")
				}
			case "ordinal", "control_revision":
				db, err := sql.Open("sqlite", path)
				must(t, err)
				//cozy:allow raw fixture update models a newer accepted attempt or operator control revision.
				_, err = db.Exec("UPDATE requests SET " + change + "=" + change + "+1 WHERE id='request'")
				must(t, err)
				must(t, db.Close())
			}
			applied, problem := store.FailQueuedPreparation(*expected, map[string]any{"error_type": "proof.old_preparation", "error": "old selection failed"})
			fatal(t, problem)
			if applied != (change == "unchanged") {
				t.Fatalf("failure applied=%v for %s", applied, change)
			}
			events, problem := store.EventsAfter("request", 0, 100)
			fatal(t, problem)
			failures := 0
			for _, event := range events {
				if event.Type == "request.failed" {
					failures++
				}
			}
			if (failures == 1) != (change == "unchanged") {
				t.Fatalf("stale failure wrote %d terminal events", failures)
			}
		})
	}
}

// Exercise an ordinary store write after the exclusive observation connection
// returned to its pool. A leaked busy_timeout=0 would fail instead of waiting.
func assertRentalObservationRestoredBusyWait(t *testing.T, store *records.Store, writer *sql.DB) {
	t.Helper()
	tx, err := writer.Begin()
	must(t, err)
	defer tx.Rollback()
	done := make(chan *exit.Error, 1)
	go func() {
		_, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "after-observation", Hub: "http://hub.example", HourlyRateUSDMicros: 1}, replacementAuthor)
		done <- problem
	}()
	select {
	case problem := <-done:
		t.Fatalf("observation leaked its SQLite wait policy: %v", problem)
	case <-time.After(150 * time.Millisecond):
	}
	must(t, tx.Commit())
	select {
	case problem := <-done:
		fatal(t, problem)
	case <-time.After(5 * time.Second):
		t.Fatal("ordinary writer did not resume")
	}
}
