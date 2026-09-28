package producttest

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func completedPublicationFixture(t *testing.T) (*records.Store, *sql.DB, string) {
	t.Helper()
	store, db, id := publicationRetryFixture(t)
	_, err := db.Exec(`UPDATE requests SET worker='rental-unavailable',rental=1`)
	must(t, err)
	changed, problem := store.RetryModelTransferPublication(id, "settlement-proof")
	fatal(t, problem)
	if !changed {
		t.Fatal("successful producer fixture did not resume publication")
	}
	return store, db, id
}

func publicationStateOwner(t *testing.T, name string, store *records.Store, releases *atomic.Int64) *owner {
	t.Helper()
	return hostOwner(t, name, func(options *orchestrator.Options) {
		options.Store = store
		options.ReleaseManagedRental = func(string) (string, *exit.Error) { releases.Add(1); return "", nil }
	})
}

func TestCompletedPublicationSucceedsWithoutWorkerSession(t *testing.T) {
	store, db, id := completedPublicationFixture(t)
	rows, problem := store.AllModelTransferWeights(id, 1)
	fatal(t, problem)
	fatal(t, store.CompleteModelTransfer(id, map[string]string{"model": rows[0].ManifestID}))
	before, problem := store.Attempts(id)
	fatal(t, problem)
	var releases atomic.Int64
	first := publicationStateOwner(t, "publication-missing-session", store, &releases)
	waited := make(chan *exit.Error, 1)
	go func() { _, problem := first.c.AwaitSettled(id, 5*time.Second); waited <- problem }()
	fatal(t, first.c.ResumeModelTransfers())
	waitUntil(t, "completed publication visible without worker ACK", func() bool {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		return row.State == "succeeded"
	})
	fatal(t, <-waited)
	first.close()
	store.Close()
	reopened, problem := records.Open(dbPathForRetry(t, db))
	fatal(t, problem)
	defer reopened.Close()
	second := publicationStateOwner(t, "publication-restarted-session", reopened, &releases)
	fatal(t, second.c.ResumeModelTransfers())
	defer second.close()
	after, problem := reopened.Attempts(id)
	fatal(t, problem)
	old, _ := json.Marshal(before)
	fresh, _ := json.Marshal(after)
	var cleaned int
	must(t, db.QueryRow(`SELECT media_cleaned FROM attempts WHERE request_id=?`, id).Scan(&cleaned))
	if !bytes.Equal(old, fresh) || len(after) != 1 || after[0].State != "terminal" || cleaned != 0 {
		t.Fatal("request settlement changed successful attempt or pretended worker cleanup occurred")
	}
	transfer, problem := reopened.ModelTransferOf(id)
	fatal(t, problem)
	if transfer.State != "completed" || transfer.Checkpoints["model"] != rows[0].ManifestID || releases.Load() != 0 {
		t.Fatalf("publication result or provider retention changed: %+v releases=%d", transfer, releases.Load())
	}
	events, problem := reopened.EventsAfter(id, 0, 100)
	fatal(t, problem)
	completions := 0
	for _, event := range events {
		if event.Type == "request.completed" {
			completions++
		}
	}
	if completions != 1 {
		t.Fatalf("got %d completion events across owner restart", completions)
	}
	pending, problem := reopened.OpenAttemptsOf(after[0].InstanceID)
	fatal(t, problem)
	if len(pending) != 1 {
		t.Fatal("settled publication lost the retained terminal cleanup obligation")
	}
}

func TestIncompleteOrFailedPublicationCannotSucceedWithoutSession(t *testing.T) {
	for _, state := range []string{"finalizing", "failed", "failed-producer"} {
		t.Run(state, func(t *testing.T) {
			store, db, id := completedPublicationFixture(t)
			if state == "failed" {
				fatal(t, store.FailModelTransfer(id, "upload_failed", "bytes are missing"))
			}
			if state == "failed-producer" {
				rows, problem := store.AllModelTransferWeights(id, 1)
				fatal(t, problem)
				fatal(t, store.CompleteModelTransfer(id, map[string]string{"model": rows[0].ManifestID}))
				_, err := db.Exec(`UPDATE attempts SET terminal_status='FAILED'`)
				must(t, err)
			}
			var releases atomic.Int64
			o := publicationStateOwner(t, "publication-incomplete-"+state, store, &releases)
			fatal(t, o.c.ResumeModelTransfers())
			o.close()
			row, problem := store.RequestRow(id)
			fatal(t, problem)
			if row.State == "succeeded" || releases.Load() != 0 {
				t.Fatalf("unproven publication succeeded: %s", row.State)
			}
		})
	}
}
