package producttest

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func idleAdmissionRequest(id, worker string) records.Request {
	return records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/idle", Entrypoint: "generate", Payload: []byte("{}"), Rental: true, Worker: worker}
}

// Two independently opened records writers race at the actual SQLite boundary.
// A fleet mutex alone cannot prove this: admission uses a separate code path.
func TestRentalIdleReleaseAndWorkAdmissionHaveOneWinner(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	expiry, problem := records.Open(path)
	fatal(t, problem)
	defer expiry.Close()
	admission, problem := records.Open(path)
	fatal(t, problem)
	defer admission.Close()
	now := time.Now().UTC()
	for i := 0; i < 16; i++ {
		row := idleRecord(t, expiry, fmt.Sprintf("race-%d", i), now.Add(-time.Hour))
		request := idleAdmissionRequest(fmt.Sprintf("request-%d", i), "")
		_, _, problem := admission.Submit(request)
		fatal(t, problem)
		type result struct {
			won     bool
			problem *exit.Error
		}
		start := make(chan struct{})
		releases := make(chan result, 1)
		pins := make(chan result, 1)
		go func() { <-start; won, p := expiry.ClaimRentalIdleRelease(row.ID, now); releases <- result{won, p} }()
		go func() { <-start; won, p := admission.PinRental(request.ID, row.ID, nil); pins <- result{won, p} }()
		close(start)
		released, pinned := <-releases, <-pins
		fatal(t, released.problem)
		if pinned.problem != nil && pinned.problem.ErrName() != "request.rental_unavailable" {
			fatal(t, pinned.problem)
		}
		if released.won == pinned.won {
			t.Fatalf("expiry=%+v admission=%+v; wanted exactly one winner", released, pinned)
		}
		current, problem := expiry.RentalRow(row.ID)
		fatal(t, problem)
		if released.won && current.State != "release_requested" {
			t.Fatal("release won without durable admission fence")
		}
		if pinned.won && current.State != "ready" {
			t.Fatal("new work lost its rental after winning admission")
		}
	}
}

func TestRentalIdleReleaseFenceRefusesFreshSubmissionAndPreparation(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	now := time.Now().UTC()
	row := idleRecord(t, store, "release-first", now.Add(-time.Hour))
	request := idleAdmissionRequest("unassigned-before-expiry", "")
	request.Kind = "job"
	request.MachineExecutionObserver = true
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	won, problem := store.ClaimRentalIdleRelease(row.ID, now)
	fatal(t, problem)
	if !won {
		t.Fatal("idle release was not admitted")
	}
	if _, _, problem := store.Submit(idleAdmissionRequest("fresh-after-expiry", row.ID)); problem == nil {
		t.Fatal("fresh non-retained work entered a committed release")
	}
	if pinned, problem := store.PinRental(request.ID, row.ID, nil); problem == nil || pinned {
		t.Fatal("queued work pinned to a committed release")
	}
	if problem := store.LinkMachineExecution(request.ID, row.ID); problem == nil {
		t.Fatal("machine execution linked after expiry committed")
	}
	if problem := store.RecordRentalPreparationStarted(row.ID); problem == nil {
		t.Fatal("preparation entered after expiry committed")
	}
}

func TestRentalIdleReleaseRechecksWorkSettlementAfterObservation(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	now := time.Now().UTC()
	row := idleRecord(t, store, "work-first", now.Add(-time.Hour))
	observed, problem := rental.ObserveIdle(store, row)
	fatal(t, problem)
	if !observed.Due(now) {
		t.Fatal("fixture was not overdue")
	}
	request := idleAdmissionRequest("arrives-after-observation", row.ID)
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	won, problem := store.ClaimRentalIdleRelease(row.ID, time.Now())
	fatal(t, problem)
	if won {
		t.Fatal("newly admitted queued work lost its rental")
	}
	changed, problem := store.CancelQueuedRequest(request.ID, nil)
	fatal(t, problem)
	if !changed {
		t.Fatal("new work did not settle")
	}
	won, problem = store.ClaimRentalIdleRelease(row.ID, time.Now())
	fatal(t, problem)
	if won {
		t.Fatal("stale observation overrode a new work settlement")
	}
	current, problem := store.RentalRow(row.ID)
	fatal(t, problem)
	if current.State != "ready" {
		t.Fatal("new grace was not preserved")
	}
}

func TestRentalReleaseFencePreventsLaterAttemptDispatch(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row := idleRecord(t, store, "closing", time.Now())
	request := idleAdmissionRequest("already-queued", row.ID)
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	row.State = "release_requested"
	fatal(t, store.RecordRental(row))
	if _, problem := store.Dispatch(records.Attempt{RequestID: request.ID}); problem == nil || problem.ErrName() != "request.rental_unavailable" {
		t.Fatalf("late dispatch not fenced: %v", problem)
	}
	attempts, problem := store.Attempts(request.ID)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("closing rental acquired a new attempt")
	}
}
