package producttest

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

func idleAdmissionRequest(id, worker string) records.Request {
	return records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/idle", Entrypoint: "generate", Payload: []byte("{}"), Rental: true, Worker: worker}
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
