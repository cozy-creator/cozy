package producttest

import (
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// An attempt that may have started on a rented machine that was lost fails naming the
// loss; it is never offered again, and nothing is bought to run it.
func TestALostRentalFailsItsStartedAttempt(t *testing.T) {
	var purchases atomic.Int32
	o := hostOwner(t, "lost-started-attempt", func(opt *orchestrator.Options) {
		opt.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			purchases.Add(1)
			return orchestrator.PlacementDecision{}, "", nil
		}
	})
	const id = "req-lost-started"
	_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + sixtyFour("1"), Package: "proof/model",
		Entrypoint: "run", Payload: []byte("{}"), Rental: true, RentalRequired: true, Worker: "gone"})
	fatal(t, problem)
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "lost", Package: "proof/model", WorkerID: "lost", Devices: []string{"cpu"}}))
	ordinal, problem := o.store.Dispatch(records.Attempt{RequestID: id, SessionID: "session", InstanceID: "lost",
		InvocationDigest: "sha256:" + sixtyFour("2"), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, ordinal, "session"))
	fatal(t, o.store.Accepted(id, ordinal, "session"))
	o.c.RecoverLostWork()
	row, problem := o.store.RequestRow(id)
	fatal(t, problem)
	attempts, problem := o.store.Attempts(id)
	fatal(t, problem)
	errorType, _, message, problem := o.store.SettledFailure(id)
	fatal(t, problem)
	if row.State != "failed" || len(attempts) != 1 || purchases.Load() != 0 ||
		errorType != "rental.lost" || !strings.Contains(message, "the rented machine gone was lost") {
		t.Fatalf("a lost started attempt was not failed with its loss: %s, %d attempt(s), %d purchase(s), %s: %s",
			row.State, len(attempts), purchases.Load(), errorType, message)
	}
}
