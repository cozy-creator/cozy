package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// An edited program is a new immutable request. Its explicit predecessor conveys
// retained custody without replacing history or allowing a different machine to
// claim bytes that only exist on the original rental.
func TestUnpublishedTransactionEditedRetryPreservesHistoryAndCustody(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const rental = "pr-edited-retry"
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: rental, MachineName: "otter", State: "ready", HourlyRateUSDMicros: 100_000}))
	prior := recordPrivateTransaction(t, store, "broken-program", rental)
	changed, problem := store.BlockRetainedWork(prior.ID, "author_exception", "step B failed")
	fatal(t, problem)
	if !changed {
		t.Fatal("failed request did not retain its work")
	}
	before, problem := store.RequestByReference(prior.ID)
	fatal(t, problem)
	retry := *before
	retry.ID, retry.IdemKey = "req-private-fixed-program", "idem-private-fixed-program"
	retry.RetryOf = prior.ID
	retry.BodyDigest = "sha256:" + strings.Repeat("d", 64)
	retry.LocalInstallationID = "sha256:" + strings.Repeat("e", 64)
	retry.PlanID = "sha256:" + strings.Repeat("f", 64)
	retry.Payload = []byte(`{"source_revision":"fixed-B","seed":17,"quality_bar":0.95}`)
	retry.Worker, retry.Rental, retry.RentalRequired = "", false, false
	admitted, fresh, problem := store.Submit(retry)
	fatal(t, problem)
	if !fresh || admitted.ID == prior.ID || admitted.Number == prior.Number || admitted.RetryOf != prior.ID ||
		admitted.ReuseScope != prior.ID || admitted.Worker != rental || !admitted.Rental || !admitted.RentalRequired ||
		admitted.LocalInstallationID != retry.LocalInstallationID || admitted.PlanID != retry.PlanID ||
		admitted.BodyDigest != retry.BodyDigest || !bytes.Equal(admitted.Payload, retry.Payload) || admitted.Ordinal != 0 {
		t.Fatalf("edited retry lost immutable program or retained custody: %+v", admitted)
	}
	assertPrivateTransactionIdentity(t, store, *before, "blocked")
	after, problem := store.RequestByReference(prior.ID)
	fatal(t, problem)
	if after.RetryOf != before.RetryOf || after.ReuseScope != before.ReuseScope {
		t.Fatal("admitting descendant rewrote predecessor lineage")
	}
	replayed, fresh, problem := store.Submit(retry)
	fatal(t, problem)
	if fresh || replayed.ID != admitted.ID || replayed.ReuseScope != admitted.ReuseScope {
		t.Fatal("retry admission replay created another request or changed custody")
	}
	// Removing the predecessor is a changed command even when its new program
	// body happens to be identical. The original idempotency key must refuse it.
	retry.RetryOf = ""
	if _, _, problem := store.Submit(retry); problem == nil {
		t.Fatal("idempotency key accepted different retry lineage")
	}
	fatal(t, store.RequestRetainedCancellation(prior.ID, "finished fixing B"))
	if _, problem := store.ReleaseRetainedWork(prior.ID); problem != nil {
		fatal(t, problem)
	}
	if _, problem := store.CompleteRetainedCancellation(prior.ID); problem != nil {
		fatal(t, problem)
	}
	held, problem := store.RentalRetainsWork(rental)
	fatal(t, problem)
	if !held {
		t.Fatal("abandoning predecessor released the descendant's rental claim")
	}
}

func TestUnpublishedTransactionRetryRefusesUnavailableCustody(t *testing.T) {
	for _, arm := range []string{"missing", "active", "canceled", "different-machine", "lost-machine", "not-retained"} {
		t.Run(arm, func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			const rental = "pr-retry-refusal"
			if arm != "lost-machine" {
				fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: rental, MachineName: "otter", State: "ready", HourlyRateUSDMicros: 100_000}))
			}
			prior := recordPrivateTransaction(t, store, arm, rental)
			if arm != "active" {
				_, problem := store.BlockRetainedWork(prior.ID, "author_exception", "step B failed")
				fatal(t, problem)
			}
			if arm == "canceled" {
				fatal(t, store.RequestRetainedCancellation(prior.ID, "abandon"))
			}
			retry := prior
			retry.ID, retry.IdemKey = prior.ID+"-retry", prior.IdemKey+"-retry"
			retry.RetryOf = prior.ID
			switch arm {
			case "missing":
				retry.RetryOf = "req-absent"
			case "different-machine":
				retry.Worker = "pr-another-rental"
			case "not-retained":
				retry.RetainWork = false
			}
			if _, _, problem := store.Submit(retry); problem == nil {
				t.Fatal("retry admitted without stopped, retained predecessor custody")
			}
			row, problem := store.RequestByReference(retry.ID)
			fatal(t, problem)
			if row != nil {
				t.Fatal("refused retry nevertheless persisted a schedulable request")
			}
		})
	}
}
