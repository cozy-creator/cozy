package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// Source preparation already occupies the worker's job mode, even before the
// producer has an attempt. Price ranking must only see mode-compatible rentals.
// The second identity is a new empty serving rental; this proves selection
// eligibility, not that a particular GPU has been provisioned or fits H3.
func TestServingSelectionExcludesOwedJobRentalBeforeAnyAttempt(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "h3-serving-job-exclusion", rentalWiring(connection, private))
	id := submitPublishedRentalJob(t, o, "cozy/h3-package", "1.0.7",
		"sha256:"+strings.Repeat("35", 32), "h3-waiting-producer")
	waitUntil(t, "job directive before the producer attempt", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.jobDirectives) > 0
	})
	attempts, problem := o.store.Attempts(id)
	fatal(t, problem)
	if len(attempts) != 0 {
		t.Fatal("fixture dispatched a producer instead of holding source preparation")
	}
	row := records.Rental{ID: podRental, ManagedRequestID: id}
	owed, problem := orchestrator.RentalOwedBy(o.store, row)
	fatal(t, problem)
	spent, problem := orchestrator.RentalSpent(o.store, row)
	fatal(t, problem)
	if !owed || spent {
		t.Fatal("the live source job lost its rental hold")
	}
	const servingRental = "pr-h200-reference"
	kept, excluded := o.c.ModeCompatibleRentalsWithExclusions([]string{podRental, servingRental}, false)
	if len(kept) != 1 || kept[0] != servingRental || len(excluded) != 1 ||
		excluded[0].RentalID != podRental || excluded[0].Reason != orchestrator.ExcludedModeConflict {
		t.Fatalf("serving can reuse the producer's rental: kept=%v excluded=%v", kept, excluded)
	}
	retained, problem := o.store.RequestRow(id)
	fatal(t, problem)
	if retained.State != "submitted" || retained.Worker != podRental {
		t.Fatal("serving selection changed the existing job's request or rental")
	}
}
