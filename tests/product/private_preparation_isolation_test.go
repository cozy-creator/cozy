package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// Independent wire peer: the queued second root must not even prepare a new
// package while the first offer still owns this worker's ordinary executor.
func TestActiveRentalJobBlocksAnotherRootBeforePreparation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	owner := hostOwner(t, "private-preparation-busy", rentalWiring(connection, private))
	first, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: "first-root", Package: "cozy/h3-package", Release: "1.0.7",
		Entrypoint: "first", PlanID: "sha256:" + strings.Repeat("35", 32), Kind: "job", Org: "paul", Payload: []byte("{}"),
		Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "first root offer", func() bool { pod.mu.Lock(); defer pod.mu.Unlock(); return len(pod.offers) == 1 })
	pod.mu.Lock()
	before := len(pod.prepares)
	pod.mu.Unlock()
	second, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: "second-root", Package: "cozy/h3-package", Release: "1.0.7",
		Entrypoint: "second", PlanID: "sha256:" + strings.Repeat("36", 32), Kind: "job", Org: "paul", Payload: []byte("{}"),
		Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "second root waits before mutating preparation", func() bool {
		pod.mu.Lock()
		prepared, offers := len(pod.prepares), len(pod.offers)
		pod.mu.Unlock()
		if prepared != before || offers != 1 {
			t.Fatalf("second root crossed active preparation boundary: prepares%d→%d offers%d", before, prepared, offers)
		}
		events, problem := owner.store.EventsAfter(second, 0, 100)
		fatal(t, problem)
		for _, event := range events {
			if strings.Contains(fmt.Sprint(event.Payload), "active request "+first) {
				return true
			}
		}
		return false
	})
	active, problem := owner.store.Attempts(first)
	fatal(t, problem)
	queued, problem := owner.store.Attempts(second)
	fatal(t, problem)
	if len(active) != 1 || active[0].State == "closed" || len(queued) != 0 {
		t.Fatal("preparation wait changed either request's execution")
	}
}
