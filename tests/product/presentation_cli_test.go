package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestCancellationAttributionSurvivesHumanCLIProjection(t *testing.T) {
	for _, arm := range []struct{ actor, display string }{
		{"cozy run cancel", "cancelled by user"},
		{"deadline", "cancelled by system"},
	} {
		t.Run(arm.actor, func(t *testing.T) {
			root := t.TempDir()
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
			req, _, problem := store.Submit(records.Request{ID: "request-projection", IdemKey: "projection", BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/projection", Entrypoint: "run", Payload: []byte(`{}`)})
			fatal(t, problem)
			settled, problem := store.CancelQueuedRequest(req.ID, map[string]any{"actor": arm.actor})
			fatal(t, problem)
			if !settled {
				t.Fatal("queued cancellation did not settle")
			}
			code, output := runCozy(t, root, "run", "watch", req.ID)
			if code != 1 || !strings.Contains(output, "was "+arm.display) {
				t.Fatalf("human watch lost cancellation attribution: exit=%d %s", code, output)
			}
			row, problem := store.RequestRow(req.ID)
			fatal(t, problem)
			actor, _, _, problem := store.CancelAttribution(req.ID)
			fatal(t, problem)
			if row.State != "canceled" || actor != arm.actor {
				t.Fatalf("human projection rewrote lifecycle facts: state=%s actor=%q", row.State, actor)
			}
		})
	}
}

func TestRentalHumanDrainingPreservesWireState(t *testing.T) {
	root, origin, peer := rentalEndRoot(t, "draining-projection")
	peer.add("pr-draining", "draining-machine")
	peer.setState("pr-draining", "release_requested", "")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{ID: "pr-draining", MachineName: "draining-machine", SKU: "cpu",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "release_requested", Hub: origin}))
	code, output := runCozy(t, root, "rental", "list", "--no-watch")
	if code != 0 || !strings.Contains(output, "draining") || strings.Contains(output, "release_requested") {
		t.Fatalf("human rental list lost draining state: exit=%d %s", code, output)
	}
	code, output = runCozy(t, root, "rental", "list", "--json")
	if code != 0 || !strings.Contains(output, `"state":"release_requested"`) {
		t.Fatalf("JSON rental list changed wire state: exit=%d %s", code, output)
	}
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if peer.released["pr-draining"] != 0 {
		t.Fatal("listing a draining rental issued a release")
	}
}

func TestRentalConflictNamesDrainingOperation(t *testing.T) {
	root, origin, stand := rentalEndRoot(t, "draining-conflict")
	stand.publishListing()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	_, _, problem = store.BeginRentalOperation(records.RentalOperation{Key: "draining-op", Hub: origin, Reason: "manual", HourlyRateUSDMicros: 100_000},
		5_000_000, 0, func(string) ([]byte, string, *exit.Error) { return []byte(`{}`), "draining", nil }, nil)
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation("draining-op", "pr-draining-op", "release_requested"))
	code, output := runCozy(t, root, "rental", "new", "cpu", "--idempotency-key", "draining-op")
	if code == 0 || !strings.Contains(output, "is draining and still names rental pr-draining-op") || strings.Contains(output, "release_requested") {
		t.Fatalf("draining operation conflict was not stated in lifecycle terms: exit=%d %s", code, output)
	}
}
