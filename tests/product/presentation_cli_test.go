package producttest

import (
	"encoding/base64"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

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
	_, _, problem = store.BeginRentalOperation(records.RentalOperation{Key: "draining-op", Hub: origin, Reason: "manual", HourlyRateUSDMicros: 100_000}, func(name string) ([]byte, string, *exit.Error) {
		body, problem := hub.RentalRequestBytes(name, "cpu", 1, strings.Repeat("ab", 32),
			base64.RawURLEncoding.EncodeToString(make([]byte, 32)), hub.DeclaredWorkload{}, nil, "", "")
		return body, "draining", problem
	})
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation("draining-op", "pr-draining-op", "release_requested"))
	code, output := runCozy(t, root, "rental", "new", "cpu", "--idempotency-key", "draining-op")
	if code == 0 || !strings.Contains(output, "is draining and still names rental pr-draining-op") || strings.Contains(output, "release_requested") {
		t.Fatalf("draining operation conflict was not stated in lifecycle terms: exit=%d %s", code, output)
	}
}
