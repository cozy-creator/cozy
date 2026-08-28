package rental

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/records"
)

func TestKnownRefusesConvergingRentalBeforeControlDecode(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	if problem := store.RecordRental(records.Rental{
		ID: "rental-1", EndpointRef: "cozy/a/v1/generate", AcceleratorModel: "H100",
		State: "converging", Hub: "https://hub.invalid", Address: "worker:443",
	}); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := Known(store)("rental-1"); problem == nil ||
		problem.ErrName() != "rental.convergence_pending" {
		t.Fatalf("converging rental was admitted for request creation: %v", problem)
	}
}
