package cli

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestRentNewExcludesOtherRequestsRentals(t *testing.T) {
	st, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer st.Close()
	for _, row := range []records.Rental{
		{ID: "pr-other", MachineName: "other", State: "ready", ManagedRequestID: "other", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1},
		{ID: "pr-own", MachineName: "own", State: "pending_acquisition", ManagedRequestID: "fresh", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1},
	} {
		if problem := st.RecordRental(row); problem != nil {
			t.Fatal(problem)
		}
	}
	m := managedRentals{store: st}
	candidates, problem := m.attachedLocked(records.Request{ID: "fresh", RentNew: true}, map[machineKey]hub.RentalSKU{}, false, rental.Constraints{})
	if problem != nil {
		t.Fatal(problem)
	}
	if len(candidates) != 2 {
		t.Fatalf("lost candidate diagnostics: %+v", candidates)
	}
	for _, candidate := range candidates {
		excluded := candidate.Verdict == orchestrator.VerdictExcluded+"fresh_rental_requested"
		if excluded != (candidate.Rental == "pr-other") {
			t.Fatalf("wrong fresh ownership decision: %+v", candidate)
		}
	}
}
