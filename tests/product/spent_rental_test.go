package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalPurposeKeepsManualAndOwedCustody(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row := records.Rental{AcceleratorCount: 1, ID: "rental-purpose"}
	spent, problem := orchestrator.RentalSpent(store, row)
	fatal(t, problem)
	if spent {
		t.Fatal("a manual rental inherited managed-job retirement")
	}
	replacementRequest(t, store, "job-purpose", row.ID)
	row.ManagedRequestID = "job-purpose"
	for _, state := range []string{"submitted", "finalizing"} {
		fatal(t, store.SettleRequest("job-purpose", state))
		spent, problem = orchestrator.RentalSpent(store, row)
		fatal(t, problem)
		if spent {
			t.Fatalf("%s source/publication custody was called spent", state)
		}
	}
}
