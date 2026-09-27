package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentNewCLIExcludesExistingFleetAndReplaysAcquisition(t *testing.T) {
	h := fleetHub(t)
	root := fleetRoot(t, h, 20)
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	plantH100(t, root, h, st, "pr-guchuko", "guchuko", true)
	startDaemonProcess(t, root)
	args := []string{"run", "proof/h3/generate", "steps=1", explicitFP8, "--rent-new", "--json", "--idempotency-key", "fresh-one"}
	// The stand-in hub fails every pod it sells, so the run may settle failed inside the
	// CLI's observation window. The fresh acquisition and its replay are the proof.
	runCozy(t, root, args...)
	var row *records.Request
	waitFor(t, root, "new acquisition instead of existing fleet reuse", func() bool {
		row, problem = st.RequestByIdempotencyKey("fresh-one")
		return problem == nil && row != nil && len(h.postedSKUs()) > 0
	})
	if !row.RentNew || row.Worker == "pr-guchuko" {
		t.Fatalf("fresh intent not applied: %+v", row)
	}
	operations, problem := st.RentalOperations()
	fatal(t, problem)
	count := len(operations)
	if _, out := runCozy(t, root, args...); !strings.Contains(out, row.ID) {
		t.Fatalf("idempotent replay did not answer run %s: %s", row.ID, out)
	}
	operations, problem = st.RentalOperations()
	fatal(t, problem)
	if len(operations) != count {
		t.Fatalf("replay opened another paid acquisition: %d -> %d", count, len(operations))
	}
	existing, problem := st.RentalRow("pr-guchuko")
	if problem != nil || existing == nil {
		t.Fatalf("existing rental disappeared: %+v %v", existing, problem)
	}
}
