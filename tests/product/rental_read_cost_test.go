package producttest

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A fleet read is one listing per Hub. Every live rental is in it and an ended one cannot
// change, so `rental list` and `rental show` read neither again by id: the owner's board
// read each of twenty-one rows in turn and took seconds to draw.
func TestRentalReadsAskEachHubForOneListing(t *testing.T) {
	root, hubURL, hub := rentalEndRoot(t, "rental-read-cost")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	var ids []string
	for i := range 6 {
		id, state := fmt.Sprintf("pr-c0ffee%015d", i), "ready"
		hub.add(id, fmt.Sprintf("kotone-%d", i))
		if i >= 3 {
			state = "failed"
			hub.setState(id, state, "container_exited_before_readiness")
		}
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: fmt.Sprintf("kotone-%d", i), AcceleratorModel: "CPU",
			AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: state, Hub: hubURL}))
		ids = append(ids, id)
	}
	store.Close()

	code, out := runCozy(t, root, "rental", "list", "--json")
	if code != 0 || !strings.Contains(out, ids[0]) || strings.Contains(out, ids[3]) {
		t.Fatalf("rental list did not show the live fleet [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "rental", "show", "kotone-1", "--json")
	if code != 0 || !strings.Contains(out, ids[1]) {
		t.Fatalf("rental show did not resolve a live rental by name [exit %d]\n%s", code, out)
	}
	for _, id := range ids {
		if n := hub.idReads(id); n != 0 {
			t.Fatalf("rental %s was read by id %d time(s); the listing answers for it", id, n)
		}
	}
}
