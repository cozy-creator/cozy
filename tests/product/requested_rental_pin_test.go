package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func requestedRentalRequest(id, rentalID string) records.Request {
	return records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package: "proof/requested", Entrypoint: "generate", Payload: []byte("{}"),
		Rental: true, RequestedRental: rentalID, MachineExecutionObserver: true}
}

// `cozy run --rental <name>` records requested_rental and leaves worker empty until the
// scheduler assigns the run. Those runs are the rental's queue in the inventory. They are
// recorded once the daemon runs: at its start it places every recorded run, and ends work
// no machine execution holds.
func TestRentalInventoryCountsRequestedRentalQueue(t *testing.T) {
	root, hubURL, hub := rentalEndRoot(t, "rental-requested-queue")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const rentalID = "pr-loran"
	hub.add(rentalID, "loran")
	fatal(t, store.RecordRental(records.Rental{ID: rentalID, MachineName: "loran", State: "ready",
		Hub: hubURL, AcceleratorCount: 1, HourlyRateUSDMicros: 100_000,
		ReadyAt: time.Now().UTC().Format(time.RFC3339Nano)}))
	live := startDaemonProcess(t, root)
	var ids []string
	for i := 0; i < 4; i++ {
		request := requestedRentalRequest(fmt.Sprintf("req-loran-%d", i), rentalID)
		_, _, problem := store.Submit(request)
		fatal(t, problem)
		ids = append(ids, request.ID)
	}
	response := live.call(t, "GET", "/v1/local/rentals", nil)
	if response.Status != http.StatusOK {
		t.Fatalf("inventory API failed: %s", response.brief())
	}
	for _, id := range ids {
		row, problem := store.RequestRow(id)
		fatal(t, problem)
		if row.Worker != "" || row.RequestedRental != rentalID || row.State != "submitted" {
			t.Fatalf("fixture is no longer an unassigned requested-rental run: %+v", row)
		}
	}
	var inventory api.RentalInventory
	must(t, json.Unmarshal(response.Body, &inventory))
	if len(inventory.Rentals) != 1 || inventory.Rentals[0].Activity == nil || inventory.Rentals[0].Activity.Queued != len(ids) {
		t.Fatalf("inventory does not count requested-rental runs as queued: %s", response.Body)
	}
	code, out := runCozy(t, root, "rental", "list", "--json", "--full")
	if code != 0 || !strings.Contains(out, fmt.Sprintf(`"queued":%d`, len(ids))) {
		t.Fatalf("rental list does not count requested-rental runs as queued: exit=%d %s", code, out)
	}
}

// A rental whose only work is a queued, still-unassigned `--rental` run is not idle.
func TestRequestedRentalQueueKeepsRentalAlive(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	now := time.Now().UTC()
	row := idleRecord(t, store, "requested-only", now.Add(-time.Hour))
	request := requestedRentalRequest("requested-queued", row.ID)
	_, _, problem = store.Submit(request)
	fatal(t, problem)
	idle, problem := rental.ObserveIdle(store, row)
	fatal(t, problem)
	if idle.Queued != 1 || idle.Running != 0 || idle.Due(now) {
		t.Fatalf("requested-rental work is invisible to idle: %+v", idle)
	}
	queued, running, problem := store.RentalRunCounts(row.ID)
	fatal(t, problem)
	if queued != 1 || running != 0 {
		t.Fatalf("rental run counts miss requested-rental work: queued=%d running=%d", queued, running)
	}
	pinned, problem := store.PinnedRentalWork(row.ID)
	fatal(t, problem)
	if len(pinned) != 1 || pinned[0].ID != request.ID {
		t.Fatalf("pinned work misses requested-rental run: %+v", pinned)
	}
}
