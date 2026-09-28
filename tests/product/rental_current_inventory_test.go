package producttest

import (
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRentalInventoryOmitsEndedMachinesButPreservesRunHistory(t *testing.T) {
	root, hubURL, hub := rentalEndRoot(t, "rental-current-inventory")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const rentalID, requestID = "rental-celty", "req-celty-history"
	hub.add(rentalID, "celty")
	fatal(t, store.RecordRental(records.Rental{ID: rentalID, MachineName: "celty",
		State: "ready", Hub: hubURL, AcceleratorCount: 1, HourlyRateUSDMicros: 100_000}))
	_, _, problem = store.Submit(records.Request{ID: requestID, IdemKey: requestID,
		BodyDigest: "sha256:" + strings.Repeat("ab", 32), Package: "fake/video", Entrypoint: "generate",
		Payload: []byte("{}"), Rental: true, Worker: rentalID})
	fatal(t, problem)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-celty",
		Package: "fake/video", WorkerID: "remote", Devices: []string{"cpu"}}))
	digest := "sha256:" + strings.Repeat("cd", 32)
	attempt, problem := store.Dispatch(records.Attempt{RequestID: requestID, SessionID: "celty-boot",
		InstanceID: "ins-celty", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(requestID, attempt, "celty-boot"))
	fatal(t, store.Accepted(requestID, attempt, "celty-boot"))
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: requestID, Attempt: attempt,
		SessionID: "celty-boot", InvocationDigest: digest, TerminalID: "celty-outcome",
		TerminalDigest: "sha256:" + strings.Repeat("ef", 32), Body: []byte("{}"),
		Status: "SUCCEEDED", RequestState: "succeeded"})
	fatal(t, problem)
	fatal(t, store.Closed(requestID, attempt))
	// This is retained history from before the rental ended, not fresh work
	// admitted to an already failed machine.
	hub.setState(rentalID, "failed", "authorization_exposure_exhausted")
	ended, problem := store.RentalRow(rentalID)
	fatal(t, problem)
	ended.State = "failed"
	fatal(t, store.RecordRental(*ended))
	before, problem := store.RequestRow(requestID)
	fatal(t, problem)
	if before.Machine != "celty" {
		t.Fatalf("fixture did not record rental history: %+v", before)
	}
	live := startDaemonProcess(t, root)
	response := live.call(t, "GET", "/v1/local/rentals", nil)
	if response.Status != http.StatusOK {
		t.Fatalf("inventory API failed: %s", response.brief())
	}
	var inventory api.RentalInventory
	must(t, json.Unmarshal(response.Body, &inventory))
	if len(inventory.Rentals)+len(inventory.Unrecorded)+len(inventory.Pending) != 0 || inventory.MachinesRunning != 0 || inventory.HourlySpendUSDMicros != 0 {
		t.Fatalf("API retained the absent rental in current fleet: %s", response.Body)
	}
	if code, out := runCozy(t, root, "rental", "list", "--json", "--full"); code != 0 || strings.Contains(out, "celty") {
		t.Fatalf("rental list retained an ended machine: exit=%d %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "list", "--json", "--full"); code != 0 || !strings.Contains(out, `"machine":"celty"`) {
		t.Fatalf("run list lost historical machine: exit=%d %s", code, out)
	}
	row, problem := store.RentalRow(rentalID)
	fatal(t, problem)
	after, problem := store.RequestRow(requestID)
	fatal(t, problem)
	if row == nil || row.State != "failed" || after == nil || after.Machine != before.Machine || after.State != before.State {
		t.Fatal("listing removed rental or request history")
	}
	if hub.releases(rentalID) != 0 {
		t.Fatal("listing issued a provider release")
	}
}
