package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

func idleRecord(t *testing.T, store *records.Store, id string, at time.Time) records.Rental {
	t.Helper()
	row := records.Rental{ID: id, MachineName: id, SKU: "cpu", State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: "https://hub.invalid", Address: "127.0.0.1:1", CertPath: "absent.pem", ReadyAt: at.UTC().Format(time.RFC3339Nano), ExpectedWorkerID: "worker", ExpectedWorkerBootID: "boot"}
	fatal(t, store.RecordRental(row))
	return row
}

// RUNNING and QUEUED count each machine's own work, never fleet work elsewhere or work
// retained after it settled; when the machine ends itself is its own to say.
func TestRentalWorkCountsScopeWorkAndRetainedState(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row := idleRecord(t, store, "idle", time.Date(2026, 9, 24, 10, 0, 0, 0, time.UTC))
	work := func() int {
		t.Helper()
		queued, running, p := store.RentalWorkCounts(row.ID, row.ManagedRequestID)
		fatal(t, p)
		return queued + running
	}
	// Unpinned fleet work is not this rental's work.
	unpinned := recordPrivateTransaction(t, store, "unpinned", "")
	if work() != 0 {
		t.Fatal("unpinned fleet work counted on this rental")
	}
	// Its own explicitly purchased queued request is this one machine's work.
	row.ManagedRequestID = unpinned.ID
	if work() == 0 {
		t.Fatal("acquisition buyer's request is not its machine's work")
	}
	row.ManagedRequestID = ""
	request := recordPrivateTransaction(t, store, "pinned", row.ID)
	if work() == 0 {
		t.Fatal("pinned work was not counted")
	}
	state, p := store.RequestPause(request.ID, "clock proof")
	fatal(t, p)
	if state != "paused" {
		changed, p := store.CompleteRequestPause(request.ID)
		fatal(t, p)
		if !changed {
			t.Fatal("pause did not settle")
		}
	}
	if work() != 0 {
		t.Fatal("retained paused work counted as work")
	}
	failed := recordPrivateTransaction(t, store, "failed", row.ID)
	changed, p := store.FailQueuedRequest(failed.ID, retainedFailure("fixture", "failed before it started"))
	fatal(t, p)
	if !changed {
		t.Fatal("failed fixture did not settle")
	}
	if work() != 0 {
		t.Fatal("failed work counted as work")
	}
	if retained, p := store.RentalRetainsWork(row.ID); p != nil || !retained {
		t.Fatal("counting destroyed or ignored retained custody", p)
	}
}

func TestRentalIdleConfigurationHasNoDurationOrDisableEscape(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("{}\n"), 0600))
	if code, out := runCozy(t, root, "help", "rental", "keepalive"); code != 0 {
		t.Fatalf("ordinary fixed-policy CLI: %d %s", code, out)
	}
	// A retired override is named as unused on every command and changes nothing: the
	// fixed policy has no escape, and the stale line no longer disables the CLI.
	for _, body := range []string{"rentals:\n  idle_release_s: 0\n", "rentals:\n  idle_release_s: 3600\n", "rentals_idle_release_s: 0\n"} {
		must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(body), 0600))
		if _, stdout, stderr := runCozyStreams(t, root, "rental", "list", "--json"); !strings.Contains(stderr, "idle_release_s; ignored") || strings.Contains(stdout, "is invalid") {
			t.Fatalf("retired override %q was not named as ignored: %s %s", body, stdout, stderr)
		}
	}
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("{}\n"), 0600))
	t.Setenv("COZY_RENTALS_IDLE_RELEASE_S", "0")
	t.Setenv("RENTALS_IDLE_RELEASE_S", "1")
	if code, out := runCozy(t, root, "help", "rental", "keepalive"); code != 0 {
		t.Fatalf("fixed policy rejected ordinary environment: %d %s", code, out)
	}
}
