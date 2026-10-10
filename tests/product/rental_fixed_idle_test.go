package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func idleRecord(t *testing.T, store *records.Store, id string, at time.Time) records.Rental {
	t.Helper()
	row := records.Rental{ID: id, MachineName: id, SKU: "cpu", State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: "https://hub.invalid", Address: "127.0.0.1:1", CertPath: "absent.pem", ReadyAt: at.UTC().Format(time.RFC3339Nano), ExpectedWorkerID: "worker", ExpectedWorkerBootID: "boot"}
	fatal(t, store.RecordRental(row))
	return row
}

// The IDLE column is this host's observation of each machine's own work; when the machine
// ends itself is its own to say.
func TestRentalIdleObservationScopesWorkAndRetainedState(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	at := time.Date(2026, 9, 24, 10, 0, 0, 0, time.UTC)
	row := idleRecord(t, store, "idle", at)
	observe := func() rental.Idleness { t.Helper(); i, p := rental.ObserveIdle(store, row); fatal(t, p); return i }
	idleSince := func(i rental.Idleness, baseline time.Time) {
		t.Helper()
		if i.Queued+i.Running != 0 || !i.Since.Equal(baseline) {
			t.Fatalf("not idle since %s: %+v", baseline, i)
		}
	}
	busy := func() bool { i := observe(); return i.Queued+i.Running > 0 }
	idleSince(observe(), at)
	// Unpinned fleet work is not this rental's activity.
	unpinned := recordPrivateTransaction(t, store, "unpinned", "")
	idleSince(observe(), at)
	// Its own explicitly purchased queued request is this one machine's work.
	row.ManagedRequestID = unpinned.ID
	if !busy() {
		t.Fatal("acquisition buyer's request is not its machine's work")
	}
	row.ManagedRequestID = ""
	request := recordPrivateTransaction(t, store, "pinned", row.ID)
	if !busy() {
		t.Fatal("pinned work was treated as idle")
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
	paused := observe()
	idleSince(paused, paused.Since)
	failed := recordPrivateTransaction(t, store, "failed", row.ID)
	changed, p := store.FailQueuedRequest(failed.ID, retainedFailure("fixture", "failed before it started"))
	fatal(t, p)
	if !changed {
		t.Fatal("failed fixture did not settle")
	}
	idle := observe()
	idleSince(idle, idle.Since)
	if retained, p := store.RentalRetainsWork(row.ID); p != nil || !retained {
		t.Fatal("idle policy destroyed or ignored retained custody", p)
	}
}

func TestRentalKeepaliveReceiptSurvivesReconnectAndRejectsInvalidAcknowledgment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	at := time.Now().UTC().Truncate(time.Millisecond)
	row := idleRecord(t, store, "manual", at.Add(-time.Hour))
	receipt := records.RentalKeepalive{WorkerID: row.ExpectedWorkerID, WorkerBootID: row.ExpectedWorkerBootID, AcknowledgedAtMS: at.UnixMilli(), IdleDeadlineMS: at.Add(900 * time.Second).UnixMilli()}
	fatal(t, store.RecordRentalKeepalive(row.ID, receipt, at))
	// Status/reconnect writes preserve both readiness and the acknowledged clock.
	row.ReadyAt = at.Add(time.Hour).Format(time.RFC3339Nano)
	fatal(t, store.RecordRental(row))
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	rowp, problem := store.RentalRow(row.ID)
	fatal(t, problem)
	row = *rowp
	check := func(want time.Time) {
		t.Helper()
		i, p := rental.ObserveIdle(store, row)
		fatal(t, p)
		if !i.Since.Equal(want) {
			t.Fatalf("idle since=%s want=%s", i.Since, want)
		}
	}
	check(at)
	fatal(t, store.RecordRentalKeepalive(row.ID, receipt, at))
	check(at)
	for _, mutate := range []func(*records.RentalKeepalive){func(r *records.RentalKeepalive) { r.WorkerID = "other" }, func(r *records.RentalKeepalive) { r.WorkerBootID = "other" }, func(r *records.RentalKeepalive) { r.AcknowledgedAtMS = 0 }, func(r *records.RentalKeepalive) { r.IdleDeadlineMS = r.AcknowledgedAtMS }} {
		invalid := receipt
		mutate(&invalid)
		if store.RecordRentalKeepalive(row.ID, invalid, at.Add(time.Minute)) == nil {
			t.Fatal("invalid worker acknowledgment renewed rental")
		}
		check(at)
	}
	later := receipt
	later.AcknowledgedAtMS += 120000
	later.IdleDeadlineMS += 120000
	fatal(t, store.RecordRentalKeepalive(row.ID, later, at.Add(120*time.Second)))
	check(at.Add(120 * time.Second))
	// A machine that names no deadline (an older TensorD) still restarted its clock.
	none := later
	none.AcknowledgedAtMS += 60000
	none.IdleDeadlineMS = 0
	fatal(t, store.RecordRentalKeepalive(row.ID, none, at.Add(180*time.Second)))
	check(at.Add(180 * time.Second))
	row.State = "release_requested"
	fatal(t, store.RecordRental(row))
	if store.RecordRentalKeepalive(row.ID, later, at.Add(120*time.Second)) == nil {
		t.Fatal("ending rental renewed")
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
