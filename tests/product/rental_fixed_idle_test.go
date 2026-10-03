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

func TestFixedRentalIdleClockScopesWorkAndRetainedState(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
	fatal(t, problem)
	defer store.Close()
	at := time.Date(2026, 9, 24, 10, 0, 0, 0, time.UTC)
	row := idleRecord(t, store, "idle", at)
	observe := func() rental.Idleness { t.Helper(); i, p := rental.ObserveIdle(store, row); fatal(t, p); return i }
	checkBoundary := func(i rental.Idleness, baseline time.Time) {
		t.Helper()
		if i.Due(baseline.Add(900*time.Second-time.Nanosecond)) || !i.Due(baseline.Add(900*time.Second)) {
			t.Fatalf("not an exact fifteen-minute deadline: %+v", i)
		}
	}
	checkBoundary(observe(), at)
	// Unpinned fleet work is not this rental's activity.
	unpinned := recordPrivateTransaction(t, store, "unpinned", "")
	checkBoundary(observe(), at)
	// Its own explicitly purchased queued request does protect this one machine.
	row.ManagedRequestID = unpinned.ID
	if observe().Due(at.Add(time.Hour)) {
		t.Fatal("acquisition buyer did not protect its machine")
	}
	row.ManagedRequestID = ""
	request := recordPrivateTransaction(t, store, "pinned", row.ID)
	if observe().Due(at.Add(time.Hour)) {
		t.Fatal("pinned preparation was treated as idle")
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
	if paused.Running != 0 || paused.Queued != 0 {
		t.Fatalf("retained paused work is activity: %+v", paused)
	}
	checkBoundary(paused, paused.Since)
	blocked := recordPrivateTransaction(t, store, "blocked", row.ID)
	changed, p := store.BlockRetainedWork(blocked.ID, "fixture", "failed work remains retained")
	fatal(t, p)
	if !changed {
		t.Fatal("blocked fixture did not settle")
	}
	idle := observe()
	checkBoundary(idle, idle.Since)
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
		due, ok := i.ReleaseAt()
		if !ok || !due.Equal(want) {
			t.Fatalf("deadline=%s eligible=%v want=%s", due, ok, want)
		}
	}
	check(at.Add(900 * time.Second))
	fatal(t, store.RecordRentalKeepalive(row.ID, receipt, at))
	check(at.Add(900 * time.Second))
	for _, mutate := range []func(*records.RentalKeepalive){func(r *records.RentalKeepalive) { r.WorkerID = "other" }, func(r *records.RentalKeepalive) { r.WorkerBootID = "other" }, func(r *records.RentalKeepalive) { r.AcknowledgedAtMS = 0 }, func(r *records.RentalKeepalive) { r.IdleDeadlineMS = r.AcknowledgedAtMS }} {
		invalid := receipt
		mutate(&invalid)
		if store.RecordRentalKeepalive(row.ID, invalid, at.Add(time.Minute)) == nil {
			t.Fatal("invalid worker acknowledgment renewed rental")
		}
		check(at.Add(900 * time.Second))
	}
	later := receipt
	later.AcknowledgedAtMS += 120000
	later.IdleDeadlineMS += 120000
	fatal(t, store.RecordRentalKeepalive(row.ID, later, at.Add(120*time.Second)))
	check(at.Add(1020 * time.Second))
	// A Host whose own idle window differs is still an acknowledgment: Creator's
	// schedule comes from its own observation, not the Host's window.
	longer := later
	longer.AcknowledgedAtMS += 60000
	longer.IdleDeadlineMS = longer.AcknowledgedAtMS + 1800000
	fatal(t, store.RecordRentalKeepalive(row.ID, longer, at.Add(180*time.Second)))
	check(at.Add(1080 * time.Second))
	row.State = "release_requested"
	fatal(t, store.RecordRental(row))
	if store.RecordRentalKeepalive(row.ID, later, at.Add(120*time.Second)) == nil {
		t.Fatal("ending rental renewed")
	}
}

func TestInterruptedExplicitPreparationDefersOnlyCreatorExpiryWithoutRenewal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "records.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	at := time.Now().Add(-time.Hour).UTC()
	row := idleRecord(t, store, "preparing", at)
	fatal(t, store.RecordRentalPreparationStarted(row.ID))
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	idle, problem := rental.ObserveIdle(store, row)
	fatal(t, problem)
	if idle.PendingPreparation != 1 || idle.Running != 0 || !idle.Since.Equal(at) || idle.Due(time.Now()) {
		t.Fatalf("unknown preparation was renewed or falsely reported active: %+v", idle)
	}
	// Only an actual finished preparation advances the local baseline. The stale
	// intent issues no worker RPC and cannot renew the independent pod deadline.
	done := time.Now().UTC()
	fatal(t, store.RecordRentalWorkFinished(row.ID, done))
	idle, problem = rental.ObserveIdle(store, row)
	fatal(t, problem)
	if idle.PendingPreparation != 0 || !idle.Due(done.Add(900*time.Second)) || idle.Due(done.Add(899*time.Second)) {
		t.Fatalf("finished preparation did not start fixed grace: %+v", idle)
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
	if rental.IdleTimeout != 900*time.Second {
		t.Fatal("environment changed fixed deadline")
	}
}

func TestRentalKeepaliveLocalDeadlineIgnoresHostClockSkew(t *testing.T) {
	for _, skew := range []time.Duration{-8 * time.Hour, 8 * time.Hour} {
		t.Run(skew.String(), func(t *testing.T) {
			store, problem := records.Open(filepath.Join(t.TempDir(), "records.sqlite"))
			fatal(t, problem)
			defer store.Close()
			received := time.Date(2026, 9, 24, 12, 0, 0, 123456789, time.UTC)
			row := idleRecord(t, store, "skew", received.Add(-time.Hour))
			ack := received.Add(skew).UnixMilli()
			receipt := records.RentalKeepalive{WorkerID: row.ExpectedWorkerID, WorkerBootID: row.ExpectedWorkerBootID, AcknowledgedAtMS: ack, IdleDeadlineMS: ack + 900000}
			fatal(t, store.RecordRentalKeepalive(row.ID, receipt, received))
			check := func(want time.Time) {
				t.Helper()
				idle, p := rental.ObserveIdle(store, row)
				fatal(t, p)
				due, ok := idle.ReleaseAt()
				if !ok || !due.Equal(want) {
					t.Fatalf("local due=%s want=%s Host skew=%s", due, want, skew)
				}
			}
			check(received.Add(900 * time.Second))
			fatal(t, store.RecordRentalKeepalive(row.ID, receipt, received.Add(10*time.Minute)))
			check(received.Add(900 * time.Second))
			newer := receipt
			newer.AcknowledgedAtMS += 1000
			newer.IdleDeadlineMS += 1000
			fatal(t, store.RecordRentalKeepalive(row.ID, newer, received.Add(time.Second)))
			fatal(t, store.RecordRentalKeepalive(row.ID, receipt, received.Add(20*time.Minute)))
			check(received.Add(901 * time.Second))
		})
	}
}
