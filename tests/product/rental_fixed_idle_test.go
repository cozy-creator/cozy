package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
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
	receipt := &pb.KeepRentalAliveResult{RequestId: "manual-1", WorkerId: row.ExpectedWorkerID, WorkerBootId: row.ExpectedWorkerBootID, AcknowledgedAtUnixMs: at.UnixMilli(), IdleDeadlineUnixMs: at.Add(900 * time.Second).UnixMilli()}
	fatal(t, store.RecordRentalKeepalive(row.ID, receipt))
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
	fatal(t, store.RecordRentalKeepalive(row.ID, receipt))
	check(at.Add(900 * time.Second))
	for _, mutate := range []func(*pb.KeepRentalAliveResult){func(r *pb.KeepRentalAliveResult) { r.WorkerId = "other" }, func(r *pb.KeepRentalAliveResult) { r.WorkerBootId = "other" }, func(r *pb.KeepRentalAliveResult) { r.IdleDeadlineUnixMs++ }, func(r *pb.KeepRentalAliveResult) { r.AcknowledgedAtUnixMs = 0 }, func(r *pb.KeepRentalAliveResult) { r.RequestId = "" }} {
		invalid := proto.Clone(receipt).(*pb.KeepRentalAliveResult)
		mutate(invalid)
		if store.RecordRentalKeepalive(row.ID, invalid) == nil {
			t.Fatal("invalid worker acknowledgment renewed rental")
		}
		check(at.Add(900 * time.Second))
	}
	later := proto.Clone(receipt).(*pb.KeepRentalAliveResult)
	later.RequestId = "manual-2"
	later.AcknowledgedAtUnixMs += 120000
	later.IdleDeadlineUnixMs += 120000
	fatal(t, store.RecordRentalKeepalive(row.ID, later))
	check(at.Add(1020 * time.Second))
	row.State = "release_requested"
	fatal(t, store.RecordRental(row))
	if store.RecordRentalKeepalive(row.ID, later) == nil {
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
	for _, value := range []string{"0", "1", "900", "3600"} {
		must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("rentals:\n  idle_release_s: "+value+"\n"), 0600))
		if code, out := runCozy(t, root, "rental", "list", "--json"); code == 0 || !strings.Contains(out, "idle_release_s") {
			t.Fatalf("retired override %s accepted or misdiagnosed: %d %s", value, code, out)
		}
	}
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("rentals_idle_release_s: 0\n"), 0600))
	if code, out := runCozy(t, root, "rental", "list", "--json"); code == 0 || !strings.Contains(out, "rentals_idle_release_s") {
		t.Fatalf("flat retired override accepted: %d %s", code, out)
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

func TestRentalKeepaliveUsesCurrentSignedClaimAndRefusesUnconfirmedResult(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	var mode string
	calls := 0
	pod.keepalive = func(request *pb.KeepRentalAliveRequest) (*pb.KeepRentalAliveResult, error) {
		calls++
		if mode == "failure" {
			return nil, status.Error(codes.Unavailable, "receipt lost")
		}
		result := &pb.KeepRentalAliveResult{RequestId: request.RequestId, WorkerId: podWorkerID, WorkerBootId: podBootID, AcknowledgedAtUnixMs: 1700000000000, IdleDeadlineUnixMs: 1700000900000}
		switch mode {
		case "request":
			result.RequestId = "different"
		case "worker":
			result.WorkerId = "different"
		case "boot":
			result.WorkerBootId = "different"
		}
		return result, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	owner := hostOwner(t, "keepalive-claim", rentalWiring(connection, private))
	result, problem := owner.c.KeepRentalAlive(context.Background(), podRental, "explicit-1")
	fatal(t, problem)
	if result.RequestId != "explicit-1" || calls != 1 {
		t.Fatal("explicit call did not reach Host once")
	}
	for _, bad := range []string{"failure", "request", "worker", "boot"} {
		mode = bad
		if _, problem := owner.c.KeepRentalAlive(context.Background(), podRental, "explicit-"+bad); problem == nil {
			t.Fatalf("unconfirmed %s acknowledgment succeeded", bad)
		}
	}
	if _, problem := owner.c.KeepRentalAlive(context.Background(), podRental, strings.Repeat("x", 129)); problem == nil {
		t.Fatal("oversized request id admitted")
	}
}
