package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

func rentalInstallCheck(t *testing.T, problem *exit.Error) {
	t.Helper()
	if problem != nil {
		t.Fatal(problem)
	}
}
func rentalInstallStore(t *testing.T) (home.Layout, *records.Store) {
	t.Helper()
	layout, problem := home.Open(t.TempDir())
	rentalInstallCheck(t, problem)
	store, problem := records.Open(layout.DB)
	rentalInstallCheck(t, problem)
	t.Cleanup(func() { store.Close() })
	return layout, store
}
func rentalInstallMachine(state string) records.Rental {
	return records.Rental{ID: "rental-install-proof", MachineName: "kirukiru", State: state, Hub: "https://hub.example", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, ExpectedWorkerBootID: "boot-proof", ReadyAt: time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano)}
}
func waitRentalInstall(t *testing.T, store *records.Store, id, state string) *records.RentalInstall {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		row, problem := store.RentalInstall(id)
		rentalInstallCheck(t, problem)
		if row != nil && row.State == state {
			return row
		}
		select {
		case <-deadline:
			t.Fatalf("installation %s did not reach %s: %+v", id, state, row)
		case <-tick.C:
		}
	}
}
func runRentalInstallQueue(t *testing.T, q *machines.Installs) func() {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); q.Run(ctx) }()
	stop := func() {
		cancel()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("installation queue did not stop")
		}
	}
	t.Cleanup(stop)
	return stop
}

func TestRentalInstallQueueSurvivesDisconnectAndReplaysExactSelection(t *testing.T) {
	layout, store := rentalInstallStore(t)
	machine := rentalInstallMachine("converging")
	rentalInstallCheck(t, store.RecordRental(machine))
	selection := records.RentalInstallSelection{Package: "paul/minimax-h3", Release: "1.15.7"}
	var calls atomic.Int32
	started := make(chan records.RentalInstall, 2)
	prepare := func(ctx context.Context, row records.RentalInstall, _ func(machines.InstallProgress)) *exit.Error {
		calls.Add(1)
		started <- row
		<-ctx.Done()
		return exit.New(exit.Canceled, "controller disconnected")
	}
	q := machines.NewInstalls(store, prepared(prepare), io.Discard)
	accepted, problem := q.Accept(machine.ID, selection)
	rentalInstallCheck(t, problem)
	replayed, problem := q.Accept(machine.ID, selection)
	rentalInstallCheck(t, problem)
	if accepted.State != "queued" || accepted.ID != replayed.ID {
		t.Fatalf("admission is not durable/idempotent: %+v %+v", accepted, replayed)
	}
	stop := runRentalInstallQueue(t, q)
	select {
	case <-started:
		t.Fatal("booting rental received installation")
	case <-time.After(20 * time.Millisecond):
	}
	if calls.Load() != 0 {
		t.Fatal("booting rental was dispatched")
	}
	idle, problem := store.RentalIdleObservation(machine)
	rentalInstallCheck(t, problem)
	if idle.Queued != 1 {
		t.Fatalf("queued installation missing from idle census: %+v", idle)
	}
	obligations, problem := store.Obligations()
	rentalInstallCheck(t, problem)
	found := false
	for _, o := range obligations {
		found = found || o.Kind == "rental_install" && o.ID == accepted.ID
	}
	if !found {
		t.Fatal("daemon could idle-exit over accepted installation")
	}
	machine.State = "ready"
	rentalInstallCheck(t, store.RecordRental(machine))
	q.Wake()
	select {
	case row := <-started:
		if !reflect.DeepEqual(row.Selection, selection) {
			t.Fatal("selection changed", row)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ready rental was not dispatched")
	}
	stop()
	waitRentalInstall(t, store, accepted.ID, "queued")
	store.Close()
	// A new database handle and queue represent the next daemon process; no CLI
	// request and no new catalog resolution are needed to finish accepted work.
	resumed, problem := records.Open(layout.DB)
	rentalInstallCheck(t, problem)
	defer resumed.Close()
	q2 := machines.NewInstalls(resumed, prepared(func(_ context.Context, row records.RentalInstall, _ func(machines.InstallProgress)) *exit.Error {
		calls.Add(1)
		if !reflect.DeepEqual(row.Selection, selection) || row.WorkerBootID != "boot-proof" {
			t.Errorf("recovery changed immutable inputs: %+v", row)
		}
		return nil
	}), io.Discard)
	stop2 := runRentalInstallQueue(t, q2)
	defer stop2()
	waitRentalInstall(t, resumed, accepted.ID, "succeeded")
	if calls.Load() != 2 {
		t.Fatalf("prepare calls=%d", calls.Load())
	}
	idle, problem = resumed.RentalIdleObservation(machine)
	rentalInstallCheck(t, problem)
	if idle.Queued+idle.Running != 0 || time.Since(idle.Since) > time.Minute {
		t.Fatalf("completion did not release activity/reset idle: %+v", idle)
	}
}

func TestRentalInstallQueueRetainsTypedTerminalFailures(t *testing.T) {
	for _, state := range []string{"failed", "release_requested", "forgotten", "worker-refused"} {
		t.Run(state, func(t *testing.T) {
			_, store := rentalInstallStore(t)
			machine := rentalInstallMachine("converging")
			rentalInstallCheck(t, store.RecordRental(machine))
			selection := records.RentalInstallSelection{Models: []records.ModelRef{{Model: "paul/minimax-h3", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + strings.Repeat("a", 64), ManifestLength: 321, CatalogRepository: "paul/minimax-h3"}}}
			q := machines.NewInstalls(store, prepared(func(context.Context, records.RentalInstall, func(machines.InstallProgress)) *exit.Error {
				if state != "worker-refused" {
					t.Error("terminal rental received work")
				}
				return exit.Named(exit.Structural, "worker.prepare_refused", "checkpoint removed")
			}), io.Discard)
			row, problem := q.Accept(machine.ID, selection)
			rentalInstallCheck(t, problem)
			want := "rental.boot_failed"
			switch state {
			case "forgotten":
				_, problem = store.ForgetRental(machine.ID)
				rentalInstallCheck(t, problem)
				want = "rental.ended"
			case "worker-refused":
				machine.State = "ready"
				rentalInstallCheck(t, store.RecordRental(machine))
				want = "worker.prepare_refused"
			default:
				machine.State = state
				rentalInstallCheck(t, store.RecordRental(machine))
				if state == "release_requested" {
					want = "rental.ended"
				}
			}
			stop := runRentalInstallQueue(t, q)
			defer stop()
			done := waitRentalInstall(t, store, row.ID, "failed")
			if done.ErrorCode != want || !reflect.DeepEqual(done.Selection, selection) {
				t.Fatalf("lost terminal failure or model grant: %+v", done)
			}
		})
	}
}

func TestRentalInstallAdmissionAndStatusAPIWhileBooting(t *testing.T) {
	layout, store := rentalInstallStore(t)
	machine := rentalInstallMachine("converging")
	rentalInstallCheck(t, store.RecordRental(machine))
	owner, problem := orchestrator.Open(orchestrator.Options{Store: store, Layout: layout})
	rentalInstallCheck(t, problem)
	defer owner.Close(time.Second)
	credential := secret.New("installation-test-only")
	q := machines.NewInstalls(store, prepared(func(context.Context, records.RentalInstall, func(machines.InstallProgress)) *exit.Error {
		t.Error("admission attempted a worker call")
		return nil
	}), io.Discard)
	server := api.New(api.Options{Orchestrator: owner, Cfg: config.Config{Home: layout.Root, HubURL: machine.Hub}, Creds: api.Credentials{CLI: credential}, Addr: "127.0.0.1:9191", RentalInstall: q.Accept})
	handler, problem := server.Handler()
	rentalInstallCheck(t, problem)
	selections := []records.RentalInstallSelection{{Package: "paul/minimax-h3", Release: "1.15.7"}, {Models: []records.ModelRef{{Model: "paul/minimax-h3", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + strings.Repeat("b", 64), ManifestLength: 123, CatalogRepository: "paul/minimax-h3"}}}}
	for _, selection := range selections {
		raw, err := json.Marshal(selection)
		if err != nil {
			t.Fatal(err)
		}
		req := httptest.NewRequest("POST", "http://127.0.0.1:9191/v1/local/rentals/"+machine.ID+"/prepare", bytes.NewReader(raw))
		req.RemoteAddr = "127.0.0.1:12345"
		api.Authorize(req, credential)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		if response.Code != http.StatusAccepted {
			t.Fatalf("booting admission=%d %s", response.Code, response.Body)
		}
		var accepted records.RentalInstall
		if err := json.Unmarshal(response.Body.Bytes(), &accepted); err != nil {
			t.Fatal(err)
		}
		held, problem := store.RentalInstall(accepted.ID)
		rentalInstallCheck(t, problem)
		// The durable intent names the hub the request came through (#893): here the default.
		want := selection
		want.Hub = machine.Hub
		if held == nil || held.State != "queued" || !reflect.DeepEqual(held.Selection, want) {
			t.Fatalf("202 was not backed by exact durable intent: %+v", held)
		}
	}
}

func TestRentalInstallQueueSerializesAndRetriesOnlyAfterWake(t *testing.T) {
	_, store := rentalInstallStore(t)
	machine := rentalInstallMachine("converging")
	rentalInstallCheck(t, store.RecordRental(machine))
	entered := make(chan string, 4)
	release := make(chan struct{})
	var calls, active atomic.Int32
	q := machines.NewInstalls(store, prepared(func(ctx context.Context, row records.RentalInstall, report func(machines.InstallProgress)) *exit.Error {
		if active.Add(1) != 1 {
			t.Error("one rental received concurrent installations")
		}
		defer active.Add(-1)
		call := calls.Add(1)
		entered <- row.Selection.Release
		if call == 1 {
			// Bytes moved before the machine became unavailable: it is tried again.
			report(machines.InstallProgress{Stage: "download", TotalBytes: 100})
			report(machines.InstallProgress{Stage: "download", TotalBytes: 100, TransferredBytes: 40})
			return exit.Unavailablef("control stream unavailable")
		}
		if call == 2 {
			select {
			case <-release:
			case <-ctx.Done():
				return exit.New(exit.Canceled, "disconnected")
			}
		}
		return nil
	}), io.Discard)
	first, problem := q.Accept(machine.ID, records.RentalInstallSelection{Package: "proof/queued", Release: "1.0.0"})
	rentalInstallCheck(t, problem)
	second, problem := q.Accept(machine.ID, records.RentalInstallSelection{Package: "proof/queued", Release: "2.0.0"})
	rentalInstallCheck(t, problem)
	machine.State = "ready"
	rentalInstallCheck(t, store.RecordRental(machine))
	stop := runRentalInstallQueue(t, q)
	defer stop()
	select {
	case got := <-entered:
		if got != "1.0.0" {
			t.Fatalf("dispatch order=%s", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("first install did not start")
	}
	waitRentalInstall(t, store, first.ID, "queued")
	select {
	case got := <-entered:
		t.Fatalf("transient failure spun or bypassed ordering: %s", got)
	case <-time.After(20 * time.Millisecond):
	}
	q.Wake()
	select {
	case got := <-entered:
		if got != "1.0.0" {
			t.Fatalf("retry changed selection: %s", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("reconciler wake did not retry")
	}
	q.Wake()
	select {
	case got := <-entered:
		t.Fatalf("second install bypassed active first install: %s", got)
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	waitRentalInstall(t, store, first.ID, "succeeded")
	waitRentalInstall(t, store, second.ID, "succeeded")
	if calls.Load() != 3 {
		t.Fatalf("prepare calls=%d", calls.Load())
	}
}

// An unavailable machine that let an installation make no progress fails it with the reason:
// it is not queued again, however often the queue wakes.
func TestRentalInstallWithNoProgressFailsInsteadOfQueuingAgain(t *testing.T) {
	_, store := rentalInstallStore(t)
	machine := rentalInstallMachine("ready")
	rentalInstallCheck(t, store.RecordRental(machine))
	var calls atomic.Int32
	q := machines.NewInstalls(store, prepared(func(_ context.Context, _ records.RentalInstall, report func(machines.InstallProgress)) *exit.Error {
		calls.Add(1)
		report(machines.InstallProgress{Stage: "connect"})
		return exit.Named(exit.Unavailable, "machine.stopped", "this computer's machine is stopped")
	}), io.Discard)
	row, problem := q.Accept(machine.ID, records.RentalInstallSelection{Package: "proof/stalled", Release: "1.0.0"})
	rentalInstallCheck(t, problem)
	runRentalInstallQueue(t, q)
	done := waitRentalInstall(t, store, row.ID, "failed")
	if done.ErrorCode != "machine.stopped" {
		t.Fatalf("the installation failed without its reason: %+v", done)
	}
	for range 5 {
		q.Wake()
		time.Sleep(10 * time.Millisecond)
	}
	if calls.Load() != 1 {
		t.Fatalf("a failed installation was tried %d times", calls.Load())
	}
	obligations, problem := store.Obligations()
	rentalInstallCheck(t, problem)
	for _, o := range obligations {
		if o.Kind == "rental_install" {
			t.Fatalf("a failed installation holds the daemon: %+v", o)
		}
	}
}

// prepared is an installation that produces no result.
func prepared(prepare func(context.Context, records.RentalInstall, func(machines.InstallProgress)) *exit.Error) machines.Prepare {
	return func(ctx context.Context, row records.RentalInstall, report func(machines.InstallProgress)) (json.RawMessage, *exit.Error) {
		return nil, prepare(ctx, row, report)
	}
}
