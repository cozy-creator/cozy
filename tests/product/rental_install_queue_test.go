package producttest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	"github.com/cozy-creator/cozy/internal/rental"
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
func waitRentalInstall(t *testing.T, store *records.Store, id, state string) *records.Operation {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		row, problem := store.Operation(id)
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
	selection := records.InstallSelection{Package: "paul/minimax-h3", Release: "1.15.7"}
	var calls atomic.Int32
	started := make(chan records.Operation, 2)
	prepare := func(ctx context.Context, row records.Operation) ([]records.ModelProgress, *exit.Error) {
		calls.Add(1)
		started <- row
		<-ctx.Done()
		return nil, exit.New(exit.Canceled, "controller disconnected")
	}
	q := machines.NewInstalls(store, prepare, io.Discard)
	accepted, _, problem := q.Accept(machine.ID, "", selection)
	rentalInstallCheck(t, problem)
	replayed, _, problem := q.Accept(machine.ID, "", selection)
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
		found = found || o.Kind == "install" && o.ID == accepted.ID
	}
	if !found {
		t.Fatal("daemon could idle-exit over accepted installation")
	}
	machine.State = "ready"
	rentalInstallCheck(t, store.RecordRental(machine))
	q.Wake()
	select {
	case row := <-started:
		if !reflect.DeepEqual(row.Install, selection) {
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
	q2 := machines.NewInstalls(resumed, func(_ context.Context, row records.Operation) ([]records.ModelProgress, *exit.Error) {
		calls.Add(1)
		if !reflect.DeepEqual(row.Install, selection) || row.BootID != "boot-proof" {
			t.Errorf("recovery changed immutable inputs: %+v", row)
		}
		return nil, nil
	}, io.Discard)
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
			selection := records.InstallSelection{Models: []records.ModelRef{{Model: "paul/minimax-h3", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + strings.Repeat("a", 64), ManifestLength: 321, CatalogRepository: "paul/minimax-h3"}}}
			q := machines.NewInstalls(store, func(context.Context, records.Operation) ([]records.ModelProgress, *exit.Error) {
				if state != "worker-refused" {
					t.Error("terminal rental received work")
				}
				return nil, exit.Named(exit.Structural, "worker.prepare_refused", "checkpoint removed")
			}, io.Discard)
			row, _, problem := q.Accept(machine.ID, "", selection)
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
			if done.ErrorCode != want || !reflect.DeepEqual(done.Install, selection) {
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
	q := machines.NewInstalls(store, func(context.Context, records.Operation) ([]records.ModelProgress, *exit.Error) {
		t.Error("admission attempted a worker call")
		return nil, nil
	}, io.Discard)
	server := api.New(api.Options{Orchestrator: owner, Cfg: config.Config{Home: layout.Root, HubURL: machine.Hub}, Creds: api.Credentials{CLI: credential}, Addr: "127.0.0.1:9191", RentalInstall: q.Accept})
	handler, problem := server.Handler()
	rentalInstallCheck(t, problem)
	selections := []records.InstallSelection{{Package: "paul/minimax-h3", Release: "1.15.7"}, {Models: []records.ModelRef{{Model: "paul/minimax-h3", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + strings.Repeat("b", 64), ManifestLength: 123, CatalogRepository: "paul/minimax-h3"}}}}
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
		var accepted api.Lifecycle
		if err := json.Unmarshal(response.Body.Bytes(), &accepted); err != nil {
			t.Fatal(err)
		}
		held, problem := store.Operation(accepted.RequestID)
		rentalInstallCheck(t, problem)
		// The durable intent names the hub the request came through (#893): here the default.
		want := selection
		want.Hub = machine.Hub
		if held == nil || held.State != "queued" || !reflect.DeepEqual(held.Install, want) {
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
	q := machines.NewInstalls(store, func(ctx context.Context, row records.Operation) ([]records.ModelProgress, *exit.Error) {
		if active.Add(1) != 1 {
			t.Error("one rental received concurrent installations")
		}
		defer active.Add(-1)
		call := calls.Add(1)
		entered <- row.Install.Release
		if call == 1 {
			return nil, exit.Unavailablef("control stream unavailable")
		}
		if call == 2 {
			select {
			case <-release:
			case <-ctx.Done():
				return nil, exit.New(exit.Canceled, "disconnected")
			}
		}
		return nil, nil
	}, io.Discard)
	first, _, problem := q.Accept(machine.ID, "", records.InstallSelection{Package: "proof/queued", Release: "1.0.0"})
	rentalInstallCheck(t, problem)
	second, _, problem := q.Accept(machine.ID, "", records.InstallSelection{Package: "proof/queued", Release: "2.0.0"})
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

// The worker restarted after the installation was claimed on its old boot. The daemon
// claims it again on the rental's current boot and prepares it there over the signed
// TLS Host; it never fails the queued selection for the restart.
func TestQueuedRentalInstallReclaimsARestartedWorker(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "install-restart")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod := &fakePod{controlKey: public}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	peer := newFakeRentalHub(t, 0)
	peer.add(podRental, "restarted")
	peer.rentals[podRental]["requested_accelerator_model"] = "fake-4090"
	release := rentalReleaseFacts()
	release.PackageInterface = []byte(`{"application":"proof:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[]}`)
	peer.packageReleases = map[string]any{"proof/restart@1": release}
	row := records.Rental{ID: podRental, MachineName: "restarted", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, Hub: peer.server.URL, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: "boot-before-restart"}
	fatal(t, store.RecordRental(row))
	queued, _, problem := store.BeginInstall(podRental, "", "", records.InstallSelection{Package: "proof/restart", Release: "1"})
	fatal(t, problem)
	_, problem = store.StartInstall(queued.ID, row.ExpectedWorkerBootID)
	fatal(t, problem)
	_, problem = store.ForgetRental(podRental)
	fatal(t, problem)
	row.Address, row.MediaAddress, row.ExpectedWorkerBootID = connection.Addr, connection.Media.Addr, podBootID
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+peer.server.URL+"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root)
	var done *records.Operation
	waitUntil(t, "the restarted worker's installation settles", func() bool {
		done, problem = store.Operation(queued.ID)
		fatal(t, problem)
		return !done.Active()
	})
	pod.mu.Lock()
	prepares := len(pod.prepares)
	pod.mu.Unlock()
	if done.State != "succeeded" || done.BootID != podBootID || prepares != 1 {
		t.Fatalf("installation was not claimed again on the new boot: %+v, %d preparations\n%s", done, prepares, tail(filepath.Join(root, "daemon.log")))
	}
}
