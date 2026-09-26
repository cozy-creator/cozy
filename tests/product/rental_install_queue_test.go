package producttest

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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
func runRentalInstallQueue(t *testing.T, q *rental.InstallQueue) func() {
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
	prepare := func(ctx context.Context, row records.RentalInstall) *exit.Error {
		calls.Add(1)
		started <- row
		<-ctx.Done()
		return exit.New(exit.Canceled, "controller disconnected")
	}
	q := rental.NewInstallQueue(store, prepare, io.Discard)
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
	claimed, problem := store.ClaimRentalIdleRelease(machine.ID, time.Now())
	rentalInstallCheck(t, problem)
	if claimed {
		t.Fatal("active installation lost its rental to idle release")
	}
	stop()
	waitRentalInstall(t, store, accepted.ID, "queued")
	store.Close()
	// A new database handle and queue represent the next daemon process; no CLI
	// request and no new catalog resolution are needed to finish accepted work.
	resumed, problem := records.Open(layout.DB)
	rentalInstallCheck(t, problem)
	defer resumed.Close()
	q2 := rental.NewInstallQueue(resumed, func(_ context.Context, row records.RentalInstall) *exit.Error {
		calls.Add(1)
		if !reflect.DeepEqual(row.Selection, selection) || row.WorkerBootID != "boot-proof" {
			t.Errorf("recovery changed immutable inputs: %+v", row)
		}
		return nil
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
	for _, state := range []string{"failed", "release_requested", "forgotten", "new-boot", "worker-refused"} {
		t.Run(state, func(t *testing.T) {
			_, store := rentalInstallStore(t)
			machine := rentalInstallMachine("converging")
			rentalInstallCheck(t, store.RecordRental(machine))
			selection := records.RentalInstallSelection{Models: []records.ModelRef{{Model: "paul/minimax-h3", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + strings.Repeat("a", 64), ManifestLength: 321, CatalogRepository: "paul/minimax-h3"}}}
			q := rental.NewInstallQueue(store, func(context.Context, records.RentalInstall) *exit.Error {
				if state != "worker-refused" {
					t.Error("terminal rental received work")
				}
				return exit.Named(exit.Structural, "worker.prepare_refused", "checkpoint removed")
			}, io.Discard)
			row, problem := q.Accept(machine.ID, selection)
			rentalInstallCheck(t, problem)
			want := "rental.boot_failed"
			switch state {
			case "forgotten":
				_, problem = store.ForgetRental(machine.ID)
				rentalInstallCheck(t, problem)
				want = "rental.ended"
			case "new-boot":
				machine.State = "ready"
				rentalInstallCheck(t, store.RecordRental(machine))
				_, problem = store.StartRentalInstall(row.ID, machine.ExpectedWorkerBootID)
				rentalInstallCheck(t, problem)
				machine.ExpectedWorkerBootID = "replacement"
				_, problem = store.ForgetRental(machine.ID)
				rentalInstallCheck(t, problem)
				rentalInstallCheck(t, store.RecordRental(machine))
				want = "rental.worker_boot_changed"
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
	q := rental.NewInstallQueue(store, func(context.Context, records.RentalInstall) *exit.Error {
		t.Error("admission attempted a worker call")
		return nil
	}, io.Discard)
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
		if held == nil || held.State != "queued" || !reflect.DeepEqual(held.Selection, selection) {
			t.Fatalf("202 was not backed by exact durable intent: %+v", held)
		}
	}
	req := httptest.NewRequest("GET", "http://127.0.0.1:9191/v1/local/rentals/"+machine.ID+"/installs", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	api.Authorize(req, credential)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, req)
	var rows []records.RentalInstall
	if err := json.Unmarshal(response.Body.Bytes(), &rows); err != nil || response.Code != http.StatusOK || len(rows) != 2 {
		t.Fatalf("status failed: %d %s %v", response.Code, response.Body, err)
	}
}

func TestRentalInstallSchema46MigrationPreservesRental(t *testing.T) {
	layout, store := rentalInstallStore(t)
	machine := rentalInstallMachine("converging")
	rentalInstallCheck(t, store.RecordRental(machine))
	store.Close()
	db, err := sql.Open("sqlite", layout.DB)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range []string{"DROP TABLE rental_installs", "PRAGMA user_version=46"} {
		if _, err := db.Exec(statement); err != nil {
			t.Fatal(err)
		}
	}
	db.Close()
	old, problem := records.Open(layout.DB)
	if old != nil {
		old.Close()
	}
	if problem == nil || problem.ErrName() != "records_schema_upgrade_required" {
		t.Fatalf("old daemon was not protected: %v", problem)
	}
	migrated, problem := records.OpenForDaemon(layout.DB, filepath.Join(layout.Root, "triage"))
	rentalInstallCheck(t, problem)
	defer migrated.Close()
	row, problem := migrated.RentalRow(machine.ID)
	rentalInstallCheck(t, problem)
	if row == nil || row.State != "converging" {
		t.Fatal("migration changed existing rental", row)
	}
	accepted, problem := migrated.BeginRentalInstall(machine.ID, records.RentalInstallSelection{Package: "paul/minimax-h3", Release: "1.15.7"})
	rentalInstallCheck(t, problem)
	if accepted.State != "queued" {
		t.Fatal(accepted)
	}
}
