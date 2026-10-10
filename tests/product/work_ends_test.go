package producttest

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"database/sql"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// idleRoot is a daemon root whose idle exit follows one second with nothing to manage.
func idleRoot(t *testing.T, name string) (string, *records.Store) {
	t.Helper()
	root := filepath.Join(scratchBase, name)
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("daemon:\n  idle_shutdown_s: 1\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	return root, store
}

// The owner's ruling: work ends completed or failed, never waits forever, and dead work never
// holds the daemon. These are the records an older daemon left on his machine — retained jobs
// `blocked` for a retry with their output export pending, a classic terminal its worker never
// acknowledged, rentals the Hub reported failed, and a weights destination of a run that had
// already failed. The next daemon fails each with its reason, keeps every row, and leaves.
func TestDaemonEndsWorkAnOlderDaemonLeftWaiting(t *testing.T) {
	root, store := idleRoot(t, "work-ends")
	blocked, _, problem := store.Submit(records.Request{ID: "job-blocked-refused", IdemKey: "job-blocked-refused", Kind: "job",
		Package: "local/sage2-lease-proof", Entrypoint: "prove", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		RetainWork: true, MachineExecutionObserver: true,
		OutputExport: &records.OutputExportIntent{Directory: filepath.Join(root, "out"),
			Outputs: []records.OutputExportEntry{{OutputID: "witness", MediaType: "application/json"}}}})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(blocked.ID, "local"))
	const reason = "runtime preparation failed: 'sage2-lease-proof' exposes 0 cozy.application entries"
	fatal(t, store.AppendEvent(blocked.ID, "request.blocked", 0, map[string]any{"status": "blocked",
		"error_type": "machine_execution.prepare_refused", "error": reason}))
	failedRun, _, problem := store.Submit(records.Request{ID: "job-failed-upload", IdemKey: "job-failed-upload", Kind: "job",
		Package: "local/sdxl", Entrypoint: "fp8", Payload: []byte(`{}`), BodyDigest: childDigest("2"), MachineExecutionObserver: true,
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "fidika/sdxl", Outputs: []records.ModelTransferOutput{{Name: "fp8"}}}})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(failedRun.ID, "local"))
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/classic", WorkerID: "worker"}))
	unacked := offerChildParent(t, store, recordPrivateTransaction(t, store, "unacked", ""))
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: unacked.ID, Attempt: 1, SessionID: "private-boot",
		InvocationDigest: childDigest("1"), TerminalID: "terminal-unacked", TerminalDigest: childDigest("2"),
		Status: "FAILED", RequestState: "failed", Body: []byte(`{}`)})
	fatal(t, problem)
	for _, rental := range []records.Rental{
		{ID: "pr-exposure", MachineName: "lancer", Failure: records.RentalFailure{Code: "authorization_exposure_exhausted"}},
		{ID: "pr-supervisor", MachineName: "fuyuno", Failure: records.RentalFailure{Code: "supervisor_never_started", ProviderState: "running"}},
	} {
		rental.SKU, rental.AcceleratorModel, rental.AcceleratorCount, rental.HourlyRateUSDMicros = "cpu", "CPU", 1, 1
		rental.State, rental.Hub = "failed", "http://127.0.0.1:1"
		fatal(t, store.RecordRental(rental))
	}
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET state='blocked' WHERE id=?`, blocked.ID)
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET state='failed' WHERE id=?`, failedRun.ID)
	must(t, err)
	must(t, db.Close())

	logPath := filepath.Join(root, "daemon.log")
	if code := awaitDaemonExit(t, startDaemonProcess(t, root), 30*time.Second); code != 0 {
		t.Fatalf("the daemon exited %d\n%s", code, tail(logPath))
	}
	log, _ := os.ReadFile(logPath)
	if !strings.Contains(string(log), "nothing to manage for 1s; stopping") || strings.Contains(string(log), "idle exit held") ||
		!strings.Contains(string(log), blocked.ID+": "+reason+"; it waited for a retry and has failed") {
		t.Fatalf("dead work held the daemon, or its end went unsaid\n%s", tail(logPath))
	}
	row, problem := store.RequestRow(blocked.ID)
	fatal(t, problem)
	errType, _, errText, problem := store.SettledFailure(blocked.ID)
	fatal(t, problem)
	if row == nil || row.State != "failed" || errType != "machine_execution.prepare_refused" || errText != reason {
		t.Fatalf("the blocked job did not fail with its own reason: %+v %s %s", row, errType, errText)
	}
	export, problem := store.OutputExportOf(blocked.ID)
	fatal(t, problem)
	if export == nil || export.State != "skipped" || !strings.Contains(export.SafeError, reason) {
		t.Fatalf("the blocked job's export still waits: %+v", export)
	}
	attempt, problem := store.AttemptRow(unacked.ID, 1)
	fatal(t, problem)
	if attempt == nil || attempt.State != "closed" || attempt.TerminalStatus != "FAILED" {
		t.Fatalf("the unacknowledged classic terminal stayed open: %+v", attempt)
	}
	transfer, problem := store.ModelTransferOf(failedRun.ID)
	fatal(t, problem)
	if transfer == nil || transfer.State != "failed" || transfer.ErrorCode != "model_transfer.run_failed" {
		t.Fatalf("a failed run's weights destination still waits: %+v", transfer)
	}
	for _, id := range []string{"pr-exposure", "pr-supervisor"} {
		if rental, problem := store.RentalRow(id); problem != nil || rental == nil || rental.State != "failed" {
			t.Fatalf("failed rental %s was not kept as history: %+v %v", id, rental, problem)
		}
	}
	obligations, problem := store.Obligations()
	fatal(t, problem)
	if len(obligations) != 0 {
		t.Fatalf("ended work still counts as managed: %+v", obligations)
	}
}

// An export whose destination cannot hold its files fails with the reason, once: it holds
// nothing, and no later daemon start tries it again.
func TestAnExportToAnInvalidDestinationFailsOnce(t *testing.T) {
	root, store := idleRoot(t, "export-invalid")
	blocker := filepath.Join(root, "not-a-directory")
	must(t, os.WriteFile(blocker, []byte("a file where the folder should be"), 0o600))
	request, _, problem := store.Submit(records.Request{ID: "req-export-invalid", IdemKey: "req-export-invalid", Kind: "job",
		Package: "local/export-proof", Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: childDigest("3"),
		OutputExport: &records.OutputExportIntent{Directory: filepath.Join(blocker, "out"),
			Outputs: []records.OutputExportEntry{{OutputID: "report", MediaType: "application/json"}}}})
	fatal(t, problem)
	body := []byte(`{"ok":true}`)
	sum := sha256.Sum256(body)
	accepted := filepath.Join(root, "accepted.json")
	must(t, os.WriteFile(accepted, body, 0o600))
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: request.Package, WorkerID: "worker"}))
	request = offerChildParent(t, store, request)
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: request.ID, Attempt: 1, SessionID: "private-boot",
		InvocationDigest: childDigest("1"), TerminalID: "terminal-export", TerminalDigest: childDigest("2"),
		Status: "SUCCEEDED", RequestState: "succeeded", Body: []byte(`{}`),
		Outputs: []records.Output{{OutputID: "report", MediaID: "media-report", Path: accepted,
			Digest: "sha256:" + hex.EncodeToString(sum[:]), Length: int64(len(body)), MimeType: "application/json"}}})
	fatal(t, problem)
	fatal(t, store.Closed(request.ID, 1))

	logPath := filepath.Join(root, "daemon.log")
	var settled string
	for start := range 2 {
		if code := awaitDaemonExit(t, startDaemonProcess(t, root), 30*time.Second); code != 0 {
			t.Fatalf("start %d: the daemon exited %d\n%s", start, code, tail(logPath))
		}
		export, problem := store.OutputExportOf(request.ID)
		fatal(t, problem)
		if export == nil || export.State != "failed" || export.SafeError == "" {
			t.Fatalf("start %d: the export to an invalid destination did not fail with its reason: %+v\n%s", start, export, tail(logPath))
		}
		if start == 1 && export.UpdatedAt != settled {
			t.Fatalf("a failed export was tried again at the next start: %+v", export)
		}
		settled = export.UpdatedAt
	}
	if raw, _ := os.ReadFile(logPath); strings.Contains(string(raw), "idle exit held") {
		t.Fatalf("a failed export held the daemon\n%s", tail(logPath))
	}
}

// A run no machine on the market can take fails with the reason instead of queuing on stock.
func TestARunTheMarketCannotServeFails(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	for _, sku := range h.market {
		h.soldOut[sku.Name] = true
	}
	root := ladderRoot(t, h)
	startDaemonProcess(t, root)
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--json", "--idempotency-key", "sold-out")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "the sold-out run failing", func() bool {
		row, problem := store.RequestByIdempotencyKey("sold-out")
		return problem == nil && row != nil && row.State == "failed"
	})
	row, problem := store.RequestByIdempotencyKey("sold-out")
	fatal(t, problem)
	if errType, _, _, problem := store.SettledFailure(row.ID); problem != nil || errType != "rental.no_inventory" {
		t.Fatalf("the sold-out run failed as %q, not for want of stock: %v\n%s", errType, problem, out)
	}
}

// A run waiting on a rental that fails before it is ever ready fails with that rental's
// cause: it is not placed again on a market that could not boot it. One that had served is.
func TestARunWhoseMachineNeverBootedFailsInsteadOfPlacingAgain(t *testing.T) {
	root := filepath.Join(scratchBase, "never-booted")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hubURL+"\n"+
		"tensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	hub.packageReleases = map[string]any{"fake/lost@1": rentalReleaseFacts()}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	hub.add("rental-unbooted", "sumireko")
	hub.setState("rental-unbooted", "acquiring", "")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: "rental-unbooted", MachineName: "sumireko", SKU: "cpu",
		AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "acquiring", Hub: hubURL}))
	const id = "req-on-unbooted"
	body, _ := canonical.Spell(canonical.Digest([]byte(id)))
	_, _, problem = store.Submit(records.Request{ID: id, IdemKey: "idem-" + id, BodyDigest: body, Package: "fake/lost",
		Release: "1", Entrypoint: "generate", Payload: []byte("{}"), Rental: true, Worker: "rental-unbooted", MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(id, "rental-unbooted"))
	startDaemonProcess(t, root)
	hub.setState("rental-unbooted", "failed", "supervisor_never_started")
	waitFor(t, root, "the run failing with its unbooted rental", func() bool {
		row, problem := store.RequestRow(id)
		return problem == nil && row != nil && row.State == "failed"
	})
	_, _, errText, problem := store.SettledFailure(id)
	fatal(t, problem)
	if !strings.Contains(errText, "supervisor_never_started") {
		t.Fatalf("the run did not fail with its rental's cause: %q", errText)
	}
	events, problem := store.EventsAfter(id, 0, 1000)
	fatal(t, problem)
	for _, event := range events {
		if event.Type == "request.queued" {
			t.Fatalf("a run whose rental never booted was placed again: %+v", event)
		}
	}
}

// A canceled run its machine answers it never took ends canceled; it does not wait forever
// on a cancellation the machine cannot apply.
type neverTookMachine struct {
	v1.UnimplementedMachineServer
	controls atomic.Int32
}

func (m *neverTookMachine) Control(context.Context, *v1.ControlRequest) (*v1.RunState, error) {
	m.controls.Add(1)
	return nil, status.Error(codes.NotFound, "no such run")
}

func (m *neverTookMachine) Run(*v1.RunRequest, grpc.ServerStreamingServer[v1.RunEvent]) error {
	return status.Error(codes.NotFound, "no such run")
}

func TestACancelTheMachineNeverTookEndsCanceled(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	must(t, os.MkdirAll(layout.Machine, 0700))
	_, problem = machines.NewHost(layout.Machine, "", nil).Owner()
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	certSource := httptest.NewTLSServer(http.NotFoundHandler())
	cert := certSource.TLS.Certificates[0]
	certSource.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow test machine transport
	must(t, err)
	machine := &neverTookMachine{}
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}})))
	v1.RegisterMachineServer(server, machine)
	go server.Serve(listener)
	defer server.Stop()
	ep := &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: listener.Addr().String(), WorkerID: "fixture-worker",
		WorkerBootID: "fixture-boot", WorkspaceID: "fixture",
		CertificatePEM: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]}))}
	startDaemonProcess(t, root)
	row, _, problem := store.SubmitWithEvent(records.Request{ID: "req-never-taken", IdemKey: "never-taken", Package: "proof/never",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true,
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "proof/never", Outputs: []records.ModelTransferOutput{{Name: "model"}}}},
		map[string]any{"machine_endpoint": ep})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(row.ID, ep.Name()))
	fatal(t, store.AppendEvent(row.ID, records.RunV1Sent, 0, map[string]any{"machine": ep.Name()}))
	if code, out := runCozy(t, root, "run", "cancel", row.ID, "--json"); code != 0 {
		t.Fatalf("cancel [%d]: %s", code, out)
	}
	waitFor(t, root, "the never-taken run ending canceled", func() bool {
		current, problem := store.RequestRow(row.ID)
		return problem == nil && current != nil && current.State == "canceled"
	})
	if machine.controls.Load() == 0 {
		t.Fatal("the cancellation never asked the machine")
	}
	transfer, problem := store.ModelTransferOf(row.ID)
	fatal(t, problem)
	if transfer == nil || transfer.State != "canceled" {
		t.Fatalf("the canceled run's weights destination still waits: %+v", transfer)
	}
}
