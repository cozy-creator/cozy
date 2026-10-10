package producttest

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
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
