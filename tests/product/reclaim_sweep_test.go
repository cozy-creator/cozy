package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

// TestReclaimSweeps is the startup sweep over the three roots that used to grow without
// bound (cl-090): attempt working directories, closed workers' roots, and transfer
// scratch. Nothing is mocked — a real layout, a real records store, real directories,
// and the product's own reclaim package. Each root proves both halves: what nothing
// references goes, and what a row or a live process still claims stays.
func TestReclaimSweeps(t *testing.T) {
	l, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(l.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()

	// ---- attempts: exported (reclaim), still holding its bytes (keep), open (keep),
	// unrecorded (reclaim), and a non-attempt entry the sweep never touches.
	if problem := store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-attempts", Package: "cozy/sweep",
		PackageRevisionDigest: "sha256:" + sixtyFour("d"), WorkerID: "local",
		Devices: []string{"cpu"}}); problem != nil {
		t.Fatal(problem)
	}
	exported := closedAttempt(t, l, store, "req-exported", true)
	resident := closedAttempt(t, l, store, "req-resident", false)
	open := l.AttemptDir("req-open", 1)
	must(t, os.MkdirAll(filepath.Join(open, "in"), 0o755))
	submitSweepRequest(t, store, "req-open")
	if _, problem := store.Dispatch(records.Attempt{RequestID: "req-open", SessionID: "s-open", InstanceID: "ins-attempts",
		InvocationDigest: "sha256:" + sixtyFour("c"), InvocationCanonical: []byte("{}")}); problem != nil {
		t.Fatal(problem)
	}
	stranded := l.AttemptDir("req-gone", 3)
	must(t, os.MkdirAll(filepath.Join(stranded, "in"), 0o755))
	must(t, os.WriteFile(filepath.Join(stranded, "in", "payload"), []byte("{}"), 0o644))
	must(t, os.WriteFile(filepath.Join(l.Attempts, "notes.txt"), []byte("keep"), 0o644))

	swept, problem := reclaim.Attempts(l, store)
	if problem != nil {
		t.Fatal(problem)
	}
	if swept.Scanned != 4 || swept.Removed != 2 || swept.Bytes <= 0 {
		t.Fatalf("attempt sweep = %+v, want 4 scanned, 2 removed, bytes freed", swept)
	}
	for _, gone := range []string{exported, stranded, l.RequestAttempts("req-exported"), l.RequestAttempts("req-gone")} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Fatalf("unreferenced attempt directory %s remains: %v", gone, err)
		}
	}
	for _, kept := range []string{resident, open, filepath.Join(l.Attempts, "notes.txt")} {
		if _, err := os.Stat(kept); err != nil {
			t.Fatalf("claimed attempt directory %s was swept: %v", kept, err)
		}
	}
	// The exported output is still served from where it went.
	outputs, problem := store.VisibleOutputs("req-exported")
	if problem != nil || len(outputs) != 1 || outputs[0].Path != filepath.Join(l.PackageOutputs("cozy/sweep"), "img.png") {
		t.Fatalf("relocated output = %+v, %v", outputs, problem)
	}

	// ---- workers: a live row keeps its root, a closed row and an unrecorded root go.
	for _, id := range []string{"ins-live", "ins-closed", "ins-unrecorded"} {
		must(t, os.MkdirAll(filepath.Join(l.WorkerDir(id), "home", "scratch"), 0o755))
		must(t, os.WriteFile(filepath.Join(l.WorkerDir(id), "home", "scratch", "copy.safetensors"),
			make([]byte, 4096), 0o644))
		must(t, os.WriteFile(filepath.Join(l.WorkerDir(id), "worker.log"), []byte("log"), 0o644))
	}
	for index, id := range []string{"ins-live", "ins-closed"} {
		if problem := store.SpawnWorker(records.WorkerProcess{InstanceID: id, Package: "cozy/sweep",
			PackageRevisionDigest: "sha256:" + sixtyFour("d"), WorkerID: "local",
			Devices: []string{"cuda:" + string(rune('1'+index))}}); problem != nil {
			t.Fatal(problem)
		}
	}
	if problem := store.CloseWorker("ins-closed"); problem != nil {
		t.Fatal(problem)
	}
	// Closing a worker reclaims its root at once but keeps the log the exit message named.
	if freed, problem := reclaim.Worker(l, "ins-closed", true); problem != nil || freed < 4096 {
		t.Fatalf("worker close reclaim = %d, %v", freed, problem)
	}
	if _, err := os.Stat(filepath.Join(l.WorkerDir("ins-closed"), "worker.log")); err != nil {
		t.Fatalf("worker close reclaim removed the log: %v", err)
	}
	if _, err := os.Stat(filepath.Join(l.WorkerDir("ins-closed"), "home")); !os.IsNotExist(err) {
		t.Fatalf("worker close reclaim kept the runtime home: %v", err)
	}
	workers, problem := reclaim.Workers(l, store)
	if problem != nil {
		t.Fatal(problem)
	}
	// ins-attempts (live, no root) is not on disk; three roots are scanned.
	if workers.Scanned != 3 || workers.Removed != 2 {
		t.Fatalf("worker sweep = %+v, want 3 scanned and 2 removed", workers)
	}
	for _, gone := range []string{"ins-closed", "ins-unrecorded"} {
		if _, err := os.Stat(l.WorkerDir(gone)); !os.IsNotExist(err) {
			t.Fatalf("closed worker root %s remains: %v", gone, err)
		}
	}
	if _, err := os.Stat(filepath.Join(l.WorkerDir("ins-live"), "home", "scratch", "copy.safetensors")); err != nil {
		t.Fatalf("live worker root was swept: %v", err)
	}

	// ---- transfer: a claim held by a live process keeps its scratch; an unheld
	// directory, a stray file, and a crashed process's leftover go; locks/ stays.
	held, problem := scratch.Temp(l.Transfer, "package-install-")
	if problem != nil {
		t.Fatal(problem)
	}
	defer held.Release()
	must(t, os.WriteFile(filepath.Join(held.Path, "wheel"), make([]byte, 100), 0o644))
	// A crashed process's scratch: the claim file is there, the kernel lock died with it.
	leftoverPath := filepath.Join(l.Transfer, "invoke-models-crashed")
	must(t, os.MkdirAll(leftoverPath, 0o700))
	must(t, os.WriteFile(filepath.Join(leftoverPath, ".claim"), nil, 0o600))
	must(t, os.WriteFile(filepath.Join(leftoverPath, "manifest.bin"), make([]byte, 2048), 0o644))
	must(t, os.MkdirAll(filepath.Join(l.Transfer, "locks"), 0o700))
	must(t, os.WriteFile(filepath.Join(l.Transfer, "locks", sixtyFour("e")+".lock"), nil, 0o600))
	must(t, os.MkdirAll(filepath.Join(l.Transfer, "unclaimed-dir"), 0o700))
	must(t, os.WriteFile(filepath.Join(l.Transfer, "stray-evidence.jsonl"), []byte("{}\n"), 0o600))

	transfer, problem := reclaim.Transfer(l)
	if problem != nil {
		t.Fatal(problem)
	}
	if transfer.Scanned != 4 || transfer.Removed != 3 || transfer.Bytes < 2048 {
		t.Fatalf("transfer sweep = %+v, want 4 scanned, 3 removed, the leftover's bytes freed", transfer)
	}
	for _, gone := range []string{leftoverPath, filepath.Join(l.Transfer, "unclaimed-dir"),
		filepath.Join(l.Transfer, "stray-evidence.jsonl")} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Fatalf("unheld transfer entry %s remains: %v", gone, err)
		}
	}
	for _, kept := range []string{filepath.Join(held.Path, "wheel"),
		filepath.Join(l.Transfer, "locks", sixtyFour("e")+".lock")} {
		if _, err := os.Stat(kept); err != nil {
			t.Fatalf("held or lock entry %s was swept: %v", kept, err)
		}
	}

	// Every sweep is idempotent: a second pass over the survivors removes nothing.
	again, _ := reclaim.Attempts(l, store)
	againWorkers, _ := reclaim.Workers(l, store)
	againTransfer, _ := reclaim.Transfer(l)
	if again.Removed+againWorkers.Removed+againTransfer.Removed != 0 {
		t.Fatalf("second sweeps removed something: %+v %+v %+v", again, againWorkers, againTransfer)
	}
}

// closedAttempt drives one request through the store to a closed attempt whose single
// output was written under its attempt directory. With exported, the export row is
// settled the way the daemon settles it, which re-points the output row at the store.
func closedAttempt(t *testing.T, l home.Layout, store *records.Store, requestID string, exported bool) string {
	t.Helper()
	submitSweepRequest(t, store, requestID)
	session := "session-" + requestID
	digest := "sha256:" + sixtyFour("a")
	attempt, problem := store.Dispatch(records.Attempt{RequestID: requestID, SessionID: session, InstanceID: "ins-attempts",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(requestID, attempt, session))
	fatal(t, store.Accepted(requestID, attempt, session, "sha256:"+sixtyFour("b"), "", ""))
	dir := l.AttemptDir(requestID, uint64(attempt))
	must(t, os.MkdirAll(filepath.Join(dir, "in"), 0o755))
	must(t, os.WriteFile(filepath.Join(dir, "in", "payload"), []byte(`{"size":32}`), 0o644))
	source := filepath.Join(dir, "image")
	must(t, os.WriteFile(source, make([]byte, 1024), 0o600))
	if _, problem := store.AcceptTerminal(records.Terminal{RequestID: requestID, Attempt: attempt,
		SessionID: session, InvocationDigest: digest, TerminalID: "out-" + requestID,
		TerminalDigest: "sha256:" + sixtyFour("f"), Status: "SUCCEEDED", Cause: "COMPLETED",
		Outputs: []records.Output{{OutputID: "image", MediaID: records.NewID("med"), Path: source,
			Digest: "sha256:" + sixtyFour("0"), Length: 1024, MimeType: "image/png"}},
		EventType: "request.succeeded", EventPayload: map[string]any{}, RequestState: "succeeded",
	}); problem != nil {
		t.Fatal(problem)
	}
	fatal(t, store.Closed(requestID, attempt))
	if exported {
		published := filepath.Join(l.PackageOutputs("cozy/sweep"), "img.png")
		fatal(t, store.BeginOutputExport(requestID))
		fatal(t, store.CompleteOutputExport(requestID,
			[]records.PublishedOutput{{OutputID: "image", Source: source, Path: published}}, true))
	}
	return dir
}

func submitSweepRequest(t *testing.T, store *records.Store, requestID string) {
	t.Helper()
	if _, _, problem := store.Submit(records.Request{
		ID: requestID, IdemKey: "idem-" + requestID, BodyDigest: "sha256:" + sixtyFour("9"),
		Package: "cozy/sweep", Entrypoint: "generate", Payload: []byte("{}"), Outputs: "image",
		OutputExport: &records.OutputExportIntent{Directory: "/tmp/sweep-out",
			Outputs: []records.OutputExportEntry{{OutputID: "image", MediaType: "image/png"}}},
	}); problem != nil {
		t.Fatal(problem)
	}
}

func sixtyFour(char string) string {
	out := ""
	for len(out) < 64 {
		out += char
	}
	return out
}
