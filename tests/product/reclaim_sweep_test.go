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

// TestReclaimSweeps is the startup sweep over the two roots that used to grow without
// bound (cl-091): closed workers' roots and tmp/. Nothing is mocked — a real layout, a
// real records store, real directories, and the product's own reclaim package. Each root
// proves both halves: what nothing references goes, and what a row or a live process
// still claims stays. There is no attempt root to sweep: a local attempt writes its
// result straight into the store and stages nothing.
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
	if _, err := os.Stat(filepath.Join(l.Root, "attempts")); !os.IsNotExist(err) {
		t.Fatalf("the layout still creates an attempt root: %v", err)
	}

	// ---- two request rows the tmp sweep reads: one settled, one still open.
	if problem := store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-attempts", Package: "cozy/sweep",
		WorkerID: "local",
		Devices:  []string{"cpu"}}); problem != nil {
		t.Fatal(problem)
	}
	settledRun(t, l, store, "req-exported")
	submitSweepRequest(t, store, "req-open")
	if _, problem := store.Dispatch(records.Attempt{RequestID: "req-open", SessionID: "s-open", InstanceID: "ins-attempts",
		InvocationDigest: "sha256:" + sixtyFour("c"), InvocationCanonical: []byte("{}")}); problem != nil {
		t.Fatal(problem)
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
			WorkerID: "local",
			Devices:  []string{"cuda:" + string(rune('1'+index))}}); problem != nil {
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

	// ---- tmp: a claim held by a live process keeps its scratch; a crashed process's
	// leftover, an unclaimed directory and a stray file go; an entry named for a request
	// that has not settled (req-open) stays, one named for a settled request goes; locks/
	// stays.
	held, problem := scratch.Temp(l.Tmp, "package-install-")
	if problem != nil {
		t.Fatal(problem)
	}
	defer held.Release()
	must(t, os.WriteFile(filepath.Join(held.Path, "wheel"), make([]byte, 100), 0o644))
	// A crashed process's scratch: the claim file is there, the kernel lock died with it.
	leftoverPath := filepath.Join(l.Tmp, "invoke-models-crashed")
	must(t, os.MkdirAll(leftoverPath, 0o700))
	must(t, os.WriteFile(filepath.Join(leftoverPath, ".claim"), nil, 0o600))
	must(t, os.WriteFile(filepath.Join(leftoverPath, "manifest.bin"), make([]byte, 2048), 0o644))
	must(t, os.MkdirAll(filepath.Join(l.Tmp, "locks"), 0o700))
	must(t, os.WriteFile(filepath.Join(l.Tmp, "locks", sixtyFour("e")+".lock"), nil, 0o600))
	must(t, os.MkdirAll(filepath.Join(l.Tmp, "unclaimed-dir"), 0o700))
	must(t, os.WriteFile(filepath.Join(l.Tmp, "stray-evidence.jsonl"), []byte("{}\n"), 0o600))
	liveRequest := filepath.Join(l.Tmp, "req-open")
	must(t, os.MkdirAll(liveRequest, 0o700))
	must(t, os.WriteFile(filepath.Join(liveRequest, "model.safetensors.part"), make([]byte, 512), 0o600))
	settledRequest := filepath.Join(l.Tmp, "req-exported")
	must(t, os.MkdirAll(settledRequest, 0o700))
	must(t, os.WriteFile(filepath.Join(settledRequest, "model.safetensors"), make([]byte, 512), 0o600))

	tmp, problem := reclaim.Tmp(l, store)
	if problem != nil {
		t.Fatal(problem)
	}
	if tmp.Scanned != 6 || tmp.Removed != 4 || tmp.Bytes < 2048+512 {
		t.Fatalf("tmp sweep = %+v, want 6 scanned, 4 removed, the leftovers' bytes freed", tmp)
	}
	for _, gone := range []string{leftoverPath, filepath.Join(l.Tmp, "unclaimed-dir"),
		filepath.Join(l.Tmp, "stray-evidence.jsonl"), settledRequest} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Fatalf("dead tmp entry %s remains: %v", gone, err)
		}
	}
	for _, kept := range []string{filepath.Join(held.Path, "wheel"), liveRequest,
		filepath.Join(l.Tmp, "locks", sixtyFour("e")+".lock")} {
		if _, err := os.Stat(kept); err != nil {
			t.Fatalf("held, live-request or lock entry %s was swept: %v", kept, err)
		}
	}

	// Every sweep is idempotent: a second pass over the survivors removes nothing.
	againWorkers, _ := reclaim.Workers(l, store)
	againTmp, _ := reclaim.Tmp(l, store)
	if againWorkers.Removed+againTmp.Removed != 0 {
		t.Fatalf("second sweeps removed something: %+v %+v", againWorkers, againTmp)
	}
}

// settledRun drives one request through the store to a closed, succeeded attempt whose
// single output sits in the package's store under its digest name — where the worker
// wrote it — and settles its export row the way the daemon does.
func settledRun(t *testing.T, l home.Layout, store *records.Store, requestID string) {
	t.Helper()
	submitSweepRequest(t, store, requestID)
	session := "session-" + requestID
	digest := "sha256:" + sixtyFour("a")
	attempt, problem := store.Dispatch(records.Attempt{RequestID: requestID, SessionID: session, InstanceID: "ins-attempts",
		InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(requestID, attempt, session))
	fatal(t, store.Accepted(requestID, attempt, session))
	published := filepath.Join(l.PackageOutputs("cozy/sweep"), sixtyFour("0")+".png")
	must(t, os.MkdirAll(filepath.Dir(published), 0o755))
	must(t, os.WriteFile(published, make([]byte, 1024), 0o644))
	if _, problem := store.AcceptTerminal(records.Terminal{RequestID: requestID, Attempt: attempt,
		SessionID: session, InvocationDigest: digest, TerminalID: "out-" + requestID,
		TerminalDigest: "sha256:" + sixtyFour("f"), Status: "SUCCEEDED", Cause: "COMPLETED",
		Outputs: []records.Output{{OutputID: "image", MediaID: records.NewID("med"), Path: published,
			Digest: "sha256:" + sixtyFour("0"), Length: 1024, MimeType: "image/png"}},
		EventType: "request.succeeded", EventPayload: map[string]any{}, RequestState: "succeeded",
	}); problem != nil {
		t.Fatal(problem)
	}
	fatal(t, store.Closed(requestID, attempt))
	fatal(t, store.SettleOutputExport(requestID, []string{published}, nil))
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
