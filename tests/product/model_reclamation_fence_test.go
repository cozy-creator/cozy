package producttest

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/units"
)

// TestReclamationWaitsOnlyForWhatHoldsBytes: stale retained runs (a blocked job, a paused
// one) no longer fence `cozy model gc` or `cozy model remove`; they keep only the writerless
// ingest session a retry may resume. A live local run refuses a pass by run number, and a
// run that uses a model refuses only that model's removal.
func TestReclamationWaitsOnlyForWhatHoldsBytes(t *testing.T) {
	root := filepath.Join(scratchBase, "reclamation-fence")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	must(t, os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	fatal(t, e)
	cfg.Home = root
	cfg.TensorFSRoot = filepath.Join(root, "tensorfs")
	tool, e := tfs.Open(cfg)
	if e != nil && e.Name == "tfs_missing" {
		t.Skipf("no tensorfs CLI on this runner: %s", e.Message)
	}
	fatal(t, e)
	fixture := tfsStore{t: t, bin: tool.Bin, env: cfg.Tool(), root: tool.Root, work: t.TempDir()}
	model := func(name string, own byte) {
		blob := fixture.put(name+".bin", bytes.Repeat([]byte{own}, 300))
		header := fmt.Sprintf(`{"format":"cozytensors/1","configs":[],"assets":[],"encodings":[%s],`+
			`"components":[["model",[["own","f32",[75],0,[["value","f32",[75],{"segments":[["sha256:%s",300]]}]]]]]]}`,
			plainEncoding, blob.sha256)
		manifest := fixture.reproduce(name, header, `[["model.cozytensors","cozytensors"]]`, `[["model","own"]]`)
		if _, e := tool.ReplaceLocal(name, "sha256:"+blob.sha256, "absent", "sha256:"+manifest.sha256,
			manifest.length); e != nil {
			t.Fatalf("local/%s: %s", name, briefly(e))
		}
	}
	model("alpha", 1)
	model("beta", 2)
	orphan := fixture.put("orphan.bin", bytes.Repeat([]byte{9}, 1000))
	resumable := fixture.put("session.bin", bytes.Repeat([]byte{8}, 700))
	// A crashed ingest's candidate root: no live writer, so only reaping abandons it.
	session := filepath.Join(tool.Root, "tmp", "ingest", "0123456789abcdef")
	must(t, os.MkdirAll(session, 0o755))
	must(t, os.WriteFile(filepath.Join(session, "session.json"), []byte(fmt.Sprintf(
		`{"candidates":[{"length":%d,"sha256":"%s"}],"session":"0123456789abcdef","tenant":"dev"}`,
		resumable.length, resumable.sha256)), 0o644))

	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	submit := func(id string, models []records.ModelRef) {
		t.Helper()
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Kind: "job", Package: "local/example",
			Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64),
			RetainWork: true, Models: models})
		fatal(t, problem)
	}
	submit("job-blocked", nil)
	if changed, problem := store.BlockRetainedWork("job-blocked", "tfs_refused", "stale since September"); problem != nil || !changed {
		t.Fatalf("the retained stop did not block: %v", problem)
	}

	// Run 1 is a retained stop: gc reclaims what nothing holds and keeps its session.
	code, out := runCozy(t, root, "model", "gc")
	if code != 0 || !strings.Contains(out, "reclaimed: "+units.Bytes(orphan.length)) ||
		!strings.Contains(out, "ingest sessions kept for retained run 1") {
		t.Fatalf("model gc beside a retained stop [exit %d]: want %s reclaimed and the session kept\n%s",
			code, units.Bytes(orphan.length), out)
	}
	if fixture.present(orphan.sha256) || !fixture.present(resumable.sha256) {
		t.Fatalf("gc beside a retained stop must take the orphan and keep the resumable session")
	}
	code, out = runCozy(t, root, "model", "remove", "local/alpha")
	if code != 0 || !strings.Contains(out, "reclaimed: ") {
		t.Fatalf("model remove beside a retained stop [exit %d]\n%s", code, out)
	}

	// Run 2 uses local/beta: its removal is refused by number; nothing else is.
	submit("job-uses-beta", []records.ModelRef{{Package: "local/example", Slot: "model", Model: "local/beta"}})
	code, out = runCozy(t, root, "model", "remove", "local/beta")
	if code == 0 || !strings.Contains(out, "run 2 (queued) still uses local/beta") ||
		!strings.Contains(out, "cozy run cancel 2") {
		t.Fatalf("model remove of a model a queued run uses [exit %d]\n%s", code, out)
	}
	if code, out = runCozy(t, root, "model", "gc"); code != 0 {
		t.Fatalf("a queued run has moved no bytes and must not fence gc [exit %d]\n%s", code, out)
	}

	// Run 2 starts: a live local attempt may be moving unnamed bytes in, so a pass waits.
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "local-worker", Package: "local/example",
		WorkerID: "local-worker", Devices: []string{"cpu"}}))
	_, problem = store.Dispatch(records.Attempt{RequestID: "job-uses-beta", InstanceID: "local-worker",
		SessionID: "boot", InvocationDigest: "sha256:" + strings.Repeat("2", 64), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	code, out = runCozy(t, root, "model", "gc")
	if code == 0 || !strings.Contains(out, "run 2 (in_progress) may still be moving bytes into the store") {
		t.Fatalf("model gc beside a live local run [exit %d]\n%s", code, out)
	}

	// Both settle: the next pass reaps the abandoned session and reclaims its bytes.
	fatal(t, store.SettleRequest("job-uses-beta", "canceled"))
	fatal(t, store.SettleRequest("job-blocked", "canceled"))
	fatal(t, store.CloseWorker("local-worker"))
	code, out = runCozy(t, root, "model", "gc")
	if code != 0 || !strings.Contains(out, "reclaimed: "+units.Bytes(resumable.length)) {
		t.Fatalf("model gc after the runs settled [exit %d]: want the session's %s\n%s",
			code, units.Bytes(resumable.length), out)
	}
	if fixture.present(resumable.sha256) {
		t.Fatalf("the abandoned session's bytes survived a pass with nothing to resume them")
	}
}
