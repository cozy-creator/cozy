package producttest

import (
	"database/sql"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// cpuSKU is the one product the stand-in hub sells these arms.
var cpuSKU = map[string]any{
	"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
	"price_usd_micros_per_hour": 100_000, "storage_usd_micros_per_hour": 10_000,
	"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86",
}

// refusedAnswer is an accepted ask whose create answer omits the width, so this host
// keeps the paid operation and records no rental row (th-198).
func refusedAnswer(id, state string, asks *atomic.Int32) func(map[string]any) map[string]any {
	return func(request map[string]any) map[string]any {
		asks.Add(1)
		return map[string]any{"rental_id": id, "name": request["name"], "state": state,
			"requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 100_000}
	}
}

// `cozy rental new` waits out another writer holding the records database, where it
// failed at 19:23Z on 2026-09-27 with "cannot begin rental operation: database is locked".
func TestRentalNewWaitsOutARecordsWriter(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-new-busy")
	stand.publishListing()
	stand.setSKUs(cpuSKU)
	var asks atomic.Int32
	stand.rent = refusedAnswer("pr-busywriterproof0001", "pending_acquisition", &asks)
	startDaemonProcess(t, root)

	writer, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite")+"?_txlock=immediate")
	must(t, err)
	defer writer.Close()
	tx, err := writer.Begin()
	must(t, err)
	defer tx.Rollback()
	cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "rental", "new", "cpu", "--json")
	cmd.Env = childEnv(t, root)
	var out strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &out
	must(t, cmd.Start())
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	// Hold the writer well past the five-second SQLite busy timeout the ask used to fail at.
	// The command's first record write (its fleet reconcile) waits; nothing is asked yet.
	select {
	case <-done:
		t.Fatalf("rental new ended while another writer held the records:\n%s", out.String())
	case <-time.After(8 * time.Second):
	}
	if asks.Load() != 0 {
		t.Fatalf("rental new asked for a pod before it could record the ask:\n%s", out.String())
	}
	must(t, tx.Commit())
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("rental new did not continue after the writer committed")
	}
	if strings.Contains(out.String(), "database is locked") || asks.Load() != 1 {
		t.Fatalf("the ask did not reach the hub once after the writer let go (%d asks):\n%s", asks.Load(), out.String())
	}
}

// A paid ask whose answer was refused has no rental row. Once the Hub reports that
// rental gone, the next reconcile settles the ask; `rental list` stops showing it.
func TestARefusedAskSettlesOnceTheHubReportsItGone(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-refused-gone")
	stand.publishListing()
	stand.setSKUs(cpuSKU)
	const id = "pr-refusedaskproof0001"
	var asks atomic.Int32
	stand.rent = refusedAnswer(id, "release_requested", &asks)
	startDaemonProcess(t, root)
	if code, out := runCozy(t, root, "rental", "new", "cpu", "--json"); code == 0 || asks.Load() != 1 {
		t.Fatalf("the width fence did not refuse the answer [exit %d]\n%s", code, out)
	}
	stand.setState(id, "released", "")

	code, board := runCozy(t, root, "rental", "list", "--json")
	var listed struct {
		Unattached int `json:"unattached_rental_operations"`
	}
	if code != 0 || json.Unmarshal([]byte(board), &listed) != nil || listed.Unattached != 0 {
		t.Fatalf("a rental the hub reports gone is still a paid ask [exit %d]\n%s", code, board)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	open, problem := store.ActiveRentalOperations()
	fatal(t, problem)
	if len(open) != 0 {
		t.Fatalf("the settled ask is still open: %+v", open)
	}
}

// A deleted editable source is reported once and no longer watched; a source whose local
// dependency is missing is reported once, not on every records write.
func TestVanishedEditableSourcesAreReportedOnce(t *testing.T) {
	root, sources := t.TempDir(), t.TempDir()
	deleted, broken, live := filepath.Join(sources, "deleted"), filepath.Join(sources, "broken"), filepath.Join(sources, "live")
	for dir, pyproject := range map[string]string{
		broken: "[project]\nname = \"broken\"\nversion = \"0.1.0\"\ndependencies = [\"gone-dep\"]\n\n[tool.uv.sources]\ngone-dep = { path = \"../gone-dep\" }\n",
		live:   "[project]\nname = \"live\"\nversion = \"0.1.0\"\n",
	} {
		must(t, os.MkdirAll(dir, 0o755))
		must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(pyproject), 0o644))
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	// A pin moves under the install writer, as `cozy package install` moves it.
	activate := func(name, source string) {
		t.Helper()
		writer, problem := home.LockWriter(layout)
		fatal(t, problem)
		defer writer.Unlock()
		_, problem = store.Activate(records.PackageInstall{ID: "editable-" + name, Package: "local/" + name, Version: "0.1.0",
			SourceKind: "local", SourceRef: source, ProjectDir: source, Dir: layout.InstallDir("editable-" + name)})
		fatal(t, problem)
	}
	activate("deleted", deleted)
	activate("broken", broken)
	startDaemonProcess(t, root)
	// Every records write rescans every source.
	for i := range 5 {
		fatal(t, store.AppendPackageEvent("local/broken", "proof.write", map[string]any{"write": i}))
	}
	activate("live", live)
	log := filepath.Join(root, "daemon.log")
	var raw []byte
	// A daemon niced on a loaded box can take longer than waitFor's 20 s to rescan.
	for deadline := time.Now().Add(3 * time.Minute); !strings.Contains(string(raw), "editable local/live: watching "+live); {
		if time.Now().After(deadline) {
			t.Fatalf("the daemon never rescanned the live source:\n%s", raw)
		}
		time.Sleep(100 * time.Millisecond)
		raw, _ = os.ReadFile(log)
	}
	if n := strings.Count(string(raw), deleted); n != 1 {
		t.Fatalf("the deleted source was mentioned %d times, want once:\n%s", n, raw)
	}
	if n := strings.Count(string(raw), "dependency watch scan refused"); n != 1 {
		t.Fatalf("the missing dependency was reported %d times, want once:\n%s", n, raw)
	}
}
