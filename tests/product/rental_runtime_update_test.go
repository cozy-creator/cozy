package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestRentalRuntimeUpdateJournalKeepsDispatchClosedAcrossRestart(t *testing.T) {
	f := updateFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	update, problem := f.store.BeginRuntimeUpdate(f.rentalID, updateBootID, nil)
	fatal(t, problem)
	update.Selection = []byte(`{"wheels":[]}`)
	update.State = "updating"
	fatal(t, f.store.SaveRuntimeUpdate(*update))
	f.store.Close()
	reopened, problem := records.Open(f.layout.DB)
	fatal(t, problem)
	defer reopened.Close()
	c, problem := orchestrator.Open(orchestrator.Options{Store: reopened, Layout: f.layout, Rentals: rental.Resolver(f.layout, reopened)})
	fatal(t, problem)
	defer c.Close(orchestrator.StopGrace)
	if _, problem := c.UseRental(f.rentalID, "run 7 reading its execution"); problem == nil || problem.ErrName() != "rental.maintenance" {
		t.Fatalf("unfinished update did not retain dispatch hold: %v", problem)
	}
	other, problem := c.UseRental("another-rental", "run 8 preparing")
	fatal(t, problem)
	other()
	row, problem := reopened.RuntimeUpdate(f.rentalID)
	fatal(t, problem)
	if row.ID != update.ID || row.State != "updating" {
		t.Fatalf("update was not durable: %+v", row)
	}
	row.State = "waiting_activation"
	fatal(t, reopened.SaveRuntimeUpdate(*row))
	if _, problem := c.UseRental(f.rentalID, "run 9 preparing"); problem == nil || problem.ErrName() != "rental.maintenance" {
		t.Fatalf("pending activation reopened dispatch: %v", problem)
	}
	row.State = "failed"
	row.Error = "candidate rolled back"
	fatal(t, reopened.SaveRuntimeUpdate(*row))
	release, problem := c.UseRental(f.rentalID, "run 7 reading its execution")
	fatal(t, problem)
	release()
	release()
	row.ID = "stale-update"
	if problem := reopened.SaveRuntimeUpdate(*row); problem == nil {
		t.Fatal("stale updater overwrote the current operation")
	}
}

// An update never waits on the daemon's own uses of the rental: the machine waits for its
// running work before it restarts. While the update's record holds the rental, new uses wait;
// other rentals are not held.
func TestRentalMaintenanceLeavesRunningWorkToTheMachine(t *testing.T) {
	f := updateFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	c, problem := orchestrator.Open(orchestrator.Options{Store: f.store, Layout: f.layout, Rentals: rental.Resolver(f.layout, f.store)})
	fatal(t, problem)
	defer c.Close(orchestrator.StopGrace)
	release, problem := c.UseRental(f.rentalID, "run 7 reading its execution")
	fatal(t, problem)
	defer release()
	_, problem = f.store.BeginRuntimeUpdate(f.rentalID, updateBootID, nil)
	fatal(t, problem)
	called := false
	fatal(t, c.MaintainRental(context.Background(), f.rentalID, func(context.Context, *orchestrator.WorkerConnection) *exit.Error {
		called = true
		if _, problem := c.UseRental(f.rentalID, "run 8 preparing"); problem == nil || problem.ErrName() != "rental.maintenance" {
			t.Fatalf("a new use was admitted during the update: %v", problem)
		}
		other, problem := c.UseRental("another-rental", "run 9 preparing")
		fatal(t, problem)
		other()
		return nil
	}))
	if !called {
		t.Fatal("the update waited on the daemon's own use of the rental")
	}
}

func TestRuntimeUpdateInitialCandidateSurvivesBeforePlan(t *testing.T) {
	f := updateFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	candidate := filepath.Join(f.layout.Tmp, "runtime-candidate-proof", "cozy_runtime.whl")
	planned := filepath.Join(f.layout.Tmp, "runtime-updates", "operation", "cozy_runtime.whl")
	tensorfs := filepath.Join(f.layout.Tmp, "runtime-candidate-tensorfs", "tensorfs.whl")
	for _, path := range []string{candidate, planned, tensorfs} {
		must(t, os.MkdirAll(filepath.Dir(path), 0700))
		must(t, os.WriteFile(path, []byte("exact frozen candidate"), 0600))
	}
	selection, err := json.Marshal(map[string]any{"local_runtime": map[string]any{"path": candidate, "digest": "sha256:exact", "length": 22}, "local_tensorfs": map[string]any{"path": tensorfs, "digest": "sha256:tensorfs", "length": 22}})
	must(t, err)
	initial, problem := f.store.BeginRuntimeUpdate(f.rentalID, updateBootID, selection)
	fatal(t, problem)
	f.store.Close()
	reopened, problem := records.Open(f.layout.DB)
	fatal(t, problem)
	defer reopened.Close()
	recovered, problem := reopened.RuntimeUpdate(f.rentalID)
	fatal(t, problem)
	_, problem = reclaim.Tmp(f.layout, reopened)
	fatal(t, problem)
	for _, path := range []string{candidate, planned, tensorfs} {
		data, err := os.ReadFile(path)
		must(t, err)
		if string(data) != "exact frozen candidate" {
			t.Fatal("restart sweep lost active update bytes")
		}
	}
	if recovered.ID != initial.ID || recovered.State != "preparing" || !bytes.Equal(recovered.Selection, selection) {
		t.Fatalf("initial local candidate lost before remote planning: %+v", recovered)
	}
	recovered.State = "failed"
	fatal(t, reopened.SaveRuntimeUpdate(*recovered))
	_, problem = reclaim.Tmp(f.layout, reopened)
	fatal(t, problem)
	for _, path := range []string{candidate, planned, tensorfs} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("settled update scratch retained: %s: %v", path, err)
		}
	}
}

// The real CLI/daemon snapshots before any remote maintenance. The independent
// peer is deliberately unavailable: this test needs no worker or wheel install.
func TestRentalRuntimeUpdateCLIFreezesLocalCandidate(t *testing.T) {
	f := updateFixtureAt(t)
	f.attach(t, "127.0.0.1:1")
	must(t, os.WriteFile(filepath.Join(f.layout.Root, "config.yaml"), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: local-update-proof\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	source := filepath.Join(t.TempDir(), "cozy_runtime-0.18.25.dev1-cp312-abi3-manylinux_2_28_x86_64.whl")
	content := []byte("bounded immutable candidate bytes; metadata checked during planning")
	must(t, os.WriteFile(source, content, 0600))
	tensorfsSource := filepath.Join(t.TempDir(), "tensorfs-0.3.54+dev.g9d02fc3-cp312-abi3-manylinux_2_28_x86_64.whl")
	must(t, os.WriteFile(tensorfsSource, content, 0600))
	startDaemonProcess(t, f.layout.Root)
	code, out := runCozy(t, f.layout.Root, "rental", "update", "proof", "--tensorfs-wheel", tensorfsSource, "--json")
	if code == 0 || !strings.Contains(out, "requires --runtime-wheel") {
		t.Fatalf("unpaired TensorFS was admitted: %d %s", code, out)
	}
	code, out = runCozy(t, f.layout.Root, "rental", "update", "proof", "--runtime-wheel", source, "--tensorfs-wheel", tensorfsSource, "--json")
	if code == 0 || !strings.Contains(out, "rental.runtime_update_failed") {
		t.Fatalf("expected remote maintenance refusal after local capture: %d %s", code, out)
	}
	row, problem := f.store.RuntimeUpdate(f.rentalID)
	fatal(t, problem)
	if row == nil {
		t.Fatal("candidate accepted without a durable row")
	}
	var selection struct {
		LocalRuntime struct {
			Path, Filename, Digest string
			Length                 int64
		} `json:"local_runtime"`
		LocalTensorFS struct {
			Path, Filename, Digest string
			Length                 int64
		} `json:"local_tensorfs"`
	}
	must(t, json.Unmarshal(row.Selection, &selection))
	if selection.LocalTensorFS.Path == tensorfsSource || selection.LocalTensorFS.Filename != filepath.Base(tensorfsSource) || selection.LocalTensorFS.Digest != fmt.Sprintf("sha256:%x", sha256.Sum256(content)) {
		t.Fatalf("TensorFS candidate was not frozen: %+v", selection.LocalTensorFS)
	}
	must(t, os.WriteFile(tensorfsSource, []byte("replaced TensorFS build output"), 0600))
	frozenTensorFS, err := os.ReadFile(selection.LocalTensorFS.Path)
	must(t, err)
	if !bytes.Equal(frozenTensorFS, content) {
		t.Fatal("mutable caller path changed the recorded TensorFS candidate")
	}
	frozen := selection.LocalRuntime
	if frozen.Path == source || frozen.Filename != filepath.Base(source) || frozen.Length != int64(len(content)) || frozen.Digest != fmt.Sprintf("sha256:%x", sha256.Sum256(content)) {
		t.Fatalf("candidate was not frozen with its exact identity: %+v", frozen)
	}
	must(t, os.WriteFile(source, []byte("replaced build output"), 0600))
	actual, err := os.ReadFile(frozen.Path)
	must(t, err)
	if !bytes.Equal(actual, content) {
		t.Fatal("mutable caller path changed the recorded candidate")
	}
	info, err := os.Stat(frozen.Path)
	must(t, err)
	if info.Mode().Perm() != 0600 {
		t.Fatalf("snapshot mode %o", info.Mode().Perm())
	}
	// The Kong path is independently checked without starting another maintenance.
	code, out = runCozy(t, f.layout.Root, "rental", "update", "--help")
	if code != 0 || (!strings.Contains(out, "--runtime-wheel") || !strings.Contains(out, "--tensorfs-wheel")) {
		t.Fatalf("local wheel flag missing: %d %s", code, out)
	}
}
