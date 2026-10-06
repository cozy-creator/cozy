package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// A run on a rental belongs to the rental's machine, not to the daemon that sent it. While the
// daemon is down the run goes on and its bytes arrive after the restart; a cancel recorded
// meanwhile reaches the machine; and no restart ends the rental, which takes new work after.
func TestV1RentalRunSurvivesDaemonRestart(t *testing.T) {
	h, root, layout, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", nativeLifecycleProject(t), "--editable"); code != 0 {
		t.Fatalf("installing authored lifecycle fixture [%d]: %s", code, out)
	}
	found := &machines.Resolver{Host: machines.NewHost(layout.Machine, "", nil), HubOrigin: h.server.URL,
		Rentals: rental.Resolver(layout, store), UseRental: func(string, string) (func(), *exit.Error) { return func() {}, nil },
		RentalKey: func(id string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentityFor(layout, id) }}
	machine, problem := found.DialV1(t.Context(), parityRental, "rental daemon restart test")
	fatal(t, problem)
	t.Cleanup(machine.Close)
	start := func(key string) (*records.Request, string, string) {
		gate, directory := filepath.Join(root, key+"-gate"), filepath.Join(root, key+"-out")
		if code, out := runCozy(t, root, "run", "local/native-lifecycle/make", "gate="+gate, "size=200000",
			"--idempotency-key="+key, "--out="+directory, "--json", "--rental=tessa"); code != 0 {
			t.Fatalf("submitting %s to the rental [%d]: %s", key, code, out)
		}
		row := nativeLifecycleRequest(t, store, key)
		landed(t, key+" running on the rental", func() bool { _, err := os.Stat(gate + ".entered"); return err == nil })
		return row, gate, directory
	}
	down := func() {
		if code, out := runCozy(t, root, "down", "--json"); code != 0 {
			t.Fatalf("stopping the daemon [%d]: %s", code, out)
		}
	}
	up := func() {
		if code, out := runCozy(t, root, "up", "--json"); code != 0 {
			t.Fatalf("restarting the daemon [%d]: %s", code, out)
		}
	}

	t.Log("rental bytes after daemon shutdown")
	row, gate, directory := start("rental-down-bytes")
	down()
	must(t, os.WriteFile(gate, nil, 0o600))
	if ended := nativeLifecycleOutcome(t, machine, row.ID); ended != "succeeded" {
		t.Fatalf("the daemon's shutdown changed the rental run's outcome: %s", ended)
	}
	up()
	if code, out := runCozy(t, root, "run", "watch", row.ID, "--json"); code != 0 {
		t.Fatalf("reattaching to the rental run [%d]: %s", code, out)
	}
	nativeLifecycleBytes(t, directory, 200000)

	t.Log("durable cancel recorded while the daemon is down")
	row, _, _ = start("rental-down-cancel")
	down()
	_, problem = store.RequestMachineCancellation(row.ID, "cozy run cancel")
	fatal(t, problem)
	up()
	landed(t, "the cancel projected after reattach", func() bool {
		row, _ := store.RequestRow(row.ID)
		return row != nil && row.State == "canceled"
	})
	if ended := nativeLifecycleOutcome(t, machine, row.ID); ended != "canceled" {
		t.Fatalf("the cancel recorded while the daemon was down did not reach the rental: %s", ended)
	}

	if released := h.releases(parityRental); released != 0 {
		t.Fatalf("a daemon restart ended the rental (%d release(s))", released)
	}
	warm := filepath.Join(root, "after-restart-gate")
	must(t, os.WriteFile(warm, nil, 0o600))
	if code, out := runCozy(t, root, "run", "local/native-lifecycle/make", "gate="+warm, "size=64",
		"--idempotency-key=rental-after-restart", "--rental=tessa", "--await", "--json"); code != 0 {
		t.Fatalf("the rental took no work after the restarts [%d]: %s", code, out)
	}
}
