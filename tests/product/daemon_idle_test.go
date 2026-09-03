package producttest

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestDaemonIdleShutdown is the owner's ruling as behaviour: the daemon leaves on its own
// only once it has had nothing to manage for the configured debounce, and never while a
// rental it owns or a request it owes is on the books. Every arm is the real daemon
// process on a real root reading its real records; the debounce is the one product knob.
func TestDaemonIdleShutdown(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-idle")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("daemon:\n  idle_shutdown_s: 1\n"), 0o600))
	probe := config.Config{Home: root, Port: config.DefaultPort}
	logPath := filepath.Join(root, "daemon.log")

	// (a) Nothing to manage: the daemon says so, leaves cleanly, and releases the root.
	idle := startDaemonProcess(t, root)
	if code := awaitDaemonExit(t, idle, 15*time.Second); code != 0 {
		t.Fatalf("an idle daemon exited %d\n%s", code, tail(logPath))
	}
	if log, _ := os.ReadFile(logPath); !strings.Contains(string(log), "nothing to manage for 1s; stopping") {
		t.Fatalf("the idle exit did not say why it left\n%s", tail(logPath))
	}
	if state := daemon.Probe(probe); state.Up {
		t.Fatalf("the root is still owned after the idle exit: %s", state.Details)
	}

	// (b) A rental row — including one the hub has not yet confirmed released — holds the
	// daemon up past the debounce, and the daemon names it. Forgetting the row, which is
	// what a confirmed release does, is what lets it go.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{
		ID: "rental-idle-arm", MachineName: "heron",
		AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		State: "release_requested", Hub: "https://hub.invalid",
	}))
	rented := startDaemonProcess(t, root)
	select {
	case code := <-rented.exited:
		t.Fatalf("the daemon left (exit %d) while it still owned a rental\n%s", code, tail(logPath))
	case <-time.After(4 * time.Second):
	}
	if log, _ := os.ReadFile(logPath); !strings.Contains(string(log),
		"idle exit held for 1s by: rental rental-idle-arm (release_requested)") {
		t.Fatalf("the daemon did not say what held it up\n%s", tail(logPath))
	}
	if _, problem := store.ForgetRental("rental-idle-arm"); problem != nil {
		t.Fatal(problem.Message)
	}
	if code := awaitDaemonExit(t, rented, 15*time.Second); code != 0 {
		t.Fatalf("the daemon exited %d after its last rental was forgotten\n%s", code, tail(logPath))
	}

	// (c) A request still owed work holds the daemon up; settling it through the daemon's
	// own cancel route is what lets it go.
	queued := startDaemonProcess(t, root)
	if _, _, problem := store.Submit(records.Request{
		ID: "req-idle-arm", IdemKey: "idem-idle-arm", BodyDigest: "sha256:" + strings.Repeat("ab", 32),
		Package: "fake/idle", Entrypoint: "generate", Payload: []byte("{}"),
	}); problem != nil {
		t.Fatal(problem.Message)
	}
	select {
	case code := <-queued.exited:
		t.Fatalf("the daemon left (exit %d) while it still owed a request\n%s", code, tail(logPath))
	case <-time.After(4 * time.Second):
	}
	if r := queued.call(t, "POST", "/v1/requests/req-idle-arm/cancel", nil); r.Status != http.StatusOK {
		t.Fatalf("cancel of the queued request: %s", r.brief())
	}
	if code := awaitDaemonExit(t, queued, 15*time.Second); code != 0 {
		t.Fatalf("the daemon exited %d after its last request settled\n%s", code, tail(logPath))
	}
	if code, out := runCozy(t, root, "down"); code != 0 || !strings.Contains(out, "stopped") {
		t.Fatalf("down after an idle exit is not the idempotent no-op [exit %d]\n%s", code, out)
	}
}

func awaitDaemonExit(t *testing.T, s *daemonProcess, within time.Duration) int {
	t.Helper()
	select {
	case code := <-s.exited:
		return code
	case <-time.After(within):
		t.Fatalf("the daemon did not leave within %s\n%s", within, tail(filepath.Join(s.root, "daemon.log")))
		return -1
	}
}
