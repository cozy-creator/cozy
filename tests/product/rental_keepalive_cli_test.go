package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
)

// Ordinary CLI -> actual daemon/records -> the rental's machine over cozy.machine.v1. Only an
// explicit keepalive moves the machine's deadline and the local clock: listing and a daemon
// restart never do, and an ending rental is not kept alive.
func TestRentalKeepaliveCLIResetsOnlyAfterAcknowledgment(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: a machine serving cozy.machine.v1")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv lays out the machine root")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czk")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		}
	})
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	launch, identity, token, _ := providerHost(t, h, layout, source, uv)
	cert := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf}))
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 1,
		"worker_address": launch.Addr, "media_address": launch.MediaAddr}
	h.mu.Unlock()
	row := records.Rental{ID: parityRental, MachineName: "tessa", SKU: "cpu", State: "ready",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: launch.Addr, MediaAddress: launch.MediaAddr, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID}
	fatal(t, rental.Attach(layout, store, row, cert, secret.New(token), identity))
	daemon := startDaemonProcess(t, root)

	// The machine's own deadline, read over its API with the rental's owner key.
	pin, err := workertls.ParsePin([]byte(cert))
	must(t, err)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	machine, err := machinev1.Dial(launch.Addr, pin.TLSConfig(), launch.WorkerID, machinev1.Signer{Public: public, Sign: identity.Sign})
	must(t, err)
	defer machine.Close()
	held := func() int64 {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		frame, err := machine.Status(ctx)
		must(t, err)
		return frame.IdleDeadlineUnixMs
	}
	keepalive := func() time.Time {
		t.Helper()
		code, out := runCozy(t, root, "rental", "keepalive", "tessa", "--json", "--full")
		if code != 0 {
			t.Fatalf("keepalive CLI: %d %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
		}
		var result struct {
			ReleaseDue string `json:"release_due"`
		}
		must(t, json.Unmarshal([]byte(out), &result))
		due, err := time.Parse(time.RFC3339Nano, result.ReleaseDue)
		if err != nil {
			t.Fatalf("missing acknowledged deadline: %s", out)
		}
		return due
	}
	baseline := func() time.Time {
		t.Helper()
		current, p := store.RentalRow(parityRental)
		fatal(t, p)
		idle, p := rental.ObserveIdle(store, *current)
		fatal(t, p)
		return idle.Since
	}

	first := keepalive()
	if first.UnixMilli() != held() {
		t.Fatalf("the CLI reported %s, the machine holds %d", first, held())
	}
	initial := baseline()
	time.Sleep(20 * time.Millisecond)
	second := keepalive()
	renewed := baseline()
	if !second.After(first) || !renewed.After(initial) || second.UnixMilli() != held() {
		t.Fatal("an explicit keepalive did not reset the machine and the local clock again")
	}
	if code, out := runCozy(t, root, "rental", "list", "--json", "--no-watch"); code != 0 {
		t.Fatalf("list: %s", out)
	}
	daemon = crashAndRestartTransactionDaemon(t, daemon)
	if !baseline().Equal(renewed) || held() != second.UnixMilli() {
		t.Fatal("listing or a daemon restart renewed the rental")
	}
	_ = daemon
	if code, _ := runCozy(t, root, "rental", "keepalive", "tessa", "--duration", "0"); code == 0 {
		t.Fatal("CLI duration override admitted")
	}

	current, problem := store.RentalRow(parityRental)
	fatal(t, problem)
	recorded := *current

	// An ending rental is never kept alive, and the machine is not asked.
	recorded.State = "release_requested"
	fatal(t, store.RecordRental(recorded))
	before := held()
	if code, out := runCozy(t, root, "rental", "keepalive", "tessa", "--json"); code == 0 {
		t.Fatalf("ending rental kept alive: %s", out)
	}
	if held() != before {
		t.Fatal("an ending rental's machine was reset")
	}
}
