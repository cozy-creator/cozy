package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
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
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// statusRental gives a fresh root a ready rental named "tessa" whose machine is the real one
// (-machine-host) behind a stand-in Hub, and a running daemon.
func statusRental(t *testing.T) (string, *records.Store, *machines.Launch, rental.CreatorIdentity, *daemonProcess) {
	t.Helper()
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
	t.Cleanup(store.Close)
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
	return root, store, launch, identity, daemon
}

// Ordinary CLI -> actual daemon/records -> the rental's machine over cozy.machine.v1. Only an
// explicit keepalive moves the machine's deadline and the local clock: listing and a daemon
// restart never do, and an ending rental is not kept alive.
func TestRentalKeepaliveCLIResetsOnlyAfterAcknowledgment(t *testing.T) {
	root, store, launch, identity, daemon := statusRental(t)
	cert := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf}))

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
	if *olderCozy != "" {
		// Version skew: a daemon that predates cozy.machine.v1 cannot keep a Rust rental
		// alive, so the command does it itself over the rental's pinned machine.
		if code, out := runCozy(t, root, "down"); code != 0 {
			t.Fatalf("down [exit %d]\n%s", code, out)
		}
		up := exec.Command(*olderCozy, "up")
		up.Env = childEnv(t, root)
		if raw, err := up.CombinedOutput(); err != nil {
			t.Fatalf("the older cozy did not start its daemon: %v\n%s", err, raw)
		}
		time.Sleep(20 * time.Millisecond)
		if third := keepalive(); !third.After(second) || third.UnixMilli() != held() || !baseline().After(renewed) {
			t.Fatal("keepalive under the older daemon did not reset the machine and the local clock")
		}
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

// `cozy rental show` names what the rental's machine reports over Status: its agent, Runtime,
// TensorFS and phase. The local reader of a running machine reads the same picture, and a
// machine that cannot answer leaves a note, never a failed show.
func TestRentalShowReportsTheMachinesStatus(t *testing.T) {
	root, _, launch, _, _ := statusRental(t)
	version := func(wheel, distribution string) string {
		name := strings.TrimPrefix(filepath.Base(wheel), distribution+"-")
		return name[:strings.Index(name, "-")]
	}
	runtime, tensorfs := version(*machineRuntimeWheel, "cozy_runtime"), version(*machineTensorFSWheel, "tensorfs")
	code, out := runCozy(t, root, "rental", "show", "tessa", "--json")
	var shown map[string]any
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil || shown["runtime_version"] != runtime ||
		shown["tensorfs_version"] != tensorfs || shown["agent_version"] == "" || shown["phase"] != "ready" {
		t.Fatalf("rental show did not name the machine's status [exit %d]:\n%s", code, out)
	}
	if code, out = runCozy(t, root, "rental", "show", "tessa"); code != 0 || !strings.Contains(out, runtime) || !strings.Contains(out, "live runs") {
		t.Fatalf("the human rental show hides the status [exit %d]:\n%s", code, out)
	}
	environ, err := os.ReadFile("/proc/" + strconv.Itoa(launch.PID) + "/environ")
	must(t, err)
	var machineRoot string
	for _, entry := range strings.Split(string(environ), "\x00") {
		if value, ok := strings.CutPrefix(entry, "COZY_MACHINE_ROOT="); ok {
			machineRoot = value
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	local, problem := machines.NewHost(filepath.Dir(machineRoot), "", nil).ReadStatus(ctx)
	fatal(t, problem)
	if local == nil || local.WorkerId != launch.WorkerID || local.Runtime != runtime {
		t.Fatalf("the local reader did not read the running machine: %v", local)
	}
	// The test's own provider process: once it is gone the show still answers, with a note.
	process, err := os.FindProcess(launch.PID)
	must(t, err)
	must(t, process.Signal(syscall.SIGTERM))
	waitUntil(t, "the machine stops", func() bool { return syscall.Kill(launch.PID, 0) != nil })
	if code, out = runCozy(t, root, "rental", "show", "tessa", "--json"); code != 0 || strings.Contains(out, "runtime_version") ||
		!strings.Contains(out, "machine status unavailable") {
		t.Fatalf("an unreachable machine must leave a note, not fail the show [exit %d]:\n%s", code, out)
	}
}

// A pod whose machine predates cozy.machine.v1 (the old agent, or a dual-arm pod on that arm)
// is kept alive all the same: the command claims it with the rental's ClaimProof over worker.v1,
// with a daemon running and with none. Version skew is allowed; an older machine is not refused.
func TestRentalKeepaliveWorksOnAMachineThatPredatesV1(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "older-machine")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	var resets int
	var deadline int64
	pod := &fakePod{controlKey: public, keepalive: func(request *pb.KeepRentalAliveRequest) *pb.KeepRentalAliveResult {
		resets++
		now := time.Now()
		deadline = now.Add(15 * time.Minute).UnixMilli()
		return &pb.KeepRentalAliveResult{RequestId: request.RequestId, WorkerId: podWorkerID, WorkerBootId: podBootID,
			AcknowledgedAtUnixMs: now.UnixMilli(), IdleDeadlineUnixMs: deadline}
	}}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "comfy")
	row := records.Rental{ID: podRental, MachineName: "comfy", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	keepalive := func(want int) {
		t.Helper()
		code, out := runCozy(t, root, "rental", "keepalive", "comfy", "--json", "--full")
		var result struct {
			ReleaseDue string `json:"release_due"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &result) != nil || resets != want {
			t.Fatalf("keepalive on an older machine [exit %d, %d resets]\n%s", code, resets, out)
		}
		due, err := time.Parse(time.RFC3339Nano, result.ReleaseDue)
		if err != nil || due.UnixMilli() != deadline {
			t.Fatalf("the command reported %q, the machine holds %d", result.ReleaseDue, deadline)
		}
		current, p := store.RentalRow(podRental)
		fatal(t, p)
		idle, p := rental.ObserveIdle(store, *current)
		fatal(t, p)
		if time.Since(idle.Since) > time.Minute {
			t.Fatalf("the local idle clock was not reset: %s", idle.Since)
		}
	}
	keepalive(1)
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); err == nil {
		t.Fatal("rental keepalive started a daemon")
	}
	startDaemonProcess(t, root)
	keepalive(2)
}
