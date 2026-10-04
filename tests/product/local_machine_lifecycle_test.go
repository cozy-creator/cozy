package producttest

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/userunit"
)

// This computer's machine is its root's systemd user unit. Whatever that unit runs is the
// root's agent, launch record or not: `cozy machine stop` ends it, `cozy machine show` reports
// it as systemd does, and starting adopts it instead of refusing (2026-10-01: a stop during the
// unit's executor window removed the record and left the agent running, and every later launch
// and install refused with machine.process_untracked).

// serveFakeMachineAgent is this test binary run as a root's cozy-machine: it serves the
// readiness receipt its launcher authenticates, under the launch's key and ports, until stopped.
// It reads its launch as adoption reads an agent's: from the process environment's file.
func serveFakeMachineAgent() {
	raw, _ := os.ReadFile("/proc/self/environ")
	env := map[string]string{}
	for _, pair := range strings.Split(string(raw), "\x00") {
		name, value, _ := strings.Cut(pair, "=")
		env[name] = value
	}
	encoded, err := os.ReadFile(env["COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_FILE"])
	if err != nil {
		os.Exit(2)
	}
	key, err := base64.RawURLEncoding.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		os.Exit(2)
	}
	leafKey, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	template := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: "127.0.0.1"},
		IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1)}, NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}}
	leaf, _ := x509.CreateCertificate(rand.Reader, template, template, &leafKey.PublicKey, leafKey)
	worker, _ := strconv.Atoi(env["COZY_WORKER_INTERNAL_PORT"])
	payload, _ := json.Marshal(map[string]any{"pod_boot_id": fmt.Sprintf("fake-boot-%d", os.Getpid()), "worker_internal_port": worker,
		"tls_certificate_der_base64": base64.StdEncoding.EncodeToString(leaf), "machine_version": "fake"})
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(machines.ReadinessReceiptDomain))
	mac.Write(payload)
	envelope, _ := json.Marshal(map[string]any{"payload": payload, "hmac_sha256": hex.EncodeToString(mac.Sum(nil))})
	listener, err := tls.Listen("tcp", "127.0.0.1:"+env["COZY_MEDIA_INTERNAL_PORT"],
		&tls.Config{Certificates: []tls.Certificate{{Certificate: [][]byte{leaf}, PrivateKey: leafKey}}})
	if err != nil {
		os.Exit(3)
	}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, os.Interrupt)
	go func() {
		_ = http.Serve(listener, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(envelope) }))
	}()
	<-stop
}

// fakeAgentMachine installs root's machine with this test binary as its bundled agent. The
// unit's program waits `executor` before it execs the agent, as systemd's executor does on a
// loaded box: until then the unit's main process is not yet the agent.
func fakeAgentMachine(t *testing.T, root string, executor time.Duration) (*machines.Host, string) {
	t.Helper()
	if !userunit.Available() {
		t.Skip("needs a systemd user manager")
	}
	must(t, os.MkdirAll(root, 0o700))
	// The reaper learns the root before its agent can exist: a test interrupted after the
	// launch, its cleanup never run, still has its agent ended.
	trackDaemonRoot(t, root)
	dir := filepath.Join(root, "machine")
	self, err := os.Executable()
	must(t, err)
	agent := filepath.Join(dir, "root/opt/cozy/python/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(agent), 0o755))
	if os.Link(self, agent) != nil {
		raw, err := os.ReadFile(self)
		must(t, err)
		must(t, os.WriteFile(agent, raw, 0o755)) //cozy:allow the test binary is the fake agent
	}
	program := filepath.Join(dir, "root/usr/local/bin/cozy-machine")
	must(t, os.MkdirAll(filepath.Dir(program), 0o755))
	if executor == 0 {
		must(t, os.Symlink(agent, program))
	} else {
		script := fmt.Sprintf("#!/bin/sh\nsleep %.1f\nexec %s\n", executor.Seconds(), agent)
		must(t, os.WriteFile(program, []byte(script), 0o755)) //cozy:allow stands in for systemd's executor
	}
	metadata := `{"host":{"name":"cozy-machine"},"host_pinned":true}`
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), []byte(metadata), 0o600))
	resolved, err := filepath.EvalSymlinks(dir)
	must(t, err)
	unit := userunit.Name("cozy-machine-agent", resolved, false)
	t.Cleanup(func() { _ = userunit.Stop(unit) })
	return machines.NewHost(dir, "", nil), unit
}

type machineShown struct {
	Running bool `json:"running"`
	PID     int  `json:"pid"`
}

// showAgreesWithSystemd is `cozy machine show` saying what systemd says of the root's unit.
func showAgreesWithSystemd(t *testing.T, root, unit string) machineShown {
	t.Helper()
	code, out := runCozy(t, root, "machine", "show", "--json")
	var shown machineShown
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil {
		t.Fatalf("machine show [exit %d]\n%s", code, out)
	}
	running, pid := userunit.Running(unit), 0
	if running {
		pid = userunit.MainPID(unit)
	}
	if shown.Running != running || shown.PID != pid {
		t.Fatalf("`cozy machine show` says running=%v pid=%d; systemd says running=%v pid=%d for %s",
			shown.Running, shown.PID, running, pid, unit)
	}
	return shown
}

func recorded(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "agent.json"))
	return err == nil
}

// A launch waits out its unit's executor: the Host there is starting, not exited (run 2311
// failed 42 ms after its unit started; TestStopEndsAnAgentStillInSystemdsExecutor sees it
// starting). A unit that ends there never became the agent: its Host exited.
func TestALaunchWhoseUnitEndsInItsExecutorReportsTheHostExited(t *testing.T) {
	root := filepath.Join(t.TempDir(), "home")
	h, unit := fakeAgentMachine(t, root, time.Second)
	program := filepath.Join(root, "machine/root/usr/local/bin/cozy-machine")
	must(t, os.WriteFile(program, []byte("#!/bin/sh\nsleep 1\n"), 0o755)) //cozy:allow an executor that ends before any agent
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, problem := h.Ensure(ctx, "", nil, true); problem == nil || problem.ErrName() != "machine.host_exited" || userunit.Running(unit) {
		t.Fatalf("a launch whose unit ended in its executor answered %v (unit running %v)", problem, userunit.Running(unit))
	}
}

// A launch that gives up while the unit is still in systemd's executor leaves the unit
// starting; `cozy machine stop` then ends that unit, not just its record.
func TestStopEndsAnAgentStillInSystemdsExecutor(t *testing.T) {
	root := filepath.Join(t.TempDir(), "home")
	h, unit := fakeAgentMachine(t, root, 3*time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	_, problem := h.Ensure(ctx, "", nil, true)
	cancel()
	if problem == nil || problem.ErrName() != "machine.host_starting" || !userunit.Running(unit) {
		t.Fatalf("the launch did not give up while its unit was starting: %v (unit running %v)", problem, userunit.Running(unit))
	}
	code, out := runCozy(t, root, "machine", "stop", "--json")
	if code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	if userunit.Running(unit) {
		t.Fatalf("`cozy machine stop` answered %s while %s kept running", strings.TrimSpace(out), unit)
	}
	if recorded(filepath.Join(root, "machine")) {
		t.Fatal("the stopped machine kept its launch record")
	}
	showAgreesWithSystemd(t, root, unit)
}

// An agent whose record an older stop removed is still this root's: show reports it, start
// adopts it in place (the same process, never a refusal), and stop ends it.
func TestAnAgentWithoutItsRecordIsAdoptedAndStopped(t *testing.T) {
	root := filepath.Join(t.TempDir(), "home")
	h, unit := fakeAgentMachine(t, root, 0)
	launched, problem := h.Ensure(context.Background(), "", nil, true)
	fatal(t, problem)
	dir := filepath.Join(root, "machine")
	must(t, os.Remove(filepath.Join(dir, "agent.json")))
	if shown := showAgreesWithSystemd(t, root, unit); !shown.Running || shown.PID != launched.PID {
		t.Fatalf("the unrecorded agent %d is shown as %+v", launched.PID, shown)
	}
	code, out := runCozy(t, root, "machine", "start", "--json")
	if code != 0 {
		t.Fatalf("machine start refused the root's own agent [exit %d]\n%s", code, out)
	}
	if shown := showAgreesWithSystemd(t, root, unit); !shown.Running || shown.PID != launched.PID || !recorded(dir) {
		t.Fatalf("start did not adopt agent %d in place: %+v, recorded %v", launched.PID, shown, recorded(dir))
	}
	if code, out := runCozy(t, root, "machine", "stop", "--json"); code != 0 || userunit.Running(unit) {
		t.Fatalf("machine stop left %s running [exit %d]\n%s", unit, code, out)
	}
	if showAgreesWithSystemd(t, root, unit).Running || recorded(dir) {
		t.Fatal("the stopped machine is still shown running or recorded")
	}
}

// Work the machine already accepted (observing, collecting, controlling) never starts it: it
// attaches while the machine runs and waits while it is stopped. The daemon used to relaunch a
// stopped machine within 3 s to re-read output bytes its Runtime refused, hundreds of times.
func TestAttachingNeverStartsAStoppedMachine(t *testing.T) {
	root := filepath.Join(t.TempDir(), "home")
	h, unit := fakeAgentMachine(t, root, 0)
	attach := machines.AttachOnly(context.Background())
	if _, problem := h.Ensure(attach, "", nil, true); problem == nil || problem.ErrName() != "machine.stopped" || userunit.Running(unit) {
		t.Fatalf("attaching to a stopped machine answered %v (unit running %v)", problem, userunit.Running(unit))
	}
	launched, problem := h.Ensure(context.Background(), "", nil, true)
	fatal(t, problem)
	if attached, problem := h.Ensure(attach, "", nil, true); problem != nil || attached.PID != launched.PID {
		t.Fatalf("attaching to running agent %d answered %+v, %v", launched.PID, attached, problem)
	}
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	if _, problem := h.Ensure(attach, "", nil, true); problem == nil || problem.ErrName() != "machine.stopped" || userunit.Running(unit) {
		t.Fatalf("attaching after stop answered %v (unit running %v)", problem, userunit.Running(unit))
	}
}
