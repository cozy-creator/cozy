package producttest

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

const olderWorkerID, olderBootID, olderRental = "wrk-older-1", "boot-older-1", "pr-older-0001"

// olderMachine is a cozy.machine.v1 machine a release behind this cozy: until it runs `serves`,
// it refuses new calls at the door with UNIMPLEMENTED, as a machine without a call does. It
// still answers the runs it accepted, and Run kind: update moves its software.
type olderMachine struct {
	v1.UnimplementedMachineServer
	mu                sync.Mutex
	runtime, tensorfs string
	serves            string
	updates, refusals int
	runs              map[string]bool
	// held is the run that stays running until release closes.
	held    string
	release chan struct{}
}

func (m *olderMachine) Status(_ *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	m.mu.Lock()
	frame := &v1.StatusFrame{WorkerId: olderWorkerID, BootId: olderBootID, Version: "0.1.0", Phase: "ready",
		Runtime: m.runtime, Tensorfs: m.tensorfs, Capabilities: []string{"status/1", "run/1", "update/1"}}
	m.mu.Unlock()
	return stream.Send(frame)
}

func (m *olderMachine) Run(request *v1.RunRequest, stream grpc.ServerStreamingServer[v1.RunEvent]) error {
	m.mu.Lock()
	spec := request.GetSpec()
	result := []byte(`{}`)
	switch {
	case spec.GetKind() == v1.RunKind_RUN_KIND_UPDATE:
		var to map[string]string
		_ = json.Unmarshal(spec.GetPayload(), &to)
		from := map[string]string{"runtime": m.runtime, "tensorfs": m.tensorfs}
		m.runtime, m.tensorfs, m.updates = to["runtime"], to["tensorfs"], m.updates+1
		result, _ = json.Marshal(map[string]any{"from": from, "to": map[string]string{"runtime": m.runtime, "tensorfs": m.tensorfs}})
	case spec != nil && m.runtime != m.serves:
		m.refusals++
		m.mu.Unlock()
		return status.Error(codes.Unimplemented, "Run takes calls, jobs, warm-ups and updates")
	case spec == nil && !m.runs[request.GetId()]:
		m.mu.Unlock()
		return status.Error(codes.NotFound, "no such run")
	}
	m.runs[request.GetId()] = true
	held := request.GetId() == m.held
	m.mu.Unlock()
	now := time.Now().UnixMilli()
	if request.GetAfter() < 1 {
		if err := stream.Send(&v1.RunEvent{Sequence: 1, AtMs: now, Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.GetId(), Number: 1, State: "running", Sequence: 1}}}); err != nil {
			return err
		}
	}
	if held {
		select {
		case <-m.release:
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
	}
	return stream.Send(&v1.RunEvent{Sequence: 2, AtMs: now, Event: &v1.RunEvent_Outcome{Outcome: &v1.Outcome{Status: "succeeded", Result: result}}})
}

func (m *olderMachine) Control(_ context.Context, request *v1.ControlRequest) (*v1.RunState, error) {
	return &v1.RunState{Id: request.GetId(), Number: 1, State: "succeeded", Sequence: 2}, nil
}

// A machine older than this cozy refuses a new call it does not serve. The daemon updates it to
// the Hub's target software and sends the call again, which it then serves; work it accepted
// before is observed and collected without any update. When the target cannot help (it already
// runs it), the call fails naming the next step.
func TestAnOlderMachineTakesTheTargetBeforeNewWork(t *testing.T) {
	t.Run("follows the target", func(t *testing.T) {
		// The machine runs a call this cozy sent while it served, then this cozy moves a release ahead.
		const accepted, fresh = "run-accepted-earlier", "run-after-the-target"
		machine := &olderMachine{runtime: "0.18.101", tensorfs: "0.3.93", serves: "0.18.101", runs: map[string]bool{},
			held: accepted, release: make(chan struct{})}
		root, store, submit := olderMachineHome(t, machine)
		submit(accepted)
		daemon := startDaemonProcess(t, root)
		eventually(t, root, "the machine accepting the call", func() bool {
			machine.mu.Lock()
			defer machine.mu.Unlock()
			return machine.runs[accepted]
		})
		machine.mu.Lock()
		machine.serves = "0.18.102"
		machine.mu.Unlock()
		submit(fresh)
		crashAndRestartTransactionDaemon(t, daemon)
		reachesState(t, root, store, fresh, "completed")
		close(machine.release)
		reachesState(t, root, store, accepted, "completed")
		machine.mu.Lock()
		updates, refusals, runtime := machine.updates, machine.refusals, machine.runtime
		machine.mu.Unlock()
		if updates != 1 || refusals != 1 || runtime != "0.18.102" {
			t.Fatalf("the older machine took %d update(s) after %d refusal(s) and runs %s; want one of each and 0.18.102", updates, refusals, runtime)
		}
		if code, out := runCozy(t, root, "run", "watch", fresh, "--json"); code != 0 || !strings.Contains(out, `"status":"completed"`) {
			t.Fatalf("watch did not follow the call sent again [exit %d]: %s", code, out)
		}
	})
	t.Run("names the next step", func(t *testing.T) {
		machine := &olderMachine{runtime: "0.18.102", tensorfs: "0.3.93", serves: "0.19.0", runs: map[string]bool{}}
		root, store, submit := olderMachineHome(t, machine)
		const stuck = "run-the-target-cannot-help"
		submit(stuck)
		startDaemonProcess(t, root)
		reachesState(t, root, store, stuck, "failed")
		_, out := runCozy(t, root, "run", "show", stuck, "--json")
		if !strings.Contains(out, "machine.upgrade_required") || !strings.Contains(out, "cozy rental update elder --runtime-version") ||
			!strings.Contains(out, "it already runs this software") {
			t.Fatalf("the refused call does not name its next step: %s", out)
		}
		if machine.updates != 0 {
			t.Fatalf("a machine already on the target was updated %d time(s)", machine.updates)
		}
	})
}

// olderMachineHome attaches the rental "elder" on machine and answers the root, its records and
// a submitter of calls to it; the Hub's target is Runtime 0.18.102.
func olderMachineHome(t *testing.T, machine *olderMachine) (string, *records.Store, func(string)) {
	t.Helper()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	t.Cleanup(func() {
		if t.Failed() {
			update, _ := store.RuntimeUpdate(olderRental)
			log, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
			t.Logf("software update: %+v\ndaemon log:\n%s", update, log)
			for _, id := range []string{"run-accepted-earlier", "run-after-the-target"} {
				_, show := runCozy(t, root, "run", "show", id, "--json")
				t.Logf("%s: %s", id, show)
			}
			machine.mu.Lock()
			t.Logf("machine: runtime %s serves %s updates %d refusals %d runs %v", machine.runtime, machine.serves, machine.updates, machine.refusals, machine.runs)
			machine.mu.Unlock()
		}
	})
	identity, problem := rental.PendingCreatorIdentity(layout, "older-machine")
	fatal(t, problem)
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	leaf := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: workertls.ServerName},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames: []string{workertls.ServerName}, IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1)}}
	der, err := x509.CreateCertificate(rand.Reader, leaf, leaf, &key.PublicKey, key)
	must(t, err)
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12,
		Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}})),
		// The cozy keeps its connection alive as it does a real machine's.
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{MinTime: time.Second, PermitWithoutStream: true}))
	v1.RegisterMachineServer(server, machine)
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the test's machine binds; the product dials
	must(t, err)
	go server.Serve(listener)
	t.Cleanup(server.Stop)

	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(olderRental, "elder")
	hub.software = map[string]string{"runtime": "0.18.102", "tensorfs": "0.3.93"}
	row := records.Rental{ID: olderRental, MachineName: "elder", State: "ready", SKU: "cpu", AcceleratorModel: "CPU",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: listener.Addr().String(),
		MediaAddress: "127.0.0.1:1", ExpectedWorkerID: olderWorkerID, ExpectedWorkerBootID: olderBootID}
	cert := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
	fatal(t, rental.Attach(layout, store, row, cert, secret.New(strings.Repeat("m", 43)), identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	return root, store, func(id string) {
		t.Helper()
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/example", Release: "1.0.0",
			Entrypoint: "main", Kind: "call", Payload: []byte(`{}`), BodyDigest: childDigest(id), MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, olderRental))
	}
}

func reachesState(t *testing.T, root string, store *records.Store, id, state string) {
	t.Helper()
	eventually(t, root, id+" "+state, func() bool {
		current, problem := store.RequestRow(id)
		fatal(t, problem)
		return current != nil && records.PublicRunStatus(current.State) == state
	})
}
