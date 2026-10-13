package producttest

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// opsMachine is a rental's machine as far as one operation goes: it takes the operation's warm
// run, reports its download, and ends it canceled when its owner cancels it.
type opsMachine struct {
	v1.UnimplementedMachineServer
	mu      sync.Mutex
	changed *sync.Cond
	specs   []*v1.RunSpec
	log     []*v1.RunEvent
}

func newOpsMachine() *opsMachine {
	m := &opsMachine{}
	m.changed = sync.NewCond(&m.mu)
	return m
}

func (m *opsMachine) Status(_ *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	return stream.Send(&v1.StatusFrame{WorkerId: "ops-worker", BootId: "ops-boot", Phase: "ready",
		Capabilities: []string{"warm/1", "warm/2", "remove/1", "upload/1", "local-models/1"}})
}

func (m *opsMachine) append(id string, event *v1.RunEvent) {
	event.Sequence = uint64(len(m.log) + 1)
	m.log = append(m.log, event)
	m.changed.Broadcast()
}

func (m *opsMachine) Run(request *v1.RunRequest, stream grpc.ServerStreamingServer[v1.RunEvent]) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if request.Spec != nil && len(m.specs) == 0 {
		m.specs = append(m.specs, request.Spec)
		m.append(request.Id, &v1.RunEvent{Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.Id, Number: 1, State: "queued", Sequence: 2}}})
		m.append(request.Id, &v1.RunEvent{Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "downloading paul/demo",
			Fraction: -1, BytesDone: 1 << 20, BytesTotal: 4 << 20}}})
	}
	if len(m.log) == 0 {
		return status.Errorf(codes.NotFound, "no run %s", request.Id)
	}
	go func() {
		<-stream.Context().Done()
		m.mu.Lock()
		m.changed.Broadcast()
		m.mu.Unlock()
	}()
	head := &v1.RunEvent{Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.Id, Number: 1, State: "queued", Sequence: uint64(len(m.log))}}}
	if err := stream.Send(head); err != nil {
		return err
	}
	for sent := request.After; ; {
		for ; sent < uint64(len(m.log)); sent++ {
			if err := stream.Send(m.log[sent]); err != nil {
				return err
			}
			if m.log[sent].GetOutcome() != nil {
				return nil
			}
		}
		if stream.Context().Err() != nil {
			return nil
		}
		m.changed.Wait()
	}
}

func (m *opsMachine) Control(_ context.Context, request *v1.ControlRequest) (*v1.RunState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if request.Action == v1.Action_ACTION_CANCEL && m.log[len(m.log)-1].GetOutcome() == nil {
		m.append(request.Id, &v1.RunEvent{Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.Id, Number: 1, State: "canceled"}}})
		m.append(request.Id, &v1.RunEvent{Event: &v1.RunEvent_Outcome{Outcome: &v1.Outcome{Status: "canceled",
			Reason: &v1.Reason{Code: "canceled", Message: "canceled by its owner", Origin: "request"}}}})
	}
	return &v1.RunState{Id: request.Id, Number: 1, State: "canceled"}, nil
}

// A package install on a rental is a run like any other: it gets the next run number, `run
// list` shows it with its kind, the same install asked again while it runs is that run, and
// `run cancel` reaches the machine and ends it canceled.
func TestAnOperationIsANumberedRunTheOwnerCancels(t *testing.T) {
	h := newFakeRentalHub(t, 0)
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/execution-environment" {
			_ = json.NewEncoder(w).Encode(map[string]any{"environment": map[string]string{
				"TENSORHUB_ORIGIN": h.server.URL, "TENSORHUB_PUBLIC_ORIGIN": h.server.URL}})
			return
		}
		served.ServeHTTP(w, r)
	})
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	must(t, os.MkdirAll(layout.Rentals, 0o700))
	owner, problem := rental.OwnerIdentityAt(layout.RentalCreatorIdentity("pr-ops"))
	fatal(t, problem)
	certServer := httptest.NewTLSServer(http.NotFoundHandler())
	cert := certServer.TLS.Certificates[0]
	certServer.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow pinned loopback machine fixture for the real daemon's runs
	must(t, err)
	machine := newOpsMachine()
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{Certificates: []tls.Certificate{cert}})))
	v1.RegisterMachineServer(server, machine)
	go server.Serve(listener)
	defer server.Stop()
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
	must(t, os.WriteFile(layout.RentalCert("pr-ops"), certPEM, 0o600))
	fatal(t, store.RecordRental(records.Rental{ID: "pr-ops", MachineName: "kirin", SKU: "cpu", State: "ready", Hub: h.server.URL,
		Address: listener.Addr().String(), CertPath: layout.RentalCert("pr-ops"), ExpectedWorkerID: "ops-worker", ExpectedWorkerBootID: "ops-boot",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, ReadyAt: time.Now().UTC().Format(time.RFC3339Nano)}))
	h.mu.Lock()
	h.rentals["pr-ops"] = map[string]any{"rental_id": "pr-ops", "machine_name": "kirin", "state": "ready", "worker_address": listener.Addr().String(),
		"cert_pem": string(certPEM), "worker_id": "ops-worker", "worker_boot_id": "ops-boot", "creator_public_key": owner.PublicKey(),
		"hourly_rate_usd_micros": int64(1), "accelerator_count": 1, "requested_accelerator_model": "CPU"}
	h.mu.Unlock()
	startDaemonProcess(t, root)

	install := func() (run struct {
		Number int64
		Kind   string
	}) {
		t.Helper()
		code, out := runCozy(t, root, "package", "install", "paul/demo", "--rental=kirin", "--no-wait", "--json")
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &run) != nil || run.Number == 0 || run.Kind != "install" {
			t.Fatalf("package install --rental [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
		}
		return run
	}
	first := install()
	number := fmt.Sprint(first.Number)
	waitFor(t, root, "the machine taking the install as a warm run", func() bool {
		machine.mu.Lock()
		defer machine.mu.Unlock()
		return len(machine.specs) == 1
	})
	machine.mu.Lock()
	spec := machine.specs[0]
	machine.mu.Unlock()
	if spec.Kind != v1.RunKind_RUN_KIND_WARM || spec.GetRelease().GetPackage() != "paul/demo" || spec.Entrypoint != "" {
		t.Fatalf("the install is not a warm run of its release: %+v", spec)
	}
	if again := install(); again.Number != first.Number {
		t.Fatalf("the same install asked again while it runs is run %d, not run %d", again.Number, first.Number)
	}
	code, out := runCozy(t, root, "run", "list", "--json")
	var listed struct {
		Invocations []map[string]any `json:"invocations"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || len(listed.Invocations) != 1 ||
		listed.Invocations[0]["kind"] != "install" || listed.Invocations[0]["target"] != "paul/demo" || listed.Invocations[0]["machine"] != "kirin" {
		t.Fatalf("run list does not show the install as a run [exit %d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "cancel", number, "--json"); code != 0 {
		t.Fatalf("run cancel %s [exit %d]: %s", number, code, out)
	}
	waitFor(t, root, "the canceled install ending canceled", func() bool {
		_, shown := runCozy(t, root, "run", "show", number, "--json")
		var run struct{ Status string }
		return json.Unmarshal([]byte(lastJSONLine(shown)), &run) == nil && run.Status == "canceled"
	})
}
