package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"encoding/base64"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// This peer exercises the real pinned-TLS/authenticated RPC stream and persistent observer,
// not inference. Its one accepted run survives an owner-lease lapse exactly as a rental's does.
type observerLeasePeer struct {
	v1.UnimplementedMachineServer
	expiring, expireRelease   chan struct{}
	public                    ed25519.PublicKey
	denyRun, denyFirst        bool
	mu                        sync.Mutex
	id                        string
	statusTimes, runTimes     []time.Time
	specs, attaches, controls atomic.Int32
}

func (p *observerLeasePeer) authorize(ctx context.Context) error {
	md, _ := metadata.FromIncomingContext(ctx)
	values := md.Get("authorization")
	if len(values) != 1 {
		return status.Error(codes.Unauthenticated, "missing owner")
	}
	grant, err := capability.Verify(strings.TrimPrefix(values[0], "Cozy-Cap "), "lease-worker", []ed25519.PublicKey{p.public}, time.Now(), "")
	if err != nil || grant.Action != "machine" {
		return status.Error(codes.Unauthenticated, "wrong owner")
	}
	return nil
}

func (p *observerLeasePeer) Status(_ *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	if err := p.authorize(stream.Context()); err != nil {
		return err
	}
	p.mu.Lock()
	p.statusTimes = append(p.statusTimes, time.Now())
	p.mu.Unlock()
	return stream.Send(&v1.StatusFrame{WorkerId: "lease-worker", BootId: "lease-boot", Phase: "ready"})
}

func (p *observerLeasePeer) Run(request *v1.RunRequest, stream grpc.ServerStreamingServer[v1.RunEvent]) error {
	if err := p.authorize(stream.Context()); err != nil {
		return err
	}
	p.mu.Lock()
	p.runTimes = append(p.runTimes, time.Now())
	first := p.id == ""
	if first {
		p.id = request.Id
	}
	id := p.id
	p.mu.Unlock()
	if request.Id != id {
		return status.Error(codes.InvalidArgument, "observer changed run ID")
	}
	if request.Spec != nil {
		p.specs.Add(1)
		if !first {
			return status.Error(codes.AlreadyExists, "observer resubmitted the accepted run")
		}
	} else {
		p.attaches.Add(1)
	}
	if p.denyFirst || (!first && p.denyRun) {
		return status.Error(codes.PermissionDenied, "owner remains revoked")
	}
	if first {
		if err := stream.Send(&v1.RunEvent{Sequence: 1, Event: &v1.RunEvent_State{State: &v1.RunState{Id: id, Number: 1, State: "running", Sequence: 1, Attempt: 1}}}); err != nil {
			return err
		}
		if p.expiring != nil {
			close(p.expiring)
			select {
			case <-p.expireRelease:
			case <-stream.Context().Done():
				return stream.Context().Err()
			}
		}
		return status.Error(codes.PermissionDenied, "the key that opened this stream no longer authorizes it")
	}
	if request.Spec != nil || request.After != 1 {
		return status.Error(codes.InvalidArgument, "not the same accepted cursor")
	}
	if err := stream.Send(&v1.RunEvent{Sequence: 2, Event: &v1.RunEvent_State{State: &v1.RunState{Id: id, Number: 1, State: "succeeded", Sequence: 2, Attempt: 1}}}); err != nil {
		return err
	}
	return stream.Send(&v1.RunEvent{Sequence: 3, Event: &v1.RunEvent_Outcome{Outcome: &v1.Outcome{Status: "succeeded", Result: []byte(`{"answer":7}`)}}})
}

func (p *observerLeasePeer) Control(context.Context, *v1.ControlRequest) (*v1.RunState, error) {
	p.controls.Add(1)
	return nil, status.Error(codes.PermissionDenied, "observation cannot control work")
}

func observerLeaseFixture(t *testing.T, peer *observerLeasePeer, rented ...bool) (string, *records.Store, records.Request, *daemonProcess) {
	t.Helper()
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	host := machines.NewHost(layout.Machine, "", nil)
	if err := os.MkdirAll(layout.Machine, 0700); err != nil {
		t.Fatal(err)
	}
	owner, problem := host.Owner()
	asRental := len(rented) > 0 && rented[0]
	if asRental {
		must(t, os.MkdirAll(layout.Rentals, 0700))
		owner, problem = rental.OwnerIdentityAt(layout.RentalCreatorIdentity("pr-observer"))
	}
	if problem != nil {
		t.Fatal(problem)
	}
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	if err != nil {
		t.Fatal(err)
	}
	peer.public = public
	certSource := httptest.NewTLSServer(http.NotFoundHandler())
	cert := certSource.TLS.Certificates[0]
	certSource.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow test machine transport
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}})))
	v1.RegisterMachineServer(server, peer)
	go server.Serve(listener)
	ep := &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: listener.Addr().String(), WorkerID: "lease-worker", WorkerBootID: "lease-boot", WorkspaceID: "observer-proof", CertificatePEM: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]}))}
	machineID, event := ep.Name(), map[string]any{"machine_endpoint": ep}
	if asRental {
		machineID, event = "pr-observer", map[string]any{}
		must(t, os.WriteFile(layout.RentalCert(machineID), []byte(ep.CertificatePEM), 0600))
		fatal(t, store.RecordRental(records.Rental{ID: machineID, MachineName: "observer", SKU: "cpu",
			State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1,
			Address: ep.Address, CertPath: layout.RentalCert(machineID), ExpectedWorkerID: ep.WorkerID,
			ExpectedWorkerBootID: ep.WorkerBootID}))
	}
	request, _, problem := store.SubmitWithEvent(records.Request{ID: "lease-run", IdemKey: "lease-run", Package: "proof/lease", Release: "1", Entrypoint: "main", Kind: "serving", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true}, event)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem = store.LinkMachineExecution(request.ID, machineID); problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(func() { server.Stop(); store.Close() })
	controller := startDaemonProcess(t, layout.Root)
	return layout.Root, store, request, controller
}

func observerEventually(t *testing.T, until func() bool) {
	t.Helper()
	limit := time.NewTimer(12 * time.Second)
	defer limit.Stop()
	for !until() {
		select {
		case <-limit.C:
			t.Fatal("observer did not reach expected state")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// A stream its rental's machine ends for lack of authority is attached again, never
// resubmitted, and without a Status probe first: the attach is itself read-only.
func TestAcceptedRunObserverAttachesAgainWithoutResubmission(t *testing.T) {
	peer := new(observerLeasePeer)
	root, store, request, _ := observerLeaseFixture(t, peer, true)
	observerEventually(t, func() bool { row, _ := store.MachineExecution(request.ID); return row != nil && row.Collected })
	row, problem := store.RequestRow(request.ID)
	if problem != nil || row.State != "succeeded" {
		t.Fatalf("terminal collection: %+v %v", row, problem)
	}
	if peer.specs.Load() != 1 || peer.attaches.Load() != 1 || peer.controls.Load() != 0 {
		t.Fatalf("specs=%d attaches=%d controls=%d", peer.specs.Load(), peer.attaches.Load(), peer.controls.Load())
	}
	if code, out := runCozy(t, root, "run", "watch", request.ID, "--json"); code != 0 || !strings.Contains(out, `"status":"completed"`) || !strings.Contains(out, `"answer":7`) {
		t.Fatalf("ordinary watch missed terminal outcome: %d %s", code, out)
	}
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if len(peer.statusTimes) != 0 || len(peer.runTimes) != 2 {
		t.Fatalf("status probes %v, streams %v", peer.statusTimes, peer.runTimes)
	}
}

func TestAcceptedRunObserverRetriesTemporaryRentalResolutionRefusal(t *testing.T) {
	peer := &observerLeasePeer{expiring: make(chan struct{}), expireRelease: make(chan struct{})}
	root, store, request, _ := observerLeaseFixture(t, peer, true)
	select {
	case <-peer.expiring:
	case <-time.After(12 * time.Second):
		t.Fatal("rental peer did not accept the first submission")
	}
	observerEventually(t, func() bool { accepted, _ := store.RunV1(request.ID); return accepted })
	// Exercise the real resolver's read of a temporary maintenance hold. No updater is
	// invoked: this test owns the retained record and releases it after the refusal.
	update, problem := store.BeginRuntimeUpdate("pr-observer", "lease-boot", nil)
	if problem != nil {
		t.Fatal(problem)
	}
	close(peer.expireRelease)
	observerEventually(t, func() bool {
		log, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
		return strings.Contains(string(log), "this rental is updating its software")
	})
	update.State = "succeeded"
	fatal(t, store.SaveRuntimeUpdate(*update))
	observerEventually(t, func() bool {
		row, _ := store.MachineExecution(request.ID)
		return row != nil && row.Collected
	})
	if peer.specs.Load() != 1 || peer.attaches.Load() != 1 || peer.controls.Load() != 0 {
		t.Fatalf("specs=%d attaches=%d controls=%d", peer.specs.Load(), peer.attaches.Load(), peer.controls.Load())
	}
}

func TestEndingRentalStopsAuthorityRecoveryWithoutControllingWork(t *testing.T) {
	peer := &observerLeasePeer{expiring: make(chan struct{}), expireRelease: make(chan struct{})}
	root, store, request, _ := observerLeaseFixture(t, peer, true)
	select {
	case <-peer.expiring:
	case <-time.After(12 * time.Second):
		t.Fatal("rental peer did not accept the first submission")
	}
	observerEventually(t, func() bool { accepted, _ := store.RunV1(request.ID); return accepted })
	row, problem := store.RentalRow("pr-observer")
	if problem != nil || row == nil {
		t.Fatalf("rental record: %+v %v", row, problem)
	}
	// An ending rental is terminal for retries, even before provider absence settles
	// the machine-execution link as lost.
	row.State = "release_requested"
	fatal(t, store.RecordRental(*row))
	close(peer.expireRelease)
	observerEventually(t, func() bool {
		log, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
		return strings.Contains(string(log), "observation stopped")
	})
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if len(peer.statusTimes) != 0 || peer.specs.Load() != 1 || peer.attaches.Load() != 0 || peer.controls.Load() != 0 {
		t.Fatal("terminal rental recovery made another remote call")
	}
}

// A key its machine keeps refusing is asked again with growing pauses, never in a busy loop,
// and a daemon that detaches meanwhile leaves the accepted work alone.
func TestRevokedRunObserverBacksOffAndDetachDoesNotControlWork(t *testing.T) {
	peer := &observerLeasePeer{denyRun: true}
	_, store, request, owner := observerLeaseFixture(t, peer, true)
	observerEventually(t, func() bool { peer.mu.Lock(); defer peer.mu.Unlock(); return len(peer.runTimes) >= 3 })
	must(t, owner.cmd.Process.Signal(os.Interrupt))
	select {
	case <-owner.exited:
	case <-time.After(5 * time.Second):
		t.Fatal("observer daemon did not detach")
	}
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if gap := peer.runTimes[2].Sub(peer.runTimes[1]); gap < 1800*time.Millisecond {
		t.Fatalf("refusal busy loop: %v", peer.runTimes)
	}
	row, _ := store.RequestRow(request.ID)
	if records.Settled(row.State) || peer.specs.Load() != 1 || peer.controls.Load() != 0 || len(peer.statusTimes) != 0 {
		t.Fatalf("revocation/detach changed accepted work: state=%s specs=%d controls=%d", row.State, peer.specs.Load(), peer.controls.Load())
	}
	if loss, _ := store.ObservationLost(request.ID); loss == nil {
		t.Fatal("the refused run does not say its observation is lost")
	}
}

func TestFirstSubmissionAuthorizationRefusalIsNotRetried(t *testing.T) {
	peer := &observerLeasePeer{denyFirst: true}
	_, store, request, _ := observerLeaseFixture(t, peer)
	observerEventually(t, func() bool { row, _ := store.RequestRow(request.ID); return row != nil && records.Settled(row.State) })
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if len(peer.statusTimes) != 0 || peer.specs.Load() != 1 || peer.attaches.Load() != 0 {
		t.Fatal("first submission refusal retried")
	}
}
