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
	expiring, expireRelease        chan struct{}
	public                         ed25519.PublicKey
	denyStatus, denyRun, denyFirst bool
	transientStatus                codes.Code
	streamFailure                  codes.Code
	disconnect                     bool
	loseFirstReply                 bool
	dropConnection                 func()
	mu                             sync.Mutex
	id                             string
	statusTimes, runTimes          []time.Time
	specs, attaches, controls      atomic.Int32
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
	denied := p.denyStatus || (!p.denyRun && len(p.statusTimes) == 1)
	transient := p.transientStatus != codes.OK && len(p.statusTimes) == 1
	p.mu.Unlock()
	if transient {
		return status.Error(p.transientStatus, "temporary failure reading renewed owner authority")
	}
	if denied {
		return status.Error(codes.PermissionDenied, "owner lease unavailable")
	}
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
		if p.loseFirstReply {
			return status.Error(codes.Unavailable, "first reply was lost")
		}
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
		if p.disconnect {
			p.dropConnection()
			<-stream.Context().Done()
			return stream.Context().Err()
		}
		if p.streamFailure != codes.OK {
			return status.Error(p.streamFailure, "read tcp: connection timed out")
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

type observerListener struct {
	net.Listener
	mu   sync.Mutex
	last net.Conn
}

func (l *observerListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err == nil {
		l.mu.Lock()
		l.last = conn
		l.mu.Unlock()
	}
	return conn, err
}

func (l *observerListener) drop() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.last != nil {
		_ = l.last.Close()
	}
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
	tracked := &observerListener{Listener: listener}
	peer.dropConnection = tracked.drop
	listener = tracked
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

func TestAcceptedRunObserverRecoversItsLeaseWithoutResubmission(t *testing.T) {
	peer := new(observerLeasePeer)
	root, store, request, _ := observerLeaseFixture(t, peer)
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
	if len(peer.statusTimes) != 2 || peer.statusTimes[1].Sub(peer.statusTimes[0]) < 1800*time.Millisecond {
		t.Fatalf("missing denial backoff: %v", peer.statusTimes)
	}
}

func TestAcceptedRunObserverRetriesTransientAuthorityProbeWithoutResubmission(t *testing.T) {
	for _, code := range []codes.Code{codes.Unavailable, codes.DeadlineExceeded} {
		t.Run(code.String(), func(t *testing.T) {
			peer := &observerLeasePeer{transientStatus: code}
			_, store, request, _ := observerLeaseFixture(t, peer)
			observerEventually(t, func() bool {
				row, _ := store.MachineExecution(request.ID)
				return row != nil && row.Collected
			})
			row, problem := store.RequestRow(request.ID)
			if problem != nil || row.State != "succeeded" {
				t.Fatalf("terminal collection: %+v %v", row, problem)
			}
			if peer.specs.Load() != 1 || peer.attaches.Load() != 1 || peer.controls.Load() != 0 {
				t.Fatalf("specs=%d attaches=%d controls=%d", peer.specs.Load(), peer.attaches.Load(), peer.controls.Load())
			}
			peer.mu.Lock()
			defer peer.mu.Unlock()
			if len(peer.statusTimes) != 2 || peer.statusTimes[1].Sub(peer.statusTimes[0]) < 1800*time.Millisecond {
				t.Fatalf("transient authority probe did not back off: %v", peer.statusTimes)
			}
		})
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
		return strings.Contains(string(log), "observation recovery: this rental is updating its software")
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
		return strings.Contains(string(log), "rental is terminal; observation recovery stopped")
	})
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if len(peer.statusTimes) != 0 || peer.specs.Load() != 1 || peer.attaches.Load() != 0 || peer.controls.Load() != 0 {
		t.Fatal("terminal rental recovery made another remote call")
	}
}

func TestRevokedRunObserverBacksOffAndDetachDoesNotControlWork(t *testing.T) {
	for _, denyRun := range []bool{false, true} {
		t.Run(map[bool]string{false: "Status-denied", true: "Status-allowed-Run-denied"}[denyRun], func(t *testing.T) {
			peer := &observerLeasePeer{denyStatus: !denyRun, denyRun: denyRun}
			_, store, request, owner := observerLeaseFixture(t, peer)
			observerEventually(t, func() bool { peer.mu.Lock(); defer peer.mu.Unlock(); return len(peer.statusTimes) >= 2 })
			time.Sleep(300 * time.Millisecond)
			must(t, owner.cmd.Process.Signal(os.Interrupt))
			select {
			case <-owner.exited:
			case <-time.After(5 * time.Second):
				t.Fatal("observer daemon did not detach")
			}
			peer.mu.Lock()
			defer peer.mu.Unlock()
			if len(peer.statusTimes) != 2 || peer.statusTimes[1].Sub(peer.statusTimes[0]) < 1800*time.Millisecond {
				t.Fatalf("refusal busy loop: %v", peer.statusTimes)
			}
			row, _ := store.RequestRow(request.ID)
			if records.Settled(row.State) || peer.specs.Load() != 1 || peer.controls.Load() != 0 {
				t.Fatalf("revocation/detach changed accepted work: state=%s specs=%d controls=%d", row.State, peer.specs.Load(), peer.controls.Load())
			}
		})
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

// The TCP arm closes a real accepted TLS connection after Creator records the
// running event. Recovery uses a new authenticated Status and the same nil-spec
// cursor attach, even when that first read-only probe is temporarily unavailable.
func TestAcceptedRunObserverRecoversBrokenStreamWithoutResubmission(t *testing.T) {
	for _, test := range []struct {
		name       string
		disconnect bool
		failure    codes.Code
	}{
		{"tcp_disconnect", true, codes.OK},
		{"deadline", false, codes.DeadlineExceeded},
	} {
		t.Run(test.name, func(t *testing.T) {
			peer := &observerLeasePeer{expiring: make(chan struct{}), expireRelease: make(chan struct{}),
				disconnect: test.disconnect, streamFailure: test.failure, transientStatus: codes.Unavailable}
			_, store, request, _ := observerLeaseFixture(t, peer, true)
			observerEventually(t, func() bool { held, _ := store.RunV1(request.ID); return held })
			close(peer.expireRelease)
			observerEventually(t, func() bool { row, _ := store.MachineExecution(request.ID); return row != nil && row.Collected })
			row, problem := store.RequestRow(request.ID)
			if problem != nil || row.State != "succeeded" || peer.specs.Load() != 1 || peer.attaches.Load() != 1 || peer.controls.Load() != 0 {
				t.Fatalf("recovery changed accepted work: row=%+v error=%v specs=%d attaches=%d controls=%d", row, problem, peer.specs.Load(), peer.attaches.Load(), peer.controls.Load())
			}
			events, problem := store.EventsAfter(request.ID, 0, 100)
			fatal(t, problem)
			for _, event := range events {
				if event.Type == "request.parked" {
					t.Fatal("accepted execution was presented as waiting for a rental")
				}
			}
			peer.mu.Lock()
			defer peer.mu.Unlock()
			if len(peer.statusTimes) != 2 || peer.statusTimes[1].Sub(peer.statusTimes[0]) < 1800*time.Millisecond {
				t.Fatalf("transport recovery skipped fresh authority/backoff: %v", peer.statusTimes)
			}
		})
	}
}

func TestUncertainFirstStreamFailureDoesNotTriggerAcceptedRecovery(t *testing.T) {
	peer := &observerLeasePeer{loseFirstReply: true}
	_, store, request, _ := observerLeaseFixture(t, peer, true)
	observerEventually(t, func() bool { return peer.specs.Load() == 1 })
	time.Sleep(1500 * time.Millisecond)
	peer.mu.Lock()
	defer peer.mu.Unlock()
	accepted, problem := store.RunV1(request.ID)
	if problem != nil || accepted || peer.specs.Load() != 1 || peer.attaches.Load() != 0 || len(peer.statusTimes) != 0 || peer.controls.Load() != 0 {
		t.Fatalf("uncertain submission was retried or promoted: accepted=%v problem=%v specs=%d attaches=%d status=%d controls=%d", accepted, problem, peer.specs.Load(), peer.attaches.Load(), len(peer.statusTimes), peer.controls.Load())
	}
}
