package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"encoding/base64"
	"io"
	"net"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// This independent protocol peer verifies the real rental signature over pinned TLS.
// It issues no byte verdicts; actual Runtime restart joins the separate wrapper proof.
type idleHoldPeer struct {
	controlDefaults
	pb.UnimplementedPodHostServer
	key                  ed25519.PublicKey
	pin                  []byte
	epoch                uint64
	workerID, bootID     string
	workerBusy, hostBusy bool
	settled              []*pb.HeldAttempt
	settledAtHost        bool
	deny                 codes.Code
	claims               atomic.Int64
	otherFrames          atomic.Int64
}

func (*idleHoldPeer) ProtocolInfo(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return &pb.ProtocolInfoResult{WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor, SupportsRentalKeepalive: true}, nil
}

func (p *idleHoldPeer) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	first, err := stream.Recv()
	if err != nil {
		return err
	}
	claim := first.GetClaim()
	if claim == nil {
		return status.Error(codes.Unauthenticated, "Claim required")
	}
	body, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: claim.RecordOwnerEpoch,
		WorkerId: claim.WorkerId, WorkerBootId: claim.WorkerBootId, WorkerTlsCertificateDigest: p.pin})
	if err != nil || !ed25519.Verify(p.key, body, claim.Proof) || claim.WorkerId != p.workerID {
		return status.Error(codes.Unauthenticated, "Claim identity mismatch")
	}
	p.claims.Add(1)
	if p.deny != codes.OK {
		return status.Error(p.deny, "controlled fixed Claim refusal")
	}
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
		Accepted: true, RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: p.epoch,
		WorkerId: p.workerID, WorkerBootId: p.bootID, WireMinor: pb.WireMinor,
	}}}); err != nil {
		return err
	}
	set, setDigest, err := canonical.Identity(&pb.PlacementSet{})
	if err != nil {
		return err
	}
	worker := &pb.WorkerSnapshotBody{AcceptedPlacementSetDigest: setDigest,
		WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED}
	host := &pb.HostSnapshotBody{}
	if p.workerBusy {
		worker.HeldAttempts = []*pb.HeldAttempt{{RequestId: "already-running", AttemptOrdinal: 1}}
	}
	if p.hostBusy {
		host.HeldOutcomes = []*pb.HeldAttempt{{RequestId: "unacknowledged", AttemptOrdinal: 1}}
	}
	if p.settledAtHost {
		host.HeldOutcomes = append(host.HeldOutcomes, p.settled...)
	} else {
		worker.HeldAttempts = append(worker.HeldAttempts, p.settled...)
	}
	workerBody, workerDigest, err := canonical.Identity(worker)
	if err != nil {
		return err
	}
	hostBody, hostDigest, err := canonical.Identity(host)
	if err != nil {
		return err
	}
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{
		RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: p.epoch, WorkerBootId: p.bootID,
		SnapshotId: "idle-proof", SnapshotCanonicalBytes: workerBody, SnapshotDigest: workerDigest,
		AcceptedPlacementSetCanonicalBytes: set, HostSnapshotCanonicalBytes: hostBody, HostSnapshotDigest: hostDigest,
	}}}); err != nil {
		return err
	}
	for {
		_, err := stream.Recv()
		if err != nil {
			return nil
		}
		p.otherFrames.Add(1)
		return status.Error(codes.FailedPrecondition, "operator sent a directive after Claim")
	}
}

func serveIdleHoldPeer(t *testing.T, peer *idleHoldPeer, cert, address string) (string, func()) {
	t.Helper()
	pair, err := tls.LoadX509KeyPair(cert, cert+".key")
	must(t, err)
	listener, err := net.Listen("tcp", address) //cozy:allow the independent test peer owns this listener
	must(t, err)
	server := grpc.NewServer(grpc.Creds(credentials.NewServerTLSFromCert(&pair)))
	pb.RegisterWorkerControlServer(server, peer)
	pb.RegisterPodHostServer(server, peer)
	go func() { _ = server.Serve(listener) }()
	return listener.Addr().String(), server.Stop
}

type developmentFixture struct {
	cfg            config.Config
	layout         home.Layout
	store          *records.Store
	peer           *idleHoldPeer
	cert, rentalID string
}

func developmentFixtureAt(t *testing.T) developmentFixture {
	t.Helper()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	id := "pr-11111111111111111111"
	identity := adoptCreatorIdentity(t, layout, id)
	key, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	cert := standInCertificate(t, root)
	pin, err := workertls.LoadPin(cert)
	must(t, err)
	return developmentFixture{cfg: config.Config{Home: root, HubURL: "http://127.0.0.1:1"}, layout: layout, store: store,
		peer: &idleHoldPeer{key: key, pin: pin.Digest(), epoch: 7, workerID: "dev-worker", bootID: "dev-boot"}, cert: cert, rentalID: id}
}

func (f developmentFixture) attach(t *testing.T, address string) {
	t.Helper()
	identity, problem := rental.CreatorIdentityFor(f.layout, f.rentalID)
	fatal(t, problem)
	token, problem := rental.PendingMediaToken(f.layout, "development-hold-proof")
	fatal(t, problem)
	certificate, err := os.ReadFile(f.cert)
	must(t, err)
	fatal(t, rental.Attach(f.layout, f.store, records.Rental{AcceleratorCount: 1, ID: f.rentalID, State: "ready", Hub: f.cfg.HubURL,
		MachineName: "proof", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, Address: address,
		ExpectedWorkerID: f.peer.workerID, ExpectedWorkerBootID: f.peer.bootID}, string(certificate), token, identity))
}

func (f developmentFixture) attempt(t *testing.T, closed bool) {
	t.Helper()
	instance := (orchestrator.WorkerLaunchSpec{Connection: &orchestrator.WorkerConnection{RentalID: f.rentalID}}).InstanceID()
	fatal(t, f.store.SpawnWorker(records.WorkerProcess{InstanceID: instance, Package: "proof/app", WorkerID: f.peer.workerID, Devices: []string{"cpu"}}))
	_, _, problem := f.store.Submit(records.Request{ID: "historical-request", IdemKey: "history", BodyDigest: "sha256:" + strings.Repeat("a", 64),
		Package: "proof/app", Entrypoint: "generate", Payload: []byte("{}"), Worker: f.rentalID, Rental: true})
	fatal(t, problem)
	digest := "sha256:" + strings.Repeat("b", 64)
	ordinal, problem := f.store.Dispatch(records.Attempt{RequestID: "historical-request", InstanceID: instance,
		SessionID: "historical-session", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	if closed {
		fatal(t, f.store.OfferDispatch("historical-request", ordinal, "historical-session"))
		fatal(t, f.store.Accepted("historical-request", ordinal, "historical-session"))
		_, problem := f.store.AcceptTerminal(records.Terminal{RequestID: "historical-request", Attempt: ordinal,
			SessionID: "historical-session", InvocationDigest: digest, TerminalID: "historical-outcome",
			TerminalDigest: "sha256:" + strings.Repeat("c", 64), Status: "SUCCEEDED", Cause: "COMPLETED"})
		fatal(t, problem)
		fatal(t, f.store.Closed("historical-request", ordinal))
	}
}

func awaitDevelopmentState(t *testing.T, updates <-chan cli.DevelopmentHoldResult, state string) cli.DevelopmentHoldResult {
	t.Helper()
	timer := time.NewTimer(10 * time.Second)
	defer timer.Stop()
	for {
		select {
		case value := <-updates:
			if value.State == state {
				return value
			}
		case <-timer.C:
			t.Fatalf("development hold did not become %s", state)
		}
	}
}

func TestDevelopmentHoldKeepsOneOwnerAcrossControlReplacement(t *testing.T) {
	f := developmentFixtureAt(t)
	address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
	defer stop()
	f.attach(t, address)
	f.attempt(t, true)
	eligible, problem := cli.InspectStoredDevelopmentHold(f.cfg, f.rentalID, f.peer.bootID)
	fatal(t, problem)
	if eligible.State != "eligible" || eligible.ControlStreamEpoch != 0 {
		t.Fatal("preflight claimed remote idleness")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	updates := make(chan cli.DevelopmentHoldResult, 16)
	finished := make(chan *exit.Error, 1)
	go func() {
		finished <- cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, func(value cli.DevelopmentHoldResult) { updates <- value })
	}()
	first := awaitDevelopmentState(t, updates, "holding")
	if first.ControlStreamEpoch != 7 || first.SnapshotAcknowledged {
		t.Fatalf("bad first hold: %+v", first)
	}
	if lock, problem := daemon.Hold(f.layout, "", ""); problem == nil {
		lock.Release()
		t.Fatal("operator did not hold daemon lock")
	}
	stop()
	awaitDevelopmentState(t, updates, "reconnecting")
	if lock, problem := daemon.Hold(f.layout, "", ""); problem == nil {
		lock.Release()
		t.Fatal("owner lock was dropped between Host processes")
	}
	second := &idleHoldPeer{key: f.peer.key, pin: f.peer.pin, epoch: 8, workerID: f.peer.workerID, bootID: f.peer.bootID}
	_, stopSecond := serveIdleHoldPeer(t, second, f.cert, address)
	defer stopSecond()
	resumed := awaitDevelopmentState(t, updates, "holding")
	if resumed.ControlStreamEpoch != 8 || resumed.SnapshotAcknowledged {
		t.Fatalf("bad resumed hold: %+v", resumed)
	}
	cancel()
	select {
	case problem := <-finished:
		fatal(t, problem)
	case <-time.After(10 * time.Second):
		t.Fatal("hold did not stop")
	}
	if f.peer.otherFrames.Load() != 0 || second.otherFrames.Load() != 0 {
		t.Fatal("holder opened admission or sent work")
	}
	attempts, problem := f.store.Attempts("historical-request")
	fatal(t, problem)
	if len(attempts) != 1 || attempts[0].State != "closed" || attempts[0].TerminalStatus != "SUCCEEDED" {
		t.Fatal("completed history was changed")
	}
	lock, problem := daemon.Hold(f.layout, "", "")
	fatal(t, problem)
	lock.Release()
}

func TestDevelopmentHoldRefusesOpenWorkAndHeldOutcomes(t *testing.T) {
	for _, kind := range []string{"local-open", "worker-held", "host-held"} {
		t.Run(kind, func(t *testing.T) {
			f := developmentFixtureAt(t)
			f.peer.workerBusy, f.peer.hostBusy = kind == "worker-held", kind == "host-held"
			address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
			defer stop()
			f.attach(t, address)
			if kind == "local-open" {
				f.attempt(t, false)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			problem := cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, nil)
			if problem == nil || problem.Code != exit.Conflict {
				t.Fatalf("busy worker was not refused: %v", problem)
			}
			if kind == "local-open" && f.peer.claims.Load() != 0 {
				t.Fatal("local open attempt reached remote Claim")
			}
		})
	}
}

func TestDevelopmentHoldDoesNotRetryPermanentClaimRefusal(t *testing.T) {
	for _, code := range []codes.Code{codes.Unauthenticated, codes.PermissionDenied, codes.FailedPrecondition} {
		t.Run(code.String(), func(t *testing.T) {
			f := developmentFixtureAt(t)
			f.peer.deny = code
			address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
			defer stop()
			f.attach(t, address)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			problem := cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, nil)
			if problem == nil || problem.Code != exit.Conflict || f.peer.claims.Load() != 1 {
				t.Fatalf("fixed Claim denial was retried or ignored: problem=%v claims=%d", problem, f.peer.claims.Load())
			}
			lock, problem := daemon.Hold(f.layout, "", "")
			fatal(t, problem)
			lock.Release()
		})
	}
}

func TestDevelopmentHoldRefusesChangedOwnerDuringReplacement(t *testing.T) {
	f := developmentFixtureAt(t)
	address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
	defer stop()
	f.attach(t, address)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	updates := make(chan cli.DevelopmentHoldResult, 16)
	finished := make(chan *exit.Error, 1)
	go func() {
		finished <- cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, func(value cli.DevelopmentHoldResult) { updates <- value })
	}()
	awaitDevelopmentState(t, updates, "holding")
	_, problem := rental.PendingCreatorIdentity(f.layout, "changed-owner")
	fatal(t, problem)
	old, err := os.ReadFile(f.layout.RentalCreatorIdentity(f.rentalID))
	must(t, err)
	changed, err := os.ReadFile(f.layout.PendingRentalCreatorIdentity("changed-owner"))
	must(t, err)
	if bytes.Equal(old, changed) {
		t.Fatal("test did not create another owner")
	}
	must(t, os.WriteFile(f.layout.RentalCreatorIdentity(f.rentalID), changed, 0600))
	stop()
	select {
	case problem := <-finished:
		if problem == nil || problem.ErrName() != "development_hold_identity_changed" {
			t.Fatalf("changed owner accepted: %v", problem)
		}
	case <-ctx.Done():
		t.Fatal("changed owner was not refused")
	}
	if f.peer.claims.Load() != 1 {
		t.Fatal("changed owner reached another remote Claim")
	}
}

func TestDevelopmentHoldRefusesChangedPeerOrResetStream(t *testing.T) {
	for _, kind := range []string{"boot", "stream"} {
		t.Run(kind, func(t *testing.T) {
			f := developmentFixtureAt(t)
			address, stop := serveIdleHoldPeer(t, f.peer, f.cert, "127.0.0.1:0")
			defer stop()
			f.attach(t, address)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			updates := make(chan cli.DevelopmentHoldResult, 16)
			finished := make(chan *exit.Error, 1)
			go func() {
				finished <- cli.HoldStoredDevelopmentWorker(ctx, f.cfg, f.rentalID, f.peer.bootID, io.Discard, func(value cli.DevelopmentHoldResult) { updates <- value })
			}()
			awaitDevelopmentState(t, updates, "holding")
			stop()
			awaitDevelopmentState(t, updates, "reconnecting")
			second := &idleHoldPeer{key: f.peer.key, pin: f.peer.pin, epoch: 8, workerID: f.peer.workerID, bootID: f.peer.bootID}
			if kind == "boot" {
				second.bootID = "different-boot"
			} else {
				second.epoch = f.peer.epoch
			}
			_, stopSecond := serveIdleHoldPeer(t, second, f.cert, address)
			defer stopSecond()
			select {
			case problem := <-finished:
				if problem == nil || problem.Code != exit.Conflict {
					t.Fatalf("changed %s was not refused: %v", kind, problem)
				}
			case <-ctx.Done():
				t.Fatalf("changed %s was not refused", kind)
			}
			if second.otherFrames.Load() != 0 {
				t.Fatal("refused replacement received a directive")
			}
		})
	}
}
