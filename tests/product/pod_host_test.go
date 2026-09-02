package producttest

// THE HOST LANE (proto-025), observed end to end against a second implementation of the pod
// side: real TLS, real protocol bytes, a real PodHost service that verifies the owner's
// ClaimProof exactly as pod-supervisor does. Nothing in the orchestrator knows this pod exists.

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/media"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const (
	podWorkerID = "wrk-pod-1"
	podBootID   = "boot-pod-1"
	podRental   = "pr-test-0001"
)

// fakePod hosts WorkerControl and PodHost on one pinned TLS listener, the way pod-supervisor
// does. It verifies the owner's Ed25519 ClaimProof on both services against the control key
// the rental's auth document would carry, records what crossed, and answers minimally.
type fakePod struct {
	pb.UnimplementedWorkerControlServer
	pb.UnimplementedPodHostServer
	controlKey ed25519.PublicKey
	leafDigest []byte
	// mutateHostDigest is the red arm: the host document's digest stops hashing its bytes.
	mutateHostDigest bool

	mu           sync.Mutex
	acks         []*pb.SnapshotAck
	desired      []*pb.DesiredWorkerState
	prepares     []*pb.PreparePackageSetCall
	prepareCodes []codes.Code
	hostDigest   []byte
	preparedSet  []byte
	preparedDig  []byte
}

func (p *fakePod) verifyClaim(claim *pb.Claim, stream bool) error {
	if claim == nil {
		return status.Error(codes.Unauthenticated, "no Claim")
	}
	if !stream && claim.ControlStreamEpoch != 0 {
		return status.Error(codes.FailedPrecondition, "a host call is not stream-scoped")
	}
	proof, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: claim.RecordOwnerEpoch,
		WorkerId: podWorkerID, WorkerBootId: podBootID, WorkerTlsCertificateDigest: p.leafDigest})
	if err != nil {
		return err
	}
	if len(claim.Proof) != ed25519.SignatureSize || !ed25519.Verify(p.controlKey, proof, claim.Proof) {
		return status.Error(codes.Unauthenticated, "the ClaimProof does not verify")
	}
	return nil
}

func (p *fakePod) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	for {
		frame, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		switch m := frame.Msg.(type) {
		case *pb.RecordOwnerFrame_Claim:
			if err := p.verifyClaim(m.Claim, true); err != nil {
				return err
			}
			if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
				RecordOwnerEpoch: m.Claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: podBootID,
				Accepted: true, WireMinor: pb.WireMinor, WorkerId: podWorkerID, WorkerInstanceId: "inst-pod-1",
				Resources: &pb.WorkerResources{Backend: "cuda", DeviceName: "fake-4090", DeviceCount: 1,
					DeviceMemoryTotalBytes: 24 << 30},
			}}}); err != nil {
				return err
			}
			body, digest, err := canonical.Identity(&pb.WorkerSnapshotBody{
				WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AdmissionEpoch: 1,
				AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED, AvailableAttemptSlots: 2})
			if err != nil {
				return err
			}
			hostBody, hostDigest, err := canonical.Identity(&pb.HostSnapshotBody{
				WeightsTransactions: []*pb.WeightsTransactionStatus{{
					WeightsTransactionId: "sha256:" + strings.Repeat("ab", 32), RequestId: "job-prior",
					AttemptOrdinal: 1, InvocationSpecDigest: "sha256:" + strings.Repeat("cd", 32),
					OutputSlot: "model", WriterEpoch: 1,
					State:                     pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT,
					TensorfsDeclarationDigest: bytes.Repeat([]byte{0xef}, 32)}}})
			if err != nil {
				return err
			}
			if p.mutateHostDigest {
				hostDigest = bytes.Repeat([]byte{0x11}, 32)
			}
			p.mu.Lock()
			p.hostDigest = hostDigest
			p.mu.Unlock()
			if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{
				RecordOwnerEpoch: m.Claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: podBootID,
				SnapshotId: "snp-pod-1", SnapshotDigest: digest, SnapshotCanonicalBytes: body,
				HostSnapshotDigest: hostDigest, HostSnapshotCanonicalBytes: hostBody,
			}}}); err != nil {
				return err
			}
		case *pb.RecordOwnerFrame_SnapshotAck:
			p.mu.Lock()
			p.acks = append(p.acks, m.SnapshotAck)
			p.mu.Unlock()
		case *pb.RecordOwnerFrame_DesiredState:
			p.mu.Lock()
			p.desired = append(p.desired, m.DesiredState)
			p.mu.Unlock()
		}
	}
}

func (p *fakePod) PreparePackageSet(call *pb.PreparePackageSetCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
	if err := p.verifyClaim(call.Claim, false); err != nil {
		p.mu.Lock()
		p.prepareCodes = append(p.prepareCodes, status.Code(err))
		p.mu.Unlock()
		return err
	}
	if call.PackageSet == nil || len(call.PackageSet.DownloadDelegationSignature) != ed25519.SignatureSize {
		return status.Error(codes.InvalidArgument, "no signed delegation")
	}
	if _, err := canonical.Read(call.PackageSet.DownloadDelegation, &pb.DownloadDelegation{}); err != nil {
		return status.Errorf(codes.InvalidArgument, "delegation: %v", err)
	}
	p.mu.Lock()
	p.prepares = append(p.prepares, call)
	p.mu.Unlock()
	setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{{
		PlacementId: "plc-pod-1",
		PackageMode: &pb.Placement_Package{Package: &pb.PackageSelection{Package: "cozy/h3-package", Release: "1.0.7",
			ReleaseDigest: bytes.Repeat([]byte{0x21}, 32),
			ProjectWheel: &pb.WheelFact{Ref: &pb.Ref{Digest: bytes.Repeat([]byte{0x22}, 32), Length: 4096},
				Distribution: "h3-package", Version: "1.0.7", Filename: "h3_package-1.0.7-py3-none-any.whl",
				ImportRoots: []string{"h3_package"}, Tags: []string{"py3-none-any"}}}},
		EnvironmentDigest: bytes.Repeat([]byte{0x23}, 32),
		PackageDescriptor: &pb.Ref{Digest: bytes.Repeat([]byte{0x24}, 32), Length: 2048},
		BindingsDigest:    bytes.Repeat([]byte{0x25}, 32),
		Environment:       &pb.Environment{},
	}}})
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.preparedSet, p.preparedDig = setBytes, setDigest
	p.mu.Unlock()
	total := uint64(18_874_368)
	for _, event := range []*pb.PrepareEvent{
		{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: total},
		{Stage: pb.PrepareStage_PREPARE_STAGE_DOWNLOADING, TotalBytes: total, TransferredBytes: total / 2},
		{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARING, TotalBytes: total, TransferredBytes: total},
		{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, TotalBytes: total, TransferredBytes: total,
			PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: setDigest, PlacementSetCanonicalBytes: setBytes}},
	} {
		if err := stream.Send(event); err != nil {
			return err
		}
	}
	return nil
}

// startFakePod mints the pod leaf, binds the pinned listener, and serves the media health
// route the owner dials before it will attach.
func startFakePod(t *testing.T, root string, pod *fakePod) (*orchestrator.WorkerConnection, string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	must(t, err)
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: workertls.ServerName},
		NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames: []string{workertls.ServerName}, IPAddresses: []net.IP{net.IPv4(127, 0, 0, 1)},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	must(t, err)
	pemPath := filepath.Join(root, "pod-leaf.pem")
	must(t, os.WriteFile(pemPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	pin, err := workertls.LoadPin(pemPath)
	must(t, err)
	pod.leafDigest = pin.Digest()
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12,
		Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}})))
	pb.RegisterWorkerControlServer(server, pod)
	pb.RegisterPodHostServer(server, pod)
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the test's POD binds; the product dials
	must(t, err)
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	rev := mediawire.ContractRev
	mediaPlane := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/health" {
			_ = json.NewEncoder(w).Encode(mediawire.Health{Service: mediawire.Service, ContractRev: &rev})
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(mediaPlane.Close)
	return &orchestrator.WorkerConnection{
		RentalID: podRental, Addr: listener.Addr().String(), CACert: pemPath,
		WorkerID: podWorkerID, WorkerBootID: podBootID,
		Media: &media.Spec{Addr: strings.TrimPrefix(mediaPlane.URL, "http://"), Token: secret.New("media-token")},
	}, pemPath
}

// rentalWiring is the production entrypoint's rental hooks with a test key: the same
// ClaimProof/1 and DownloadDelegation/1 documents, signed the same way.
func rentalWiring(connection *orchestrator.WorkerConnection, signer ed25519.PrivateKey) func(*orchestrator.Options) {
	return func(o *orchestrator.Options) {
		o.Rentals = func(id string) (*orchestrator.RemoteTarget, *exit.Error) {
			if id != podRental {
				return nil, exit.New(exit.NotFound, "no rental %s", id)
			}
			return &orchestrator.RemoteTarget{Connection: connection}, nil
		}
		o.ObserveRental = func(orchestrator.RentalObservation) *exit.Error { return nil }
		o.RentalClaimProof = func(c *orchestrator.WorkerConnection, epoch uint64) ([]byte, *exit.Error) {
			pin, err := workertls.LoadPin(c.CACert)
			if err != nil {
				return nil, exit.Internalf("%v", err)
			}
			body, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: epoch, WorkerId: c.WorkerID,
				WorkerBootId: c.WorkerBootID, WorkerTlsCertificateDigest: pin.Digest()})
			if err != nil {
				return nil, exit.Internalf("%v", err)
			}
			return ed25519.Sign(signer, body), nil
		}
		o.RentalPackageSet = func(c *orchestrator.WorkerConnection, packages []*pb.DownloadPackageRef,
			models []*pb.DownloadModelRef) ([]byte, []byte, *exit.Error) {
			pin, err := workertls.LoadPin(c.CACert)
			if err != nil {
				return nil, nil, exit.Internalf("%v", err)
			}
			body, err := canonical.Bytes(&pb.DownloadDelegation{ExpiresAtUnix: 2_000_000_000,
				Models: models, Packages: packages, RentalId: c.RentalID, WorkerBootId: c.WorkerBootID,
				WorkerId: c.WorkerID, WorkerTlsCertificateDigest: pin.Digest()})
			if err != nil {
				return nil, nil, exit.Internalf("%v", err)
			}
			return body, ed25519.Sign(signer, body), nil
		}
	}
}

func waitUntil(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// TestPodHostThreeStepSequence: the owner prepares through PodHost, then sends the exact
// prepared placement_set bytes itself on WorkerControl. No package_set ever crosses the
// control stream, and the snapshot ack echoes the host document's digest.
func TestPodHostThreeStepSequence(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost", rentalWiring(connection, private))

	instance, _, _, e := o.c.EnsureRental(podRental)
	fatal(t, e)
	pod.mu.Lock()
	acks, hostDigest := append([]*pb.SnapshotAck(nil), pod.acks...), pod.hostDigest
	pod.mu.Unlock()
	if len(acks) != 1 || !bytes.Equal(acks[0].HostSnapshotDigest, hostDigest) {
		t.Fatalf("the snapshot ack did not echo the host document digest: %d ack(s) %x vs %x",
			len(acks), acks[0].GetHostSnapshotDigest(), hostDigest)
	}

	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{
		Package: "cozy/h3-package", Release: "1.0.7", ReleaseDigest: "sha256:" + strings.Repeat("21", 32)}}, nil))
	waitUntil(t, "the prepared placement_set on WorkerControl", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) >= 1
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.prepares) != 1 {
		t.Fatalf("PodHost.PreparePackageSet was called %d times, want 1", len(pod.prepares))
	}
	if got := pod.prepares[0].Claim; got.ControlStreamEpoch != 0 || got.WorkerBootId != podBootID {
		t.Fatalf("the host call's Claim is stream-scoped or misaddressed: %+v", got)
	}
	for _, d := range pod.desired {
		if d.GetPackageSet() != nil || d.GetPrivatePackageSet() != nil || d.GetPrivatePlacementSet() != nil {
			t.Fatalf("a host mode crossed WorkerControl: %T", d.Mode)
		}
	}
	sent := pod.desired[0].GetPlacementSet()
	if sent == nil || !bytes.Equal(sent.PlacementSetCanonicalBytes, pod.preparedSet) ||
		!bytes.Equal(sent.PlacementSetDigest, pod.preparedDig) {
		t.Fatalf("the desired state does not carry the exact bytes the host prepared")
	}
	facts := o.c.Worker(instance)
	if facts == nil || facts.DesiredRevision != pod.desired[0].Revision {
		t.Fatalf("the owner's desired revision %v is not the one it sent (%d)", facts, pod.desired[0].Revision)
	}
	log, err := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
	must(t, err)
	for _, want := range []string{"PodHost prepare package_set", "PREPARED", "prepared by the host as package_set"} {
		if !strings.Contains(string(log), want) {
			t.Errorf("the owner log does not show %q", want)
		}
	}
}

// TestPodHostRefusesUnverifiedHostDocument: a host document whose digest does not hash its
// bytes is refused at the barrier exactly like a worker document would be. Dispatch never
// opens, and the pod sees no ack.
func TestPodHostRefusesUnverifiedHostDocument(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, mutateHostDigest: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	o := hostOwner(t, "podhost-red", rentalWiring(connection, private))

	done := make(chan *exit.Error, 1)
	go func() { _, _, _, e := o.c.EnsureRental(podRental); done <- e }()
	waitUntil(t, "the owner's refusal of the host document", func() bool {
		log, _ := os.ReadFile(filepath.Join(o.root, "orchestrator.log"))
		return strings.Contains(string(log), "host_snapshot_digest") &&
			strings.Contains(string(log), "NOT acknowledged")
	})
	pod.mu.Lock()
	acks := len(pod.acks)
	pod.mu.Unlock()
	if acks != 0 {
		t.Fatalf("the owner acknowledged a host document whose digest does not hash its bytes")
	}
	select {
	case e := <-done:
		t.Fatalf("EnsureRental returned (%s) although the barrier never opened", briefly(e))
	default:
	}
}
