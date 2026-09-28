package producttest

import (
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

type maintenancePeer struct {
	info   *pb.ProtocolInfoResult
	public ed25519.PublicKey
	digest []byte
	held   []*pb.HeldAttempt
	claims atomic.Int64
	extra  atomic.Int64
}

type maintenanceHost struct {
	pb.UnimplementedPodHostServer
	peer *maintenancePeer
}
type maintenanceWorker struct {
	pb.UnimplementedWorkerControlServer
	peer *maintenancePeer
}

func (h *maintenanceHost) ProtocolInfo(ctx context.Context, request *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return h.peer.ProtocolInfo(ctx, request)
}
func (w *maintenanceWorker) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	return w.peer.Control(stream)
}
func (p *maintenancePeer) ProtocolInfo(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return p.info, nil
}
func (p *maintenancePeer) Control(stream grpc.BidiStreamingServer[pb.RecordOwnerFrame, pb.WorkerFrame]) error {
	frame, err := stream.Recv()
	if err != nil {
		return err
	}
	claim := frame.GetClaim()
	if claim == nil {
		return status.Error(codes.Unauthenticated, "missing Claim")
	}
	p.claims.Add(1)
	transcript, _ := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: claim.RecordOwnerEpoch, WorkerId: "worker", WorkerBootId: "boot", WorkerTlsCertificateDigest: p.digest})
	if !ed25519.Verify(p.public, transcript, claim.Proof) {
		return status.Error(codes.Unauthenticated, "wrong signer")
	}
	if claim.WireMinor != min(p.info.WireMinor, pb.WireMinor) {
		return status.Error(codes.FailedPrecondition, "Claim did not negotiate maintenance range")
	}
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{Accepted: true, RecordOwnerEpoch: claim.RecordOwnerEpoch, WorkerId: "worker", WorkerBootId: "boot", ControlStreamEpoch: 1, WireMinor: p.info.WireMinor}}}); err != nil {
		return err
	}
	body, digest, err := canonical.Identity(&pb.WorkerSnapshotBody{HeldAttempts: p.held})
	if err != nil {
		return err
	}
	host, hostDigest, err := canonical.Identity(&pb.HostSnapshotBody{})
	if err != nil {
		return err
	}
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: "boot", SnapshotId: "snapshot", SnapshotCanonicalBytes: body, SnapshotDigest: digest, HostSnapshotCanonicalBytes: host, HostSnapshotDigest: hostDigest}}}); err != nil {
		return err
	}
	for {
		if _, err := stream.Recv(); err != nil {
			return err
		}
		p.extra.Add(1)
	}
}

func maintenanceFixture(t *testing.T, p *maintenancePeer) (*orchestrator.WorkerConnection, orchestrator.RentalClaimProofSource) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	cert := &x509.Certificate{SerialNumber: big.NewInt(1), NotBefore: time.Now().Add(-time.Minute), NotAfter: time.Now().Add(time.Hour), DNSNames: []string{workertls.ServerName}, KeyUsage: x509.KeyUsageDigitalSignature, ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}}
	der, err := x509.CreateCertificate(rand.Reader, cert, cert, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "pin.pem")
	if err := os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0600); err != nil {
		t.Fatal(err)
	}
	pin, err := workertls.LoadPin(path)
	if err != nil {
		t.Fatal(err)
	}
	p.digest = pin.Digest()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	p.public = public
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}}})))
	pb.RegisterPodHostServer(server, &maintenanceHost{peer: p})
	pb.RegisterWorkerControlServer(server, &maintenanceWorker{peer: p})
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow isolated TLS maintenance peer; product only dials
	if err != nil {
		t.Fatal(err)
	}
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	remote := &orchestrator.WorkerConnection{RentalID: "rental", Addr: listener.Addr().String(), CACert: path, WorkerID: "worker", WorkerBootID: "boot"}
	sign := func(remote *orchestrator.WorkerConnection, epoch uint64) ([]byte, *exit.Error) {
		transcript, err := canonical.Bytes(&pb.ClaimProof{RecordOwnerEpoch: epoch, WorkerId: remote.WorkerID, WorkerBootId: remote.WorkerBootID, WorkerTlsCertificateDigest: p.digest})
		if err != nil {
			t.Fatal(err)
		}
		return ed25519.Sign(private, transcript), nil
	}
	return remote, sign
}

func TestMaintenanceControlProtocolAndSafety(t *testing.T) {
	for _, row := range []struct {
		name                      string
		minor, floor              uint32
		unsafe, wrongSigner, busy bool
		accepted                  bool
	}{
		{name: "old", minor: 60, floor: 59, accepted: true},
		{name: "current", minor: pb.WireMinor, floor: pb.MinCompatibleWireMinor, accepted: true},
		{name: "pre keepalive", minor: 59, floor: 59},
		{name: "newer floor", minor: pb.WireMinor + 1, floor: pb.WireMinor + 1, accepted: true},
		{name: "missing activity", minor: 60, floor: 59, unsafe: true},
		{name: "wrong signer", minor: 60, floor: 59, wrongSigner: true},
		{name: "active attempt", minor: 60, floor: 59, busy: true},
	} {
		t.Run(row.name, func(t *testing.T) {
			peer := &maintenancePeer{info: &pb.ProtocolInfoResult{WireMinor: row.minor, MinimumWireMinor: row.floor}}
			if row.busy {
				peer.held = []*pb.HeldAttempt{{RequestId: "active", AttemptOrdinal: 1}}
			}
			remote, sign := maintenanceFixture(t, peer)
			if row.wrongSigner {
				sign = func(*orchestrator.WorkerConnection, uint64) ([]byte, *exit.Error) {
					return make([]byte, ed25519.SignatureSize), nil
				}
			}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			control, problem := orchestrator.DialMaintenanceControl(ctx, remote, sign, nil)
			if control != nil {
				control.Close()
			}
			if (problem == nil) != row.accepted {
				t.Fatalf("accepted=%t problem=%v", row.accepted, problem)
			}
			if peer.extra.Load() != 0 {
				t.Fatal("maintenance sent dispatch/admission frames")
			}
			if row.minor < pb.MinCompatibleWireMinor && orchestrator.ValidateWorkerProtocol(peer.info, podRental) == nil {
				t.Fatal("maintenance weakened execution gate")
			}
		})
	}
}
