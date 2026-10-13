package host

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"database/sql"
	"io"
	"net"
	"path/filepath"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
	_ "modernc.org/sqlite"
)

// A separate WorkerControl implementation journals closure in SQLite. The Host
// under test is the production authenticated router, running on a real gRPC socket.
type journalClosurePeer struct {
	pb.UnimplementedWorkerControlServer
	db *sql.DB
}

func (p *journalClosurePeer) CloseMachineSubmission(_ context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if q.Claim.RecordOwnerId != runtimeOwnerID || q.ExpectedExecutionWorkspaceId != "workspace" {
		return nil, status.Error(codes.PermissionDenied, "Host did not translate the authenticated owner")
	}
	_, err := p.db.Exec(`INSERT INTO closures(owner,submission,request) VALUES(?,?,?) ON CONFLICT DO NOTHING`, q.Claim.RecordOwnerId, q.SubmissionId, q.RequestId)
	if err != nil {
		return nil, err
	}
	closed := &pb.MachineSubmissionClosure{RequestId: q.RequestId, SubmissionId: q.SubmissionId, ExecutionWorkspaceId: q.ExpectedExecutionWorkspaceId}
	var raw []byte
	err = p.db.QueryRow(`SELECT receipt FROM accepted WHERE request=?`, q.RequestId).Scan(&raw)
	if err == nil {
		closed.Receipt = &pb.MachineExecutionReceipt{}
		err = proto.Unmarshal(raw, closed.Receipt)
	}
	if err == sql.ErrNoRows {
		err = nil
	}
	return closed, err
}
func testGRPC(t *testing.T, s *grpc.Server) *grpc.ClientConn {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go s.Serve(l)
	t.Cleanup(s.Stop)
	c, err := grpc.NewClient(l.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}
func TestClosureTraversesAuthenticatedHostAndPreservesAcceptedWork(t *testing.T) {
	db, err := sql.Open("sqlite", filepath.Join(t.TempDir(), "runtime.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, stmt := range []string{`CREATE TABLE closures(owner TEXT,submission TEXT,request TEXT,PRIMARY KEY(owner,submission))`, `CREATE TABLE accepted(request TEXT PRIMARY KEY,receipt BLOB)`} {
		if _, err = db.Exec(stmt); err != nil {
			t.Fatal(err)
		}
	}
	receipt := &pb.MachineExecutionReceipt{RequestId: "accepted", SubmissionId: "accepted", WorkerId: "worker", WorkerBootId: "boot", ExecutionWorkspaceId: "workspace"}
	raw, _ := proto.Marshal(receipt)
	if _, err = db.Exec(`INSERT INTO accepted VALUES(?,?)`, receipt.RequestId, raw); err != nil {
		t.Fatal(err)
	}
	runtimeServer := grpc.NewServer()
	pb.RegisterWorkerControlServer(runtimeServer, &journalClosurePeer{db: db})
	runtimeConn := testGRPC(t, runtimeServer)
	pub, key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	authority, err := newClaims("worker", "boot", make([]byte, 32), []ed25519.PublicKey{pub})
	if err != nil {
		t.Fatal(err)
	}
	ready := make(chan struct{})
	close(ready)
	machine := &Machine{claims: authority, idle: &idle{}, owned: true, proc: &runtimeProcess{done: make(chan struct{})}, ready: ready, conn: runtimeConn, log: io.Discard}
	hostServer := grpc.NewServer(grpc.UnknownServiceHandler(machine.serveRPC))
	hostConn := testGRPC(t, hostServer)
	proof, err := authority.transcript(7)
	if err != nil {
		t.Fatal(err)
	}
	claim := &pb.Claim{WorkerId: "worker", WorkerBootId: "boot", RecordOwnerId: "client", RecordOwnerEpoch: 7, WireMinor: 1, Proof: ed25519.Sign(key, proof)}
	// Reconciliation is available even to an older wire peer; it is not new work.
	for _, id := range []string{"absent", "accepted"} {
		request := &pb.MachineSubmissionClose{Claim: claim, RequestId: id, SubmissionId: id, ExpectedExecutionWorkspaceId: "workspace"}
		for _, call := range []func(context.Context, *pb.MachineSubmissionClose, ...grpc.CallOption) (*pb.MachineSubmissionClosure, error){pb.NewPodHostClient(hostConn).CloseMachineSubmission, pb.NewWorkerControlClient(hostConn).CloseMachineSubmission} {
			closed, err := call(context.Background(), request)
			if err != nil {
				t.Fatal(err)
			}
			if closed.RequestId != id || closed.SubmissionId != id || closed.ExecutionWorkspaceId != "workspace" {
				t.Fatalf("Host changed closure identity: %v", closed)
			}
			if (closed.Receipt != nil) != (id == "accepted") {
				t.Fatalf("Host lost accepted work: %v", closed)
			}
		}
	}
	var n int
	if err = db.QueryRow(`SELECT count(*) FROM closures`).Scan(&n); err != nil || n != 2 {
		t.Fatalf("closures not retained idempotently: %d %v", n, err)
	}
	bad := proto.Clone(claim).(*pb.Claim)
	bad.Proof = []byte("not signed")
	_, err = pb.NewPodHostClient(hostConn).CloseMachineSubmission(context.Background(), &pb.MachineSubmissionClose{Claim: bad, RequestId: "forged", SubmissionId: "forged", ExpectedExecutionWorkspaceId: "workspace"})
	if status.Code(err) != codes.Unauthenticated {
		t.Fatalf("unsigned closure reached Runtime: %v", err)
	}
	if err = db.QueryRow(`SELECT count(*) FROM closures`).Scan(&n); err != nil || n != 2 {
		t.Fatal("unauthorized closure changed Runtime journal")
	}
}
