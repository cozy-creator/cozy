package cli

import (
	"context"
	"net"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

type closureCapabilityPeer struct {
	pb.UnimplementedPodHostServer
	advertised, routed bool
	closed             []*pb.MachineSubmissionClose
}

func (p *closureCapabilityPeer) GetMachineExecutionWorkspace(_ context.Context, q *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: q.Claim.WorkerId, WorkerBootId: q.Claim.WorkerBootId, ExecutionWorkspaceId: "workspace", RunOutputLog: true, SubmissionClose: p.advertised}, nil
}
func (p *closureCapabilityPeer) CloseMachineSubmission(_ context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if !p.routed {
		return nil, status.Error(codes.Unimplemented, "old Host has no CloseMachineSubmission route")
	}
	p.closed = append(p.closed, q)
	return &pb.MachineSubmissionClosure{RequestId: q.RequestId, SubmissionId: q.SubmissionId, ExecutionWorkspaceId: q.ExpectedExecutionWorkspaceId}, nil
}
func closurePeerConnection(t *testing.T, p *closureCapabilityPeer) *machineConnection {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	server := grpc.NewServer()
	pb.RegisterPodHostServer(server, p)
	go server.Serve(listener)
	t.Cleanup(server.Stop)
	conn, err := grpc.NewClient(listener.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return &machineConnection{Machine: &machines.Machine{Host: pb.NewPodHostClient(conn), Claim: &pb.Claim{WorkerId: "worker", WorkerBootId: "boot"}}}
}
func TestNewSubmissionRequiresEndToEndClosureBeforeFreeze(t *testing.T) {
	for _, test := range []struct {
		name               string
		advertised, routed bool
	}{
		{"old Runtime", false, true}, {"new Runtime behind old Host", true, false}, {"closure capable pair", true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			peer := &closureCapabilityPeer{advertised: test.advertised, routed: test.routed}
			connection := closurePeerConnection(t, peer)
			workspace, problem := newExecutionWorkspace(context.Background(), connection)
			if !test.advertised || !test.routed {
				if workspace != nil || problem == nil || problem.Code != exit.Structural || !strings.Contains(problem.Message, "no submission was sent") {
					t.Fatalf("not refused before transmission: %v %+v", workspace, problem)
				}
			} else {
				if problem != nil || workspace == nil {
					t.Fatalf("capable peer refused: %v", problem)
				}
				if len(peer.closed) != 1 || !strings.HasPrefix(peer.closed[0].RequestId, "closure-probe-") {
					t.Fatalf("probe identity: %+v", peer.closed)
				}
				if _, problem = newExecutionWorkspace(context.Background(), connection); problem != nil || len(peer.closed) != 1 {
					t.Fatal("same connection repeated capability probe")
				}
			}
			// Reading/reconciling already-frozen work remains possible on every arm.
			if _, problem = currentExecutionWorkspace(context.Background(), connection); problem != nil {
				t.Fatalf("observation was gated: %v", problem)
			}
		})
	}
}
func TestPermanentRefusalKeepsCauseAndUncertainAcceptance(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	req, _, problem := store.Submit(records.Request{ID: "request", IdemKey: "request", Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("a", 64), MachineExecutionObserver: true})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem = store.LinkMachineExecution(req.ID, "rental"); problem != nil {
		t.Fatal(problem)
	}
	captured, spec := []byte(`{"capture":true}`), []byte(`{"invocation":true}`)
	frozen := &pb.MachineExecutionSubmit{SubmissionId: req.IdemKey, ExpectedExecutionWorkspaceId: "workspace", CaptureCanonicalBytes: captured, CaptureDigest: canonical.Digest(captured), Offer: &pb.AttemptOffer{RequestId: req.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	if problem = store.RecordMachineSubmission(req.ID, frozen); problem != nil {
		t.Fatal(problem)
	}
	m := &machineRuns{store: store}
	connection := closurePeerConnection(t, &closureCapabilityPeer{advertised: true})
	problem = m.settleSubmissionRefusal(context.Background(), connection, req.ID, machineTransport(status.Error(codes.InvalidArgument, "reference model is missing its required image")))
	if problem == nil || !strings.Contains(problem.Message, "reference model is missing") || !strings.Contains(problem.Message, "old Host") || problem.Details["submission_refusal"] == nil {
		t.Fatalf("original refusal masked: %+v", problem)
	}
	row, p := store.RequestRow(req.ID)
	if p != nil {
		t.Fatal(p)
	}
	link, p := store.MachineExecution(req.ID)
	if p != nil {
		t.Fatal(p)
	}
	owed, p := store.MachineExecutionOwesWork(req.ID)
	if p != nil {
		t.Fatal(p)
	}
	if row.State != req.State || len(link.Submission) == 0 || len(link.Receipt) != 0 || link.SubmissionClosed || !owed {
		t.Fatalf("missing closure settled unknown acceptance: state=%s submission=%d receipt=%d closed=%v owed=%v", row.State, len(link.Submission), len(link.Receipt), link.SubmissionClosed, owed)
	}
}
