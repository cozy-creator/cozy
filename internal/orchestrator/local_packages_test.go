package orchestrator

import (
	"bytes"
	"context"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// The real upload RPC waits for its initial acknowledgement. Ending the owner
// session must interrupt it without confusing transport loss with user intent.
type pendingUpload struct {
	pb.UnimplementedPodHostServer
	started chan struct{}
	stopped chan struct{}
}

func (p *pendingUpload) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	defer close(p.stopped)
	if _, err := stream.Recv(); err != nil {
		return err
	}
	close(p.started)
	<-stream.Context().Done()
	return stream.Context().Err()
}

func TestUploadSessionLossIsRetryableButRequestCancellationIsTerminal(t *testing.T) {
	for _, requestCanceled := range []bool{false, true} {
		name := "session_lost"
		if requestCanceled {
			name = "request_canceled"
		}
		t.Run(name, func(t *testing.T) {
			listener := bufconn.Listen(1 << 20)
			server := grpc.NewServer()
			peer := &pendingUpload{started: make(chan struct{}), stopped: make(chan struct{})}
			pb.RegisterPodHostServer(server, peer)
			go func() { _ = server.Serve(listener) }()
			defer server.Stop()
			defer listener.Close()
			connection, err := grpc.NewClient("passthrough:///owned-test-buffer",
				grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
				grpc.WithTransportCredentials(insecure.NewCredentials()))
			if err != nil {
				t.Fatal(err)
			}
			defer connection.Close()
			controlContext, endControl := context.WithCancel(context.Background())
			defer endControl()
			current := &session{ctx: controlContext, host: pb.NewPodHostClient(connection), claim: &pb.Claim{}}
			transfer := &localTransfer{source: bytes.Repeat([]byte{1}, 32)}
			file := filepath.Join(t.TempDir(), "one.whl")
			if err := os.WriteFile(file, []byte("held bytes"), 0600); err != nil {
				t.Fatal(err)
			}
			selected := localTransferFile{digest: bytes.Repeat([]byte{2}, 32), filename: "one.whl", length: 10, path: file}
			owner := &Orchestrator{}
			result := make(chan *exit.Error, 1)
			go func() { result <- owner.uploadLocalWheel(current, "same-operation", transfer, selected) }()
			select {
			case <-peer.started:
			case <-time.After(5 * time.Second):
				t.Fatal("upload did not reach its real gRPC receiver")
			}
			if requestCanceled {
				owner.mu.Lock()
				transfer.canceled = true
				owner.mu.Unlock()
			}
			endControl()
			select {
			case problem := <-result:
				want := exit.Unavailable
				if requestCanceled {
					want = exit.Canceled
				}
				if problem == nil || problem.Code != want {
					t.Fatalf("request canceled=%v: got %v, want %v", requestCanceled, problem, want)
				}
				if !requestCanceled && problem.ErrName() != "local_package_upload_interrupted" {
					t.Fatalf("transport loss classification: %v", problem)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("session loss left an upload running")
			}
			select {
			case <-peer.stopped:
			case <-time.After(5 * time.Second):
				t.Fatal("canceled client left its receiver running")
			}
		})
	}
}
