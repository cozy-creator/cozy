package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// End the real pinned TLS control stream while its upload RPC waits for an ACK.
// The same request must survive that session cancellation and reach serving after
// reconnect. The existing explicit-cancellation product test covers user intent.
func TestLocalWheelSessionLossRetriesWithoutUserCancellation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	root := t.TempDir()
	started := make(chan struct{})
	var firstAck atomic.Bool
	var uploads atomic.Int32
	pod := &fakePod{controlKey: public, serve: true}
	pod.localUpload = func(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
		n := uploads.Add(1)
		if n == 1 {
			if _, err := stream.Recv(); err != nil {
				return err
			}
			close(started)
			<-stream.Context().Done()
			return stream.Context().Err()
		}
		receiver := &uploadReceiver{path: filepath.Join(root, fmt.Sprintf("received-%d", n)), batch: 1}
		return receiver.LocalPackageUpload(stream)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetSnapshotAck() == nil || !firstAck.CompareAndSwap(false, true) {
			return false, nil
		}
		select {
		case <-started:
			return false, status.Error(codes.Unavailable, "control transport interrupted")
		case <-time.After(5 * time.Second):
			return false, context.DeadlineExceeded
		}
	}
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	owner := hostOwner(t, "upload-session-loss", rentalWiring(connection, private), func(options *orchestrator.Options) {
		options.Packages = localLauncher{revision: revision}
	})
	requestID := submitPrivateRental(t, owner, revision, "upload-session-loss")
	waitUntil(t, "same request serving after control-session loss", func() bool {
		owner.c.WakeQueue()
		row, problem := owner.store.RequestRow(requestID)
		fatal(t, problem)
		if row.State == "failed" || row.State == "canceled" {
			t.Fatalf("transport loss terminalized the request: %s", row.State)
		}
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) > 0
	})
	if uploads.Load() < 2 {
		t.Fatal("the interrupted upload was not retried")
	}
	events, problem := owner.store.EventsAfter(requestID, 0, 100)
	fatal(t, problem)
	for _, event := range events {
		if event.Type == "request.failed" || event.Type == "request.canceled" {
			t.Fatalf("transport loss was published as terminal: %s", event.Type)
		}
	}
}
