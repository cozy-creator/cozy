package producttest

import (
	"context"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type loseVerifiedUploadACK struct {
	pb.UnimplementedPodHostServer
	receiver *uploadReceiver
	lost     atomic.Bool
}

type verifiedACKStream struct {
	grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]
	lost *atomic.Bool
}

func (s *verifiedACKStream) Send(row *pb.LocalPackageFileStatus) error {
	if row.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED && s.lost.CompareAndSwap(false, true) {
		return status.Error(codes.Unavailable, "lost final acknowledgement after durable verification")
	}
	return s.BidiStreamingServer.Send(row)
}

func (r *loseVerifiedUploadACK) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	return r.receiver.LocalPackageUpload(&verifiedACKStream{stream, &r.lost})
}

// A verified upload whose final acknowledgement is lost replays its header without
// resending the carrier.
func TestUploadLostFinalACKReplaysVerifiedHeaderWithoutCarrier(t *testing.T) {
	_, carrier, header := uploadFixture(t, 5<<20+31)
	receiver := &uploadReceiver{path: filepath.Join(t.TempDir(), "received")}
	peer := &loseVerifiedUploadACK{receiver: receiver}
	client, _ := uploadClient(t, peer)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if problem := localpackage.UploadFile(ctx, client, header, carrier, nil); problem == nil || problem.Code != exit.Unavailable || !peer.lost.Load() {
		t.Fatalf("final ACK loss did not remain resumable: %v", problem)
	}
	must(t, os.Remove(carrier))
	if problem := localpackage.UploadFile(ctx, client, header, carrier, nil); problem != nil {
		t.Fatalf("verified header replay reopened a missing carrier: %v", problem)
	}
	receiver.mu.Lock()
	defer receiver.mu.Unlock()
	if len(receiver.streams) != 2 || len(receiver.streams[1]) != 0 {
		t.Fatalf("verified replay resent chunk bytes: %+v", receiver.streams)
	}
}
