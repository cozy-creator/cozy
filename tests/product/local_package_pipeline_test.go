package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
)

// This receiver writes and fsyncs actual file bytes before each acknowledgement,
// validates their digest, and reopens the retained prefix on a new gRPC stream.
// Adversarial variants alter the wire after custody to exercise the client fence.
type uploadReceiver struct {
	pb.UnimplementedPodHostServer
	path      string
	header    *pb.LocalPackageUploadHeader
	batch     int
	gate      <-chan struct{}
	filled    chan struct{}
	dropAfter int
	dropCode  codes.Code
	refuse    codes.Code
	mutate    func(*pb.LocalPackageFileStatus, int)
	stopped   chan struct{}
	mu        sync.Mutex
	streams   [][]uint64
}

func (r *uploadReceiver) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	if r.stopped != nil {
		defer close(r.stopped)
	}
	if r.refuse != codes.OK {
		return status.Error(r.refuse, "receiver refused exact upload")
	}
	first, err := stream.Recv()
	if err != nil {
		return err
	}
	h := first.GetHeader()
	if h == nil {
		return status.Error(codes.InvalidArgument, "missing header")
	}
	r.mu.Lock()
	if r.header != nil && !proto.Equal(r.header, h) {
		r.mu.Unlock()
		return status.Error(codes.FailedPrecondition, "header changed")
	}
	r.header = proto.Clone(h).(*pb.LocalPackageUploadHeader)
	index := len(r.streams)
	r.streams = append(r.streams, nil)
	r.mu.Unlock()
	file, err := os.OpenFile(r.path, os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	offset := uint64(info.Size())
	ack := func() *pb.LocalPackageFileStatus {
		state := pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING
		if offset == h.File.Length {
			state = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		}
		return &pb.LocalPackageFileStatus{OperationId: h.OperationId,
			Digest: h.File.Digest, Filename: h.File.Filename, Length: h.File.Length,
			ReceivedBytes: offset, State: state}
	}
	initial := ack()
	if r.mutate != nil {
		r.mutate(initial, 0)
	}
	if err := stream.Send(initial); err != nil || offset == h.File.Length {
		return err
	}
	pending := []*pb.LocalPackageFileStatus{}
	chunks := 0
	for {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		chunk := frame.GetChunk()
		if chunk == nil || chunk.Offset != offset || len(chunk.Data) == 0 || len(chunk.Data) > 1<<20 || offset+uint64(len(chunk.Data)) > h.File.Length {
			return status.Error(codes.FailedPrecondition, "chunk changed exact offset or bound")
		}
		if _, err := file.WriteAt(chunk.Data, int64(offset)); err != nil {
			return err
		}
		if err := file.Sync(); err != nil {
			return err
		}
		r.mu.Lock()
		r.streams[index] = append(r.streams[index], offset)
		r.mu.Unlock()
		offset += uint64(len(chunk.Data))
		chunks++
		if offset == h.File.Length {
			data, err := os.ReadFile(r.path)
			if err != nil {
				return err
			}
			sum := sha256.Sum256(data)
			if !bytes.Equal(sum[:], h.File.Digest) {
				return status.Error(codes.FailedPrecondition, "file digest changed")
			}
		}
		next := ack()
		if r.mutate != nil {
			r.mutate(next, chunks)
		}
		pending = append(pending, next)
		if len(pending) < max(r.batch, 1) && offset < h.File.Length {
			continue
		}
		if r.gate != nil && chunks == r.batch {
			close(r.filled)
			select {
			case <-r.gate:
			case <-stream.Context().Done():
				return stream.Context().Err()
			}
		}
		for _, row := range pending {
			if err := stream.Send(row); err != nil {
				return err
			}
		}
		pending = pending[:0]
		if r.dropAfter != 0 && chunks == r.dropAfter {
			code := r.dropCode
			if code == codes.OK {
				code = codes.Unavailable
			}
			return status.Error(code, "stream ended after durable acknowledgement")
		}
		if offset == h.File.Length {
			return nil
		}
	}
}

type uploadMessages struct {
	mu     sync.Mutex
	chunks []*pb.LocalPackageUploadChunk
}

func (*uploadMessages) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context   { return ctx }
func (*uploadMessages) TagConn(ctx context.Context, _ *stats.ConnTagInfo) context.Context { return ctx }
func (*uploadMessages) HandleConn(context.Context, stats.ConnStats)                       {}
func (m *uploadMessages) HandleRPC(_ context.Context, event stats.RPCStats) {
	if sent, ok := event.(*stats.OutPayload); ok {
		if frame, ok := sent.Payload.(*pb.LocalPackageUploadFrame); ok && frame.GetChunk() != nil {
			m.mu.Lock()
			m.chunks = append(m.chunks, frame.GetChunk()) // intentionally retain SendMsg's original
			m.mu.Unlock()
		}
	}
}

func uploadClient(t *testing.T, receiver pb.PodHostServer) (pb.PodHostClient, *uploadMessages) {
	t.Helper()
	listener := bufconn.Listen(4 << 20)
	server := grpc.NewServer()
	pb.RegisterPodHostServer(server, receiver)
	go func() { _ = server.Serve(listener) }()
	messages := &uploadMessages{}
	connection, err := grpc.NewClient("passthrough:///wheel-upload",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }),
		grpc.WithStatsHandler(messages))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { connection.Close(); server.Stop(); listener.Close() })
	return pb.NewPodHostClient(connection), messages
}

func uploadFixture(t *testing.T, length int) ([]byte, string, *pb.LocalPackageUploadHeader) {
	t.Helper()
	data := make([]byte, length)
	for i := range data {
		data[i] = byte(i/(1<<20) + i%251)
	}
	path := filepath.Join(t.TempDir(), "source.whl")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return data, path, &pb.LocalPackageUploadHeader{Claim: &pb.Claim{WorkerBootId: "boot"},
		OperationId: "operation",
		File:        &pb.LocalPackageFileRef{Digest: digest[:], Filename: "example-1.0-py3-none-any.whl", Length: uint64(length)}}
}

func TestUploadPipelinesFourDurableChunksAndNeverMutatesSentMessages(t *testing.T) {
	data, path, header := uploadFixture(t, 9<<20+731)
	gate := make(chan struct{})
	receiver := &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), batch: 4, gate: gate, filled: make(chan struct{})}
	client, messages := uploadClient(t, receiver)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result := make(chan *exit.Error, 1)
	var acknowledged []uint64
	go func() {
		result <- localpackage.UploadFile(ctx, client, header, path, func(ack *pb.LocalPackageFileStatus) {
			acknowledged = append(acknowledged, ack.ReceivedBytes)
		})
	}()
	select {
	case <-receiver.filled:
	case problem := <-result:
		t.Fatalf("upload stopped before pipelining four chunks: %v", problem)
	case <-ctx.Done():
		t.Fatal("sender waited for ACK before filling its four-chunk window")
	}
	time.Sleep(20 * time.Millisecond)
	messages.mu.Lock()
	count := len(messages.chunks)
	messages.mu.Unlock()
	if count != 4 {
		t.Fatalf("sent %d chunks without an ACK, want exactly four", count)
	}
	close(gate)
	if problem := <-result; problem != nil {
		t.Fatal(problem)
	}
	held, err := os.ReadFile(receiver.path)
	if err != nil || !bytes.Equal(held, data) {
		t.Fatalf("durable receiver bytes differ: %v", err)
	}
	if len(acknowledged) != 11 || acknowledged[0] != 0 || acknowledged[10] != uint64(len(data)) {
		t.Fatalf("callback did not report exact durable ACKs: %v", acknowledged)
	}
	messages.mu.Lock()
	defer messages.mu.Unlock()
	for _, chunk := range messages.chunks {
		if !bytes.Equal(chunk.Data, data[chunk.Offset:chunk.Offset+uint64(len(chunk.Data))]) {
			t.Fatalf("SendMsg buffer at %d was mutated after sending", chunk.Offset)
		}
	}
}

func TestUploadResumesDurablePrefixAndVerifiedReplayNeedsNoCarrier(t *testing.T) {
	data, path, header := uploadFixture(t, 7<<20+31)
	receiver := &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), dropAfter: 2}
	client, _ := uploadClient(t, receiver)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if problem := localpackage.UploadFile(ctx, client, header, path, nil); problem == nil || problem.Code != exit.Unavailable {
		t.Fatalf("missing resumable interruption: %v", problem)
	}
	receiver.dropAfter = 0
	if problem := localpackage.UploadFile(ctx, client, header, path, nil); problem != nil {
		t.Fatal(problem)
	}
	if receiver.streams[1][0] != 2<<20 {
		t.Fatalf("resumed at %d instead of durable 2 MiB", receiver.streams[1][0])
	}
	held, err := os.ReadFile(receiver.path)
	if err != nil || !bytes.Equal(held, data) {
		t.Fatalf("resumed bytes differ: %v", err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if problem := localpackage.UploadFile(ctx, client, header, path, nil); problem != nil {
		t.Fatalf("verified replay reopened deleted local carrier: %v", problem)
	}
}

func TestUploadPreservesRefusalsAndRejectsChangedAcknowledgements(t *testing.T) {
	for _, code := range []codes.Code{codes.InvalidArgument, codes.FailedPrecondition, codes.PermissionDenied, codes.DataLoss, codes.Unimplemented, codes.Unavailable} {
		t.Run(code.String(), func(t *testing.T) {
			_, path, header := uploadFixture(t, 3<<20)
			client, _ := uploadClient(t, &uploadReceiver{refuse: code})
			problem := localpackage.UploadFile(context.Background(), client, header, path, nil)
			want := exit.Structural
			if code == codes.Unavailable {
				want = exit.Unavailable
			}
			if problem == nil || problem.Code != want {
				t.Fatalf("server status %s became %v", code, problem)
			}
		})
	}
	t.Run("refusal after pipelined sends", func(t *testing.T) {
		_, path, header := uploadFixture(t, 9<<20)
		client, _ := uploadClient(t, &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), dropAfter: 2, dropCode: codes.PermissionDenied})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		problem := localpackage.UploadFile(ctx, client, header, path, nil)
		if problem == nil || problem.Name != "local_package_upload_refused" || problem.Code != exit.Structural {
			t.Fatalf("pipelined Send lost terminal refusal: %v", problem)
		}
	})
	changes := map[string]func(*pb.LocalPackageFileStatus){
		"operation":       func(a *pb.LocalPackageFileStatus) { a.OperationId = "other" },
		"source filename": func(a *pb.LocalPackageFileStatus) { a.Filename = "../foreign.whl" },
		"digest":          func(a *pb.LocalPackageFileStatus) { a.Digest = bytes.Repeat([]byte{8}, 32) },
		"filename":        func(a *pb.LocalPackageFileStatus) { a.Filename = "other.whl" },
		"length":          func(a *pb.LocalPackageFileStatus) { a.Length++ },
		"offset":          func(a *pb.LocalPackageFileStatus) { a.ReceivedBytes++ },
		"duplicate":       func(a *pb.LocalPackageFileStatus) { a.ReceivedBytes = 0 },
		"skipped":         func(a *pb.LocalPackageFileStatus) { a.ReceivedBytes += 1 << 20 },
		"beyond sent": func(a *pb.LocalPackageFileStatus) {
			a.ReceivedBytes = a.Length
			a.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		},
		"incomplete": func(a *pb.LocalPackageFileStatus) {
			a.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		},
		"refused": func(a *pb.LocalPackageFileStatus) {
			a.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_REFUSED
			a.SafeCode = "exact_receiver_refusal"
			a.SafeDetail = "wrong wheel"
		},
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			_, path, header := uploadFixture(t, 6<<20)
			receiver := &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), mutate: func(ack *pb.LocalPackageFileStatus, number int) {
				if number == 1 {
					change(ack)
				}
			}}
			client, _ := uploadClient(t, receiver)
			problem := localpackage.UploadFile(context.Background(), client, header, path, nil)
			if problem == nil || (name != "refused" && problem.Code != exit.Conflict) || (name == "refused" && problem.Name != "exact_receiver_refusal") {
				t.Fatalf("changed ACK %s was accepted or lost: %v", name, problem)
			}
		})
	}
}

func TestUploadRequiresCompleteInitialIdentityAndFinalVerification(t *testing.T) {
	for name, change := range map[string]func(*pb.LocalPackageFileStatus){
		"operation":         func(a *pb.LocalPackageFileStatus) { a.OperationId = "other" },
		"missing operation": func(a *pb.LocalPackageFileStatus) { a.OperationId = "" },
		"digest":            func(a *pb.LocalPackageFileStatus) { a.Digest = nil },
		"filename":          func(a *pb.LocalPackageFileStatus) { a.Filename = "other.whl" },
		"length":            func(a *pb.LocalPackageFileStatus) { a.Length++ },
		"offset":            func(a *pb.LocalPackageFileStatus) { a.ReceivedBytes = a.Length + 1 },
		"incomplete": func(a *pb.LocalPackageFileStatus) {
			a.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, path, header := uploadFixture(t, 23)
			client, messages := uploadClient(t, &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), mutate: func(ack *pb.LocalPackageFileStatus, number int) {
				if number == 0 {
					change(ack)
				}
			}})
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if problem := localpackage.UploadFile(ctx, client, header, path, nil); problem == nil || problem.Code != exit.Conflict {
				t.Fatalf("invalid initial ACK accepted: %v", problem)
			}
			messages.mu.Lock()
			defer messages.mu.Unlock()
			if len(messages.chunks) != 0 {
				t.Fatal("sent wheel bytes under invalid initial identity")
			}
		})
	}
	t.Run("full bytes without verification", func(t *testing.T) {
		_, path, header := uploadFixture(t, 4<<20+17)
		client, _ := uploadClient(t, &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), mutate: func(ack *pb.LocalPackageFileStatus, _ int) {
			ack.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING
		}})
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if problem := localpackage.UploadFile(ctx, client, header, path, nil); problem == nil || problem.Name != "local_package_upload_unverified" {
			t.Fatalf("accepted unverified full file: %v", problem)
		}
	})
}

type blockedUploadReceiver struct {
	pb.UnimplementedPodHostServer
	beforeInitial bool
	ready         chan struct{}
}

func (r *blockedUploadReceiver) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	frame, err := stream.Recv()
	if err != nil {
		return err
	}
	if !r.beforeInitial {
		h := frame.GetHeader()
		if err := stream.Send(&pb.LocalPackageFileStatus{OperationId: h.OperationId,
			Digest: h.File.Digest, Filename: h.File.Filename, Length: h.File.Length,
			State: pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING}); err != nil {
			return err
		}
	}
	close(r.ready)
	<-stream.Context().Done()
	return stream.Context().Err()
}

func TestUploadCancellationInterruptsInitialReceiveAndBlockedSend(t *testing.T) {
	for _, beforeInitial := range []bool{true, false} {
		name := "blocked Send"
		if beforeInitial {
			name = "initial Recv"
		}
		t.Run(name, func(t *testing.T) {
			_, path, header := uploadFixture(t, 12<<20)
			receiver := &blockedUploadReceiver{beforeInitial: beforeInitial, ready: make(chan struct{})}
			client, messages := uploadClient(t, receiver)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result := make(chan *exit.Error, 1)
			go func() { result <- localpackage.UploadFile(ctx, client, header, path, nil) }()
			select {
			case <-receiver.ready:
			case <-time.After(3 * time.Second):
				t.Fatal("receiver did not read upload header")
			}
			if !beforeInitial {
				waitUntil(t, "first Send accepted by gRPC", func() bool {
					messages.mu.Lock()
					defer messages.mu.Unlock()
					return len(messages.chunks) > 0
				})
				time.Sleep(20 * time.Millisecond)
				messages.mu.Lock()
				count := len(messages.chunks)
				messages.mu.Unlock()
				if count >= 4 {
					t.Fatalf("test did not block Send before filling the credit window: sent %d chunks", count)
				}
			}
			cancel()
			select {
			case problem := <-result:
				if problem == nil || problem.Code != exit.Canceled {
					t.Fatalf("cancellation was lost: %v", problem)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("cancellation did not interrupt gRPC")
			}
		})
	}
}

func TestUploadCancellationInterruptsBlockedWindow(t *testing.T) {
	data, path, header := uploadFixture(t, 12<<20)
	receiver := &uploadReceiver{path: filepath.Join(t.TempDir(), "received"), batch: 4, gate: make(chan struct{}), filled: make(chan struct{}), stopped: make(chan struct{})}
	client, _ := uploadClient(t, receiver)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan *exit.Error, 1)
	go func() { result <- localpackage.UploadFile(ctx, client, header, path, nil) }()
	select {
	case <-receiver.filled:
	case <-time.After(3 * time.Second):
		t.Fatal("window did not fill")
	}
	cancel()
	select {
	case problem := <-result:
		if problem == nil || problem.Code != exit.Canceled {
			t.Fatalf("cancel was not preserved: %v", problem)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancellation did not join the blocked sender and receiver")
	}
	<-receiver.stopped
	// Canceling a stream does not discard bytes the receiver already fsynced,
	// even when its ACKs had not reached the client yet.
	receiver.gate, receiver.stopped = nil, nil
	ctx, resumeCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer resumeCancel()
	if problem := localpackage.UploadFile(ctx, client, header, path, nil); problem != nil {
		t.Fatal(problem)
	}
	if receiver.streams[1][0] != 4<<20 {
		t.Fatalf("canceled stream lost retained prefix: %v", receiver.streams)
	}
	held, err := os.ReadFile(receiver.path)
	if err != nil || !bytes.Equal(held, data) {
		t.Fatalf("resumed canceled stream bytes differ: %v", err)
	}
}
