package machinev1

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"sync"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// Real gRPC streams with controlled authority endings; no inference is simulated.
type capServer struct {
	pb.UnimplementedMachineServer
	read  func(*pb.ReadRequest, grpc.ServerStreamingServer[pb.ReadFrame]) error
	write func(grpc.ClientStreamingServer[pb.WriteFrame, pb.WriteResult]) error
	run   func(*pb.RunRequest, grpc.ServerStreamingServer[pb.RunEvent]) error
}

func (s *capServer) Read(r *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
	return s.read(r, stream)
}
func (s *capServer) Write(stream grpc.ClientStreamingServer[pb.WriteFrame, pb.WriteResult]) error {
	return s.write(stream)
}
func (s *capServer) Run(r *pb.RunRequest, stream grpc.ServerStreamingServer[pb.RunEvent]) error {
	return s.run(r, stream)
}

func renewalClient(t *testing.T, server *capServer) *Client {
	t.Helper()
	listener := bufconn.Listen(4 << 20)
	srv := grpc.NewServer()
	pb.RegisterMachineServer(srv, server)
	go srv.Serve(listener)
	conn, err := grpc.NewClient("passthrough:///caps",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close(); srv.Stop(); listener.Close() })
	return &Client{Machine: pb.NewMachineClient(conn)}
}
func expired() error {
	return status.Error(codes.Unauthenticated, "capability_expired: this stream's short cap ended")
}

func TestReadOutputRenewsOnlyExpiryAndPinsTheSameRevision(t *testing.T) {
	calls := 0
	client := renewalClient(t, &capServer{read: func(r *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
		calls++
		if calls == 1 {
			if r.Offset != 0 || r.IfRev != 0 {
				t.Errorf("first request: %v", r)
			}
			if err := stream.Send(&pb.ReadFrame{Rev: 7, Length: 8}); err != nil {
				return err
			}
			if err := stream.Send(&pb.ReadFrame{Data: []byte("123")}); err != nil {
				return err
			}
			return expired()
		}
		if r.Offset != 3 || r.IfRev != 7 {
			t.Errorf("resumed request: %v", r)
		}
		if err := stream.Send(&pb.ReadFrame{Rev: 7, Length: 8}); err != nil {
			return err
		}
		return stream.Send(&pb.ReadFrame{Data: []byte("45678")})
	}})
	var out bytes.Buffer
	meta, n, err := client.ReadOutput(context.Background(), "run", "image", 0, 0, 0, &out)
	if err != nil || n != 8 || meta.GetRev() != 7 || out.String() != "12345678" || calls != 2 {
		t.Fatalf("meta=%v n=%d err=%v bytes=%q calls=%d", meta, n, err, out.String(), calls)
	}
}

func TestReadOutputDoesNotRenewARevokedKeyAndRejectsShortBytes(t *testing.T) {
	for _, kind := range []string{"revoked", "short"} {
		t.Run(kind, func(t *testing.T) {
			calls := 0
			client := renewalClient(t, &capServer{read: func(r *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
				calls++
				if err := stream.Send(&pb.ReadFrame{Rev: 1, Length: 8}); err != nil {
					return err
				}
				if err := stream.Send(&pb.ReadFrame{Data: []byte("123")}); err != nil {
					return err
				}
				if kind == "revoked" {
					return status.Error(codes.Unauthenticated, "this key no longer authorizes")
				}
				return nil
			}})
			var out bytes.Buffer
			_, _, err := client.ReadOutput(context.Background(), "run", "image", 0, 0, 0, &out)
			if calls != 1 {
				t.Fatalf("retried %s %d times", kind, calls)
			}
			if kind == "revoked" && status.Code(err) != codes.Unauthenticated {
				t.Fatal(err)
			}
			if kind == "short" && !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatal(err)
			}
		})
	}
}

func TestWriteRenewsExpiryAndResumesActualStaging(t *testing.T) {
	for _, method := range []string{"reader-at", "open"} {
		t.Run(method, func(t *testing.T) {
			data := bytes.Repeat([]byte{42}, 2*writeChunk+7)
			var held []byte
			var mu sync.Mutex
			probes, uploads, expiredOnce := 0, 0, false
			client := renewalClient(t, &capServer{write: func(stream grpc.ClientStreamingServer[pb.WriteFrame, pb.WriteResult]) error {
				first, err := stream.Recv()
				if err != nil {
					return err
				}
				frame, err := stream.Recv()
				mu.Lock()
				defer mu.Unlock()
				if errors.Is(err, io.EOF) {
					probes++
					return stream.SendAndClose(&pb.WriteResult{Digest: first.Digest, Held: uint64(len(held))})
				}
				if err != nil {
					return err
				}
				uploads++
				if first.Offset != uint64(len(held)) {
					t.Errorf("offset=%d held=%d", first.Offset, len(held))
				}
				held = append(held, frame.Data...)
				if !expiredOnce {
					expiredOnce = true
					return expired()
				}
				for {
					frame, err = stream.Recv()
					if errors.Is(err, io.EOF) {
						return stream.SendAndClose(&pb.WriteResult{Held: uint64(len(held))})
					}
					if err != nil {
						return err
					}
					held = append(held, frame.Data...)
				}
			}})
			var err error
			if method == "reader-at" {
				err = client.Write(context.Background(), "sha256:fixture", uint64(len(data)), bytes.NewReader(data))
			} else {
				err = Write(context.Background(), client.Machine, "sha256:fixture", int64(len(data)),
					func() (io.ReadSeekCloser, error) { return nopCloser{bytes.NewReader(data)}, nil })
			}
			mu.Lock()
			defer mu.Unlock()
			if err != nil || !bytes.Equal(held, data) || probes != 2 || uploads != 2 {
				t.Fatalf("err=%v bytes=%d/%d probes=%d uploads=%d", err, len(held), len(data), probes, uploads)
			}
		})
	}
}

func TestUpdateReattachesAfterExpiryWithoutAnotherSpec(t *testing.T) {
	calls := 0
	client := renewalClient(t, &capServer{run: func(r *pb.RunRequest, stream grpc.ServerStreamingServer[pb.RunEvent]) error {
		calls++
		if calls == 1 {
			if r.Spec == nil || r.After != 0 {
				t.Errorf("submission: %v", r)
			}
			if err := stream.Send(&pb.RunEvent{Event: &pb.RunEvent_State{State: &pb.RunState{Id: r.Id}}}); err != nil {
				return err
			}
			if err := stream.Send(&pb.RunEvent{Sequence: 1, Event: &pb.RunEvent_Progress{Progress: &pb.Progress{Stage: "preparing"}}}); err != nil {
				return err
			}
			return expired()
		}
		if r.Spec != nil || r.After != 1 {
			t.Errorf("reattach: %v", r)
		}
		return stream.Send(&pb.RunEvent{Sequence: 2, Event: &pb.RunEvent_Outcome{Outcome: &pb.Outcome{Status: "succeeded"}}})
	}})
	outcome, err := client.Update(context.Background(), "update", Cohort{Runtime: &Member{Version: "0.19.0"}}, nil)
	if err != nil || outcome.GetStatus() != "succeeded" || calls != 2 {
		t.Fatalf("outcome=%v err=%v calls=%d", outcome, err, calls)
	}
}
