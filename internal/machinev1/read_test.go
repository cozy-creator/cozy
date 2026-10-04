package machinev1

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// shortServer announces 8 bytes and sends 3: a machine whose output ended early.
type shortServer struct{ pb.UnimplementedMachineServer }

func (shortServer) Read(_ *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
	if err := stream.Send(&pb.ReadFrame{Rev: 1, Length: 8}); err != nil {
		return err
	}
	return stream.Send(&pb.ReadFrame{Data: []byte("123")})
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestReadRefusesShortBytesAndShortWrites(t *testing.T) {
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	pb.RegisterMachineServer(server, shortServer{})
	go server.Serve(listener)
	conn, err := grpc.NewClient("passthrough:///short", grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	must(t, err)
	t.Cleanup(func() { conn.Close(); server.Stop() })
	client := &Client{Machine: pb.NewMachineClient(conn)}

	var out bytes.Buffer
	if _, n, err := client.ReadOutput(context.Background(), "run", "image", 0, 0, 0, &out); !errors.Is(err, io.ErrUnexpectedEOF) || n != 3 {
		t.Fatalf("a short output read returned n=%d err=%v", n, err)
	}
	if _, _, err := client.ReadOutput(context.Background(), "run", "image", 0, 0, 0, shortWriter{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("a short destination write returned %v", err)
	}
}
