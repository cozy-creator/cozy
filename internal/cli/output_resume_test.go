package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// cutMachine serves one output at revision 2 the way a machine's Read does, from the asked
// offset; its first read ends partway as a dropped connection does, and while `down` every
// later read fails before a byte.
type cutMachine struct {
	pb.UnimplementedMachineServer
	data    []byte
	mu      sync.Mutex
	offsets []uint64
	cut     bool
	down    bool
}

func (m *cutMachine) Read(request *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
	m.mu.Lock()
	m.offsets = append(m.offsets, request.Offset)
	first, down := !m.cut, m.down
	m.cut = true
	m.mu.Unlock()
	if down && !first {
		return status.Error(codes.Unavailable, "connection refused")
	}
	if request.IfRev != 2 {
		return status.Error(codes.FailedPrecondition, "the output is at revision 2")
	}
	sum := sha256.Sum256(m.data)
	if err := stream.Send(&pb.ReadFrame{Rev: 2, Length: uint64(len(m.data)), Digest: "sha256:" + hex.EncodeToString(sum[:])}); err != nil {
		return err
	}
	tail := m.data[request.Offset:]
	if first {
		if err := stream.Send(&pb.ReadFrame{Data: tail[:len(tail)/2]}); err != nil {
			return err
		}
		return status.Error(codes.Unavailable, "connection reset")
	}
	return stream.Send(&pb.ReadFrame{Data: tail})
}

// A download the connection cuts asks again at once for the rest alone.
func TestACutOutputReadAsksAgainForTheRest(t *testing.T) {
	data := bytes.Repeat([]byte("the run's output "), 1<<14)
	machine := &cutMachine{data: data}
	v1 := serveCut(t, machine)
	directory := t.TempDir()
	product := cutProduct(data, directory)
	if problem := writeOutputV1(t.Context(), v1, "run", directory, product, -1, nil); problem != nil {
		t.Fatalf("the cut read failed: %v", problem)
	}
	if want := []uint64{0, uint64(len(data) / 2)}; !slices.Equal(machine.offsets, want) {
		t.Fatalf("the machine was asked from offsets %v, want %v", machine.offsets, want)
	}
	if held, err := os.ReadFile(product.Path); err != nil || !bytes.Equal(held, data) {
		t.Fatalf("the placed output is not the revision's bytes: %v", err)
	}
}

func serveCut(t *testing.T, machine *cutMachine) *machines.V1 {
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	pb.RegisterMachineServer(server, machine)
	go server.Serve(listener)
	conn, err := grpc.NewClient("passthrough:///cut", grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close(); server.Stop() })
	return &machines.V1{Client: &machinev1.Client{Machine: pb.NewMachineClient(conn)}}
}

func cutProduct(data []byte, directory string) records.Product {
	sum := sha256.Sum256(data)
	return records.Product{Item: "7/video", Output: "video", Rev: 2, Length: int64(len(data)),
		Digest: "sha256:" + hex.EncodeToString(sum[:]), Path: filepath.Join(directory, "7-video.mp4")}
}

// A download the machine stops answering keeps the bytes it read; the next read of the same
// revision asks the machine only for the rest, and the folder ends with the whole file alone.
func TestAnInterruptedOutputReadResumesFromItsBytes(t *testing.T) {
	data := bytes.Repeat([]byte("the run's output "), 1<<14)
	machine := &cutMachine{data: data, down: true}
	v1 := serveCut(t, machine)
	directory := t.TempDir()
	product := cutProduct(data, directory)
	if problem := writeOutputV1(t.Context(), v1, "run", directory, product, -1, nil); problem == nil ||
		problem.ErrName() != "machine_execution.transport_unavailable" {
		t.Fatalf("a cut read answered %v", problem)
	}
	if _, err := os.Stat(product.Path); !os.IsNotExist(err) {
		t.Fatal("a cut read placed the output")
	}
	machine.mu.Lock()
	machine.down, machine.offsets = false, nil
	machine.mu.Unlock()
	if problem := writeOutputV1(t.Context(), v1, "run", directory, product, -1, nil); problem != nil {
		t.Fatalf("the second read failed: %v", problem)
	}
	if want := []uint64{uint64(len(data) / 2)}; !slices.Equal(machine.offsets, want) {
		t.Fatalf("the machine was asked from offsets %v, want %v", machine.offsets, want)
	}
	held, err := os.ReadFile(product.Path)
	if err != nil || !bytes.Equal(held, data) {
		t.Fatalf("the placed output is not the revision's bytes: %v", err)
	}
	if entries, _ := os.ReadDir(directory); len(entries) != 1 {
		t.Fatalf("the folder holds %d entries, want the output alone", len(entries))
	}
}
