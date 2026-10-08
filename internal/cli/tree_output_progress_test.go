package cli

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

type progressTreeMachine struct {
	pb.UnimplementedMachineServer
	files map[string][]byte
	reads []string
}

func (m *progressTreeMachine) Read(request *pb.ReadRequest, stream grpc.ServerStreamingServer[pb.ReadFrame]) error {
	member := request.GetOutput().Member
	m.reads = append(m.reads, member)
	data := m.files[member]
	sum := sha256.Sum256(data)
	if err := stream.Send(&pb.ReadFrame{Rev: 1, Length: uint64(len(data)), Digest: "sha256:" + hex.EncodeToString(sum[:])}); err != nil {
		return err
	}
	for start := 0; start < len(data); start += 16 << 10 {
		if err := stream.Send(&pb.ReadFrame{Data: data[start:min(start+16<<10, len(data))]}); err != nil {
			return err
		}
	}
	return nil
}

// Tree members use the ordinary output transport and progress stream. A shared blob
// counts once, and no progress callback exposes a partially published destination.
func TestTreeOutputReportsMemberBytesBeforeAtomicPublication(t *testing.T) {
	first, second := bytes.Repeat([]byte("trace"), 20000), bytes.Repeat([]byte("summary"), 10000)
	machine := &progressTreeMachine{files: map[string][]byte{
		"nested/trace.gz": first, "same-trace.gz": first, "summary.json": second, "empty": {},
	}}
	var entries []map[string]any
	for _, name := range []string{"nested/trace.gz", "same-trace.gz", "summary.json", "empty"} {
		data := machine.files[name]
		sum := sha256.Sum256(data)
		entries = append(entries, map[string]any{"kind": "file", "path": name,
			"blob": map[string]any{"length": len(data), "sha256": hex.EncodeToString(sum[:])}})
	}
	manifest, err := json.Marshal(map[string]any{"entries": entries})
	if err != nil {
		t.Fatal(err)
	}
	machine.files[""] = manifest
	listener := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	pb.RegisterMachineServer(server, machine)
	go server.Serve(listener)
	conn, err := grpc.NewClient("passthrough:///tree", grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return listener.Dial() }))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close(); server.Stop() })
	v1 := &machines.V1{Client: &machinev1.Client{Machine: pb.NewMachineClient(conn)}}
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	request, _, problem := store.Submit(records.Request{ID: "tree-run", IdemKey: "tree-run", BodyDigest: "tree-run", Package: "test/tree", Entrypoint: "main", Payload: []byte(`{}`)})
	if problem != nil {
		t.Fatal(problem)
	}
	directory := t.TempDir()
	sum := sha256.Sum256(manifest)
	product := records.Product{Item: "1/trace", Output: "trace", Rev: 1, Length: int64(len(manifest)),
		Digest: "sha256:" + hex.EncodeToString(sum[:]), Path: filepath.Join(directory, "1-trace"), MediaType: resultfiles.TreeMediaType}
	fetcher := &fetcherV1{m: &machineRuns{store: store}, request: request, finishing: true}
	emit := fetcher.progress(product)
	var samples []int64
	totalBytes := int64(len(first) + len(second))
	progress := func(held, total int64) {
		if _, err := os.Stat(product.Path); !os.IsNotExist(err) {
			t.Fatal("Tree was published before all received bytes were verified")
		}
		if total != totalBytes || held < 0 || held > total || len(samples) > 0 && held < samples[len(samples)-1] {
			t.Fatalf("invalid transfer progress %d/%d after %v", held, total, samples)
		}
		samples = append(samples, held)
		emit(held, total)
	}
	if problem := writeOutputV1(t.Context(), v1, request.ID, directory, product, -1, progress); problem != nil {
		t.Fatal(problem)
	}
	if len(samples) < 3 || samples[0] != 0 || samples[1] <= 0 || samples[1] >= totalBytes || samples[len(samples)-1] != totalBytes {
		t.Fatalf("member bytes did not report incremental progress: %v", samples)
	}
	for _, name := range []string{"nested/trace.gz", "same-trace.gz", "summary.json", "empty"} {
		got, err := os.ReadFile(filepath.Join(product.Path, name))
		if err != nil || !bytes.Equal(got, machine.files[name]) {
			t.Fatalf("exported %s differs: %v", name, err)
		}
	}
	if len(machine.reads) != 4 { // Manifest plus three distinct blobs, including empty.
		t.Fatalf("duplicate blob was downloaded again: %v", machine.reads)
	}
	events, problem := store.EventsAfter(request.ID, 0, 100)
	if problem != nil {
		t.Fatal(problem)
	}
	if len(events) < 2 {
		t.Fatalf("download progress did not reach the normal event stream: %+v", events)
	}
	var payload struct {
		Payload struct {
			Stage, Unit     string
			Position, Total int64
			Rate            float64
		}
	}
	raw, err := json.Marshal(events[len(events)-1].Payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Payload.Stage != "saving trace" || payload.Payload.Unit != "bytes" || payload.Payload.Position != totalBytes || payload.Payload.Total != totalBytes || payload.Payload.Rate <= 0 {
		t.Fatalf("final progress omitted member count or rate: %+v", payload)
	}
	// Final collection follows the live fetcher. The same complete revision must
	// verify existing local files and only re-read its small signed manifest.
	previousReads := len(machine.reads)
	if problem := writeOutputV1(t.Context(), v1, request.ID, directory, product, -1, nil); problem != nil {
		t.Fatal(problem)
	}
	if len(machine.reads) != previousReads+1 || machine.reads[previousReads] != "" {
		t.Fatalf("collection downloaded an already verified Tree again: %v", machine.reads[previousReads:])
	}
	// Directory existence is insufficient: edited bytes require the accepted
	// revision again, just as an edited ordinary file does.
	if err := os.WriteFile(filepath.Join(product.Path, "nested/trace.gz"), []byte("edited"), 0600); err != nil {
		t.Fatal(err)
	}
	previousReads = len(machine.reads)
	if problem := writeOutputV1(t.Context(), v1, request.ID, directory, product, -1, nil); problem != nil {
		t.Fatal(problem)
	}
	got, err := os.ReadFile(filepath.Join(product.Path, "nested/trace.gz"))
	if err != nil || !bytes.Equal(got, first) || len(machine.reads) != previousReads+4 {
		t.Fatalf("changed Tree was incorrectly reused: reads=%v error=%v", machine.reads[previousReads:], err)
	}
}
