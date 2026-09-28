package producttest

import (
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// peerMachine is a newer or careless Runtime: mid-run it reports a state this Creator does not
// know, journals one unreadable event, and publishes an output whose label and media type are
// not printable ASCII. It serves that output's bytes.
type peerMachine struct {
	*runtimeMachine
	blob []byte
}

func (m *peerMachine) misbehave() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state.State = "thawing"
	m.state.Sequence++
	m.events = append(m.events, &pb.MachineExecutionEvent{Sequence: m.state.Sequence, AttemptOrdinal: 1, Kind: "log",
		BodyCanonicalBytes: []byte("not json")})
	sum := sha256.Sum256(m.blob)
	source := &pb.NativeByteRetentionRequest{RetentionId: childDigest("9"), Source: &pb.NativeByteTreeRef{
		ProducerRootId: childDigest("5"), ReceiptDigest: sum[:], ContentBytes: uint64(len(m.blob)),
		Manifest: &pb.Ref{Digest: sum[:], Length: uint64(len(m.blob))}}}
	body, _ := json.Marshal(map[string]string{"output": "image"})
	m.state.Sequence++
	m.events = append(m.events, &pb.MachineExecutionEvent{Sequence: m.state.Sequence, AttemptOrdinal: 1, Kind: "product",
		BodyCanonicalBytes: body, Product: &pb.RunProduct{Output: "image", Op: pb.RunProductOp_RUN_PRODUCT_OP_SET,
			Content: &pb.Ref{Digest: sum[:], Length: uint64(len(m.blob))}, MediaType: "image/png\x00", Label: "café \x01", Source: source}})
}

func (m *peerMachine) ReadByteTreeObject(call *pb.NativeByteReadCall, stream grpc.ServerStreamingServer[pb.NativeByteReadChunk]) error {
	if sum := sha256.Sum256(m.blob); string(call.GetObject().GetDigest()) != string(sum[:]) {
		return status.Error(codes.NotFound, "no such object")
	}
	return stream.Send(&pb.NativeByteReadChunk{Offset: call.Offset, Data: m.blob[call.Offset:]})
}

// A machine is an independently upgraded peer: an unknown state, an unreadable event, and an
// output whose words are not printable each cost one note, never the page. The run is
// followed to its end, its output recorded under printable words, and nothing wedges.
func TestAPeerMachineNeverWedgesItsRun(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &peerMachine{runtimeMachine: &runtimeMachine{triage: bundle}, blob: []byte("\x89PNG peer output")}
	pod := &fakePod{machine: machine, deviceCount: 4}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json",
		"--idempotency-key", "peer-tolerance"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	machine.misbehave()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("peer-tolerance")
	fatal(t, problem)
	waitFor(t, root, "the output revision past the unreadable event", func() bool {
		products, _ := store.Products(request.ID)
		return len(products) == 1
	})
	machine.finish()
	waitFor(t, root, "the run's end", func() bool {
		row, _ := store.RequestRow(request.ID)
		link, _ := store.MachineExecution(request.ID)
		return row != nil && row.State == "succeeded" && link != nil && link.Collected
	})
	products, problem := store.Products(request.ID)
	fatal(t, problem)
	if len(products) != 1 || products[0].Label != "caf? ?" || products[0].MediaType != "application/octet-stream" || products[0].Output != "image" {
		t.Fatalf("the output was not kept under printable words: %+v", products)
	}
	events, problem := store.EventsAfter(request.ID, 0, 1000)
	fatal(t, problem)
	var notes []string
	for _, event := range events {
		if event.Type == "request.warning" {
			notes = append(notes, event.Payload["message"].(string))
		}
	}
	joined := strings.Join(notes, "\n")
	if strings.Count(joined, `state "thawing"`) != 1 || !strings.Contains(joined, "was unreadable and is skipped") {
		t.Fatalf("the peer's unknown state and unreadable event were not each noted once:\n%s", joined)
	}
}
