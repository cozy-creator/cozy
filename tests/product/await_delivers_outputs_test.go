package producttest

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// finishingMachine is a Runtime whose run ends while its observer reads, as a real one's last
// burst does: the state it reports is behind the page that follows it, and the run's output
// and outcome land between that page and the state read after.
type finishingMachine struct {
	*runtimeMachine
	blob  []byte
	armed int // 1: journal a log after the next state read; 2: finish after the next page
}

func (m *finishingMachine) arm() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.armed = 1
}

func (m *finishingMachine) GetMachineExecution(ctx context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	state, err := m.runtimeMachine.GetMachineExecution(ctx, query)
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.armed == 1 && state.State == "running" {
		m.record("log", []byte(`{"type":"log","payload":{"name":"decode_video"}}`))
		m.armed = 2
	}
	return state, err
}

func (m *finishingMachine) ListMachineExecutionEvents(ctx context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	page, err := m.runtimeMachine.ListMachineExecutionEvents(ctx, query)
	m.mu.Lock()
	finish := m.armed == 2 && !query.Wait
	if finish {
		m.armed = 3
		sum := sha256.Sum256(m.blob)
		source := &pb.NativeByteRetentionRequest{RetentionId: childDigest("9"), Source: &pb.NativeByteTreeRef{
			ProducerRootId: childDigest("5"), ReceiptDigest: sum[:], ContentBytes: uint64(len(m.blob)),
			Manifest: &pb.Ref{Digest: sum[:], Length: uint64(len(m.blob))}}}
		m.state.Sequence++
		m.events = append(m.events, &pb.MachineExecutionEvent{Sequence: m.state.Sequence, AttemptOrdinal: 1, Kind: "product",
			BodyCanonicalBytes: []byte(`{"output":"image"}`), Product: &pb.RunProduct{Output: "image", Op: pb.RunProductOp_RUN_PRODUCT_OP_SET,
				Content: &pb.Ref{Digest: sum[:], Length: uint64(len(m.blob))}, MediaType: "image/png", Source: source}})
	}
	m.mu.Unlock()
	if finish {
		m.finish()
	}
	return page, err
}

func (m *finishingMachine) ReadByteTreeObject(call *pb.NativeByteReadCall, stream grpc.ServerStreamingServer[pb.NativeByteReadChunk]) error {
	if sum := sha256.Sum256(m.blob); string(call.GetObject().GetDigest()) != string(sum[:]) {
		return status.Error(codes.NotFound, "no such object")
	}
	return stream.Send(&pb.NativeByteReadChunk{Offset: call.Offset, Data: m.blob[call.Offset:]})
}

// awaitRun runs `cozy run … --rental=tessa --await --json --out <out>`, run 1591's shape, and
// finishes the machine's run in its observer's read once the run is on the machine, after
// `meanwhile`.
func awaitRun(t *testing.T, out string, meanwhile func()) (*finishingMachine, *records.Store, int, string, string) {
	t.Helper()
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.workflow = []byte(strings.Replace(string(workflowInterface), `"result":{"fields":[]}}],"format"`,
		`"result":{"fields":[{"name":"image","type":{"asset":"image"},"asset_bound":{"max_bytes":1048576,"media_types":["image/png"]}}]}}],"format"`, 1))
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &finishingMachine{runtimeMachine: &runtimeMachine{triage: bundle}, blob: []byte("\x89PNG the run's last output")}
	pod := &fakePod{machine: machine, deviceCount: 4}
	root, layout := rentedLadderMachine(t, h, pod, nil)
	input := filepath.Join(t.TempDir(), "input.json")
	must(t, os.WriteFile(input, []byte(`{"steps":1}`), 0o600))
	command := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "run", ladderPackage+"/generate", "--input", input,
		"--rental=tessa", "--await", "--json", "--out", out, "--idempotency-key", "awaited")
	command.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	command.Stdout, command.Stderr = &stdout, &stderr
	must(t, command.Start())
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	waitFor(t, root, "the run on its machine", func() bool {
		request, _ := store.RequestByIdempotencyKey("awaited")
		link, _ := store.MachineExecution(request.ID)
		return link != nil && link.RemoteCursor > 0
	})
	meanwhile()
	machine.arm()
	_ = command.Wait()
	return machine, store, command.ProcessState.ExitCode(), stdout.String(), stderr.String()
}

// `cozy run --await` returns once every output is in the chosen folder at its final revision,
// sha256 verified, even when the run's output and outcome land while the observer reads.
func TestAwaitReturnsOnceEveryOutputIsInItsFolder(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out")
	machine, store, code, stdout, stderr := awaitRun(t, out, func() {})
	if code != 0 {
		t.Fatalf("the awaited run exited %d:\n%s\n%s", code, stdout, stderr)
	}
	request, problem := store.RequestByIdempotencyKey("awaited")
	fatal(t, problem)
	numbered, problem := store.RequestByReference(request.ID)
	fatal(t, problem)
	sum := sha256.Sum256(machine.blob)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	path := filepath.Join(out, fmt.Sprintf("%d-image.png", numbered.Number))
	entries, _ := os.ReadDir(out)
	data, err := os.ReadFile(path)
	if len(entries) != 1 || err != nil || digestOf(data) != digest {
		events, _ := store.EventsAfter(request.ID, 0, 1000)
		var kinds []string
		for _, event := range events {
			kinds = append(kinds, event.Type)
		}
		t.Fatalf("the folder does not hold exactly the run's output at %s: %d entries: %v\nevents: %s\n%s\n%s", path, len(entries), err,
			strings.Join(kinds, " "), stdout, stderr)
	}
	var result struct {
		Saved []struct {
			Output, Path, Digest string
		} `json:"saved"`
	}
	must(t, json.Unmarshal([]byte(stdout), &result))
	if len(result.Saved) != 1 || result.Saved[0].Path != path || result.Saved[0].Digest != digest || result.Saved[0].Output != "image" {
		t.Fatalf("the awaited result does not name the saved output: %s", stdout)
	}
}

// An output the chosen folder refuses once the run is under way is not delivered: `--await`
// exits non-zero, naming the output and the machine that still holds its bytes.
func TestAwaitNamesAnOutputItCouldNotDeliver(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out")
	must(t, os.MkdirAll(out, 0o700))
	t.Cleanup(func() { _ = os.Chmod(out, 0o700) })
	_, _, code, stdout, stderr := awaitRun(t, out, func() { must(t, os.Chmod(out, 0o500)) })
	var failure struct {
		Error struct {
			Code    string `json:"code"`
			Details struct {
				Missing []string `json:"missing"`
				BytesOn string   `json:"bytes_on"`
			} `json:"details"`
		} `json:"error"`
	}
	if code == 0 || json.Unmarshal([]byte(stdout), &failure) != nil || failure.Error.Code != "run.outputs_undelivered" ||
		strings.Join(failure.Error.Details.Missing, ",") != "1/image" || failure.Error.Details.BytesOn != "machine tessa" {
		t.Fatalf("an undelivered output exited %d without naming it and its machine:\n%s\n%s", code, stdout, stderr)
	}
}
