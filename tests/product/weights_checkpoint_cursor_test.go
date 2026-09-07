package producttest

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestWeightsCheckpointCursorPreservesCustodyAcrossOwnerAndWorkerReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	defer func() { store.Close() }()
	const request, instance, session = "checkpoint-request", "checkpoint-worker", "checkpoint-session"
	digest := "sha256:" + strings.Repeat("a", 64)
	outputs := `[{"output_id":"model","mime_type":"application/vnd.cozy.model-manifest","max_bytes":8192}]`
	_, _, problem = store.Submit(records.Request{ID: request, IdemKey: request, BodyDigest: digest, Package: "test/producer", Entrypoint: "derive", State: "queued", Kind: "job", Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: outputs,
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "test/model", Outputs: []records.ModelTransferOutput{{Name: "model"}}}})
	fatal(t, problem)
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: instance, Package: "test/producer", WorkerID: "remote", Devices: []string{"cpu"}}))
	dispatch := func() int64 {
		ordinal, problem := store.Dispatch(records.Attempt{RequestID: request, InstanceID: instance, SessionID: session,
			InvocationDigest: digest, InvocationCanonical: []byte("{}"), WeightsOutputs: outputs})
		fatal(t, problem)
		fatal(t, store.OfferDispatch(request, ordinal, session))
		fatal(t, store.Accepted(request, ordinal, session))
		return ordinal
	}
	ordinal := dispatch()
	status := &pb.WeightsTransactionStatus{WeightsTransactionId: "sha256:" + strings.Repeat("b", 64), RequestId: request, AttemptOrdinal: uint64(ordinal), InvocationSpecDigest: digest, OutputSlot: "model", WriterEpoch: 1, TensorfsDeclarationDigest: bytes.Repeat([]byte{0xc}, 32), State: pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT}
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, "boot1", status))
	point := func(index uint64, marker byte) *pb.CheckpointRef {
		return &pb.CheckpointRef{Head: &pb.Ref{Digest: bytes.Repeat([]byte{marker}, 32), Length: 123}, PlanDigest: bytes.Repeat([]byte{0xd}, 32), Index: index, Bytes: index * 2048}
	}
	status.Checkpoint = point(1, 1)
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, "boot1", status))
	rows, problem := store.ModelWeightsProgress(request)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].Acknowledged != nil {
		t.Fatal("local observation claimed custody")
	}
	first := rows[0]
	fatal(t, store.AcknowledgeModelWeightsCheckpoint(first.Subject, ordinal, "boot1", "", first.Observed))
	revision, problem := store.NextCheckpointGrantRevision(request, "weights", "model")
	fatal(t, problem)
	status.Checkpoint = point(2, 2)
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, "boot1", status))
	for _, change := range []string{"declaration", "transaction", "owner", "slot", "boot", "plan", "counter"} {
		altered := proto.Clone(status).(*pb.WeightsTransactionStatus)
		owner, boot := instance, "boot1"
		switch change {
		case "declaration":
			altered.TensorfsDeclarationDigest = bytes.Repeat([]byte{0xe}, 32)
		case "transaction":
			altered.WeightsTransactionId = "sha256:" + strings.Repeat("f", 64)
		case "owner":
			owner = "foreign"
		case "slot":
			altered.OutputSlot = "unknown"
		case "boot":
			boot = "foreign"
		case "plan":
			altered.Checkpoint.PlanDigest = bytes.Repeat([]byte{0xe}, 32)
		case "counter":
			altered.Checkpoint.Bytes++
		}
		if problem := store.ObserveModelWeightsCheckpoint(owner, boot, altered); problem == nil {
			t.Fatalf("changed %s accepted", change)
		}
	}
	before, problem := store.ModelWeightsProgress(request)
	fatal(t, problem)
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	after, problem := store.ModelWeightsProgress(request)
	fatal(t, problem)
	a, _ := json.Marshal(before)
	b, _ := json.Marshal(after)
	if !bytes.Equal(a, b) {
		t.Fatal("owner restart changed checkpoint custody")
	}
	next, problem := store.NextCheckpointGrantRevision(request, "weights", "model")
	fatal(t, problem)
	if next != revision+1 {
		t.Fatal("owner restart reset grant fence")
	}
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: request, Attempt: ordinal, SessionID: session, InvocationDigest: digest, TerminalID: "lost-worker", TerminalDigest: "sha256:" + strings.Repeat("e", 64), Status: "FAILED", Cause: "worker_lost", RequestState: "requeue_pending"})
	fatal(t, problem)
	fatal(t, store.Closed(request, ordinal))
	_, started, _, problem := store.BeginRequeue(request, 2, false)
	fatal(t, problem)
	if !started {
		t.Fatal("request did not requeue")
	}
	ordinal = dispatch()
	status.AttemptOrdinal = uint64(ordinal)
	status.WriterEpoch = 1
	status.Checkpoint = nil
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, "boot2", status))
	rows, problem = store.ModelWeightsProgress(request)
	fatal(t, problem)
	if rows[0].Observed.HeadID != "" || rows[0].Acknowledged == nil || *rows[0].Acknowledged != first.Observed {
		t.Fatal("replacement adopted unacknowledged head or lost custody")
	}
	if problem := store.AcknowledgeModelWeightsCheckpoint(first.Subject, first.Attempt, "boot1", first.Observed.HeadID, before[0].Observed); problem == nil {
		t.Fatal("old uploader changed replacement custody")
	}
	status.Checkpoint = point(1, 1)
	fatal(t, store.ObserveModelWeightsCheckpoint(instance, "boot2", status))
	rows, problem = store.ModelWeightsProgress(request)
	fatal(t, problem)
	fatal(t, store.AcknowledgeModelWeightsCheckpoint(rows[0].Subject, ordinal, "boot2", first.Observed.HeadID, rows[0].Observed))
	source, problem := store.ModelSourceProgress(request)
	fatal(t, problem)
	if len(source) != 0 {
		t.Fatal("weights cursor leaked into source preparation")
	}
}
