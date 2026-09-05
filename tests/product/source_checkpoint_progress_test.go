package producttest

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// An authenticated independent worker can report local progress while source
// bodies are still missing. The owner must retain it across a database reopen
// without calling it remotely durable, and must reject a changed frozen plan.
func TestIncompleteSourceCheckpointIsDurableObservation(t *testing.T) {
	const requestID = "job-source-checkpoint-progress"
	ready := make(chan struct{})
	frames := make(chan *pb.ModelSourceCheckpoint, 4)
	stop := make(chan struct{})
	defer close(stop)
	pod := &standInPod{}
	pod.onSession = func(send func(*pb.WorkerFrame) error, ownerEpoch, controlEpoch uint64, bootID string) {
		select {
		case <-ready:
		case <-stop:
			return
		}
		for {
			select {
			case checkpoint := <-frames:
				if checkpoint == nil {
					_ = send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ModelSourcePrepared{ModelSourcePrepared: &pb.ModelSourcePrepared{
						RecordOwnerEpoch: ownerEpoch, ControlStreamEpoch: controlEpoch, WorkerBootId: bootID,
						OperationId: requestID, SourceSelectionDigest: bytes.Repeat([]byte{0x22}, 32),
						Outcome:  pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED,
						SafeCode: "model_source_preparer_unavailable"}}})
					continue
				}
				_ = send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ModelSourcePrepared{
					ModelSourcePrepared: &pb.ModelSourcePrepared{
						RecordOwnerEpoch: ownerEpoch, ControlStreamEpoch: controlEpoch,
						WorkerBootId: bootID, OperationId: requestID,
						SourceSelectionDigest: bytes.Repeat([]byte{0x22}, 32),
						Outcome:               pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE,
						Checkpoints:           []*pb.ModelSourceCheckpoint{checkpoint}}}})
			case <-stop:
				return
			}
		}
	}
	o, _ := attachStandInRental(t, "source-checkpoint-progress", pod)
	header := []byte("{\"weight_map\":{\"x\":\"shard.safetensors\"}}")
	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "paul/minimax-h3",
		Source: "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40), SourceSelection: verdictSelection,
		SourceFiles: []records.ModelTransferSourceFile{{Member: "model.safetensors.index.json",
			SHA256: verdictGoodSHA, Length: int64(len(header)), Header: header}},
		SourceProfiles: map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
		Outputs:        []records.ModelTransferOutput{{Name: "full"}},
	}
	_, created, problem := o.store.Submit(records.Request{
		ID: requestID, IdemKey: requestID, BodyDigest: "sha256:" + strings.Repeat("c", 64),
		Package: "paul/minimax-h3-tools", Entrypoint: "four-lane", State: "queued", Kind: "job",
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]", Worker: "rental-source-checkpoint-progress",
		Rental: true, ModelTransfer: intent})
	fatal(t, problem)
	if !created {
		t.Fatal("the source operation was not created")
	}
	o.c.ObservePhase(requestID, orchestrator.PhaseSample{Name: orchestrator.PhaseWarming})
	close(ready)
	checkpoint := func(head, plan byte, index, size uint64) *pb.ModelSourceCheckpoint {
		return &pb.ModelSourceCheckpoint{Slot: "shared", Head: &pb.Ref{Digest: bytes.Repeat([]byte{head}, 32), Length: 500},
			PlanDigest: bytes.Repeat([]byte{plan}, 32), Index: index, Bytes: size}
	}
	frames <- checkpoint(0x33, 0x44, 4, 100)
	awaitSourceCheckpointIndex(t, o.store, requestID, 4)
	frames <- checkpoint(0x22, 0x44, 3, 75) // an older replay must not regress progress
	frames <- checkpoint(0x55, 0x44, 5, 125)
	awaitSourceCheckpointIndex(t, o.store, requestID, 5)
	phase, present := o.c.PhaseOf(requestID)
	if !present || phase.Name != orchestrator.PhasePreparing || !phase.HasBytes || phase.Moved != 125 {
		t.Fatalf("source conversion did not replace stale warming with measured bytes: %+v", phase)
	}
	frames <- nil
	for deadline := time.Now().Add(10 * time.Second); ; {
		phase, _ := o.c.PhaseOf(requestID)
		if strings.Contains(phase.Detail, "model_source_preparer_unavailable") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("preparer disconnect was not exposed in phase")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// Read through a newly opened database handle, after all temporary caller
	// byte buffers could have disappeared. Neither header nor head lives only in memory.
	reopened, problem := records.Open(filepath.Join(o.root, "creator.sqlite"))
	fatal(t, problem)
	defer reopened.Close()
	transfer, problem := reopened.ModelTransferOf(requestID)
	fatal(t, problem)
	if transfer == nil || transfer.State == "failed" || !bytes.Equal(transfer.SourceFiles[0].Header, header) {
		t.Fatalf("incomplete preparation lost the accepted header or failed the source: %+v", transfer)
	}
	progress, problem := reopened.ModelSourceProgress(requestID)
	fatal(t, problem)
	if len(progress) != 1 || progress[0].Observed.Index != 5 || progress[0].Acknowledged != nil {
		t.Fatalf("local progress disappeared or was falsely acknowledged: %+v", progress)
	}

	frames <- checkpoint(0x66, 0x77, 6, 150) // same source and slot, changed TensorFS plan
	deadline := time.Now().Add(10 * time.Second)
	for {
		transfer, problem = reopened.ModelTransferOf(requestID)
		fatal(t, problem)
		if transfer != nil && transfer.State == "failed" {
			if transfer.ErrorCode != "model_transfer.source_checkpoint_plan_changed" {
				t.Fatalf("wrong source refusal: %s", transfer.ErrorCode)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("worker changed the frozen source plan without a terminal refusal")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func awaitSourceCheckpointIndex(t *testing.T, store *records.Store, requestID string, index int64) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		progress, problem := store.ModelSourceProgress(requestID)
		fatal(t, problem)
		if len(progress) == 1 && progress[0].Observed.Index == index {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("source checkpoint index %d never reached durable owner state: %+v", index, progress)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
