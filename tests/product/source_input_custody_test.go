package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Ready Model bindings and a free producer seat do not authorize dispatch while
// the source's latest checkpoint is only a local observation.
func TestProducerWaitsForAcknowledgedSourceInputs(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	configured, acknowledged, stop := make(chan struct{}), make(chan struct{}), make(chan struct{})
	defer close(stop)
	var requestID string
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	pod.onJobReady = func(ready *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		select {
		case <-configured:
		case <-stop:
			return context.Canceled
		}
		r := ready.GetObservedState()
		metadata := &pb.WorkerFrame{Msg: &pb.WorkerFrame_ModelSourceFileStatus{ModelSourceFileStatus: &pb.ModelSourceFileStatus{
			RecordOwnerEpoch: r.RecordOwnerEpoch, ControlStreamEpoch: r.ControlStreamEpoch, WorkerBootId: r.WorkerBootId,
			OperationId: requestID, SourceSelectionDigest: bytes.Repeat([]byte{0x22}, 32),
			Member: "source.index.json", ObjectId: "sha256:" + verdictGoodSHA, Length: 2, CapabilityRevision: 1,
			State: pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED}}}
		if err := send(metadata); err != nil {
			return err
		}
		if err := send(ready); err != nil {
			return err
		}
		go func() {
			select {
			case <-acknowledged:
			case <-stop:
				return
			}
			// A later source observation wakes the owner after the uploader's durable
			// acknowledgment. This uses the real claimed control stream.
			metadata.GetModelSourceFileStatus().CapabilityRevision = 2
			_ = send(metadata)
		}()
		return nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "source-input-custody", rentalWiring(connection, private))
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: "paul/minimax-h3",
		Source: "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40), SourceSelection: verdictSelection,
		SourceProfiles: map[string]string{"shared": "hf/minimax-h3/shared-bf16/1"},
		SourceFiles: []records.ModelTransferSourceFile{{Member: "source.index.json", SHA256: verdictGoodSHA,
			Length: 2, Header: []byte("{}")}}, Outputs: []records.ModelTransferOutput{{Name: "model"}}}
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "source-input-custody", Package: "cozy/h3-package", Entrypoint: "four-lane",
		PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7", Kind: "job", Org: "paul",
		Payload: []byte("{}"), Outputs: []string{"model"}, Worker: podRental, Rental: true, RentalRequired: true,
		ModelTransfer: intent,
	})
	fatal(t, problem)
	fatal(t, o.store.BeginModelTransferMaterialization(requestID))
	checkpoint := records.ModelCheckpoint{Slot: "shared", HeadID: "sha256:" + strings.Repeat("3", 64),
		HeadLength: 500, PlanDigest: "sha256:" + strings.Repeat("4", 64), Index: 0, Bytes: 256}
	fatal(t, o.store.ObserveModelSourceCheckpoints(requestID, verdictSelection, podBootID, []records.ModelCheckpoint{checkpoint}))
	fatal(t, o.store.CompleteModelTransferMaterialization(requestID, []records.ModelRef{{Package: "cozy/h3-package",
		Slot: "shared", Model: "source/shared", Manifest: "sha256:" + strings.Repeat("5", 64), ManifestLength: 164}}, podBootID))
	close(configured)
	waitUntil(t, "source custody barrier before producer dispatch", func() bool {
		pod.mu.Lock()
		dispatched := len(pod.offers)
		pod.mu.Unlock()
		if dispatched != 0 {
			t.Fatal("producer received an offer before its source checkpoint was acknowledged")
		}
		phase, exists := o.c.PhaseOf(requestID)
		return exists && strings.Contains(phase.Detail, "retaining converted source checkpoints")
	})
	pod.mu.Lock()
	before := len(pod.offers)
	pod.mu.Unlock()
	if before != 0 {
		t.Fatal("producer received an offer before its source checkpoint was acknowledged")
	}
	// This is the RecordOwner boundary the verified uploader commits after Hub custody.
	fatal(t, o.store.AcknowledgeModelSourceCheckpoint(requestID, verdictSelection, podBootID, "", checkpoint))
	close(acknowledged)
	waitUntil(t, "producer offer after source custody acknowledgment", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) == 1
	})
}
