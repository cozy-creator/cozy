package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// An independent claimed Host status drives the real Creator Ready consumer.
// No model publication exists for this job, so Ready must explicitly choose no restore.
func TestWeightsIntentStatusTriggersExactReadyWithoutAPublication(t *testing.T) {
	for _, maximum := range []uint64{0, 8 << 20} {
		t.Run(strconv.FormatUint(maximum, 10), func(t *testing.T) {
			weightsIntentReady(t, maximum)
		})
	}
}

func weightsIntentReady(t *testing.T, maximum uint64) {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	ready := make(chan *pb.WeightsIntentReadyRequest, 1)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	var sent *pb.WeightsTransactionStatus
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		offer := frame.GetAttemptOffer()
		if offer == nil {
			return false, nil
		}
		digest, err := canonical.Spell(offer.InvocationSpecDigest)
		if err != nil {
			return true, err
		}
		sent = &pb.WeightsTransactionStatus{WeightsTransactionId: "sha256:" + strings.Repeat("d", 64), RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: digest, OutputSlot: "model", WriterEpoch: 7, TensorfsDeclarationDigest: bytes.Repeat([]byte{0xe}, 32), State: pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT}
		return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_WeightsTransaction{WeightsTransaction: sent}})
	}
	pod.weightsReady = func(request *pb.WeightsIntentReadyRequest) (*pb.WeightsHostAck, error) {
		ready <- request
		subject := request.Weights
		return &pb.WeightsHostAck{RecordOwnerEpoch: request.RecordOwnerEpoch, WorkerBootId: request.WorkerBootId, RequestId: subject.RequestId, AttemptOrdinal: request.AttemptOrdinal,
			InvocationSpecDigest: subject.InvocationSpecDigest, OutputSlot: subject.OutputSlot, WeightsTransactionId: subject.WeightsTransactionId, WriterEpoch: subject.WriterEpoch,
			TensorfsDeclarationDigest: subject.TensorfsDeclarationDigest, Stage: pb.WeightsHostStage_WEIGHTS_HOST_STAGE_INTENT, Outcome: pb.WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_RECORDED}, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	owner := hostOwner(t, "weights-ready-consumer-"+strconv.FormatUint(maximum, 10), rentalWiring(connection, private))
	id, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: "weights-ready", Package: "cozy/h3-package", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7", Kind: "job", Org: "paul", Payload: []byte(`{}`), Outputs: []string{"model"},
		WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model", MimeType: orchestrator.WeightsManifestMime, MaxBytes: maximum}}, Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "actual Creator Ready call", func() bool { return len(ready) > 0 })
	request := <-ready
	if request.Checkpoint != nil || request.ControlStreamEpoch != 0 || request.WorkerBootId != podBootID || request.Weights.RequestId != id || request.Weights.OutputSlot != sent.OutputSlot || request.Weights.WeightsTransactionId != sent.WeightsTransactionId || request.Weights.WriterEpoch != sent.WriterEpoch || request.AttemptOrdinal != sent.AttemptOrdinal {
		t.Fatalf("Creator changed current intent or fabricated restore: %v", request)
	}
	if !bytes.Equal(request.Weights.TensorfsDeclarationDigest, sent.TensorfsDeclarationDigest) {
		t.Fatal("Creator reauthored native declaration identity")
	}
}
