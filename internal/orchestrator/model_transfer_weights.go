package orchestrator

import (
	"encoding/base64"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type WeightsTransferDecision struct {
	ObjectID      string            `json:"object_id"`
	Length        int64             `json:"length"`
	URL           string            `json:"url,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`
	ExpiresAtUnix uint64            `json:"expires_at_unix,omitempty"`
	Held          bool              `json:"held,omitempty"`
}

func (c *Orchestrator) onModelTransferWeightsReceipt(s *session, frame *pb.WeightsReceiptFrame) {
	transfer, problem := c.opt.Store.ModelTransferOf(frame.RequestId)
	if problem != nil || transfer == nil {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.RequestId)
	if problem != nil || request == nil || !request.IsJob() {
		return
	}
	attempt, problem := c.opt.Store.AttemptRow(frame.RequestId, int64(frame.AttemptOrdinal))
	if problem != nil || attempt == nil || attempt.InstanceID != s.instanceID {
		return
	}
	invocationDigest, err := canonical.Spell(frame.InvocationSpecDigest)
	manifestDigest := ""
	if frame.Manifest != nil {
		manifestDigest, _ = canonical.Spell(frame.Manifest.Digest)
	}
	if err != nil || invocationDigest != attempt.InvocationDigest || frame.WeightsReceipt == nil ||
		frame.Manifest == nil || manifestDigest == "" || frame.Manifest.Length == 0 ||
		frame.WriterGeneration == 0 || frame.WeightsTransactionId == "" {
		return
	}
	receipt, problem := parseWeightsReceiptRef(canonical.Doc{
		"weights_receipt_digest": canonicalSpell(frame.WeightsReceipt.WeightsReceiptDigest),
		"weights_receipt_canonical_bytes": base64.StdEncoding.EncodeToString(
			frame.WeightsReceipt.WeightsReceiptCanonicalBytes),
	})
	if problem != nil || receipt.RequestID != frame.RequestId ||
		receipt.InvocationDigest != invocationDigest || receipt.OutputSlot != frame.OutputSlot {
		return
	}
	receiptDoc, err := canonical.Read(receipt.ReceiptBytes, &pb.WeightsReceipt{})
	if err != nil || receiptDoc.Str("weights_transaction_id") != frame.WeightsTransactionId {
		return
	}
	evidence, err := base64.StdEncoding.Strict().DecodeString(
		receiptDoc.Str("checkpoint_evidence_canonical_bytes"))
	if err != nil || len(evidence) == 0 || len(evidence) > pb.MaxCheckpointEvidenceBytes {
		return
	}
	objects := make([]records.ModelTransferObject, 0, len(frame.Objects))
	previous := ""
	manifestPresent := false
	for _, object := range frame.Objects {
		if object == nil || object.ObjectId <= previous || object.Length == 0 ||
			object.Length > uint64(^uint64(0)>>1) || object.SourceRef == "" {
			return
		}
		if _, digestErr := canonical.Raw(object.ObjectId); digestErr != nil {
			return
		}
		previous = object.ObjectId
		manifestPresent = manifestPresent || object.ObjectId == manifestDigest &&
			object.Length == frame.Manifest.Length
		objects = append(objects, records.ModelTransferObject{ObjectID: object.ObjectId,
			Length: int64(object.Length), SourceRef: object.SourceRef})
	}
	if len(objects) == 0 || !manifestPresent {
		return
	}
	problem = c.opt.Store.RecordModelTransferWeights(records.ModelTransferWeights{
		RequestID: frame.RequestId, OutputSlot: frame.OutputSlot, ManifestID: manifestDigest,
		ManifestLength: int64(frame.Manifest.Length), Evidence: evidence, Objects: objects,
		Attempt: int64(frame.AttemptOrdinal), InvocationDigest: invocationDigest,
		TransactionID: frame.WeightsTransactionId, ReceiptDigest: receipt.ReceiptDigest,
		Receipt: receipt.ReceiptBytes})
	if problem == nil {
		c.signalTransfer(frame.RequestId)
	}
}

func (c *Orchestrator) onModelTransferWeightsStatus(s *session,
	frame *pb.WeightsTransferStatus,
) {
	weights, problem := c.opt.Store.ModelTransferWeights(frame.RequestId,
		int64(frame.AttemptOrdinal), frame.OutputSlot)
	if problem != nil || weights == nil || frame.Length == 0 || frame.Length > uint64(^uint64(0)>>1) {
		return
	}
	invocation := canonicalSpell(frame.InvocationSpecDigest)
	if weights.Attempt != int64(frame.AttemptOrdinal) || weights.InvocationDigest != invocation ||
		weights.TransactionID != frame.WeightsTransactionId {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.RequestId)
	attempt, attemptProblem := c.opt.Store.AttemptRow(frame.RequestId, int64(frame.AttemptOrdinal))
	if problem != nil || attemptProblem != nil || request == nil || attempt == nil ||
		request.Worker == "" || rentalInstanceID(request.Worker) != s.instanceID ||
		attempt.InstanceID != s.instanceID {
		return
	}
	state := trimEnum(pb.WeightsTransferState_name[int32(frame.State)], "WEIGHTS_TRANSFER_STATE_")
	state = map[string]string{"ACCEPTED": "accepted", "READING": "reading",
		"UPLOADING": "uploading", "UPLOADED": "uploaded", "ALREADY_PRESENT": "already_present",
		"HELD": "held", "FAILED": "failed"}[state]
	if state == "" || frame.TransferredBytes > frame.Length {
		return
	}
	problem = c.opt.Store.RecordModelTransferObjectStatus(records.ModelTransferObject{
		RequestID: frame.RequestId, Attempt: int64(frame.AttemptOrdinal),
		OutputSlot: frame.OutputSlot, ObjectID: frame.ObjectId, Length: int64(frame.Length),
		OperationID: frame.OperationId, GrantRevision: int64(frame.GrantRevision),
		UpdateSequence: int64(frame.UpdateSequence), State: state,
		Transferred: int64(frame.TransferredBytes), SafeCode: frame.SafeCode,
		SafeDetail: frame.SafeDetail})
	if problem != nil {
		return
	}
	c.signalTransfer(frame.RequestId)
}

func canonicalSpell(raw []byte) string {
	value, _ := canonical.Spell(raw)
	return value
}
