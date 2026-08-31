package orchestrator

import (
	"context"
	"encoding/base64"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type ArtifactTransferDecision struct {
	ObjectID      string            `json:"object_id"`
	Length        int64             `json:"length"`
	URL           string            `json:"url,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`
	ExpiresAtUnix uint64            `json:"expires_at_unix,omitempty"`
	Held          bool              `json:"held,omitempty"`
}

// TransferProductionArtifact drives only the renter-owned decision lane. Tensorhub
// authored the bounded grants and pod-supervisor moves bytes directly from Runtime to
// object storage; Creator receives neither model bytes nor a bucket credential.
func (c *Orchestrator) TransferProductionArtifact(ctx context.Context, operationID,
	nodeName, outputSlot, rentalID, transferOperationID string,
	decisions []ArtifactTransferDecision,
) *exit.Error {
	artifacts, problem := c.opt.Store.ModelProductionArtifacts(operationID)
	if problem != nil {
		return problem
	}
	var artifact *records.ModelProductionArtifact
	for i := range artifacts {
		if artifacts[i].NodeName == nodeName && artifacts[i].OutputSlot == outputSlot {
			artifact = &artifacts[i]
			break
		}
	}
	if artifact == nil {
		return exit.New(exit.NotFound, "model production output %s.%s has no artifact receipt",
			nodeName, outputSlot)
	}
	objects, problem := c.opt.Store.ModelProductionObjects(operationID, nodeName, outputSlot)
	if problem != nil {
		return problem
	}
	byObject := make(map[string]ArtifactTransferDecision, len(decisions))
	for _, decision := range decisions {
		if decision.ObjectID == "" || decision.Length <= 0 || byObject[decision.ObjectID].ObjectID != "" {
			return exit.Named(exit.Validation, "model_production.transfer_decision_invalid",
				"artifact transfer decisions must name one positive-length row per object")
		}
		byObject[decision.ObjectID] = decision
	}
	for objectID, decision := range byObject {
		found := false
		for _, object := range objects {
			found = found || object.ObjectID == objectID && object.Length == decision.Length
		}
		if !found {
			return exit.Named(exit.Conflict, "model_production.transfer_decision_changed",
				"artifact transfer decision changed object %s", objectID)
		}
	}

	for {
		objects, problem = c.opt.Store.ModelProductionObjects(operationID, nodeName, outputSlot)
		if problem != nil {
			return problem
		}
		complete := len(byObject) > 0
		for _, object := range objects {
			if _, selected := byObject[object.ObjectID]; !selected {
				continue
			}
			if object.State == "failed" {
				return exit.Named(exit.Failed, object.SafeCode,
					"worker artifact transfer %s failed: %s", object.ObjectID, object.SafeDetail)
			}
			complete = complete && (object.State == "uploaded" ||
				object.State == "already_present" || object.State == "held")
		}
		if complete {
			return nil
		}
		s, problem := c.rentalControl(rentalID)
		if problem != nil {
			if wait := c.waitProduction(ctx, operationID); wait != nil {
				return wait
			}
			continue
		}
		specDigest, err := canonical.Raw(artifact.InvocationDigest)
		if err != nil {
			return exit.Internalf("persisted artifact invocation digest is malformed: %s", err)
		}
		receiptDigest, err := canonical.Raw(artifact.ReceiptDigest)
		if err != nil {
			return exit.Internalf("persisted artifact receipt digest is malformed: %s", err)
		}
		for _, object := range objects {
			if _, selected := byObject[object.ObjectID]; !selected {
				continue
			}
			if object.State == "uploaded" || object.State == "already_present" || object.State == "held" {
				continue
			}
			decision, ok := byObject[object.ObjectID]
			if !ok || decision.Length != object.Length {
				return exit.Named(exit.Conflict, "model_production.transfer_decision_changed",
					"artifact transfer decision changed object %s", object.ObjectID)
			}
			revision := uint64(object.GrantRevision + 1)
			request := &pb.ArtifactTransferRequest{
				RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: s.generation,
				WorkerBootId: s.bootID, RequestId: artifact.RequestID,
				AttemptOrdinal: uint64(artifact.Attempt), InvocationSpecDigest: specDigest,
				OutputSlot: artifact.OutputSlot, ArtifactTransactionId: artifact.TransactionID,
				ArtifactReceiptDigest: receiptDigest, OperationId: transferOperationID,
				GrantRevision: revision,
			}
			if decision.Held {
				request.Decision = &pb.ArtifactTransferRequest_Held{Held: &pb.ArtifactObjectRef{
					ObjectId: object.ObjectID, Length: uint64(object.Length),
				}}
			} else {
				names := make([]string, 0, len(decision.Headers))
				for name := range decision.Headers {
					names = append(names, name)
				}
				sort.Strings(names)
				headers := make([]*pb.ArtifactUploadHeader, 0, len(names))
				for _, name := range names {
					headers = append(headers, &pb.ArtifactUploadHeader{Name: name,
						Value: decision.Headers[name]})
				}
				request.Decision = &pb.ArtifactTransferRequest_UploadGrant{UploadGrant: &pb.ArtifactUploadGrant{
					ObjectId: object.ObjectID, Length: uint64(object.Length), Url: decision.URL,
					RequiredHeaders: headers, ExpiresAtUnix: decision.ExpiresAtUnix,
				}}
			}
			if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ArtifactTransferRequest{
				ArtifactTransferRequest: request,
			}}) {
				break
			}
		}
		if wait := c.waitProduction(ctx, operationID); wait != nil {
			return wait
		}
	}
}

func (c *Orchestrator) onProductionArtifactReceipt(s *session, frame *pb.ArtifactReceiptFrame) {
	node, problem := c.opt.Store.ModelProductionNodeByRequest(frame.RequestId)
	if problem != nil || node == nil {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.RequestId)
	if problem != nil || request == nil || !request.IsJob() || request.Worker == "" {
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
	if err != nil || invocationDigest != attempt.InvocationDigest || frame.ArtifactReceipt == nil ||
		frame.Manifest == nil || manifestDigest == "" || frame.Manifest.Length == 0 ||
		frame.WriterGeneration == 0 || frame.ArtifactTransactionId == "" {
		return
	}
	receipt, problem := parseArtifactReceiptRef(canonical.Doc{
		"artifact_receipt_digest": canonicalSpell(frame.ArtifactReceipt.ArtifactReceiptDigest),
		"artifact_receipt_canonical_bytes": base64.StdEncoding.EncodeToString(
			frame.ArtifactReceipt.ArtifactReceiptCanonicalBytes),
	})
	if problem != nil || receipt.RequestID != frame.RequestId ||
		receipt.InvocationDigest != invocationDigest || receipt.OutputSlot != frame.OutputSlot {
		return
	}
	receiptDoc, err := canonical.Read(receipt.ReceiptBytes, &pb.ArtifactReceipt{})
	if err != nil || receiptDoc.Str("artifact_transaction_id") != frame.ArtifactTransactionId {
		return
	}
	evidence, err := base64.StdEncoding.Strict().DecodeString(
		receiptDoc.Str("release_evidence_canonical_bytes"))
	if err != nil || len(evidence) == 0 || len(evidence) > pb.MaxReleaseEvidenceBytes {
		return
	}
	objects := make([]records.ModelProductionObject, 0, len(frame.Objects))
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
		objects = append(objects, records.ModelProductionObject{
			OperationID: node.OperationID, NodeName: node.NodeName, OutputSlot: frame.OutputSlot,
			ObjectID: object.ObjectId, Length: int64(object.Length), SourceRef: object.SourceRef,
		})
	}
	if len(objects) == 0 || !manifestPresent {
		return
	}
	problem = c.opt.Store.RecordModelProductionArtifact(records.ModelProductionArtifact{
		OperationID: node.OperationID, NodeName: node.NodeName, OutputSlot: frame.OutputSlot,
		RequestID: frame.RequestId, Attempt: int64(frame.AttemptOrdinal),
		InvocationDigest: invocationDigest, TransactionID: frame.ArtifactTransactionId,
		WriterGeneration: int64(frame.WriterGeneration), ReceiptDigest: receipt.ReceiptDigest,
		Receipt: receipt.ReceiptBytes, ManifestID: manifestDigest,
		ManifestLength: int64(frame.Manifest.Length), ReleaseEvidence: evidence,
	}, objects)
	if problem != nil {
		c.logf("ArtifactReceiptFrame %s/%s refused: %s", frame.RequestId, frame.OutputSlot,
			problem.Message)
		return
	}
	c.signalProduction(node.OperationID)
}

func (c *Orchestrator) onProductionArtifactTransferStatus(s *session,
	frame *pb.ArtifactTransferStatus,
) {
	node, problem := c.opt.Store.ModelProductionNodeByRequest(frame.RequestId)
	if problem != nil || node == nil || frame.Length == 0 || frame.Length > uint64(^uint64(0)>>1) {
		return
	}
	artifacts, problem := c.opt.Store.ModelProductionArtifacts(node.OperationID)
	if problem != nil {
		return
	}
	var artifact *records.ModelProductionArtifact
	for i := range artifacts {
		if artifacts[i].RequestID == frame.RequestId && artifacts[i].OutputSlot == frame.OutputSlot {
			artifact = &artifacts[i]
			break
		}
	}
	invocation := canonicalSpell(frame.InvocationSpecDigest)
	if artifact == nil || artifact.Attempt != int64(frame.AttemptOrdinal) ||
		artifact.InvocationDigest != invocation || artifact.TransactionID != frame.ArtifactTransactionId {
		return
	}
	state := trimEnum(pb.ArtifactTransferState_name[int32(frame.State)],
		"ARTIFACT_TRANSFER_STATE_")
	state = map[string]string{"ACCEPTED": "accepted", "READING": "reading", "UPLOADING": "uploading",
		"UPLOADED": "uploaded", "ALREADY_PRESENT": "already_present", "HELD": "held",
		"FAILED": "failed"}[state]
	if state == "" || frame.TransferredBytes > frame.Length {
		return
	}
	problem = c.opt.Store.RecordModelProductionObjectStatus(records.ModelProductionObject{
		OperationID: node.OperationID, NodeName: node.NodeName, OutputSlot: frame.OutputSlot,
		ObjectID: frame.ObjectId, Length: int64(frame.Length),
		TransferOperationID: frame.OperationId, GrantRevision: int64(frame.GrantRevision),
		UpdateSequence: int64(frame.UpdateSequence), State: state,
		TransferredBytes: int64(frame.TransferredBytes), SafeCode: frame.SafeCode,
		SafeDetail: frame.SafeDetail,
	})
	if problem == nil {
		c.signalProduction(node.OperationID)
	}
}

func canonicalSpell(raw []byte) string {
	value, _ := canonical.Spell(raw)
	return value
}
