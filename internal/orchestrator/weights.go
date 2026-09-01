package orchestrator

import (
	"bytes"
	"encoding/base64"
	"fmt"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// weightsReceiptsFromOutcome validates the RecordOwner-specific joins the protocol's
// generic canonical reader cannot know: job mode, local owner authority, exact request/spec,
// and membership in the weights-output subset persisted beside this InvocationSpec.
func weightsReceiptsFromOutcome(req records.Request, attempt records.Attempt,
	doc canonical.Doc) ([]records.WeightsReceipt, map[string]records.WeightsReceipt, *exit.Error) {
	refs := doc.List("weights_receipts")
	if len(refs) == 0 {
		return nil, map[string]records.WeightsReceipt{}, nil
	}
	if !req.IsJob() {
		return nil, nil, exit.Named(exit.Validation, "weights_receipt_not_job",
			"serving outcome %s#%d carries %d weights receipt(s)", req.ID, attempt.Attempt, len(refs))
	}
	declared, e := decodeWeightsOutputs(attempt.WeightsOutputs)
	if e != nil {
		return nil, nil, e
	}
	allowed := map[string]bool{}
	for _, output := range declared {
		allowed[output.OutputID] = true
	}
	if len(declared) > 0 && len(doc.Sub("output_manifest").List("outputs")) > 0 {
		return nil, nil, exit.Named(exit.Validation, "mixed_weights_asset_outcome",
			"weights-only job %s#%d also returned ordinary output-manifest entries",
			req.ID, attempt.Attempt)
	}
	out := make([]records.WeightsReceipt, 0, len(refs))
	bySlot := make(map[string]records.WeightsReceipt, len(refs))
	for _, ref := range refs {
		receipt, e := parseWeightsReceiptRef(ref)
		if e != nil {
			return nil, nil, e
		}
		if receipt.OwnerScope != recordOwnerID || receipt.RequestID != req.ID ||
			receipt.InvocationDigest != attempt.InvocationDigest {
			return nil, nil, exit.Named(exit.Validation, "weights_receipt_identity_mismatch",
				"weights receipt %s names owner/request/spec %q/%q/%s, expected %q/%q/%s",
				receipt.OutputSlot, receipt.OwnerScope, receipt.RequestID,
				shortDigest(receipt.InvocationDigest), recordOwnerID, req.ID,
				shortDigest(attempt.InvocationDigest))
		}
		if !allowed[receipt.OutputSlot] {
			return nil, nil, exit.Named(exit.Validation, "weights_receipt_output_undeclared",
				"weights receipt slot %q is not in the persisted weights-output subset",
				receipt.OutputSlot)
		}
		receipt.Attempt = attempt.Attempt
		out = append(out, receipt)
		bySlot[receipt.OutputSlot] = receipt
	}
	return out, bySlot, nil
}

func parseWeightsReceiptRef(ref canonical.Doc) (records.WeightsReceipt, *exit.Error) {
	var out records.WeightsReceipt
	data, err := base64.StdEncoding.Strict().DecodeString(ref.Str("weights_receipt_canonical_bytes"))
	if err != nil || len(data) == 0 || len(data) > pb.MaxWeightsReceiptBytes {
		return out, exit.Named(exit.Validation, "weights_receipt_bytes",
			"weights receipt bytes are malformed or outside the 1..%d-byte cap",
			pb.MaxWeightsReceiptBytes)
	}
	digest := ref.Str("weights_receipt_digest")
	rawDigest, err := canonical.Raw(digest)
	if err != nil || !bytes.Equal(canonical.Digest(data), rawDigest) {
		return out, exit.Named(exit.Validation, "weights_receipt_digest_mismatch",
			"weights receipt digest does not hash the exact carried bytes")
	}
	doc, err := canonical.Read(data, &pb.WeightsReceipt{})
	if err != nil {
		return out, exit.Named(exit.Validation, "weights_receipt_invalid",
			"weights receipt is not an admissible WeightsReceipt/1: %s", err)
	}
	out = records.WeightsReceipt{
		RequestID: doc.Str("request_id"), OwnerScope: doc.Str("owner_authority_scope"),
		InvocationDigest: doc.Str("invocation_spec_digest"), OutputSlot: doc.Str("output_slot"),
		ReceiptDigest: digest,
		ReceiptBytes:  append([]byte(nil), data...),
	}
	return out, nil
}

func weightsFinalizationIntents(req records.Request, attempt records.Attempt, status string,
	requeuing bool, receipts map[string]records.WeightsReceipt) ([]records.WeightsFinalization, *exit.Error) {
	declared, e := decodeWeightsOutputs(attempt.WeightsOutputs)
	if e != nil {
		return nil, e
	}
	if requeuing || len(declared) == 0 {
		return nil, nil
	}
	if status == "SUCCEEDED" && len(receipts) != len(declared) {
		return nil, exit.Named(exit.Validation, "weights_receipt_required",
			"successful weights job %s#%d returned %d receipt(s) for %d required slot(s)",
			req.ID, attempt.Attempt, len(receipts), len(declared))
	}
	out := make([]records.WeightsFinalization, 0, len(declared))
	for _, output := range declared {
		receipt, committed := receipts[output.OutputID]
		disposition := pb.WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED
		receiptDigest, scratchRoot := "", ""
		switch {
		case status == "SUCCEEDED":
			disposition = pb.WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ADOPT
			receiptDigest = receipt.ReceiptDigest
			var err error
			scratchRoot, err = weightsScratchRootID(req.ID, output.OutputID)
			if err != nil {
				return nil, exit.Internalf("cannot derive the weights scratch root: %s", err)
			}
		case committed:
			disposition = pb.WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON
			receiptDigest = receipt.ReceiptDigest
		}
		out = append(out, records.WeightsFinalization{
			RequestID: req.ID, Attempt: attempt.Attempt, InstanceID: attempt.InstanceID,
			OwnerScope: recordOwnerID, InvocationDigest: attempt.InvocationDigest,
			OutputSlot: output.OutputID,
			Disposition: trimEnum(pb.WeightsFinalizeDisposition_name[int32(disposition)],
				"WEIGHTS_FINALIZE_DISPOSITION_"),
			ReceiptDigest: receiptDigest, ScratchRootID: scratchRoot,
		})
	}
	return out, nil
}

// weightsScratchRootID is Runtime's exact private-root identity. It is a semantic id,
// never a path and never the public `<org>/_job-*` repository name.
func weightsScratchRootID(requestID, outputSlot string) (string, error) {
	data, err := canonical.Write(map[string]canonical.Value{
		"format":                "cozy.runtime.WeightsScratchRootIdentity/1",
		"owner_authority_scope": recordOwnerID,
		"request_id":            requestID,
		"output_slot":           outputSlot,
	})
	if err != nil {
		return "", err
	}
	digest, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", err
	}
	return "weights-scratch-" + digest[len("sha256:"):], nil
}

// sendPendingWeightsFinalizations replays exact decisions in slot order. A closed stream
// leaves the rows pending; the replayed outcome on the next claim drives them again.
func (c *Orchestrator) sendPendingWeightsFinalizations(s *session, requestID string,
	attempt int64) (int, *exit.Error) {
	rows, e := c.opt.Store.PendingWeightsFinalizations(requestID, attempt)
	if e != nil {
		return 0, e
	}
	for _, row := range rows {
		if row.InstanceID != s.instanceID {
			return len(rows), exit.New(exit.Conflict,
				"weights finalization %s/%s belongs to worker %s, not %s",
				row.RequestID, row.OutputSlot, row.InstanceID, s.instanceID)
		}
		specDigest, err := canonical.Raw(row.InvocationDigest)
		if err != nil {
			return len(rows), exit.Internalf("persisted invocation digest is malformed: %s", err)
		}
		disposition, ok := pb.WeightsFinalizeDisposition_value["WEIGHTS_FINALIZE_DISPOSITION_"+row.Disposition]
		if !ok {
			return len(rows), exit.Internalf("persisted finalize disposition %q is malformed",
				row.Disposition)
		}
		var receiptDigest []byte
		if row.ReceiptDigest != "" {
			receiptDigest, err = canonical.Raw(row.ReceiptDigest)
			if err != nil {
				return len(rows), exit.Internalf("persisted receipt digest is malformed: %s", err)
			}
		}
		request := &pb.WeightsFinalizeRequest{
			RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: s.generation,
			WorkerBootId: s.bootID, RequestId: row.RequestID,
			InvocationSpecDigest: specDigest, OutputSlot: row.OutputSlot,
			Disposition:          pb.WeightsFinalizeDisposition(disposition),
			WeightsReceiptDigest: receiptDigest, ScratchRootId: row.ScratchRootID,
			OwnerAuthorityScope: row.OwnerScope,
		}
		if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_WeightsFinalizeRequest{
			WeightsFinalizeRequest: request,
		}}) {
			return len(rows), exit.Unavailablef("the stream closed before weights finalization %s/%s was sent",
				row.RequestID, row.OutputSlot)
		}
		c.logf("WeightsFinalizeRequest %s/%s %s -> %s", row.RequestID,
			row.OutputSlot, row.Disposition, s.bootID)
	}
	return len(rows), nil
}

func (c *Orchestrator) onWeightsFinalizeResult(s *session, frame *pb.WeightsFinalizeResult) {
	refuse := func(format string, args ...any) {
		c.logf("WeightsFinalizeResult %s/%s REFUSED: "+format,
			append([]any{frame.RequestId, frame.OutputSlot}, args...)...)
	}
	spelledSpec, err := canonical.Spell(frame.InvocationSpecDigest)
	if err != nil || frame.RequestId == "" || frame.OutputSlot == "" ||
		frame.OwnerAuthorityScope != recordOwnerID {
		refuse("incomplete identity or owner divergence")
		return
	}
	intent, e := c.opt.Store.WeightsFinalization(frame.RequestId, spelledSpec, frame.OutputSlot)
	if e != nil || intent == nil {
		if e != nil {
			refuse("%s", e.Message)
		} else {
			refuse("no persisted first-wins intent")
		}
		return
	}
	if intent.InstanceID != s.instanceID {
		refuse("worker %s sent a result for %s's transaction", s.instanceID, intent.InstanceID)
		return
	}
	outcome := trimEnum(pb.WeightsFinalizeOutcome_name[int32(frame.Outcome)],
		"WEIGHTS_FINALIZE_OUTCOME_")
	if (intent.Disposition == "ADOPT" && outcome != "ADOPTED") ||
		(intent.Disposition != "ADOPT" && outcome != "ABANDONED") {
		refuse("result %s contradicts persisted %s intent", outcome, intent.Disposition)
		return
	}
	resultReceiptDigest := ""
	var resultReceiptBytes []byte
	if receiptRef := frame.WeightsReceipt; receiptRef != nil {
		digest, digestErr := canonical.Spell(receiptRef.WeightsReceiptDigest)
		if digestErr != nil {
			refuse("returned receipt digest is malformed")
			return
		}
		receipt, problem := parseWeightsReceiptRef(canonical.Doc{
			"weights_receipt_digest":          digest,
			"weights_receipt_canonical_bytes": base64.StdEncoding.EncodeToString(receiptRef.WeightsReceiptCanonicalBytes),
		})
		if problem != nil {
			refuse("%s", problem.Message)
			return
		}
		if receipt.OwnerScope != recordOwnerID || receipt.RequestID != frame.RequestId ||
			receipt.InvocationDigest != spelledSpec || receipt.OutputSlot != frame.OutputSlot {
			refuse("returned receipt does not close this owner/request/spec/slot")
			return
		}
		if intent.ReceiptDigest != "" && receipt.ReceiptDigest != intent.ReceiptDigest {
			refuse("returned receipt %s differs from persisted %s",
				shortDigest(receipt.ReceiptDigest), shortDigest(intent.ReceiptDigest))
			return
		}
		resultReceiptDigest, resultReceiptBytes = receipt.ReceiptDigest, receipt.ReceiptBytes
	}
	if intent.Disposition == "ADOPT" && resultReceiptDigest == "" {
		refuse("ADOPTED omitted the exact adopted receipt")
		return
	}
	applied, e := c.opt.Store.RecordWeightsFinalizeResult(records.WeightsFinalization{
		RequestID: frame.RequestId, Attempt: intent.Attempt, InstanceID: s.instanceID,
		InvocationDigest: spelledSpec, OutputSlot: frame.OutputSlot,
		ResultOutcome:       outcome,
		ResultReceiptDigest: resultReceiptDigest, ResultReceiptBytes: resultReceiptBytes,
	})
	if e != nil {
		refuse("%s", e.Message)
		return
	}
	if applied {
		c.logf("WeightsFinalizeResult %s/%s %s persisted before outcome ack",
			frame.RequestId, frame.OutputSlot, outcome)
	}
	pending, e := c.opt.Store.PendingWeightsFinalizations(frame.RequestId, intent.Attempt)
	if e != nil {
		refuse("cannot test ack readiness: %s", e.Message)
		return
	}
	if len(pending) == 0 {
		c.ackSettledOutcome(s, frame.RequestId, uint64(intent.Attempt))
	}
}

// ackSettledOutcome is the one Ack boundary for immediate outcomes and weights outcomes.
// The latter reach it only after every exact finalize result is durable.
func (c *Orchestrator) ackSettledOutcome(s *session, requestID string, ordinal uint64) {
	attempt, e := c.opt.Store.AttemptRow(requestID, int64(ordinal))
	if e != nil || attempt == nil {
		c.logf("OutcomeAck %s#%d not sent: the durable attempt cannot be read", requestID, ordinal)
		return
	}
	req, e := c.opt.Store.RequestRow(requestID)
	if e != nil || req == nil {
		c.logf("OutcomeAck %s#%d not sent: the durable request cannot be read", requestID, ordinal)
		return
	}
	c.mu.Lock()
	holder := c.workers[s.instanceID]
	c.mu.Unlock()
	if req.ModelTransfer != nil && attempt.TerminalStatus == "SUCCEEDED" {
		transfer, problem := c.opt.Store.ModelTransferOf(requestID)
		if problem != nil {
			c.logf("OutcomeAck %s#%d not sent: model transfer cannot be read", requestID, ordinal)
			return
		}
		if transfer != nil && transfer.State != "completed" && transfer.State != "failed" &&
			transfer.State != "canceled" {
			c.kickModelTransferFinalizer(s, requestID, int64(ordinal))
			return
		}
	}
	specDigest, err := canonical.Raw(attempt.InvocationDigest)
	if err != nil {
		c.logf("OutcomeAck %s#%d not sent: malformed persisted invocation digest", requestID, ordinal)
		return
	}
	outcomeDigest, err := canonical.Raw(attempt.TerminalDigest)
	if err != nil {
		c.logf("OutcomeAck %s#%d not sent: malformed persisted outcome digest", requestID, ordinal)
		return
	}
	ack := &pb.AttemptOutcomeAck{
		RequestId: requestID, AttemptOrdinal: ordinal, InvocationSpecDigest: specDigest,
		OutcomeId: attempt.TerminalID, OutcomeDigest: outcomeDigest,
		RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: s.generation,
		WorkerBootId: s.bootID,
	}
	if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_OutcomeAck{OutcomeAck: ack}}) {
		c.logf("OutcomeAck %s#%d was not queued: the closed stream owes a replay", requestID, ordinal)
		return
	}
	if e := c.opt.Store.Closed(requestID, int64(ordinal)); e != nil {
		c.logf("OutcomeAck %s#%d was queued but closure is still owed: %s", requestID, ordinal, e.Message)
		return
	}
	if req.ModelTransfer != nil {
		go c.finishModelTransferRequest(requestID, int64(ordinal))
		return
	}
	c.afterAck(*req, *attempt, holder)
}

func weightsReceiptSummary(receipts []records.WeightsReceipt) string {
	return fmt.Sprintf("%d weights receipt(s)", len(receipts))
}
