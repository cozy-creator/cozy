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

// artifactReceiptsFromOutcome validates the RecordOwner-specific joins the protocol's
// generic canonical reader cannot know: job mode, local owner authority, exact request/spec,
// and membership in the artifact-output subset persisted beside this InvocationSpec.
func artifactReceiptsFromOutcome(req records.Request, attempt records.Attempt,
	doc canonical.Doc) ([]records.ArtifactReceipt, map[string]records.ArtifactReceipt, *exit.Error) {
	refs := doc.List("artifact_receipts")
	if len(refs) == 0 {
		return nil, map[string]records.ArtifactReceipt{}, nil
	}
	if !req.IsJob() {
		return nil, nil, exit.Named(exit.Validation, "artifact_receipt_not_job",
			"serving outcome %s#%d carries %d artifact receipt(s)", req.ID, attempt.Attempt, len(refs))
	}
	declared, e := decodeArtifactOutputs(attempt.ArtifactOutputs)
	if e != nil {
		return nil, nil, e
	}
	allowed := map[string]bool{}
	for _, output := range declared {
		allowed[output.OutputID] = true
	}
	if len(declared) > 0 && len(doc.Sub("output_manifest").List("outputs")) > 0 {
		return nil, nil, exit.Named(exit.Validation, "mixed_artifact_asset_outcome",
			"artifact-only job %s#%d also returned ordinary output-manifest entries",
			req.ID, attempt.Attempt)
	}
	out := make([]records.ArtifactReceipt, 0, len(refs))
	bySlot := make(map[string]records.ArtifactReceipt, len(refs))
	for _, ref := range refs {
		receipt, e := parseArtifactReceiptRef(ref)
		if e != nil {
			return nil, nil, e
		}
		if receipt.OwnerScope != recordOwnerID || receipt.RequestID != req.ID ||
			receipt.InvocationDigest != attempt.InvocationDigest {
			return nil, nil, exit.Named(exit.Validation, "artifact_receipt_identity_mismatch",
				"artifact receipt %s names owner/request/spec %q/%q/%s, expected %q/%q/%s",
				receipt.OutputSlot, receipt.OwnerScope, receipt.RequestID,
				shortDigest(receipt.InvocationDigest), recordOwnerID, req.ID,
				shortDigest(attempt.InvocationDigest))
		}
		if !allowed[receipt.OutputSlot] {
			return nil, nil, exit.Named(exit.Validation, "artifact_receipt_output_undeclared",
				"artifact receipt slot %q is not in the persisted artifact-output subset",
				receipt.OutputSlot)
		}
		receipt.Attempt = attempt.Attempt
		out = append(out, receipt)
		bySlot[receipt.OutputSlot] = receipt
	}
	return out, bySlot, nil
}

func parseArtifactReceiptRef(ref canonical.Doc) (records.ArtifactReceipt, *exit.Error) {
	var out records.ArtifactReceipt
	data, err := base64.StdEncoding.Strict().DecodeString(ref.Str("artifact_receipt_canonical_bytes"))
	if err != nil || len(data) == 0 || len(data) > pb.MaxArtifactReceiptBytes {
		return out, exit.Named(exit.Validation, "artifact_receipt_bytes",
			"artifact receipt bytes are malformed or outside the 1..%d-byte cap",
			pb.MaxArtifactReceiptBytes)
	}
	digest := ref.Str("artifact_receipt_digest")
	rawDigest, err := canonical.Raw(digest)
	if err != nil || !bytes.Equal(canonical.Digest(data), rawDigest) {
		return out, exit.Named(exit.Validation, "artifact_receipt_digest_mismatch",
			"artifact receipt digest does not hash the exact carried bytes")
	}
	doc, err := canonical.Read(data, &pb.ArtifactReceipt{})
	if err != nil {
		return out, exit.Named(exit.Validation, "artifact_receipt_invalid",
			"artifact receipt is not an admissible ArtifactReceipt/1: %s", err)
	}
	out = records.ArtifactReceipt{
		RequestID: doc.Str("request_id"), OwnerScope: doc.Str("owner_authority_scope"),
		InvocationDigest: doc.Str("invocation_spec_digest"), OutputSlot: doc.Str("output_slot"),
		ReceiptDigest: digest,
		ReceiptBytes:  append([]byte(nil), data...),
	}
	return out, nil
}

func artifactFinalizationIntents(req records.Request, attempt records.Attempt, status string,
	requeuing bool, receipts map[string]records.ArtifactReceipt) ([]records.ArtifactFinalization, *exit.Error) {
	declared, e := decodeArtifactOutputs(attempt.ArtifactOutputs)
	if e != nil {
		return nil, e
	}
	if requeuing || len(declared) == 0 {
		return nil, nil
	}
	if status == "SUCCEEDED" && len(receipts) != len(declared) {
		return nil, exit.Named(exit.Validation, "artifact_receipt_required",
			"successful artifact job %s#%d returned %d receipt(s) for %d required slot(s)",
			req.ID, attempt.Attempt, len(receipts), len(declared))
	}
	out := make([]records.ArtifactFinalization, 0, len(declared))
	for _, output := range declared {
		receipt, committed := receipts[output.OutputID]
		disposition := pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED
		receiptDigest, scratchRoot := "", ""
		switch {
		case status == "SUCCEEDED":
			disposition = pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ADOPT
			receiptDigest = receipt.ReceiptDigest
			var err error
			scratchRoot, err = artifactScratchRootID(req.ID, output.OutputID)
			if err != nil {
				return nil, exit.Internalf("cannot derive the artifact scratch root: %s", err)
			}
		case committed:
			disposition = pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON
			receiptDigest = receipt.ReceiptDigest
		}
		out = append(out, records.ArtifactFinalization{
			RequestID: req.ID, Attempt: attempt.Attempt, InstanceID: attempt.InstanceID,
			OwnerScope: recordOwnerID, InvocationDigest: attempt.InvocationDigest,
			OutputSlot: output.OutputID,
			Disposition: trimEnum(pb.ArtifactFinalizeDisposition_name[int32(disposition)],
				"ARTIFACT_FINALIZE_DISPOSITION_"),
			ReceiptDigest: receiptDigest, ScratchRootID: scratchRoot,
		})
	}
	return out, nil
}

// artifactScratchRootID is Runtime's exact private-root identity. It is a semantic id,
// never a path and never the public `<org>/_job-*` repository name.
func artifactScratchRootID(requestID, outputSlot string) (string, error) {
	data, err := canonical.Write(map[string]canonical.Value{
		"format":                "cozy.runtime.ArtifactScratchRootIdentity/1",
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
	return "artifact-scratch-" + digest[len("sha256:"):], nil
}

// sendPendingArtifactFinalizations replays exact decisions in slot order. A closed stream
// leaves the rows pending; the replayed outcome on the next claim drives them again.
func (c *Orchestrator) sendPendingArtifactFinalizations(s *session, requestID string,
	attempt int64) (int, *exit.Error) {
	rows, e := c.opt.Store.PendingArtifactFinalizations(requestID, attempt)
	if e != nil {
		return 0, e
	}
	for _, row := range rows {
		if row.InstanceID != s.instanceID {
			return len(rows), exit.New(exit.Conflict,
				"artifact finalization %s/%s belongs to worker %s, not %s",
				row.RequestID, row.OutputSlot, row.InstanceID, s.instanceID)
		}
		specDigest, err := canonical.Raw(row.InvocationDigest)
		if err != nil {
			return len(rows), exit.Internalf("persisted invocation digest is malformed: %s", err)
		}
		disposition, ok := pb.ArtifactFinalizeDisposition_value["ARTIFACT_FINALIZE_DISPOSITION_"+row.Disposition]
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
		request := &pb.ArtifactFinalizeRequest{
			RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: s.generation,
			WorkerBootId: s.bootID, RequestId: row.RequestID,
			InvocationSpecDigest: specDigest, OutputSlot: row.OutputSlot,
			Disposition:           pb.ArtifactFinalizeDisposition(disposition),
			ArtifactReceiptDigest: receiptDigest, ScratchRootId: row.ScratchRootID,
			OwnerAuthorityScope: row.OwnerScope,
		}
		if !s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ArtifactFinalizeRequest{
			ArtifactFinalizeRequest: request,
		}}) {
			return len(rows), exit.Unavailablef("the stream closed before artifact finalization %s/%s was sent",
				row.RequestID, row.OutputSlot)
		}
		c.logf("ArtifactFinalizeRequest %s/%s %s -> %s", row.RequestID,
			row.OutputSlot, row.Disposition, s.bootID)
	}
	return len(rows), nil
}

func (c *Orchestrator) onArtifactFinalizeResult(s *session, frame *pb.ArtifactFinalizeResult) {
	refuse := func(format string, args ...any) {
		c.logf("ArtifactFinalizeResult %s/%s REFUSED: "+format,
			append([]any{frame.RequestId, frame.OutputSlot}, args...)...)
	}
	spelledSpec, err := canonical.Spell(frame.InvocationSpecDigest)
	if err != nil || frame.RequestId == "" || frame.OutputSlot == "" ||
		frame.OwnerAuthorityScope != recordOwnerID {
		refuse("incomplete identity or owner divergence")
		return
	}
	intent, e := c.opt.Store.ArtifactFinalization(frame.RequestId, spelledSpec, frame.OutputSlot)
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
	outcome := trimEnum(pb.ArtifactFinalizeOutcome_name[int32(frame.Outcome)],
		"ARTIFACT_FINALIZE_OUTCOME_")
	if (intent.Disposition == "ADOPT" && outcome != "ADOPTED") ||
		(intent.Disposition != "ADOPT" && outcome != "ABANDONED") {
		refuse("result %s contradicts persisted %s intent", outcome, intent.Disposition)
		return
	}
	resultReceiptDigest := ""
	var resultReceiptBytes []byte
	if receiptRef := frame.ArtifactReceipt; receiptRef != nil {
		digest, digestErr := canonical.Spell(receiptRef.ArtifactReceiptDigest)
		if digestErr != nil {
			refuse("returned receipt digest is malformed")
			return
		}
		receipt, problem := parseArtifactReceiptRef(canonical.Doc{
			"artifact_receipt_digest":          digest,
			"artifact_receipt_canonical_bytes": base64.StdEncoding.EncodeToString(receiptRef.ArtifactReceiptCanonicalBytes),
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
	applied, e := c.opt.Store.RecordArtifactFinalizeResult(records.ArtifactFinalization{
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
		c.logf("ArtifactFinalizeResult %s/%s %s persisted before outcome ack",
			frame.RequestId, frame.OutputSlot, outcome)
	}
	pending, e := c.opt.Store.PendingArtifactFinalizations(frame.RequestId, intent.Attempt)
	if e != nil {
		refuse("cannot test ack readiness: %s", e.Message)
		return
	}
	if len(pending) == 0 {
		c.ackSettledOutcome(s, frame.RequestId, uint64(intent.Attempt))
	}
}

// ackSettledOutcome is the one Ack boundary for immediate outcomes and artifact outcomes.
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

func artifactReceiptSummary(receipts []records.ArtifactReceipt) string {
	return fmt.Sprintf("%d artifact receipt(s)", len(receipts))
}
