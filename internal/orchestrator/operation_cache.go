package orchestrator

import (
	"bytes"
	"context"
	"encoding/base64"
	"math"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

func operationCacheProblem(err error) *exit.Error {
	switch status.Code(err) {
	case codes.InvalidArgument, codes.PermissionDenied, codes.FailedPrecondition, codes.Unimplemented:
		return exit.Named(exit.Structural, "operation.cache_refused", "worker operation cache refused its captured identity")
	default:
		return exit.Unavailablef("worker operation cache is temporarily unavailable")
	}
}

// Cache completion precedes OutcomeAck: the workspace still has the actual terminal
// to verify, including scalar results whose ordinary ACK releases that journal.
func (c *Orchestrator) recordOperationResult(s *session, request records.Request, attempt records.Attempt) *exit.Error {
	// Closed follows the cache decision and queues its ACK; it does not prove
	// delivery. Historical replay therefore resends only the ACK, even if the
	// workspace compacted its source. ClosedAt timestamps terminal acceptance
	// before this decision, so only the preserved closed state is this marker.
	if attempt.State == "closed" {
		return nil
	}
	if !request.ChildReusable || request.ParentRequestID == "" || request.ReusedFrom != "" || attempt.TerminalStatus != "SUCCEEDED" || request.State == "canceling" || request.State == "canceled" || request.State == "releasing" {
		return nil
	}
	context, problem := c.qualifyOperation(s, request)
	if problem != nil {
		c.logf("%s completed without memoization: its numerical environment could not be revalidated", request.ID)
		return nil
	}
	key := context.Key
	doc, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return exit.Internalf("operation terminal is not canonical")
	}
	inline, err := base64.StdEncoding.Strict().DecodeString(doc.Sub("result").Str("inline_result"))
	if err != nil {
		return exit.Internalf("operation result is not its bounded JSON payload")
	}
	if len(inline) == 0 {
		inline = []byte(`{}`)
	}
	artifacts, problem := c.childResultArtifacts(request, inline)
	if problem != nil {
		return problem
	}
	for _, artifact := range artifacts {
		// The workspace index adopts outputs this operation actually wrote.
		// Returning a borrowed handle still executes and retains normally.
		if artifact.ProducerRequestID != request.ID {
			return nil
		}
		if _, problem := c.opt.Store.ArtifactOutput(artifact); problem != nil {
			return problem
		}
	}
	keyBytes, _ := canonical.Raw(key)
	invocation, _ := canonical.Raw(attempt.InvocationDigest)
	outcome, _ := canonical.Raw(attempt.TerminalDigest)
	workspace, problem := s.operationWorkspace()
	if problem != nil {
		return problem
	}
	answer, err := workspace.RecordOperationResult(s.ctx, &pb.RecordOperationResultCall{Claim: s.claim, ComputationDigest: keyBytes, RequestId: request.ID, AttemptOrdinal: uint64(attempt.Attempt), InvocationSpecDigest: invocation, OutcomeId: attempt.TerminalID, OutcomeDigest: outcome})
	if err != nil {
		return operationCacheProblem(err)
	}
	if answer == nil || !bytes.Equal(answer.ComputationDigest, keyBytes) {
		return exit.Named(exit.Structural, "operation.cache_changed", "worker operation cache changed the completed computation identity")
	}
	if !answer.Recorded {
		c.logf("%s completed without memoization: the workspace declined this optional cache entry", request.ID)
	}
	return nil
}

func (c *Orchestrator) lookupOperationOn(request records.Request, selected *session) (bool, *exit.Error) {
	if !request.ChildReusable || request.ParentRequestID == "" {
		return false, nil
	}
	if _, problem := c.qualifyOperation(selected, request); problem != nil {
		return false, problem
	}
	return c.lookupOperationWithSession(selected.ctx, request, false, selected)
}

func (c *Orchestrator) lookupOperationPending(ctx context.Context, request records.Request, pendingOnly bool) (bool, *exit.Error) {
	return c.lookupOperationWithSession(ctx, request, pendingOnly, nil)
}

func (c *Orchestrator) lookupOperationWithSession(ctx context.Context, request records.Request, pendingOnly bool, selected *session) (bool, *exit.Error) {
	if !request.ChildReusable || request.ParentRequestID == "" || request.Ordinal != 0 {
		return false, nil
	}
	c.mu.Lock()
	if c.operationLookups[request.ID] {
		c.mu.Unlock()
		return false, exit.Unavailablef("operation lookup is already in progress")
	}
	if c.operationLookups == nil {
		c.operationLookups = map[string]bool{}
	}
	c.operationLookups[request.ID] = true
	c.mu.Unlock()
	defer func() { c.mu.Lock(); delete(c.operationLookups, request.ID); c.mu.Unlock() }()
	lookup, problem := c.opt.Store.OperationLookup(request.ID)
	if problem != nil {
		return false, problem
	}
	if pendingOnly && (lookup == nil || lookup.State != "pending") {
		return false, nil
	}
	if !pendingOnly && lookup != nil && lookup.State == "hit" && request.ReusedFrom != "" {
		return true, c.opt.Store.CompleteReusedChild(request.ID)
	}
	// A MISS releases the lookup obligation and authorizes ordinary execution.
	// Do not start another native lookup while that dispatch may be crossing its
	// offer boundary. New requests still query the current workspace cache.
	if lookup != nil && lookup.State == "miss" {
		return false, nil
	}
	key := ""
	if lookup != nil {
		key = lookup.Key
	} else {
		context, problem := c.opt.Store.OperationContext(request.ID)
		if problem != nil {
			return false, problem
		}
		if context == nil {
			return false, exit.Named(exit.Conflict, "operation.context_absent", "operation lookup has no selected callee environment")
		}
		key = context.Key
	}
	if !pendingOnly {
		if problem := c.opt.Store.BeginOperationLookup(request.ID, key); problem != nil {
			return false, problem
		}
	}
	if pendingOnly && request.Worker != "" {
		rental, problem := c.opt.Store.RentalRow(request.Worker)
		if problem != nil {
			return false, problem
		}
		if rental != nil && (rental.State == "released" || rental.State == "failed") {
			return false, c.opt.Store.CompleteOperationMiss(request.ID, key)
		}
	}
	s := selected
	if s == nil {
		// Pause/cancellation reconciles its already-pinned lookup obligation.
		// It does not select a new computation or execute the observed result.
		s, problem = c.workspaceControlContext(ctx, request.Worker)
		if problem != nil {
			return false, problem
		}
	}
	keyBytes, _ := canonical.Raw(key)
	workspace, problem := s.operationWorkspace()
	if problem != nil {
		return false, problem
	}
	if ctx == nil {
		ctx = s.ctx
	}
	answer, err := workspace.LookupOperation(ctx, &pb.LookupOperationCall{Claim: s.claim, ComputationDigest: keyBytes, ConsumerRequestId: request.ID})
	if err != nil {
		return false, operationCacheProblem(err)
	}
	if answer == nil || !bytes.Equal(answer.ComputationDigest, keyBytes) || answer.ConsumerRequestId != request.ID || proto.Size(answer) > pb.MaxInlineControlBytes {
		return false, exit.Named(exit.Structural, "operation.cache_changed", "worker operation cache changed its lookup subject or result bound")
	}
	if !answer.Found {
		return false, c.opt.Store.CompleteOperationMiss(request.ID, key)
	}
	source := answer.Source
	if source == nil || source.AttemptOrdinal == 0 || source.AttemptOrdinal > math.MaxInt64 || !bytes.Equal(canonical.Digest(source.OutcomeCanonicalBytes), source.OutcomeDigest) {
		return false, exit.Named(exit.Structural, "operation.source_changed", "cached operation has no exact original terminal")
	}
	doc, err := canonical.Read(source.OutcomeCanonicalBytes, &pb.AttemptOutcomeBody{})
	invocation, _ := canonical.Spell(source.InvocationSpecDigest)
	if err != nil || doc.Str("request_id") != source.RequestId || doc.Int("attempt_ordinal") != int64(source.AttemptOrdinal) || doc.Str("invocation_spec_digest") != invocation || doc.Int("status") != int64(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED) {
		return false, exit.Named(exit.Structural, "operation.source_changed", "cached operation differs from its original successful terminal")
	}
	outputs, problem := c.opt.Store.AllModelTransferWeights(source.RequestId, int64(source.AttemptOrdinal))
	if problem != nil {
		return false, problem
	}
	if len(outputs) != len(answer.Retentions) {
		return false, exit.Named(exit.Structural, "operation.retention_changed", "cached operation changed its native output inventory")
	}
	byTransaction := map[string]records.ModelTransferWeights{}
	for _, output := range outputs {
		byTransaction[output.TransactionID] = output
	}
	outcome, _ := canonical.Spell(source.OutcomeDigest)
	cached := records.CachedOperation{Key: key, SourceRequestID: source.RequestId, SourceAttempt: int64(source.AttemptOrdinal), InvocationDigest: invocation, OutcomeID: source.OutcomeId, OutcomeDigest: outcome, OutcomeBody: source.OutcomeCanonicalBytes}

	byteOutputs, problem := c.opt.Store.ByteOutputs(source.RequestId, int64(source.AttemptOrdinal))
	if problem != nil {
		return false, problem
	}
	if len(byteOutputs) != len(answer.ByteRetentions) || len(answer.ByteRetentions)+len(answer.Retentions) > pb.MaxChildArtifactGrants {
		return false, exit.New(exit.Structural, "cached byte output inventory changed")
	}
	for _, output := range byteOutputs {
		var matched *pb.NativeByteRetentionResult
		for _, held := range answer.ByteRetentions {
			if held != nil && proto.Equal(held.Source, output.NativeRef()) {
				if matched != nil {
					return false, exit.New(exit.Structural, "cached byte recipient is duplicated")
				}
				matched = held
			}
		}
		if matched == nil || matched.Released {
			return false, exit.New(exit.Structural, "cached byte result lacks exact independent custody")
		}
		cached.ByteRetentions = append(cached.ByteRetentions, records.NativeArtifactRetention{ArtifactKind: "tree", ProducerAttempt: output.Attempt, ProducerOutputID: output.OutputID, ContentBytes: output.ContentBytes, ConsumerID: request.ID, ParentRequestID: request.ParentRequestID, Kind: "result", Slot: output.OutputID, ProducerID: source.RequestId, ManifestID: output.ManifestID, ManifestLength: output.ManifestLength, ReceiptDigest: output.ReceiptDigest, TransactionID: output.ProducerRootID, OwnerRequestID: source.RequestId, OwnerWorker: request.Worker, RetentionID: matched.RetentionId, InstanceID: s.instanceID, WorkerBootID: s.bootID, State: "held"})
	}
	for _, hold := range answer.Retentions {
		if hold == nil {
			return false, exit.New(exit.Structural, "cached operation returned an absent native retention")
		}
		output, ok := byTransaction[hold.WeightsTransactionId]
		if !ok || hold.Released || hold.Manifest == nil {
			return false, exit.Named(exit.Structural, "operation.retention_changed", "cached operation returned foreign or released native ownership")
		}
		delete(byTransaction, hold.WeightsTransactionId)
		receipt, err := canonical.Read(output.Receipt, &pb.WeightsReceipt{})
		nativeDigest, _ := canonical.Spell(hold.TensorfsReceiptDigest)
		manifest, _ := canonical.Spell(hold.Manifest.Digest)
		if err != nil || nativeDigest != receipt.Str("tensorfs_receipt_digest") || manifest != output.ManifestID || hold.Manifest.Length != uint64(output.ManifestLength) {
			return false, exit.Named(exit.Structural, "operation.retention_changed", "cached operation changed its original native receipt or manifest")
		}
		cached.Retentions = append(cached.Retentions, records.WeightsRetention{RequestID: request.ID, Kind: "result", Slot: "weights/" + output.OutputSlot, ProducerRequestID: source.RequestId, ProducerAttempt: int64(source.AttemptOrdinal), ProducerOutputSlot: output.OutputSlot, RetentionID: hold.RetentionId, InstanceID: s.instanceID, WorkerBootID: s.bootID, State: "held"})
	}
	if !pendingOnly {
		if problem := c.retainChildInputs(request); problem != nil {
			return false, problem
		}
	}
	if problem := c.opt.Store.AdoptCachedOperation(request.ID, cached); problem != nil {
		return false, problem
	}
	c.logf("%s reused completed operation %s from %s without an execution attempt", request.ID, key, source.RequestId)
	return true, nil
}

func (c *Orchestrator) retryOperationAck(s *session, request string, ordinal uint64) {
	time.AfterFunc(ReportCadence, func() {
		select {
		case <-s.ctx.Done():
			return
		case <-c.done:
			return
		default:
			c.ackSettledOutcome(s, request, ordinal)
		}
	})
}
