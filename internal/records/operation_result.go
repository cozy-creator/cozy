package records

import (
	"bytes"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// CachedOperation is an authenticated Runtime workspace lookup observation. Its original
// terminal stays historical; only the new request's result ownership is added.
type CachedOperation struct {
	Key, SourceRequestID, InvocationDigest, OutcomeID, OutcomeDigest string
	SourceAttempt                                                    int64
	OutcomeBody                                                      []byte
	Retentions                                                       []WeightsRetention
	ByteRetentions                                                   []NativeArtifactRetention
}

func (s *Store) AdoptCachedOperation(id string, cached CachedOperation) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin operation result adoption: %s", err)
	}
	defer tx.Rollback()
	request, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, id))
	if err != nil {
		return exit.Internalf("cannot read operation consumer: %s", err)
	}
	context, problem := scanOperationContext(tx.QueryRow(`SELECT numerical_environment_digest,computation_digest FROM request_operation_contexts WHERE request_id=?`, id))
	if problem != nil {
		return problem
	}
	// Older pending lookups carried only a prequalified target. Cancellation may
	// recover and release their receipts, but cannot turn them into a new result.
	legacyCancellation := context == nil && request.State == "canceling"
	if context == nil && !legacyCancellation {
		return exit.Named(exit.Conflict, "operation.context_absent", "cached computation has no bound callee environment")
	}
	key := ""
	if legacyCancellation {
		key, problem = OperationKey(request)
	} else {
		key, problem = QualifiedOperationKey(request, context.NumericalEnvironment)
	}
	if problem != nil {
		return problem
	}
	if key != cached.Key || (context != nil && key != context.Key) || !request.ChildReusable || request.ParentRequestID == "" || request.Ordinal != 0 {
		return exit.Named(exit.Conflict, "operation.consumer_changed", "cached computation does not match this unoffered private child")
	}
	if request.State == "succeeded" && request.ReusedFrom == cached.SourceRequestID {
		return nil
	}
	active := request.State == "submitted" || request.State == "queued"
	if !active && request.State != "pausing" && request.State != "paused" && request.State != "canceling" {
		return exit.Named(exit.Conflict, "operation.consumer_stopped", "the cached result consumer already stopped or started execution")
	}
	var pending string
	if err := tx.QueryRow(`SELECT computation_digest FROM request_operation_lookups WHERE request_id=? AND state IN ('pending','hit')`, id).Scan(&pending); err != nil || pending != key {
		return exit.Named(exit.Conflict, "operation.lookup_absent", "cached result has no exact pending lookup intent")
	}
	source, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, cached.SourceRequestID))
	if err != nil {
		return exit.Named(exit.Conflict, "operation.source_unavailable", "cached result provenance is unavailable in this owner history")
	}
	sourceContext, problem := scanOperationContext(tx.QueryRow(`SELECT numerical_environment_digest,computation_digest FROM request_operation_contexts WHERE request_id=?`, source.ID))
	if problem != nil {
		return problem
	}
	if sourceContext == nil && !legacyCancellation {
		return exit.Named(exit.Conflict, "operation.source_changed", "cached source has no recorded callee environment")
	}
	sourceKey := ""
	if sourceContext == nil {
		sourceKey, problem = OperationKey(source)
	} else {
		sourceKey, problem = QualifiedOperationKey(source, sourceContext.NumericalEnvironment)
	}
	if problem != nil {
		return problem
	}
	if sourceKey != key || (sourceContext != nil && sourceKey != sourceContext.Key) || !source.ChildReusable || source.Worker != request.Worker {
		return exit.Named(exit.Conflict, "operation.source_changed", "cached result changed its computation or workspace")
	}
	var state, status, invocation, outcome, digest string
	var body []byte
	if err := tx.QueryRow(`SELECT state,terminal_status,invocation_digest,terminal_id,terminal_digest,terminal_body FROM attempts WHERE request_id=? AND attempt=?`, source.ID, cached.SourceAttempt).Scan(&state, &status, &invocation, &outcome, &digest, &body); err != nil {
		return exit.Named(exit.Conflict, "operation.source_unavailable", "cached result has no original terminal observation")
	}
	if state != "closed" && state != "terminal" {
		return exit.Unavailablef("cached result awaits original attempt closure")
	}
	if status != "SUCCEEDED" || invocation != cached.InvocationDigest || outcome != cached.OutcomeID || digest != cached.OutcomeDigest || !bytes.Equal(body, cached.OutcomeBody) {
		return exit.Named(exit.Conflict, "operation.source_changed", "cached result differs from its immutable successful terminal")
	}
	var expected int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM request_model_transfer_outputs WHERE request_id=? AND attempt=?`, source.ID, cached.SourceAttempt).Scan(&expected); err != nil {
		return exit.Internalf("cannot read cached native result inventory: %s", err)
	}
	var expectedBytes int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM byte_outputs WHERE request_id=? AND attempt=? AND native_service_id IS NULL`, source.ID, cached.SourceAttempt).Scan(&expectedBytes); err != nil {
		return exit.Internalf("cannot read byte cache inventory: %s", err)
	}
	if len(cached.ByteRetentions) != expectedBytes || len(cached.Retentions) != expected || (request.ChildArtifacts && expected+expectedBytes == 0) {
		return exit.Named(exit.Conflict, "operation.retention_incomplete", "cached result did not independently retain every native output")
	}
	seen := map[string]bool{}
	for _, hold := range cached.Retentions {
		if hold.RequestID != request.ID || hold.Kind != "result" || hold.ProducerRequestID != source.ID || hold.ProducerAttempt != cached.SourceAttempt || hold.Slot != "weights/"+hold.ProducerOutputSlot || hold.InstanceID == "" || hold.WorkerBootID == "" {
			return exit.New(exit.Validation, "cached native retention does not name its exact consumer and producer")
		}
		if seen[hold.ProducerOutputSlot] {
			return exit.New(exit.Validation, "cached native output was repeated")
		}
		seen[hold.ProducerOutputSlot] = true
		if _, err := canonical.Raw(hold.RetentionID); err != nil {
			return exit.New(exit.Validation, "cached native retention identity is malformed")
		}
		_, err := tx.Exec(`INSERT INTO request_weights_retentions(`+weightsRetentionCols+`) VALUES(?,?,?,?,?,?,?,?,?,'held') ON CONFLICT(request_id,kind,slot) DO NOTHING`, hold.RequestID, hold.Kind, hold.Slot, hold.ProducerRequestID, hold.ProducerAttempt, hold.ProducerOutputSlot, hold.RetentionID, hold.InstanceID, hold.WorkerBootID)
		if err != nil {
			return exit.Internalf("cannot record adopted native ownership: %s", err)
		}
		row, err := scanWeightsRetention(tx.QueryRow(`SELECT `+weightsRetentionCols+` FROM request_weights_retentions WHERE request_id=? AND kind='result' AND slot=?`, id, hold.Slot))
		if err != nil || row.RetentionID != hold.RetentionID || row.ProducerRequestID != hold.ProducerRequestID || row.ProducerAttempt != hold.ProducerAttempt || row.ProducerOutputSlot != hold.ProducerOutputSlot || row.State != "held" {
			return exit.Named(exit.Conflict, "operation.retention_changed", "cached result ownership changed or was already released")
		}
	}

	seenBytes := map[string]bool{}
	for _, h := range cached.ByteRetentions {
		if seenBytes[h.ProducerOutputID] {
			return exit.New(exit.Validation, "cached byte output was repeated")
		}
		seenBytes[h.ProducerOutputID] = true
		if h.ConsumerID != request.ID || h.ParentRequestID != request.ParentRequestID || h.Kind != "result" || h.ArtifactKind != "tree" || h.ProducerID != source.ID || h.ProducerAttempt != cached.SourceAttempt || h.Slot != h.ProducerOutputID || h.OwnerRequestID != source.ID || h.OwnerWorker != request.Worker || h.InstanceID == "" || h.WorkerBootID == "" || h.State != "held" {
			return exit.New(exit.Validation, "cached byte recipient differs from exact request and producer")
		}
		b, err := scanByteOutput(tx.QueryRow(`SELECT `+byteOutputCols+` FROM byte_outputs WHERE request_id=? AND attempt=? AND output_id=?`, h.ProducerID, h.ProducerAttempt, h.ProducerOutputID))
		if err != nil || b.ProducerRootID != h.TransactionID || b.ReceiptDigest != h.ReceiptDigest || b.ManifestID != h.ManifestID || b.ManifestLength != h.ManifestLength || b.ContentBytes != h.ContentBytes {
			return exit.New(exit.Conflict, "cached byte retention differs from original output")
		}
		if _, err := canonical.Raw(h.RetentionID); err != nil {
			return exit.New(exit.Validation, "cached byte retention identity is invalid")
		}
		_, err = tx.Exec(`INSERT INTO native_artifact_retentions(`+nativeArtifactCols+`) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(consumer_id,kind,slot) DO NOTHING`, h.ArtifactKind, h.ProducerAttempt, h.ProducerOutputID, h.ContentBytes, h.ConsumerID, h.ParentRequestID, h.Kind, h.Slot, h.ProducerID, h.ManifestID, h.ManifestLength, h.ReceiptDigest, h.TransactionID, h.OwnerRequestID, h.OwnerWorker, h.RetentionID, h.InstanceID, h.WorkerBootID, h.State)
		if err != nil {
			return exit.Internalf("cannot record cached byte retention: %s", err)
		}
		held, err := scanNativeArtifact(tx.QueryRow(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? AND kind='result' AND slot=?`, request.ID, h.Slot))
		if err != nil || held != h {
			return exit.New(exit.Conflict, "cached byte ownership changed or was canceled")
		}
	}
	if request.State != "canceling" {
		state := request.State
		if active {
			state = "finalizing"
		}
		if _, err := tx.Exec(`UPDATE requests SET reused_from=?,state=? WHERE id=?`, source.ID, state, id); err != nil {
			return exit.Internalf("cannot record cached result provenance: %s", err)
		}
	}
	if _, err := tx.Exec(`UPDATE request_operation_lookups SET state='hit' WHERE request_id=? AND computation_digest=?`, id, key); err != nil {
		return exit.Internalf("cannot settle operation lookup receipt: %s", err)
	}
	if err := appendEventTx(tx, id, "request.memoized", 0, map[string]any{"operation_key": key, "source_request_id": source.ID, "source_attempt": cached.SourceAttempt}); err != nil {
		return exit.Internalf("cannot journal cached result adoption: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit cached result adoption: %s", err)
	}
	if active {
		return s.CompleteReusedChild(id)
	}
	return nil
}
