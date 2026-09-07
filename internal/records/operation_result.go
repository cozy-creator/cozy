package records

import (
	"bytes"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// CachedOperation is an authenticated Host lookup observation. Its original
// terminal stays historical; only the new request's result ownership is added.
type CachedOperation struct {
	Key, SourceRequestID, InvocationDigest, OutcomeID, OutcomeDigest string
	SourceAttempt                                                    int64
	OutcomeBody                                                      []byte
	Retentions                                                       []WeightsRetention
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
	key, problem := OperationKey(request)
	if problem != nil {
		return problem
	}
	if key != cached.Key || !request.ChildReusable || request.ParentRequestID == "" || request.Worker == "" || request.Ordinal != 0 {
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
	if err := tx.QueryRow(`SELECT computation_digest FROM request_operation_lookups WHERE request_id=? AND state='pending'`, id).Scan(&pending); err != nil || pending != key {
		return exit.Named(exit.Conflict, "operation.lookup_absent", "cached result has no exact pending lookup intent")
	}
	source, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, cached.SourceRequestID))
	if err != nil {
		return exit.Named(exit.Conflict, "operation.source_unavailable", "cached result provenance is unavailable in this owner history")
	}
	sourceKey, problem := OperationKey(source)
	if problem != nil {
		return problem
	}
	if sourceKey != key || !source.ChildReusable || source.Worker != request.Worker {
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
	if len(cached.Retentions) != expected || (request.ChildArtifacts && expected == 0) {
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
