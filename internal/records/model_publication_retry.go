package records

import (
	"bytes"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/archive"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// RetryModelTransferPublication changes only the unfinished destination transfer.
// Successful attempts, native receipts and custody proofs remain fixed. A failed
// request is restored only when all of its exact outputs are already banked.
func (s *Store) RetryModelTransferPublication(requestID, actor string) (bool, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return false, exit.Internalf("cannot begin publication retry: %s", err)
	}
	defer tx.Rollback()
	var requestState, transferState, outcome, attemptState string
	var ordinal int64
	err = tx.QueryRow(`SELECT r.state,t.state,r.ordinal,a.terminal_status,a.state
		FROM requests r JOIN request_model_transfers t ON t.request_id=r.id
		JOIN attempts a ON a.request_id=r.id AND a.attempt=r.ordinal WHERE r.id=?`, requestID).
		Scan(&requestState, &transferState, &ordinal, &outcome, &attemptState)
	if err != nil {
		return false, exit.New(exit.Conflict, "request has no retained producer publication")
	}
	if (requestState != "finalizing" && requestState != "failed") || outcome != "SUCCEEDED" || (attemptState != "terminal" && attemptState != "closed") {
		return false, exit.Named(exit.Conflict, "model_transfer.publication_not_retryable", "publication retry requires the same successful unfinished producer request")
	}
	if transferState == "finalizing" && requestState == "finalizing" {
		return false, nil
	}
	if transferState != "failed" {
		return false, exit.Named(exit.Conflict, "model_transfer.publication_not_retryable", "this publication has no blocked transfer to retry")
	}
	if requestState == "failed" {
		if problem := verifiedProducerPublication(tx, requestID, ordinal); problem != nil {
			return false, problem
		}
		result, err := tx.Exec(`UPDATE requests SET state='finalizing' WHERE id=? AND state='failed' AND ordinal=?`, requestID, ordinal)
		if err != nil {
			return false, exit.Internalf("cannot restore verified publication: %s", err)
		}
		if changed, _ := result.RowsAffected(); changed != 1 {
			return false, exit.New(exit.Conflict, "request changed before publication recovery")
		}
	}
	result, err := tx.Exec(`UPDATE request_model_transfers SET state='finalizing',error_code='',safe_error='',updated_at=?
		WHERE request_id=? AND state='failed'`, now(), requestID)
	if err != nil {
		return false, exit.Internalf("cannot retry publication: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return false, exit.New(exit.Conflict, "publication changed before retry")
	}
	if requestState == "finalizing" {
		if _, err = tx.Exec(`UPDATE request_model_transfer_objects SET state='pending',safe_code='',safe_detail=''
		WHERE request_id=? AND attempt=? AND state='failed'`, requestID, ordinal); err != nil {
			return false, exit.Internalf("cannot retry unfinished publication objects: %s", err)
		}
	}
	if err = appendEventTx(tx, requestID, "request.publication_retry_requested", ordinal, map[string]any{"actor": actor, "recovered_request_failure": requestState == "failed"}); err != nil {
		return false, exit.Internalf("cannot record publication retry: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit publication retry: %s", err)
	}
	return true, nil
}

// A pre-attempt waiter can outlive successful execution. Explicit recovery of
// that request failure is permitted only after the normal publication owner has
// banked every declared checkpoint; this never retries the producer itself.
// The native TensorFS receipt is opaque here. RecordModelTransferWeights retains
// the verified worker frame's immutable root and object roster; the publication
// owner verifies that exact closure with Hub before recording its operation ID.
func verifiedProducerPublication(tx *sql.Tx, requestID string, ordinal int64) *exit.Error {
	refuse := func() *exit.Error {
		return exit.Named(exit.Conflict, "model_transfer.publication_not_verified", "failed request recovery requires every exact successful producer checkpoint to be retained")
	}
	var kind, intentJSON, invocationDigest, outcomeDigest string
	var invocationBytes, outcomeBytes []byte
	if err := tx.QueryRow(`SELECT r.kind,t.intent,a.invocation_digest,a.invocation,a.terminal_digest,a.terminal_body
 FROM requests r JOIN request_model_transfers t ON t.request_id=r.id JOIN attempts a ON a.request_id=r.id AND a.attempt=r.ordinal
 WHERE r.id=? AND r.ordinal=?`, requestID, ordinal).Scan(&kind, &intentJSON, &invocationDigest, &invocationBytes, &outcomeDigest, &outcomeBytes); err != nil {
		return exit.Internalf("cannot read retained publication identity: %s", err)
	}
	var intent ModelTransferIntent
	if kind != "job" || json.Unmarshal([]byte(intentJSON), &intent) != nil || intent.Kind != "model-upload" || len(intent.Outputs) == 0 {
		return refuse()
	}
	for _, pair := range []struct {
		raw    []byte
		digest string
	}{{invocationBytes, invocationDigest}, {outcomeBytes, outcomeDigest}} {
		digest, err := canonical.Raw(pair.digest)
		if err != nil || !bytes.Equal(digest, canonical.Digest(pair.raw)) {
			return refuse()
		}
	}
	invocation, err := archive.Read(invocationBytes, archive.Invocation)
	if err != nil || len(invocation.Sub("job")) == 0 {
		return refuse()
	}
	outcome, err := archive.Read(outcomeBytes, archive.TerminalBody)
	if err != nil || outcome.Str("request_id") != requestID || outcome.Int("attempt_ordinal") != ordinal || outcome.Str("invocation_spec_digest") != invocationDigest || outcome.Int("status") != int64(1) {
		return refuse()
	}
	receiptRefs := make(map[string][]byte)
	for _, ref := range outcome.List("weights_receipts") {
		raw, err := base64.StdEncoding.Strict().DecodeString(ref.Str("weights_receipt_canonical_bytes"))
		if err != nil {
			return refuse()
		}
		digest := ref.Str("weights_receipt_digest")
		if _, found := receiptRefs[digest]; found {
			return refuse()
		}
		receiptRefs[digest] = raw
	}
	invoked := make(map[string]bool)
	for _, output := range invocation.List("outputs") {
		invoked[output.Str("output_id")] = true
	}
	declared := make(map[string]bool, len(intent.Outputs))
	for _, output := range intent.Outputs {
		if output.Name == "" || declared[output.Name] || !invoked[output.Name] {
			return refuse()
		}
		declared[output.Name] = true
	}
	rows, err := tx.Query(`SELECT o.output_slot,o.manifest_id,o.manifest_length,o.invocation_digest,o.transaction_id,o.receipt_digest,o.receipt,o.final_id,
 EXISTS(SELECT 1 FROM request_model_transfer_objects b WHERE b.request_id=o.request_id AND b.attempt=o.attempt AND b.output_slot=o.output_slot AND b.object_id=o.manifest_id AND b.length=o.manifest_length)
 FROM request_model_transfer_outputs o WHERE o.request_id=? AND o.attempt=?`, requestID, ordinal)
	if err != nil {
		return exit.Internalf("cannot read verified output roster: %s", err)
	}
	defer rows.Close()
	count := 0
	for rows.Next() {
		var slot, manifest, invocation, transaction, digest, finalID string
		var length int64
		var rootPresent bool
		var raw []byte
		if err := rows.Scan(&slot, &manifest, &length, &invocation, &transaction, &digest, &raw, &finalID, &rootPresent); err != nil {
			return exit.Internalf("cannot read verified output: %s", err)
		}
		if !declared[slot] || finalID != ModelTransferOutputOperation(requestID, slot) || !rootPresent || length <= 0 || invocation != invocationDigest || !bytes.Equal(receiptRefs[digest], raw) {
			return refuse()
		}
		if _, err := canonical.Raw(manifest); err != nil {
			return refuse()
		}
		receipt, err := archive.Read(raw, archive.WeightsReceipt)
		if err != nil || receipt.Str("request_id") != requestID || receipt.Str("output_slot") != slot || receipt.Str("invocation_spec_digest") != invocationDigest || receipt.Str("weights_transaction_id") != transaction {
			return refuse()
		}
		delete(declared, slot)
		count++
	}
	if err := rows.Err(); err != nil {
		return exit.Internalf("cannot finish verified output roster: %s", err)
	}
	if len(declared) != 0 || count != len(receiptRefs) {
		return refuse()
	}
	return nil
}

func (s *Store) CompleteModelTransferCancellation(requestID string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_model_transfers SET state='canceled',updated_at=? WHERE request_id=? AND state='canceling'`, now(), requestID)
	if err != nil {
		return exit.Internalf("cannot complete publication cancellation: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed == 1 {
		return nil
	}
	transfer, problem := s.ModelTransferOf(requestID)
	if problem == nil && transfer != nil && transfer.State == "canceled" {
		return nil
	}
	return exit.New(exit.Conflict, "publication cancellation changed before completion")
}
