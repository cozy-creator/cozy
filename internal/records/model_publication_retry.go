package records

import (
	"github.com/cozy-creator/cozy/internal/exit"
)

// RetryModelTransferPublication changes only the unfinished destination transfer.
// The request, attempt, native receipts and successful object proofs remain fixed.
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
	if requestState != "finalizing" || outcome != "SUCCEEDED" || (attemptState != "terminal" && attemptState != "closed") {
		return false, exit.Named(exit.Conflict, "model_transfer.publication_not_retryable", "publication retry requires the same successful unfinished producer request")
	}
	if transferState == "finalizing" {
		return false, nil
	}
	if transferState != "failed" {
		return false, exit.Named(exit.Conflict, "model_transfer.publication_not_retryable", "this publication has no blocked transfer to retry")
	}
	result, err := tx.Exec(`UPDATE request_model_transfers SET state='finalizing',error_code='',safe_error='',updated_at=?
		WHERE request_id=? AND state='failed'`, now(), requestID)
	if err != nil {
		return false, exit.Internalf("cannot retry publication: %s", err)
	}
	if changed, _ := result.RowsAffected(); changed != 1 {
		return false, exit.New(exit.Conflict, "publication changed before retry")
	}
	if _, err = tx.Exec(`UPDATE request_model_transfer_objects SET state='pending',safe_code='',safe_detail=''
		WHERE request_id=? AND attempt=? AND state='failed'`, requestID, ordinal); err != nil {
		return false, exit.Internalf("cannot retry unfinished publication objects: %s", err)
	}
	if err = appendEventTx(tx, requestID, "request.publication_retry_requested", ordinal, map[string]any{"actor": actor}); err != nil {
		return false, exit.Internalf("cannot record publication retry: %s", err)
	}
	if err = tx.Commit(); err != nil {
		return false, exit.Internalf("cannot commit publication retry: %s", err)
	}
	return true, nil
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
