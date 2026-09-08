package records

import (
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

func ArtifactRetentionID(request, kind, slot string, artifact ModelArtifact) string {
	raw, _ := json.Marshal(map[string]any{"request_id": request, "kind": kind, "slot": slot, "producer_request_id": artifact.ProducerRequestID, "output_slot": artifact.OutputSlot, "tensorfs_receipt_digest": artifact.TensorFSReceiptDigest})
	raw, _ = canonical.NormalizeJCS(raw)
	id, _ := canonical.Spell(canonical.Digest(raw))
	return id
}

const weightsRetentionsDDL = `
CREATE TABLE IF NOT EXISTS request_weights_retentions (
 request_id TEXT NOT NULL REFERENCES requests(id),
 kind TEXT NOT NULL CHECK(kind IN ('input','result')),
 slot TEXT NOT NULL,
 producer_request_id TEXT NOT NULL,
 producer_attempt INTEGER NOT NULL,
 producer_output_slot TEXT NOT NULL,
 retention_id TEXT NOT NULL UNIQUE,
 instance_id TEXT NOT NULL DEFAULT '',
 worker_boot_id TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','held','releasing','released')),
 PRIMARY KEY(request_id,kind,slot),
 FOREIGN KEY(producer_request_id,producer_attempt,producer_output_slot) REFERENCES request_model_transfer_outputs(request_id,attempt,output_slot)
)`

type WeightsRetention struct {
	RequestID          string
	Kind               string
	Slot               string
	ProducerRequestID  string
	ProducerAttempt    int64
	ProducerOutputSlot string
	RetentionID        string
	InstanceID         string
	WorkerBootID       string
	State              string
}

const weightsRetentionCols = `request_id,kind,slot,producer_request_id,producer_attempt,producer_output_slot,retention_id,instance_id,worker_boot_id,state`

func scanWeightsRetention(row interface{ Scan(...any) error }) (WeightsRetention, error) {
	var r WeightsRetention
	err := row.Scan(&r.RequestID, &r.Kind, &r.Slot, &r.ProducerRequestID, &r.ProducerAttempt, &r.ProducerOutputSlot, &r.RetentionID, &r.InstanceID, &r.WorkerBootID, &r.State)
	return r, err
}

func (s *Store) RecordWeightsRetention(r WeightsRetention) (WeightsRetention, *exit.Error) {
	if r.RequestID == "" || (r.Kind != "input" && r.Kind != "result") || r.Slot == "" || len(r.Slot) > 512 || r.ProducerRequestID == "" || r.ProducerAttempt <= 0 || r.ProducerOutputSlot == "" {
		return r, exit.New(exit.Validation, "weights retention needs an exact consumer and original output")
	}
	if _, err := canonical.Raw(r.RetentionID); err != nil {
		return r, exit.New(exit.Validation, "weights retention identity is malformed")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return r, exit.Internalf("cannot begin weights retention: %s", err)
	}
	defer tx.Rollback()
	held, problem := recordWeightsRetentionTx(tx, r)
	if problem != nil {
		return r, problem
	}
	if err := tx.Commit(); err != nil {
		return r, exit.Internalf("cannot commit artifact retention: %s", err)
	}
	return held, nil
}

func recordWeightsRetentionTx(tx *sql.Tx, r WeightsRetention) (WeightsRetention, *exit.Error) {
	var allowed bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests consumer JOIN requests producer ON consumer.worker=producer.worker
		WHERE consumer.id=? AND producer.id=? AND consumer.retain_work=1 AND consumer.reuse_scope<>'' AND consumer.worker=producer.worker
		AND (consumer.reuse_scope=producer.reuse_scope OR EXISTS(SELECT 1 FROM request_weights_retentions adopted JOIN requests keeper ON keeper.id=adopted.request_id
		WHERE adopted.producer_request_id=producer.id AND adopted.producer_attempt=? AND adopted.producer_output_slot=? AND adopted.state='held' AND keeper.reuse_scope=consumer.reuse_scope AND keeper.state NOT IN ('canceling','canceled','releasing')))
		AND consumer.state IN ('submitted','queued','dispatching','finalizing','succeeded')
		AND (producer.state NOT IN ('canceling','canceled','releasing') OR (consumer.reused_from=producer.id AND consumer.state='finalizing' AND producer.state='canceling')
		OR EXISTS(SELECT 1 FROM request_weights_retentions reserved WHERE reserved.request_id=consumer.id AND reserved.producer_request_id=producer.id AND reserved.state='pending' AND producer.state='canceling')
		OR EXISTS(SELECT 1 FROM request_weights_retentions live JOIN requests keeper ON keeper.id=live.request_id
		WHERE live.producer_request_id=producer.id AND live.producer_attempt=? AND live.producer_output_slot=? AND live.state='held' AND keeper.reuse_scope=consumer.reuse_scope AND keeper.state NOT IN ('canceling','canceled','releasing'))))`, r.RequestID, r.ProducerRequestID, r.ProducerAttempt, r.ProducerOutputSlot, r.ProducerAttempt, r.ProducerOutputSlot).Scan(&allowed); err != nil {
		return r, exit.Internalf("cannot validate artifact retention scope: %s", err)
	}
	if !allowed {
		return r, exit.Named(exit.Conflict, "child.artifact_scope", "artifact retention requires the same retained scope and physical store")
	}
	if _, err := tx.Exec(`INSERT INTO request_weights_retentions(request_id,kind,slot,producer_request_id,producer_attempt,producer_output_slot,retention_id)
		VALUES(?,?,?,?,?,?,?) ON CONFLICT(request_id,kind,slot) DO NOTHING`, r.RequestID, r.Kind, r.Slot, r.ProducerRequestID, r.ProducerAttempt, r.ProducerOutputSlot, r.RetentionID); err != nil {
		return r, exit.Internalf("cannot record artifact retention intent: %s", err)
	}
	held, err := scanWeightsRetention(tx.QueryRow(`SELECT `+weightsRetentionCols+` FROM request_weights_retentions WHERE request_id=? AND kind=? AND slot=?`, r.RequestID, r.Kind, r.Slot))
	if err != nil {
		return r, exit.Internalf("cannot read artifact retention intent: %s", err)
	}
	if held.ProducerRequestID != r.ProducerRequestID || held.ProducerAttempt != r.ProducerAttempt || held.ProducerOutputSlot != r.ProducerOutputSlot || held.RetentionID != r.RetentionID || held.State == "releasing" || held.State == "released" {
		return r, exit.Named(exit.Conflict, "child.artifact_retention_changed", "this retention already names different or released ownership")
	}
	return held, nil
}

func (s *Store) WeightsRetentions(requestID string) ([]WeightsRetention, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+weightsRetentionCols+` FROM request_weights_retentions WHERE request_id=? ORDER BY kind,slot`, requestID)
	if err != nil {
		return nil, exit.Internalf("cannot read artifact retentions: %s", err)
	}
	defer rows.Close()
	var out []WeightsRetention
	for rows.Next() {
		r, err := scanWeightsRetention(rows)
		if err != nil {
			return nil, exit.Internalf("cannot read artifact retention: %s", err)
		}
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish artifact retention census: %s", err)
	}
	return out, nil
}

func (s *Store) ConfirmWeightsRetention(id, instance, boot string) *exit.Error {
	result, err := s.db.Exec(`UPDATE request_weights_retentions SET state='held',instance_id=?,worker_boot_id=? WHERE retention_id=? AND state IN ('pending','held')`, instance, boot, id)
	if err != nil {
		return exit.Internalf("cannot confirm native artifact retention: %s", err)
	}
	if n, _ := result.RowsAffected(); n != 1 {
		return exit.Named(exit.Conflict, "child.artifact_retention_closed", "artifact retention was released while native adoption completed")
	}
	return nil
}

func (s *Store) BeginWeightsRetentionRelease(requestID string, inputsOnly bool) *exit.Error {
	filter := ""
	if inputsOnly {
		filter = " AND kind='input'"
	}
	_, err := s.db.Exec(`UPDATE request_weights_retentions SET state='releasing' WHERE request_id=? AND state IN ('pending','held')`+filter, requestID)
	if err != nil {
		return exit.Internalf("cannot record artifact retention release: %s", err)
	}
	return nil
}

func (s *Store) CompleteWeightsRetentionRelease(id string) *exit.Error {
	_, err := s.db.Exec(`UPDATE request_weights_retentions SET state='released' WHERE retention_id=? AND state='releasing'`, id)
	if err != nil {
		return exit.Internalf("cannot close artifact retention release: %s", err)
	}
	return nil
}

func (s *Store) ArtifactHasCustody(producer string, attempt int64, slot, scope string) (bool, *exit.Error) {
	var found bool
	err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests p WHERE p.id=? AND p.reuse_scope=? AND p.retain_work=1 AND p.state NOT IN ('canceled','releasing'))
		OR EXISTS(SELECT 1 FROM request_weights_retentions h JOIN requests r ON r.id=h.request_id WHERE h.producer_request_id=? AND h.producer_attempt=? AND h.producer_output_slot=? AND h.state='held' AND r.reuse_scope=? AND r.state NOT IN ('canceled','releasing'))`, producer, scope, producer, attempt, slot, scope).Scan(&found)
	if err != nil {
		return false, exit.Internalf("cannot validate model artifact custody: %s", err)
	}
	return found, nil
}

func (s *Store) PendingArtifactBorrowers(producer string) (bool, *exit.Error) {
	var found bool
	if err := s.db.QueryRow(`SELECT EXISTS(SELECT 1 FROM request_weights_retentions h JOIN requests r ON r.id=h.request_id WHERE h.producer_request_id=? AND h.state='pending' AND r.state NOT IN ('canceling','releasing','canceled'))
		OR EXISTS(SELECT 1 FROM requests WHERE reused_from=? AND state='finalizing')
		OR EXISTS(SELECT 1 FROM native_artifact_retentions h JOIN requests p ON p.id=h.parent_request_id
		 LEFT JOIN requests c ON c.id=h.consumer_id WHERE h.owner_request_id=? AND h.state='pending'
		 AND COALESCE(c.state,p.state) NOT IN ('canceling','releasing','canceled'))`, producer, producer, producer).Scan(&found); err != nil {
		return false, exit.Internalf("cannot read pending artifact borrowers: %s", err)
	}
	return found, nil
}

func (s *Store) NativeArtifactRetention(id string) (*WeightsRetention, *exit.Error) {
	r, err := scanWeightsRetention(s.db.QueryRow(`SELECT `+weightsRetentionCols+` FROM request_weights_retentions WHERE retention_id=?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read artifact retention identity: %s", err)
	}
	return &r, nil
}
