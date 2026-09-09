package records

import (
	"bytes"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// NativeByteProducerRoot follows the ordinary Runtime byte ledger identity. The
// producer is the recorded original attempt, independently of reply correlation.
func NativeByteProducerRoot(owner, request string, attempt int64, spec []byte, slot string) string {
	raw, _ := json.Marshal([]any{"byte-output", owner, request, attempt, hex.EncodeToString(spec), slot})
	raw, _ = canonical.NormalizeJCS(raw)
	digest, _ := canonical.Spell(canonical.Digest(raw))
	return digest
}

func (s *Store) NativeByteOutput(service string) (*ByteOutput, *exit.Error) {
	b, err := scanByteOutput(s.db.QueryRow(`SELECT `+byteOutputCols+` FROM byte_outputs WHERE native_service_id=?`, service))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot read native service byte output: %s", err)
	}
	return &b, nil
}

// CompleteNativeByteCall records immutable output provenance and recipient intent
// together. The recipient must be retained physically before exposing the value.
// Service outputs live in the common byte ledger but are not parent return fields.
func (s *Store) CompleteNativeByteCall(service string, currentAttempt int64, currentSpec, currentBoot, producerSpec string, b ByteOutput, result, receipt []byte, instance string) (NativeArtifactRetention, *exit.Error) {
	fail := func(message string) (NativeArtifactRetention, *exit.Error) {
		return NativeArtifactRetention{}, exit.Named(exit.Conflict, "native.byte_authority", "%s", message)
	}
	raw, err := canonical.NormalizeJCS(result)
	if err != nil || !bytes.Equal(raw, result) || len(result) > 48<<10 || len(receipt) == 0 || len(receipt) > 1<<20 || b.Attempt <= 0 || b.Attempt > currentAttempt {
		return fail("native byte completion lacks bounded immutable facts")
	}
	if problem := ValidateByteRef(b.NativeRef()); problem != nil {
		return NativeArtifactRetention{}, problem
	}
	digest, _ := canonical.Spell(canonical.Digest(receipt))
	if digest != b.ReceiptDigest {
		return fail("native receipt differs from its byte reference")
	}
	tx, err := s.db.Begin()
	if err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot begin native byte completion: %s", err)
	}
	defer tx.Rollback()
	call, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE id=?`, service))
	if err != nil || call.Kind != "source" || (call.Operation != "source_files" && call.Operation != "commit_file") || call.ParentRequestID != b.RequestID || b.OutputID != fmt.Sprintf("runtime.%s.%d", call.Operation, call.CallIndex) {
		return fail("native byte output has no exact accepted service")
	}
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, call.ParentRequestID))
	if err != nil || parent.State != "dispatching" || !parent.RetainWork || parent.Ordinal != currentAttempt || parent.Worker != call.Worker {
		return fail("native byte recipient has no current private parent")
	}
	var owned bool
	if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM attempts WHERE request_id=? AND attempt=? AND invocation_digest=? AND session_id=? AND state IN ('offered','accepted','recovered_open'))`, parent.ID, currentAttempt, currentSpec, currentBoot).Scan(&owned); err != nil || !owned {
		return fail("native byte reply does not belong to the current attempt")
	}
	var originalSpec string
	if err := tx.QueryRow(`SELECT invocation_digest FROM attempts WHERE request_id=? AND attempt=?`, parent.ID, b.Attempt).Scan(&originalSpec); err != nil || originalSpec != producerSpec {
		return fail("native byte producer differs from its recorded original attempt")
	}
	if call.State == "succeeded" {
		if !bytes.Equal(call.Result, result) || !bytes.Equal(call.NativeReceipt, receipt) {
			return fail("native byte completion changed its immutable result")
		}
	} else if call.State != "accepted" && call.State != "frozen" && call.State != "executing" {
		return fail("native byte call stopped before completion")
	}
	old, err := scanByteOutput(tx.QueryRow(`SELECT `+byteOutputCols+` FROM byte_outputs WHERE native_service_id=?`, service))
	if err == nil {
		if old != b {
			return fail("native service changed its original byte producer")
		}
	} else if errors.Is(err, sql.ErrNoRows) {
		if _, err := tx.Exec(`INSERT INTO byte_outputs(`+byteOutputCols+`,native_service_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`, b.RequestID, b.Attempt, b.OutputID, b.Digest, b.Length, b.MimeType, b.ProducerRootID, b.ReceiptDigest, b.ManifestID, b.ManifestLength, b.ContentBytes, service); err != nil {
			return NativeArtifactRetention{}, exit.Internalf("cannot record native service byte producer: %s", err)
		}
	} else {
		return NativeArtifactRetention{}, exit.Internalf("cannot read native service producer: %s", err)
	}
	slot := b.OutputID
	id := ByteRetentionID(parent.ID, "result", slot, b)
	if _, err := tx.Exec(`INSERT INTO native_artifact_retentions(artifact_kind,producer_attempt,producer_output_id,content_bytes,consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,owner_worker,retention_id) VALUES('tree',?,?,?,?,?,'result',?,?,?,?,?,?,?,?,?) ON CONFLICT(consumer_id,kind,slot) DO NOTHING`, b.Attempt, b.OutputID, b.ContentBytes, parent.ID, parent.ID, slot, parent.ID, b.ManifestID, b.ManifestLength, b.ReceiptDigest, b.ProducerRootID, parent.ID, parent.Worker, id); err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot reserve native byte recipient: %s", err)
	}
	h, err := scanNativeArtifact(tx.QueryRow(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? AND kind='result' AND slot=?`, parent.ID, slot))
	if err != nil || h.RetentionID != id || h.State == "releasing" || h.State == "released" {
		return fail("native byte recipient changed or was released")
	}
	if call.State != "succeeded" {
		if _, err := tx.Exec(`UPDATE native_calls SET state='succeeded',result=?,native_receipt=?,instance_id=?,worker_boot_id=? WHERE id=?`, result, receipt, instance, currentBoot, service); err != nil {
			return NativeArtifactRetention{}, exit.Internalf("cannot complete native byte call: %s", err)
		}
		if err := appendEventTx(tx, parent.ID, "native.completed", currentAttempt, map[string]any{"call_index": call.CallIndex, "service_id": service, "producer_attempt": b.Attempt}); err != nil {
			return NativeArtifactRetention{}, exit.Internalf("cannot journal native byte completion: %s", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot commit native byte completion: %s", err)
	}
	return h, nil
}

// CompletedNativeByteRecipients are disposal obligations from closed successful
// parents. Intermediate service views are not returned-output or cache owners.
func (s *Store) CompletedNativeByteRecipients() ([]string, *exit.Error) {
	rows, err := s.db.Query(`SELECT DISTINCT h.consumer_id FROM native_artifact_retentions h
 JOIN byte_outputs b ON b.request_id=h.producer_id AND b.attempt=h.producer_attempt AND b.output_id=h.producer_output_id
 JOIN requests r ON r.id=h.consumer_id JOIN attempts a ON a.request_id=r.id AND a.attempt=r.ordinal
 WHERE h.consumer_id=h.parent_request_id AND h.kind='result' AND h.artifact_kind='tree' AND h.state<>'released'
 AND b.native_service_id IS NOT NULL AND r.state='succeeded' AND a.state='closed' AND a.terminal_status='SUCCEEDED'`)
	if err != nil {
		return nil, exit.Internalf("cannot read completed native byte recipients: %s", err)
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, exit.Internalf("cannot read native byte cleanup owner: %s", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, exit.Internalf("cannot finish native byte cleanup census: %s", err)
	}
	return ids, nil
}

func (s *Store) ReleaseCompletedNativeBytes(parent string) *exit.Error {
	_, err := s.db.Exec(`UPDATE native_artifact_retentions SET state='releasing'
 WHERE consumer_id=? AND consumer_id=parent_request_id AND kind='result' AND artifact_kind='tree' AND state IN ('pending','held')
 AND EXISTS(SELECT 1 FROM requests r JOIN attempts a ON a.request_id=r.id AND a.attempt=r.ordinal WHERE r.id=? AND r.state='succeeded' AND a.state='closed' AND a.terminal_status='SUCCEEDED')
 AND EXISTS(SELECT 1 FROM byte_outputs b WHERE b.request_id=producer_id AND b.attempt=producer_attempt AND b.output_id=producer_output_id AND b.native_service_id IS NOT NULL)`, parent, parent)
	if err != nil {
		return exit.Internalf("cannot release completed native byte recipients: %s", err)
	}
	return nil
}

// CompleteNativeRootResult completes the existing finalizing lifecycle only after
// every explicit final output has independent, confirmed native custody.
func (s *Store) CompleteNativeRootResult(request string, attempt int64) *exit.Error {
	tx, err := s.db.Begin()
	if err != nil {
		return exit.Internalf("cannot begin native result completion: %s", err)
	}
	defer tx.Rollback()
	row, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, request))
	if err != nil || row.ParentRequestID != "" || !row.RetainsLocalOutputs() || row.Ordinal != attempt {
		return exit.New(exit.Conflict, "native result changed its owner")
	}
	if row.State == "succeeded" {
		return nil
	}
	if row.State != "finalizing" {
		return exit.New(exit.Conflict, "native result owner stopped")
	}
	var completed, total, missing int
	if err := tx.QueryRow(`SELECT count(*) FROM attempts WHERE request_id=? AND attempt=? AND terminal_status='SUCCEEDED' AND state IN ('terminal','closed')`, request, attempt).Scan(&completed); err != nil || completed != 1 {
		return exit.New(exit.Conflict, "native result has no successful computation")
	}
	if err := tx.QueryRow(`SELECT count(*) FROM byte_outputs WHERE request_id=? AND attempt=? AND native_service_id IS NULL`, request, attempt).Scan(&total); err != nil || total == 0 {
		return exit.New(exit.Conflict, "native result has no recorded final outputs")
	}
	if err := tx.QueryRow(`SELECT count(*) FROM byte_outputs b WHERE b.request_id=? AND b.attempt=? AND b.native_service_id IS NULL AND NOT EXISTS(
 SELECT 1 FROM native_artifact_retentions h WHERE h.consumer_id=b.request_id AND h.parent_request_id=b.request_id AND h.kind='result' AND h.artifact_kind='tree' AND h.state='held'
 AND h.producer_id=b.request_id AND h.producer_attempt=b.attempt AND h.producer_output_id=b.output_id AND h.transaction_id=b.producer_root_id AND h.receipt_digest=b.receipt_digest AND h.manifest_id=b.manifest_id AND h.manifest_length=b.manifest_length AND h.content_bytes=b.content_bytes)`, request, attempt).Scan(&missing); err != nil || missing != 0 {
		return exit.Unavailablef("native result still awaits independent custody")
	}
	var raw string
	if err := tx.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND attempt=? AND type='request.finalizing' ORDER BY seq DESC LIMIT 1`, request, attempt).Scan(&raw); err != nil {
		return exit.New(exit.Conflict, "native result has no finalizing event")
	}
	var payload map[string]any
	if json.Unmarshal([]byte(raw), &payload) != nil {
		return exit.New(exit.Conflict, "native finalizing event is invalid")
	}
	payload["status"] = "SUCCEEDED"
	delete(payload, "execution_status")
	if _, err := tx.Exec(`UPDATE requests SET state='succeeded' WHERE id=? AND state='finalizing'`, request); err != nil {
		return exit.Internalf("cannot complete native result: %s", err)
	}
	if err := appendEventTx(tx, request, "request.completed", attempt, payload); err != nil {
		return exit.Internalf("cannot record native result completion: %s", err)
	}
	if err := tx.Commit(); err != nil {
		return exit.Internalf("cannot commit native result completion: %s", err)
	}
	return nil
}

func (s *Store) NoteNativeResultWait(request string, attempt int64, code, detail string) *exit.Error {
	if len(code) > 128 || len(detail) > 1024 {
		return exit.New(exit.Validation, "native retention diagnostic exceeds its bound")
	}
	raw, err := json.Marshal(map[string]string{"error_type": code, "error": detail})
	if err != nil {
		return exit.Internalf("cannot encode native retention diagnostic: %s", err)
	}
	_, err = s.db.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) SELECT ?,'native.result_wait',?,?,? WHERE EXISTS(SELECT 1 FROM requests WHERE id=? AND state='finalizing')
 AND COALESCE((SELECT payload FROM request_events WHERE request_id=? AND attempt=? AND type='native.result_wait' ORDER BY seq DESC LIMIT 1),'')<>?`, request, attempt, string(raw), now(), request, request, attempt, string(raw))
	if err != nil {
		return exit.Internalf("cannot record native retention wait: %s", err)
	}
	return nil
}

func (s *Store) NativeResultWait(request string, attempt int64) (string, string, *exit.Error) {
	var raw string
	err := s.db.QueryRow(`SELECT payload FROM request_events WHERE request_id=? AND attempt=? AND type='native.result_wait' ORDER BY seq DESC LIMIT 1`, request, attempt).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return "", "", nil
	}
	if err != nil {
		return "", "", exit.Internalf("cannot read native retention wait: %s", err)
	}
	var value map[string]string
	if json.Unmarshal([]byte(raw), &value) != nil {
		return "", "", exit.New(exit.Internal, "native retention diagnostic is invalid")
	}
	return value["error_type"], value["error"], nil
}
