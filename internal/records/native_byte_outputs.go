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
