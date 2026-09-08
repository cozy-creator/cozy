package records

import (
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// ReserveAssessmentBytes captures two actually received file results for this
// accepted effect in one transaction. The report's original producing request is
// also the only permitted render-composition owner; author payloads cannot pick it.
func (s *Store) ReserveAssessmentBytes(callID, parentID, report, workloads string) ([]NativeArtifactRetention, string, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, "", exit.Internalf("cannot begin assessment custody: %s", err)
	}
	defer tx.Rollback()
	call, err := scanNativeCall(tx.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE id=?`, callID))
	if err != nil || call.ParentRequestID != parentID || call.Kind != "effect" || call.Operation != "attach_assessment" || call.CancelRequested || (call.State != "accepted" && call.State != "frozen") {
		return nil, "", exit.Named(exit.Conflict, "assessment.effect_stopped", "assessment has no active accepted effect")
	}

	var requested struct {
		Report    string `json:"report"`
		Workloads string `json:"workloads"`
	}
	if json.Unmarshal(call.Request, &requested) != nil || requested.Report != report || requested.Workloads != workloads {
		return nil, "", exit.New(exit.Conflict, "assessment file inputs differ from accepted effect")
	}
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, parentID))
	if err != nil || parent.State != "dispatching" || !parent.RetainWork {
		return nil, "", exit.Named(exit.Conflict, "assessment.parent_stopped", "assessment parent no longer holds execution authority")
	}

	existing, err := tx.Query(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? AND kind='effect' ORDER BY slot`, call.ID)
	if err != nil {
		return nil, "", exit.Internalf("cannot inspect assessment retention replay: %s", err)
	}
	var prior []NativeArtifactRetention
	for existing.Next() {
		h, err := scanNativeArtifact(existing)
		if err != nil {
			existing.Close()
			return nil, "", exit.Internalf("cannot read assessment retention replay: %s", err)
		}
		prior = append(prior, h)
	}
	replayErr := existing.Err()
	existing.Close()
	if replayErr != nil {
		return nil, "", exit.Internalf("cannot finish assessment retention replay: %s", replayErr)
	}
	if len(prior) > 0 {
		if len(prior) != 2 {
			return nil, "", exit.New(exit.Conflict, "assessment retention inventory changed")
		}
		producer := ""
		seen := map[string]bool{}
		for _, h := range prior {
			expected := report
			if h.Slot == "workloads" {
				expected = workloads
			} else if h.Slot != "report" {
				return nil, "", exit.New(exit.Conflict, "assessment input slot changed")
			}
			b, err := scanByteOutput(tx.QueryRow(`SELECT `+byteOutputCols+` FROM byte_outputs WHERE request_id=? AND attempt=? AND output_id=?`, h.ProducerID, h.ProducerAttempt, h.ProducerOutputID))
			if err != nil || seen[h.Slot] || h.ArtifactKind != "tree" || h.ParentRequestID != parentID || (h.State != "pending" && h.State != "held") || b.Digest != expected || b.ProducerRootID != h.TransactionID || b.ReceiptDigest != h.ReceiptDigest || b.ManifestID != h.ManifestID || b.ManifestLength != h.ManifestLength || b.ContentBytes != h.ContentBytes || (producer != "" && producer != h.ProducerID) {
				return nil, "", exit.New(exit.Conflict, "assessment retained input changed or was released")
			}
			seen[h.Slot] = true
			producer = h.ProducerID
		}
		return prior, producer, nil
	}
	var holds []NativeArtifactRetention
	producer := ""
	for _, input := range []struct{ slot, digest string }{{"report", report}, {"workloads", workloads}} {
		if _, err := canonical.Raw(input.digest); err != nil {
			return nil, "", exit.New(exit.Validation, "assessment input is not an immutable file reference")
		}
		rows, err := tx.Query(`SELECT b.`+`request_id,b.attempt,b.output_id,b.digest,b.length,b.mime_type,b.producer_root_id,b.receipt_digest,b.manifest_id,b.manifest_length,b.content_bytes
   FROM native_artifact_retentions h JOIN byte_outputs b ON b.request_id=h.producer_id AND b.attempt=h.producer_attempt AND b.output_id=h.producer_output_id
   JOIN requests c ON c.id=h.consumer_id WHERE h.parent_request_id=? AND h.kind='result' AND h.artifact_kind='tree' AND h.state='held' AND b.digest=? AND c.worker=? AND c.state='succeeded' ORDER BY b.request_id,b.attempt,b.output_id`, parentID, input.digest, parent.Worker)
		if err != nil {
			return nil, "", exit.Internalf("cannot resolve assessment file grant: %s", err)
		}
		var candidates []ByteOutput
		for rows.Next() {
			b, err := scanByteOutput(rows)
			if err != nil {
				rows.Close()
				return nil, "", exit.Internalf("cannot decode assessment file grant: %s", err)
			}
			candidates = append(candidates, b)
		}
		readErr := rows.Err()
		rows.Close()
		if readErr != nil {
			return nil, "", exit.Internalf("cannot finish assessment grant lookup: %s", readErr)
		}
		var found *ByteOutput
		for _, b := range candidates {
			if producer != "" && b.RequestID != producer {
				continue
			}
			if found != nil && found.RequestID != b.RequestID {
				return nil, "", exit.New(exit.Conflict, "assessment file digest has ambiguous producing execution")
			}
			copy := b
			found = &copy
		}
		if found == nil || found.Length <= 0 || found.Length > 8<<20 || found.ContentBytes != found.Length || found.MimeType == "application/vnd.cozy.tree-manifest" {
			return nil, "", exit.Named(exit.Conflict, "assessment.file_ungranted", "assessment requires received bounded report and workload files from one producing execution")
		}
		b := *found
		producer = b.RequestID
		id := ByteRetentionID(call.ID, "effect", input.slot, b)
		_, err = tx.Exec(`INSERT INTO native_artifact_retentions(artifact_kind,producer_attempt,producer_output_id,content_bytes,consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,owner_worker,retention_id) VALUES('tree',?,?,?,?,?,'effect',?,?,?,?,?,?,?,?,?) ON CONFLICT(consumer_id,kind,slot) DO NOTHING`, b.Attempt, b.OutputID, b.ContentBytes, call.ID, parentID, input.slot, b.RequestID, b.ManifestID, b.ManifestLength, b.ReceiptDigest, b.ProducerRootID, b.RequestID, parent.Worker, id)
		if err != nil {
			return nil, "", exit.Internalf("cannot reserve assessment file custody: %s", err)
		}
		h, err := scanNativeArtifact(tx.QueryRow(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? AND kind='effect' AND slot=?`, call.ID, input.slot))
		if err != nil || h.RetentionID != id || h.TransactionID != b.ProducerRootID || h.ReceiptDigest != b.ReceiptDigest || h.ManifestID != b.ManifestID || h.ManifestLength != b.ManifestLength || h.ProducerID != b.RequestID || h.ProducerAttempt != b.Attempt || h.ProducerOutputID != b.OutputID || h.ContentBytes != b.ContentBytes || (h.State != "pending" && h.State != "held") {
			return nil, "", exit.Named(exit.Conflict, "assessment.custody_changed", "assessment already names changed or released file custody")
		}
		holds = append(holds, h)
	}
	if err := tx.Commit(); err != nil {
		return nil, "", exit.Internalf("cannot commit assessment file custody: %s", err)
	}
	return holds, producer, nil
}

func (s *Store) ByteOutputForRetention(h NativeArtifactRetention) (ByteOutput, *exit.Error) {
	b, err := scanByteOutput(s.db.QueryRow(`SELECT `+byteOutputCols+` FROM byte_outputs WHERE request_id=? AND attempt=? AND output_id=?`, h.ProducerID, h.ProducerAttempt, h.ProducerOutputID))
	if err == sql.ErrNoRows {
		return b, exit.New(exit.Conflict, "assessment native output was not retained")
	}
	if err != nil {
		return b, exit.Internalf("cannot read assessment native output: %s", err)
	}
	if h.ArtifactKind != "tree" || b.ProducerRootID != h.TransactionID || b.ReceiptDigest != h.ReceiptDigest || b.ManifestID != h.ManifestID || b.ManifestLength != h.ManifestLength || b.ContentBytes != h.ContentBytes {
		return b, exit.New(exit.Conflict, "assessment native output changed")
	}
	return b, nil
}
