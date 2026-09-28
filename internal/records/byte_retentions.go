package records

import (
	"database/sql"
	"reflect"

	"github.com/cozy-creator/cozy/internal/exit"
)

func reserveByteRetentionTx(tx *sql.Tx, consumer Request, kind, slot string, b ByteOutput, donor string) (NativeArtifactRetention, *exit.Error) {
	fail := func(message string) (NativeArtifactRetention, *exit.Error) {
		return NativeArtifactRetention{}, exit.Named(exit.Conflict, "child.byte_scope", "%s", message)
	}
	rootResult := consumer.ParentRequestID == "" && consumer.RetainsLocalOutputs() && kind == "result"
	if (consumer.ParentRequestID == "" && !rootResult) || !consumer.RetainWork || (kind != "input" && kind != "result") || len(slot) == 0 || len(slot) > 512 {
		return fail("byte custody requires a bounded private child obligation")
	}
	parentID := consumer.ParentRequestID
	if rootResult {
		parentID = consumer.ID
	}
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, parentID))
	expectedState := "dispatching"
	if rootResult {
		expectedState = "succeeded"
	}
	if err != nil || (parent.State != expectedState && !(rootResult && parent.State == "finalizing")) || !parent.RetainWork {
		return fail("byte recipient parent is not executing")
	}
	if consumer.State == "canceling" || consumer.State == "canceled" || consumer.State == "releasing" {
		return fail("byte recipient is canceled")
	}
	source, err := scanByteOutput(tx.QueryRow(`SELECT `+byteOutputCols+` FROM byte_outputs WHERE request_id=? AND attempt=? AND output_id=?`, b.RequestID, b.Attempt, b.OutputID))
	if err != nil || source != b {
		return fail("byte output differs from its immutable terminal")
	}
	var sourceWorker, sourceState string
	if err := tx.QueryRow(`SELECT worker,state FROM requests WHERE id=?`, b.RequestID).Scan(&sourceWorker, &sourceState); err != nil || sourceWorker != consumer.Worker || parent.Worker != consumer.Worker {
		return fail("byte custody requires the same private native workspace")
	}
	if kind == "input" {
		var allowed bool
		if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM native_artifact_retentions h JOIN requests c ON c.id=h.consumer_id WHERE h.retention_id=? AND h.artifact_kind='tree' AND h.state='held' AND h.producer_id=? AND h.producer_attempt=? AND h.producer_output_id=? AND c.state NOT IN ('canceling','canceled','releasing') AND ((h.kind='result' AND h.parent_request_id=?) OR (h.kind='input' AND h.consumer_id=?)))`, donor, b.RequestID, b.Attempt, b.OutputID, parent.ID, parent.ID).Scan(&allowed); err != nil || !allowed {
			return fail("byte input was not actually granted to this parent")
		}
	} else if consumer.ID != b.RequestID && consumer.ReusedFrom != b.RequestID {
		return fail("byte result does not belong to this child")
	} else if consumer.ReusedFrom != b.RequestID && sourceState != "succeeded" && !(rootResult && sourceState == "finalizing") {
		return fail("byte result producer was released")
	}
	id := ByteRetentionID(consumer.ID, kind, slot, b)
	_, err = tx.Exec(`INSERT INTO native_artifact_retentions(artifact_kind,producer_attempt,producer_output_id,content_bytes,consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,owner_worker,retention_id) VALUES('tree',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(consumer_id,kind,slot) DO NOTHING`, b.Attempt, b.OutputID, b.ContentBytes, consumer.ID, parent.ID, kind, slot, b.RequestID, b.ManifestID, b.ManifestLength, b.ReceiptDigest, b.ProducerRootID, b.RequestID, sourceWorker, id)
	if err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot reserve byte recipient: %s", err)
	}
	h, err := scanNativeArtifact(tx.QueryRow(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? AND kind=? AND slot=?`, consumer.ID, kind, slot))
	if err != nil {
		return h, exit.Internalf("cannot read byte recipient: %s", err)
	}
	if h.ArtifactKind != "tree" || (h.RetentionID != id && consumer.ReusedFrom != b.RequestID) || h.ProducerID != b.RequestID || h.ProducerAttempt != b.Attempt || h.ProducerOutputID != b.OutputID || h.ManifestID != b.ManifestID || h.ManifestLength != b.ManifestLength || h.ContentBytes != b.ContentBytes || h.ReceiptDigest != b.ReceiptDigest || h.TransactionID != b.ProducerRootID || h.State == "released" || h.State == "releasing" {
		return fail("byte retention changed or was released")
	}
	return h, nil
}
func (s *Store) ReserveByteResult(consumer string, b ByteOutput) (NativeArtifactRetention, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot begin byte retention: %s", err)
	}
	defer tx.Rollback()
	request, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, consumer))
	if err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot read byte recipient: %s", err)
	}
	h, problem := reserveByteRetentionTx(tx, request, "result", b.OutputID, b, "")
	if problem != nil {
		return h, problem
	}
	if err := tx.Commit(); err != nil {
		return h, exit.Internalf("cannot commit byte retention: %s", err)
	}
	return h, nil
}
func sameAssetCustody(a, b AssetBinding) bool {
	return a.Digest == b.Digest && a.Length == b.Length && a.MediaType == b.MediaType && a.LocalPath == b.LocalPath && reflect.DeepEqual(a.Native, b.Native)
}
