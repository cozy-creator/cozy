package records

import (
	"database/sql"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
)

const nativeArtifactRetentionsDDL = `CREATE TABLE IF NOT EXISTS native_artifact_retentions (
 consumer_id TEXT NOT NULL, parent_request_id TEXT NOT NULL REFERENCES requests(id),
 kind TEXT NOT NULL CHECK(kind IN ('input','result','effect')),slot TEXT NOT NULL,
 producer_id TEXT NOT NULL, manifest_id TEXT NOT NULL, manifest_length INTEGER NOT NULL,
 receipt_digest TEXT NOT NULL, transaction_id TEXT NOT NULL, owner_request_id TEXT NOT NULL,
 retention_id TEXT NOT NULL UNIQUE, instance_id TEXT NOT NULL DEFAULT '',worker_boot_id TEXT NOT NULL DEFAULT '',
 state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','held','releasing','released')),
 PRIMARY KEY(consumer_id,kind,slot)
)`

type NativeArtifactRetention struct {
	ConsumerID, ParentRequestID, Kind, Slot                                                    string
	ProducerID, ManifestID                                                                     string
	ManifestLength                                                                             int64
	ReceiptDigest, TransactionID, OwnerRequestID, RetentionID, InstanceID, WorkerBootID, State string
}

const nativeArtifactCols = `consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,retention_id,instance_id,worker_boot_id,state`

func scanNativeArtifact(row interface{ Scan(...any) error }) (NativeArtifactRetention, error) {
	var h NativeArtifactRetention
	err := row.Scan(&h.ConsumerID, &h.ParentRequestID, &h.Kind, &h.Slot, &h.ProducerID, &h.ManifestID, &h.ManifestLength, &h.ReceiptDigest, &h.TransactionID, &h.OwnerRequestID, &h.RetentionID, &h.InstanceID, &h.WorkerBootID, &h.State)
	return h, err
}

func reserveNativeArtifactTx(tx *sql.Tx, consumerID, parentID, kind, slot string, artifact ModelArtifact) (NativeArtifactRetention, *exit.Error) {
	if consumerID == "" || parentID == "" || (kind != "input" && kind != "result" && kind != "effect") || slot == "" || len(slot) > 512 {
		return NativeArtifactRetention{}, exit.New(exit.Validation, "native retention needs a bounded consumer and slot")
	}
	parent, err := scanRequest(tx.QueryRow(`SELECT `+requestCols+` FROM requests WHERE id=?`, parentID))
	if err != nil {
		return NativeArtifactRetention{}, exit.Named(exit.Conflict, "child.artifact_scope", "native retention parent is absent")
	}
	if !parent.RetainWork || parent.ReuseScope == "" || parent.State == "canceling" || parent.State == "canceled" || parent.State == "releasing" {
		return NativeArtifactRetention{}, exit.Named(exit.Conflict, "child.artifact_scope", "native retention parent is stopped")
	}
	if kind == "effect" {
		var active bool
		if err := tx.QueryRow(`SELECT EXISTS(SELECT 1 FROM native_calls WHERE id=? AND parent_request_id=? AND kind='effect' AND state IN ('accepted','frozen','executing'))`, consumerID, parentID).Scan(&active); err != nil || !active {
			return NativeArtifactRetention{}, exit.Named(exit.Conflict, "native.effect_stopped", "native effect cannot acquire artifact custody")
		}
	} else if consumerID != parentID {
		return NativeArtifactRetention{}, exit.New(exit.Validation, "request retention uses its own request authority")
	}
	source, problem := nativeArtifactSource(tx, artifact)
	if problem != nil || source == nil {
		return NativeArtifactRetention{}, exit.Named(exit.Conflict, "child.artifact_unowned", "native artifact has no original service provenance")
	}
	allowed, problem := nativeArtifactAllowed(tx, parent, artifact)
	if problem != nil {
		return NativeArtifactRetention{}, problem
	}
	if !allowed {
		return NativeArtifactRetention{}, exit.Named(exit.Conflict, "child.artifact_scope", "native artifact was not received in this private scope")
	}
	id := ArtifactRetentionID(consumerID, kind, slot, artifact)
	_, err = tx.Exec(`INSERT INTO native_artifact_retentions(consumer_id,parent_request_id,kind,slot,producer_id,manifest_id,manifest_length,receipt_digest,transaction_id,owner_request_id,retention_id) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(consumer_id,kind,slot) DO NOTHING`, consumerID, parentID, kind, slot, artifact.ProducerRequestID, artifact.Manifest.Digest, artifact.Manifest.Length, artifact.TensorFSReceiptDigest, source.TransactionID, source.OwnerRequestID, id)
	if err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot reserve native artifact: %s", err)
	}
	h, err := scanNativeArtifact(tx.QueryRow(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? AND kind=? AND slot=?`, consumerID, kind, slot))
	if err != nil {
		return h, exit.Internalf("cannot read native artifact reservation: %s", err)
	}
	if h.RetentionID != id || h.ProducerID != artifact.ProducerRequestID || h.ManifestID != artifact.Manifest.Digest || h.ManifestLength != artifact.Manifest.Length || h.ReceiptDigest != artifact.TensorFSReceiptDigest || h.State == "released" || h.State == "releasing" {
		return h, exit.Named(exit.Conflict, "child.artifact_retention_changed", "native retention already names different or released custody")
	}
	return h, nil
}
func (s *Store) ReserveNativeArtifact(consumer, parent, kind, slot string, artifact ModelArtifact) (NativeArtifactRetention, *exit.Error) {
	tx, err := s.db.Begin()
	if err != nil {
		return NativeArtifactRetention{}, exit.Internalf("cannot begin native retention: %s", err)
	}
	defer tx.Rollback()
	h, problem := reserveNativeArtifactTx(tx, consumer, parent, kind, slot, artifact)
	if problem != nil {
		return h, problem
	}
	if err = tx.Commit(); err != nil {
		return h, exit.Internalf("cannot commit native retention: %s", err)
	}
	return h, nil
}
func (s *Store) NativeArtifactRetentions(consumer string) ([]NativeArtifactRetention, *exit.Error) {
	rows, err := s.db.Query(`SELECT `+nativeArtifactCols+` FROM native_artifact_retentions WHERE consumer_id=? ORDER BY kind,slot`, consumer)
	if err != nil {
		return nil, exit.Internalf("cannot list native retentions: %s", err)
	}
	defer rows.Close()
	var out []NativeArtifactRetention
	for rows.Next() {
		h, err := scanNativeArtifact(rows)
		if err != nil {
			return nil, exit.Internalf("cannot decode native retention: %s", err)
		}
		out = append(out, h)
	}
	return out, nil
}
func (s *Store) ConfirmNativeArtifact(id, instance, boot string) *exit.Error {
	result, err := s.db.Exec(`UPDATE native_artifact_retentions SET state='held',instance_id=?,worker_boot_id=? WHERE retention_id=? AND state IN ('pending','held')`, instance, boot, id)
	if err != nil {
		return exit.Internalf("cannot acknowledge native retention: %s", err)
	}
	if n, _ := result.RowsAffected(); n == 0 {
		return exit.Named(exit.Conflict, "child.artifact_release_pending", "native retention was canceled before acknowledgement")
	}
	return nil
}
func (s *Store) BeginNativeArtifactRelease(consumer string, inputsOnly bool) *exit.Error {
	filter := ""
	if inputsOnly {
		filter = " AND kind='input'"
	}
	_, err := s.db.Exec(`UPDATE native_artifact_retentions SET state='releasing' WHERE consumer_id=? AND state IN ('pending','held')`+filter, consumer)
	if err != nil {
		return exit.Internalf("cannot begin native artifact release: %s", err)
	}
	return nil
}
func (s *Store) CompleteNativeArtifactRelease(id string) *exit.Error {
	_, err := s.db.Exec(`UPDATE native_artifact_retentions SET state='released' WHERE retention_id=? AND state='releasing'`, id)
	if err != nil {
		return exit.Internalf("cannot finish native artifact release: %s", err)
	}
	return nil
}

// NativeArtifactInput reserves the native branch atomically with package-child admission.
func nativeArtifactInput(tx *sql.Tx, requestID, slot string, raw json.RawMessage) (bool, *exit.Error) {
	artifact, problem := DecodeModelArtifact(raw)
	if problem != nil {
		return false, problem
	}
	if artifact == nil {
		return false, nil
	}
	native, problem := nativeArtifactSource(tx, *artifact)
	if problem != nil {
		return false, problem
	}
	if native == nil {
		return false, nil
	}
	_, problem = reserveNativeArtifactTx(tx, requestID, requestID, "input", "result/"+slot, *artifact)
	return true, problem
}
