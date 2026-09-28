package records

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// ArtifactSource resolves genuine native storage provenance without inventing a
// package request or an outcome for a built-in service result.
type ArtifactSource struct {
	Artifact        ModelArtifact
	OwnerRequestID  string
	Worker          string
	NativeServiceID string
	TransactionID   string
	NativeReceipt   []byte
	Weights         *ModelTransferWeights
}

type artifactQuery interface{ QueryRow(string, ...any) *sql.Row }

func nativeArtifactSource(q artifactQuery, artifact ModelArtifact) (*ArtifactSource, *exit.Error) {
	call, err := scanNativeCall(q.QueryRow(`SELECT `+nativeCallColumns+` FROM native_calls WHERE id=? AND kind='source' AND state='succeeded'`, artifact.ProducerRequestID))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, exit.Internalf("cannot resolve native artifact: %s", err)
	}
	observed, problem := DecodeModelArtifact(call.Result)
	if problem != nil || observed == nil || *observed != artifact {
		return nil, exit.Named(exit.Conflict, "child.artifact_unowned", "native result does not match the exact artifact handle")
	}
	receipt, err := canonical.NormalizeJCS(call.NativeReceipt)
	if err != nil || !bytes.Equal(receipt, call.NativeReceipt) {
		return nil, exit.Named(exit.Conflict, "child.artifact_provenance_lost", "native service receipt is not canonical")
	}
	digest, _ := canonical.Spell(canonical.Digest(receipt))
	var identity struct {
		TransactionID string `json:"transaction_id"`
		Manifest      struct {
			SHA256 string `json:"sha256"`
			Length int64  `json:"length"`
		} `json:"manifest"`
	}
	if json.Unmarshal(receipt, &identity) != nil || digest != artifact.TensorFSReceiptDigest || "sha256:"+identity.Manifest.SHA256 != artifact.Manifest.Digest || identity.Manifest.Length != artifact.Manifest.Length {
		return nil, exit.Named(exit.Conflict, "child.artifact_provenance_lost", "native receipt identity differs from the original artifact")
	}
	if _, err := canonical.Raw(identity.TransactionID); err != nil {
		return nil, exit.Named(exit.Conflict, "child.artifact_provenance_lost", "native receipt has no exact transaction")
	}
	return &ArtifactSource{Artifact: artifact, OwnerRequestID: call.ParentRequestID, Worker: call.Worker, NativeServiceID: call.ID, TransactionID: identity.TransactionID, NativeReceipt: receipt}, nil
}

func (s *Store) ResolveArtifactSource(artifact ModelArtifact) (*ArtifactSource, *exit.Error) {
	native, problem := nativeArtifactSource(s.db, artifact)
	if problem != nil || native != nil {
		return native, problem
	}
	weights, problem := s.ArtifactOutput(artifact)
	if problem != nil {
		return nil, problem
	}
	owner, problem := s.RequestRow(weights.RequestID)
	if problem != nil || owner == nil {
		return nil, exit.Named(exit.Conflict, "child.artifact_provenance_lost", "artifact owner request is absent")
	}
	return &ArtifactSource{Artifact: artifact, OwnerRequestID: owner.ID, Worker: owner.Worker, TransactionID: weights.TransactionID, Weights: weights}, nil
}

func nativeArtifactAllowed(q artifactQuery, parent Request, artifact ModelArtifact) (bool, *exit.Error) {
	raw, err := json.Marshal(artifact)
	if err != nil {
		return false, exit.Internalf("cannot encode native artifact: %s", err)
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return false, exit.Internalf("cannot normalize native artifact: %s", err)
	}
	var allowed bool
	err = q.QueryRow(`SELECT EXISTS(SELECT 1 FROM native_calls n JOIN requests r ON r.id=n.parent_request_id
        WHERE n.kind='source' AND n.state='succeeded' AND n.result=? AND n.worker=? AND r.worker=n.worker AND r.reuse_scope=?
        AND r.state NOT IN ('canceling','canceled','releasing'))
        OR EXISTS(SELECT 1 FROM native_artifact_retentions h JOIN requests r ON r.id=h.parent_request_id
        WHERE h.producer_id=? AND h.manifest_id=? AND h.receipt_digest=? AND h.state='held'
        AND h.owner_worker=? AND r.worker=h.owner_worker AND r.reuse_scope=? AND r.state NOT IN ('canceling','canceled','releasing'))`, raw, parent.Worker, parent.ReuseScope, artifact.ProducerRequestID, artifact.Manifest.Digest, artifact.TensorFSReceiptDigest, parent.Worker, parent.ReuseScope).Scan(&allowed)
	if err != nil {
		return false, exit.Internalf("cannot authorize native artifact scope: %s", err)
	}
	return allowed, nil
}

func (s *Store) ArtifactSourceAllowed(parent Request, source ArtifactSource) (bool, *exit.Error) {
	if source.NativeServiceID != "" {
		return nativeArtifactAllowed(s.db, parent, source.Artifact)
	}
	if source.Worker != parent.Worker {
		return false, nil
	}
	return s.ArtifactHasCustody(source.Weights.RequestID, source.Weights.Attempt, source.Weights.OutputSlot, parent.ReuseScope)
}

func packageArtifactSource(q artifactQuery, artifact ModelArtifact) (*ArtifactSource, *exit.Error) {
	var weights ModelTransferWeights
	var worker string
	err := q.QueryRow(`SELECT o.request_id,o.attempt,o.output_slot,o.transaction_id,o.manifest_id,o.manifest_length,r.worker
        FROM request_model_transfer_outputs o JOIN requests r ON r.id=o.request_id
        WHERE o.request_id=? AND o.output_slot=? AND o.manifest_id=? AND o.manifest_length=?
        AND CASE WHEN json_valid(o.receipt) THEN json_extract(CAST(o.receipt AS TEXT),'$.tensorfs_receipt_digest') ELSE '' END=? ORDER BY o.attempt DESC LIMIT 1`, artifact.ProducerRequestID, artifact.OutputSlot, artifact.Manifest.Digest, artifact.Manifest.Length, artifact.TensorFSReceiptDigest).Scan(&weights.RequestID, &weights.Attempt, &weights.OutputSlot, &weights.TransactionID, &weights.ManifestID, &weights.ManifestLength, &worker)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, exit.Named(exit.Conflict, "child.artifact_unowned", "effect artifact has no observed package/native producer")
	}
	if err != nil {
		return nil, exit.Internalf("cannot inspect effect artifact provenance: %s", err)
	}
	return &ArtifactSource{Artifact: artifact, OwnerRequestID: weights.RequestID, Worker: worker, TransactionID: weights.TransactionID, Weights: &weights}, nil
}
func packageArtifactAllowed(q artifactQuery, parent Request, source ArtifactSource) (bool, *exit.Error) {
	if source.Worker != parent.Worker {
		return false, nil
	}
	var allowed bool
	err := q.QueryRow(`SELECT EXISTS(SELECT 1 FROM requests p WHERE p.id=? AND p.reuse_scope=? AND p.retain_work=1 AND p.state NOT IN ('canceling','canceled','releasing'))
        OR EXISTS(SELECT 1 FROM request_weights_retentions h JOIN requests r ON r.id=h.request_id WHERE h.producer_request_id=? AND h.producer_attempt=? AND h.producer_output_slot=? AND h.state='held' AND r.reuse_scope=? AND r.worker=? AND r.state NOT IN ('canceling','canceled','releasing'))`, source.Weights.RequestID, parent.ReuseScope, source.Weights.RequestID, source.Weights.Attempt, source.Weights.OutputSlot, parent.ReuseScope, parent.Worker).Scan(&allowed)
	if err != nil {
		return false, exit.Internalf("cannot authorize effect artifact custody: %s", err)
	}
	return allowed, nil
}
