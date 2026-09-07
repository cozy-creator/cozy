package records

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"io"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

type ModelArtifact struct {
	ProducerRequestID     string            `json:"producer_request_id"`
	OutputSlot            string            `json:"output_slot"`
	Manifest              ArtifactObjectRef `json:"manifest"`
	TensorFSReceiptDigest string            `json:"tensorfs_receipt_digest"`
}

type ArtifactObjectRef struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
}

func DecodeModelArtifact(raw []byte) (*ModelArtifact, *exit.Error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	var artifact ModelArtifact
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&artifact) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		return nil, exit.New(exit.Validation, "model input is not a closed ModelArtifact reference")
	}
	if artifact.ProducerRequestID == "" || len(artifact.ProducerRequestID) > 256 || artifact.OutputSlot == "" || len(artifact.OutputSlot) > 128 || artifact.Manifest.Length <= 0 || artifact.Manifest.Length > (1<<53)-1 {
		return nil, exit.New(exit.Validation, "model artifact reference has invalid identity or bounds")
	}
	for _, digest := range []string{artifact.Manifest.Digest, artifact.TensorFSReceiptDigest} {
		raw, err := canonical.Raw(digest)
		if err != nil {
			return nil, exit.New(exit.Validation, "model artifact reference has malformed digests")
		}
		spelled, _ := canonical.Spell(raw)
		if spelled != digest {
			return nil, exit.New(exit.Validation, "model artifact digests must be canonical")
		}
	}
	return &artifact, nil
}

// ArtifactOutput resolves provenance through the original immutable native
// receipt observation. A JSON lookalike or digest alone grants no model bytes.
func (s *Store) ArtifactOutput(artifact ModelArtifact) (*ModelTransferWeights, *exit.Error) {
	var attempt int64
	err := s.db.QueryRow(`SELECT attempt FROM request_model_transfer_outputs WHERE request_id=? AND output_slot=? AND manifest_id=? AND manifest_length=?
		AND CASE WHEN json_valid(receipt) THEN json_extract(CAST(receipt AS TEXT),'$.tensorfs_receipt_digest') ELSE '' END=?
		ORDER BY attempt DESC LIMIT 1`, artifact.ProducerRequestID, artifact.OutputSlot, artifact.Manifest.Digest, artifact.Manifest.Length, artifact.TensorFSReceiptDigest).Scan(&attempt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, exit.Named(exit.Conflict, "child.artifact_unowned", "model artifact does not name an observed native result")
	}
	if err != nil {
		return nil, exit.Internalf("cannot resolve model artifact provenance: %s", err)
	}
	return s.ModelTransferWeights(artifact.ProducerRequestID, attempt, artifact.OutputSlot)
}
