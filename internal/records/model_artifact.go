package records

import (
	"bytes"
	"encoding/json"

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
	// Runtime and Creator ship independently; members another version adds are ignored.
	var artifact ModelArtifact
	if json.Unmarshal(raw, &artifact) != nil {
		return nil, exit.New(exit.Validation, "model input is not a ModelArtifact reference")
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
