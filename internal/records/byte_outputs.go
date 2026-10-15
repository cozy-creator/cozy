package records

import (
	"database/sql"
	"math"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/custody"
	"github.com/cozy-creator/cozy/internal/exit"
)

const byteOutputsDDL = `CREATE TABLE IF NOT EXISTS byte_outputs (
 request_id TEXT NOT NULL, attempt INTEGER NOT NULL, output_id TEXT NOT NULL,
 digest TEXT NOT NULL,length INTEGER NOT NULL,mime_type TEXT NOT NULL,
 producer_root_id TEXT NOT NULL,receipt_digest TEXT NOT NULL,manifest_id TEXT NOT NULL,
 manifest_length INTEGER NOT NULL,content_bytes INTEGER NOT NULL,
 native_service_id TEXT REFERENCES native_calls(id),
 PRIMARY KEY(request_id,attempt,output_id),
 FOREIGN KEY(request_id,attempt) REFERENCES attempts(request_id,attempt)
)`

const nativeByteOutputIndex = `CREATE UNIQUE INDEX IF NOT EXISTS byte_outputs_native_service ON byte_outputs(native_service_id) WHERE native_service_id IS NOT NULL`

// ByteOutput is a terminal-owned native byte tree, never a local pathname or weights receipt.
type ByteOutput struct {
	RequestID                                           string
	Attempt                                             int64
	OutputID, Digest                                    string
	Length                                              int64
	MimeType, ProducerRootID, ReceiptDigest, ManifestID string
	ManifestLength, ContentBytes                        int64
}

const byteOutputCols = `request_id,attempt,output_id,digest,length,mime_type,producer_root_id,receipt_digest,manifest_id,manifest_length,content_bytes`

func (b ByteOutput) NativeRef() *custody.TreeRef {
	receipt, _ := canonical.Raw(b.ReceiptDigest)
	manifest, _ := canonical.Raw(b.ManifestID)
	return &custody.TreeRef{ProducerRootID: b.ProducerRootID, ReceiptDigest: receipt, Manifest: &custody.ObjectRef{Digest: manifest, Length: uint64(b.ManifestLength)}, ContentBytes: uint64(b.ContentBytes)}
}
func ValidateByteRef(ref *custody.TreeRef) *exit.Error {
	if ref == nil || ref.Manifest == nil || len(ref.ReceiptDigest) != 32 || len(ref.Manifest.Digest) != 32 || ref.Manifest.Length == 0 || ref.Manifest.Length > uint64(custody.MaxMetadataBytes) || ref.ContentBytes > math.MaxInt64 {
		return exit.New(exit.Validation, "native byte output needs bounded exact receipt, manifest and content size")
	}
	if _, err := canonical.Raw(ref.ProducerRootID); err != nil {
		return exit.New(exit.Validation, "native byte producer root is malformed")
	}
	return nil
}
func recordByteOutputsTx(tx *sql.Tx, t Terminal) *exit.Error {
	if len(t.ByteOutputs) > custody.MaxChildArtifacts {
		return exit.New(exit.Validation, "too many native byte outputs")
	}
	for _, b := range t.ByteOutputs {
		if b.RequestID != t.RequestID || b.Attempt != t.Attempt || b.OutputID == "" || b.Length < 0 {
			return exit.New(exit.Validation, "byte output differs from terminal authority")
		}
		if problem := ValidateByteRef(b.NativeRef()); problem != nil {
			return problem
		}
		if _, err := canonical.Raw(b.Digest); err != nil {
			return exit.New(exit.Validation, "byte output digest is malformed")
		}
		_, err := tx.Exec(`INSERT INTO byte_outputs(`+byteOutputCols+`) VALUES(?,?,?,?,?,?,?,?,?,?,?)`, b.RequestID, b.Attempt, b.OutputID, b.Digest, b.Length, b.MimeType, b.ProducerRootID, b.ReceiptDigest, b.ManifestID, b.ManifestLength, b.ContentBytes)
		if err != nil {
			return exit.Internalf("cannot commit native byte output: %s", err)
		}
	}
	return nil
}

// ByteAssetBinding is recorded outside author payloads, after parent-grant verification.
type ByteAssetBinding struct {
	Output      ByteOutput `json:"output"`
	RetentionID string     `json:"retention_id"`
}
