package records

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The worker has verified the private spool and native receipt. Creator matches
// the resulting immutable byte identity to the already accepted own-file intent.
func CommittedFileOutput(owner string, row NativeCall, status *pb.NativeSourceStatus) (ByteOutput, *exit.Error) {
	fail := func() (ByteOutput, *exit.Error) {
		return ByteOutput{}, exit.New(exit.Validation, "committed file differs from its bounded owned-file intent")
	}
	ref := status.ByteOutput
	if problem := ValidateByteRef(ref); problem != nil {
		return ByteOutput{}, problem
	}
	if row.Operation != "commit_file" || status.ByteOutputAttemptOrdinal == 0 || status.ByteOutputAttemptOrdinal > status.ParentAttemptOrdinal || len(status.ByteOutputInvocationSpecDigest) != 32 || ref.ContentBytes > 256<<20 {
		return fail()
	}
	var request struct {
		Slot      string `json:"slot"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size_bytes"`
		MediaType string `json:"media_type"`
	}
	decoder := json.NewDecoder(bytes.NewReader(row.Request))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil || request.Size < 0 || uint64(request.Size) != ref.ContentBytes || len(request.MediaType) == 0 || len(request.MediaType) > 255 {
		return fail()
	}
	if _, err := canonical.Raw(request.Digest); err != nil {
		return fail()
	}
	for _, letter := range request.MediaType {
		if letter < 32 || letter > 126 {
			return fail()
		}
	}
	tail, ok := strings.CutPrefix(request.Slot, "file/")
	index, err := strconv.ParseUint(tail, 10, 64)
	if !ok || err != nil || index == 0 || len(tail) > 16 || fmt.Sprintf("%04d", index) != tail {
		return fail()
	}
	manifest, _ := json.Marshal(map[string]any{"entries": []any{map[string]any{"path": "payload", "kind": "file", "blob": map[string]any{"sha256": strings.TrimPrefix(request.Digest, "sha256:"), "length": request.Size}}}})
	manifest, _ = canonical.NormalizeJCS(manifest)
	if !bytes.Equal(canonical.Digest(manifest), ref.Manifest.Digest) || uint64(len(manifest)) != ref.Manifest.Length {
		return fail()
	}
	value, _ := json.Marshal(map[string]any{"file": map[string]any{"asset_ref": request.Digest, "kind": "file", "digest": request.Digest, "size_bytes": request.Size, "media_type": request.MediaType}})
	value, _ = canonical.NormalizeJCS(value)
	if !bytes.Equal(value, status.ResultCanonicalBytes) {
		return fail()
	}
	manifestID, _ := canonical.Spell(ref.Manifest.Digest)
	receiptID, _ := canonical.Spell(ref.ReceiptDigest)
	b := ByteOutput{RequestID: row.ParentRequestID, Attempt: int64(status.ByteOutputAttemptOrdinal), OutputID: fmt.Sprintf("runtime.commit_file.%d", row.CallIndex), Digest: request.Digest, Length: request.Size, MimeType: request.MediaType, ProducerRootID: ref.ProducerRootId, ReceiptDigest: receiptID, ManifestID: manifestID, ManifestLength: int64(ref.Manifest.Length), ContentBytes: request.Size}
	if b.ProducerRootID != NativeByteProducerRoot(owner, b.RequestID, b.Attempt, status.ByteOutputInvocationSpecDigest, b.OutputID) {
		return fail()
	}
	return b, nil
}
