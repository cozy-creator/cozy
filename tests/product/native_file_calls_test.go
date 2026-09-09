package producttest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestCommittedFileKeepsOriginalProducerAndExactOwnedBytes(t *testing.T) {
	for _, fault := range []string{"", "foreign_handle", "changed_size", "changed_mime", "changed_manifest", "wrong_producer", "wrong_attempt", "wrong_kind", "extra_result", "missing_ref"} {
		t.Run(fault, func(t *testing.T) {
			encode := func(value any) []byte {
				raw, err := json.Marshal(value)
				if err != nil {
					t.Fatal(err)
				}
				raw, err = canonical.NormalizeJCS(raw)
				if err != nil {
					t.Fatal(err)
				}
				return raw
			}
			digest := "sha256:" + strings.Repeat("ab", 32)
			spec := []byte(strings.Repeat("s", 32))
			request := map[string]any{"slot": "file/0001", "digest": digest, "size_bytes": 80000, "media_type": "application/json"}
			result := map[string]any{"file": map[string]any{"asset_ref": digest, "kind": "file", "digest": digest, "size_bytes": 80000, "media_type": "application/json"}}
			manifest := encode(map[string]any{"entries": []any{map[string]any{"path": "payload", "kind": "file", "blob": map[string]any{"sha256": strings.TrimPrefix(digest, "sha256:"), "length": 80000}}}})
			row := records.NativeCall{ParentRequestID: "parent", CallIndex: 80, Operation: "commit_file"}
			status := &pb.NativeSourceStatus{ParentAttemptOrdinal: 2, ByteOutputAttemptOrdinal: 1, ByteOutputInvocationSpecDigest: spec, ByteOutput: &pb.NativeByteTreeRef{
				ProducerRootId: records.NativeByteProducerRoot("cozy-local-client", "parent", 1, spec, "runtime.commit_file.80"), ReceiptDigest: []byte(strings.Repeat("r", 32)),
				Manifest: &pb.Ref{Digest: canonical.Digest(manifest), Length: uint64(len(manifest))}, ContentBytes: 80000,
			}}
			switch fault {
			case "foreign_handle":
				request["slot"] = "attempt:other#1/file/0001"
			case "changed_size":
				request["size_bytes"] = 80001
			case "changed_mime":
				request["media_type"] = "text/plain"
			case "changed_manifest":
				status.ByteOutput.Manifest.Digest[0] ^= 1
			case "wrong_producer":
				status.ByteOutput.ProducerRootId = digest
			case "wrong_attempt":
				status.ByteOutputAttemptOrdinal = 3
			case "wrong_kind":
				result["file"].(map[string]any)["kind"] = "tree"
			case "extra_result":
				result["private"] = "unexpected"
			case "missing_ref":
				status.ByteOutput = nil
			}
			row.Request, status.ResultCanonicalBytes = encode(request), encode(result)
			output, problem := records.CommittedFileOutput("cozy-local-client", row, status)
			if fault != "" {
				if problem == nil {
					t.Fatal("accepted mismatched owned file")
				}
				return
			}
			if problem != nil {
				t.Fatal(problem)
			}
			if output.Attempt != 1 || output.ContentBytes != 80000 || output.Length != 80000 || output.MimeType != "application/json" || output.OutputID != "runtime.commit_file.80" {
				t.Fatalf("changed native file facts: %+v", output)
			}
		})
	}
}
