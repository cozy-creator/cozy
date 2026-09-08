package orchestrator

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func nativeSourceByteOutput(row records.NativeCall, status *pb.NativeSourceStatus) (records.ByteOutput, *exit.Error) {
	if row.Operation == "commit_file" {
		return nativeCommittedFileOutput(row, status)
	}
	fail := func(message string) (records.ByteOutput, *exit.Error) {
		return records.ByteOutput{}, exit.New(exit.Validation, "%s", message)
	}
	ref := status.ByteOutput
	if problem := records.ValidateByteRef(ref); problem != nil {
		return records.ByteOutput{}, problem
	}
	if status.ByteOutputAttemptOrdinal == 0 || status.ByteOutputAttemptOrdinal > status.ParentAttemptOrdinal || len(status.ByteOutputInvocationSpecDigest) != 32 || ref.ContentBytes > 64<<20 {
		return fail("source view has invalid original producer or exceeds 64 MiB")
	}
	manifest, _ := canonical.Spell(ref.Manifest.Digest)
	receipt, _ := canonical.Spell(ref.ReceiptDigest)
	b := records.ByteOutput{RequestID: row.ParentRequestID, Attempt: int64(status.ByteOutputAttemptOrdinal), OutputID: fmt.Sprintf("runtime.%s.%d", row.Operation, row.CallIndex), Digest: manifest, Length: int64(ref.Manifest.Length), MimeType: "application/vnd.cozy.tree-manifest", ProducerRootID: ref.ProducerRootId, ReceiptDigest: receipt, ManifestID: manifest, ManifestLength: int64(ref.Manifest.Length), ContentBytes: int64(ref.ContentBytes)}
	if row.Operation != "source_files" {
		return fail("native byte result requires an explicit operation verifier")
	}
	var request struct {
		Source struct {
			Manifest records.ArtifactObjectRef `json:"manifest"`
		} `json:"source"`
	}
	if err := json.Unmarshal(row.Request, &request); err != nil || request.Source.Manifest.Digest != manifest || request.Source.Manifest.Length != b.ManifestLength {
		return fail("source view changed the accepted source manifest")
	}
	if b.ProducerRootID != records.NativeByteProducerRoot(recordOwnerID, b.RequestID, b.Attempt, status.ByteOutputInvocationSpecDigest, b.OutputID) {
		return fail("source view changed its original byte producer identity")
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(status.ResultCanonicalBytes))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return fail("source view result is invalid")
	}
	root, ok := value.(map[string]any)
	if !ok || len(root) != 1 {
		return fail("source view result has an unexpected shape")
	}
	binding := b
	binding.OutputID = "files"
	if problem := verifyByteResultRow(value, binding); problem != nil {
		return records.ByteOutput{}, problem
	}
	return b, nil
}

func (c *Orchestrator) sendNativeByteResult(s *session, call *pb.ChildCallRequest, row records.NativeCall, b records.ByteOutput, producerSpec string, result, receipt []byte) {
	currentSpec, _ := canonical.Spell(call.ParentInvocationSpecDigest)
	h, problem := c.opt.Store.CompleteNativeByteCall(row.ID, int64(call.ParentAttemptOrdinal), currentSpec, s.bootID, producerSpec, b, result, receipt, s.instanceID)
	if problem == nil && h.State != "held" {
		problem = c.changeByteRetention(s.ctx, h, false)
	}
	if problem != nil {
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, problem)
		return
	}
	if _, problem := c.childParent(s, call.ParentRequestId, call.ParentAttemptOrdinal, call.ParentInvocationSpecDigest); problem != nil {
		return
	}
	field := "files"
	if row.Operation == "commit_file" {
		field = "file"
	}
	c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_SUCCEEDED, result, nil, &pb.ChildCallResult{ByteResultGrants: []*pb.ChildByteResultGrant{{OutputId: field, Source: b.NativeRef(), RetentionId: h.RetentionID}}})
}

func (c *Orchestrator) replayNativeByteResult(s *session, call *pb.ChildCallRequest, row records.NativeCall) {
	output, problem := c.opt.Store.NativeByteOutput(row.ID)
	if problem != nil || output == nil {
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, exit.Unavailablef("native byte result lacks its recorded producer"))
		return
	}
	attempt, problem := c.opt.Store.AttemptRow(output.RequestID, output.Attempt)
	if problem != nil || attempt == nil {
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, exit.Unavailablef("native byte result lacks its original attempt"))
		return
	}
	c.sendNativeByteResult(s, call, row, *output, attempt.InvocationDigest, row.Result, row.NativeReceipt)
}
