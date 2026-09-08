package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/nativeinterface"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func nativeServiceID(parent string, index uint32) string {
	raw, _ := json.Marshal([]any{recordOwnerID, parent, index})
	raw, _ = canonical.NormalizeJCS(raw)
	digest := sha256.Sum256(raw)
	return "source-" + hex.EncodeToString(digest[:])[:48]
}
func nativeSourceOperation(name string) pb.NativeSourceOperation {
	switch name {
	case "download_huggingface":
		return pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_HUGGINGFACE
	case "download_civitai":
		return pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_CIVITAI
	case "convert_cozytensors":
		return pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_CONVERT
	}
	return pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_UNSPECIFIED
}

// onNativeSourceCall runs only after common parent/intent validation. It performs no IO-heavy work.
func (c *Orchestrator) onNativeSourceCall(s *session, parent *records.Request, call *pb.ChildCallRequest) bool {
	if call.Module != nativeinterface.SourceModule {
		return false
	}
	refuse := func(problem *exit.Error) {
		c.sendChildResult(s, call, "", pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, problem)
	}
	if !bytes.Equal(call.InterfaceDigest, nativeinterface.SourceDigest()) || nativeSourceOperation(call.Export) == 0 {
		refuse(exit.New(exit.Validation, "source call is not a fixed native interface"))
		return true
	}
	if eligibility, ok := c.opt.Packages.(interface {
		NativeSourceEligible(records.Request, string) *exit.Error
	}); ok {
		if problem := eligibility.NativeSourceEligible(*parent, call.Export); problem != nil {
			refuse(problem)
			return true
		}
	} else {
		refuse(exit.Unavailablef("source owner cannot validate the captured caller descriptor"))
		return true
	}
	spec, _ := canonical.Spell(call.ParentInvocationSpecDigest)
	intent, _ := canonical.Spell(call.IntentDigest)
	row, _, problem := c.opt.Store.AcceptNativeCall(records.NativeCall{ID: nativeServiceID(parent.ID, call.CallIndex), ParentRequestID: parent.ID, CallIndex: int64(call.CallIndex), Kind: "source", Operation: call.Export, IntentDigest: intent, Request: call.RequestCanonicalBytes}, int64(call.ParentAttemptOrdinal), spec, s.bootID)
	if problem != nil {
		refuse(problem)
		return true
	}
	if row.State == "succeeded" {
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_SUCCEEDED, row.Result, nil)
		return true
	}
	if row.State == "canceled" || row.State == "failed" || row.State == "stopped" {
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_FAILED, nil, exit.Named(exit.Failed, "native.source_stopped", "native source call stopped"))
		return true
	}
	c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_PENDING, nil, nil)
	phase := pb.NativeSourcePhase_NATIVE_SOURCE_PHASE_RESOLVE
	if row.Operation == "convert_cozytensors" {
		phase = pb.NativeSourcePhase_NATIVE_SOURCE_PHASE_EXECUTE
	}
	c.sendNativeSource(s, call, row.ID, phase, nil)
	return true
}
func (c *Orchestrator) sendNativeSource(s *session, call *pb.ChildCallRequest, id string, phase pb.NativeSourcePhase, selection *pb.NativeSourceSelection) {
	command := &pb.NativeSourceCommand{RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID, ParentCall: proto.Clone(call).(*pb.ChildCallRequest), ServiceId: id, Operation: nativeSourceOperation(call.Export), Phase: phase, Selection: selection}
	if credentials, ok := c.opt.Packages.(interface{ NativeSourceCredential(string) string }); ok {
		command.Credential = credentials.NativeSourceCredential(call.Export)
	}
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_NativeSourceCommand{NativeSourceCommand: command}})
}
func sourcePin(selection *pb.NativeSourceSelection) ([]byte, error) {
	if selection == nil || len(selection.Members) == 0 || len(selection.Members) > 4096 || selection.ContentManifest == nil || len(selection.ContentManifest.Digest) != 32 || len(selection.SelectionDigest) != 32 {
		return nil, fmt.Errorf("source selection missing bounded exact identity")
	}
	members := make([]map[string]any, 0, len(selection.Members))
	prior := ""
	for _, member := range selection.Members {
		if member.Member <= prior || member.Object == nil || len(member.Object.Digest) != 32 || member.Object.Length == 0 {
			return nil, fmt.Errorf("source members lack exact ordered identities")
		}
		prior = member.Member
		digest, _ := canonical.Spell(member.Object.Digest)
		members = append(members, map[string]any{"member": member.Member, "digest": digest, "length": member.Object.Length})
	}
	content, _ := canonical.Spell(selection.ContentManifest.Digest)
	provenance, _ := canonical.Spell(selection.SelectionDigest)
	raw, _ := json.Marshal(map[string]any{"canonical": selection.Canonical, "selection_digest": provenance, "content_manifest": map[string]any{"digest": content, "length": selection.ContentManifest.Length}, "members": members})
	return canonical.NormalizeJCS(raw)
}
func (c *Orchestrator) onNativeSourceStatus(s *session, status *pb.NativeSourceStatus) {
	if status == nil || c.fenced(s, status.RecordOwnerEpoch, status.ControlStreamEpoch, status.WorkerBootId) {
		return
	}
	parent, problem := c.childParent(s, status.ParentRequestId, status.ParentAttemptOrdinal, status.ParentInvocationSpecDigest)
	if problem != nil {
		return
	}
	row, problem := c.opt.Store.NativeCall(parent.ID, int64(status.CallIndex))
	if problem != nil || row == nil || row.Kind != "source" || row.ID != status.ServiceId {
		return
	}
	intent, _ := canonical.Spell(status.IntentDigest)
	if row.IntentDigest != intent {
		return
	}
	call := &pb.ChildCallRequest{ParentRequestId: parent.ID, ParentAttemptOrdinal: status.ParentAttemptOrdinal, ParentInvocationSpecDigest: status.ParentInvocationSpecDigest, CallIndex: status.CallIndex, InterfaceDigest: nativeinterface.SourceDigest(), Module: nativeinterface.SourceModule, Export: row.Operation, RequestCanonicalBytes: row.Request, IntentDigest: status.IntentDigest}
	switch status.State {
	case pb.NativeSourceState_NATIVE_SOURCE_STATE_RESOLVED:
		pinned, err := sourcePin(status.Selection)
		if err != nil {
			return
		}
		if problem = c.opt.Store.FreezeNativeCall(row.ID, pinned); problem != nil {
			c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, problem)
			return
		}
		if problem = c.opt.Store.StartNativeCall(row.ID); problem != nil {
			return
		}
		c.sendNativeSource(s, call, row.ID, pb.NativeSourcePhase_NATIVE_SOURCE_PHASE_EXECUTE, status.Selection)
	case pb.NativeSourceState_NATIVE_SOURCE_STATE_SUCCEEDED:
		if len(status.ResultCanonicalBytes) > childCallMaxBytes || len(status.NativeReceiptCanonicalBytes) > 1<<20 || len(status.ComputationDigest) != 32 {
			return
		}
		if problem = c.opt.Store.CompleteNativeCallAt(row.ID, status.ResultCanonicalBytes, status.NativeReceiptCanonicalBytes, s.instanceID, s.bootID); problem != nil {
			c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, problem)
			return
		}
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_SUCCEEDED, status.ResultCanonicalBytes, nil)
	case pb.NativeSourceState_NATIVE_SOURCE_STATE_FAILED:
		_ = c.opt.Store.StopNativeCall(row.ID, "failed", "native_source_failed")
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_FAILED, nil, exit.Named(exit.Failed, "native.source_failed", "native source operation failed; retained bytes remain available"))
	case pb.NativeSourceState_NATIVE_SOURCE_STATE_CANCELED:
		_ = c.opt.Store.StopNativeCall(row.ID, "stopped", "native_source_stopped")
		c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_CANCELED, nil, nil)
	}
}

func (c *Orchestrator) cancelNativeSource(s *session, call *pb.ChildCallCancel) bool {
	row, problem := c.opt.Store.NativeCall(call.ParentRequestId, int64(call.CallIndex))
	if problem != nil || row == nil || row.Kind != "source" {
		return false
	}
	intent, err := canonical.Spell(call.IntentDigest)
	if err != nil || intent != row.IntentDigest {
		return true
	}
	if row.State == "succeeded" {
		return true
	}
	_ = c.opt.Store.StopNativeCall(row.ID, "stopped", "native_source_stopped")
	parent := &pb.ChildCallRequest{ParentRequestId: call.ParentRequestId, ParentAttemptOrdinal: call.ParentAttemptOrdinal, ParentInvocationSpecDigest: call.ParentInvocationSpecDigest,
		CallIndex: call.CallIndex, InterfaceDigest: nativeinterface.SourceDigest(), Module: nativeinterface.SourceModule, Export: row.Operation, RequestCanonicalBytes: row.Request, IntentDigest: call.IntentDigest}
	c.sendNativeSource(s, parent, row.ID, pb.NativeSourcePhase_NATIVE_SOURCE_PHASE_CANCEL, nil)
	return true
}
