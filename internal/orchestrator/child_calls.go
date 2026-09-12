package orchestrator

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

const childCallMaxBytes = 48 * 1024

type unpublishedChildResolver interface {
	ResolveUnpublishedChild(records.Request, string, string, string, []byte) (Submission, string, *exit.Error)
}

func (c *Orchestrator) childParent(s *session, id string, ordinal uint64, digest []byte) (*records.Request, *exit.Error) {
	if ordinal == 0 || ordinal > math.MaxInt64 || len(digest) != 32 {
		return nil, exit.Named(exit.Validation, "child.parent_invalid", "child call has invalid parent identity")
	}
	parent, problem := c.opt.Store.RequestRow(id)
	if problem != nil || parent == nil {
		return nil, exit.Named(exit.Conflict, "child.parent_absent", "child call has no recorded parent")
	}
	attempt, problem := c.opt.Store.AttemptRow(id, int64(ordinal))
	spelled, _ := canonical.Spell(digest)
	if problem != nil || attempt == nil || attempt.InstanceID != s.instanceID || attempt.SessionID != s.bootID || attempt.InvocationDigest != spelled || !openAttempt(attempt.State) || parent.Ordinal != int64(ordinal) {
		return nil, exit.Named(exit.Conflict, "child.parent_fenced", "child call does not belong to this worker's current parent attempt")
	}
	if !parent.IsJob() || !parent.RetainWork || parent.InstallID == "" {
		return nil, exit.Named(exit.Conflict, "child.parent_not_private", "child calls require a captured unpublished package job")
	}
	return parent, nil
}

func (c *Orchestrator) onChildCall(s *session, call *pb.ChildCallRequest) {
	if call == nil || c.fenced(s, call.RecordOwnerEpoch, call.ControlStreamEpoch, call.WorkerBootId) {
		return
	}
	refuse := func(problem *exit.Error) {
		c.sendChildResult(s, call, "", pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, problem)
	}
	parent, problem := c.childParent(s, call.ParentRequestId, call.ParentAttemptOrdinal, call.ParentInvocationSpecDigest)
	if problem != nil {
		refuse(problem)
		return
	}
	if parent.State != "dispatching" {
		refuse(exit.Named(exit.Conflict, "child.parent_stopped", "stopped parent cannot start child work"))
		return
	}
	if len(call.InterfaceDigest) != 32 || len(call.IntentDigest) != 32 || len(call.RequestCanonicalBytes) > childCallMaxBytes || len(call.Module) == 0 || len(call.Export) == 0 || len(call.Module) > 256 || len(call.Export) > 128 {
		refuse(exit.New(exit.Validation, "child call exceeds its closed identity or payload bounds"))
		return
	}
	payload, err := canonical.NormalizeJCS(call.RequestCanonicalBytes)
	if err != nil || !bytes.Equal(payload, call.RequestCanonicalBytes) || len(payload) == 0 || payload[0] != '{' {
		refuse(exit.New(exit.Validation, "child input must be one canonical JSON object"))
		return
	}
	iface, _ := canonical.Spell(call.InterfaceDigest)
	intent := map[string]any{"interface_digest": iface, "module": call.Module, "export": call.Export, "request": json.RawMessage(payload)}
	capture, problem := captureOptions(call.Capture)
	if problem != nil {
		refuse(problem)
		return
	}
	if capture != "" {
		intent["capture"] = json.RawMessage(capture)
	}
	identity, _ := json.Marshal(intent)
	identity, err = canonical.NormalizeJCS(identity)
	if err != nil || !bytes.Equal(canonical.Digest(identity), call.IntentDigest) {
		refuse(exit.Named(exit.Conflict, "child.intent_changed", "child intent digest does not match its exact target and input"))
		return
	}
	if capture != "" && (call.Module == "cozy_runtime.author.sources" || call.Module == "cozy_runtime.author.checkpoints") {
		refuse(exit.New(exit.Validation, "capture requires an ordinary model execution"))
		return
	}
	if c.onNativeSourceCall(s, parent, call) {
		return
	}
	if c.onNativeEffect(s, *parent, call) {
		return
	}
	existing, problem := c.opt.Store.ChildAt(parent.ID, int64(call.CallIndex))
	if problem != nil {
		refuse(problem)
		return
	}
	intentDigest, _ := canonical.Spell(call.IntentDigest)
	if existing != nil {
		if existing.ChildIntentDigest != intentDigest {
			refuse(exit.Named(exit.Conflict, "child.intent_changed", "the parent call index already names different inputs"))
			return
		}
		// Same-parent replay continues its durable call, even when a transient
		// environment probe is unavailable. Cross-parent reuse is qualified below.
		c.sendChildResult(s, call, existing.ID, pb.ChildCallState_CHILD_CALL_STATE_PENDING, nil, nil)
		if existing.State == "paused" {
			go func() { _ = c.ResumeRequest(existing.ID, "parent resumed its child call") }()
		}
		c.watchChildCall(s, proto.Clone(call).(*pb.ChildCallRequest), existing.ID)
		return
	}
	// A frozen callable is permission to resolve it, not spare execution capacity.
	// Rented children currently run on their parent's worker. Refuse an actual
	// nested call before queueing if that parent occupies its ordinary job slot;
	// otherwise both requests can wait forever for that same slot. Native services
	// above have their own admission and do not take the ordinary job slot.
	if parent.Worker != "" {
		if _, problem := c.retainedOrchestrationParent(records.Request{ParentRequestID: parent.ID, Worker: parent.Worker}); problem != nil {
			refuse(problem)
			return
		}
	}
	resolver, ok := c.opt.Packages.(unpublishedChildResolver)
	if !ok {
		refuse(exit.Unavailablef("this package owner cannot resolve frozen child interfaces"))
		return
	}
	spec, target, problem := resolver.ResolveUnpublishedChild(*parent, iface, call.Module, call.Export, payload)
	if problem != nil {
		refuse(problem)
		return
	}
	spec.IdemKey = fmt.Sprintf("child/%s/%d", parent.ID, call.CallIndex)
	request, _, problem := requestRecord(spec)
	if problem != nil {
		refuse(problem)
		return
	}
	request.ParentRequestID, request.ParentCallIndex = parent.ID, int64(call.CallIndex)
	request.ChildIntentDigest, _ = canonical.Spell(call.IntentDigest)
	request.ChildTargetDigest = target
	request.ChildReusable = spec.ChildReusable
	request.ChildArtifacts = spec.ChildArtifacts
	request.Capture = capture
	if capture != "" {
		request.ChildArtifacts = true
		for _, id := range splitList(request.Outputs) {
			if id == "runtime.capture" {
				refuse(exit.New(exit.Validation, "author output collides with runtime.capture"))
				return
			}
		}
		if request.Outputs != "" {
			request.Outputs += ","
		}
		request.Outputs += "runtime.capture"
	}
	parentDigest, _ := canonical.Spell(call.ParentInvocationSpecDigest)
	child, fresh, problem := c.opt.Store.SubmitChild(request, int64(call.ParentAttemptOrdinal), parentDigest, s.bootID, call.RequestCanonicalBytes)
	if problem != nil {
		refuse(problem)
		return
	}
	c.sendChildResult(s, call, child.ID, pb.ChildCallState_CHILD_CALL_STATE_PENDING, nil, nil)
	if fresh && child.State == "submitted" {
		go func() { _, _ = c.activateRecorded(child) }()
	}
	if !fresh && child.State == "paused" {
		go func() { _ = c.ResumeRequest(child.ID, "parent resumed its child call") }()
	}
	c.watchChildCall(s, proto.Clone(call).(*pb.ChildCallRequest), child.ID)
}

func (c *Orchestrator) sendChildResult(s *session, call *pb.ChildCallRequest, child string, state pb.ChildCallState, result []byte, problem *exit.Error, extras ...*pb.ChildCallResult) {
	frame := &pb.ChildCallResult{RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: s.epoch, WorkerBootId: s.bootID,
		ParentRequestId: call.ParentRequestId, ParentAttemptOrdinal: call.ParentAttemptOrdinal, ParentInvocationSpecDigest: call.ParentInvocationSpecDigest,
		CallIndex: call.CallIndex, IntentDigest: call.IntentDigest, ChildRequestId: child, State: state, ResultCanonicalBytes: result}
	if len(extras) > 0 && extras[0] != nil {
		frame.ByteResultGrants = extras[0].ByteResultGrants
		frame.Observation = extras[0].Observation
	}
	if problem != nil {
		frame.ResultCanonicalBytes = nil
		frame.ByteResultGrants = nil
		frame.Observation = nil
		frame.SafeCode = problem.ErrName()
		frame.SafeDetail = problem.Message
		if len(frame.SafeDetail) > 1024 {
			frame.SafeDetail = strings.ToValidUTF8(frame.SafeDetail[:1024], "")
		}
	}
	if len(frame.ByteResultGrants) > pb.MaxChildArtifactGrants || proto.Size(frame) > pb.MaxInlineControlBytes {
		frame.State = pb.ChildCallState_CHILD_CALL_STATE_FAILED
		frame.ResultCanonicalBytes = nil
		frame.ByteResultGrants = nil
		frame.Observation = nil
		frame.SafeCode = "child.result_bounds"
		frame.SafeDetail = "child result exceeds bounded control metadata"
	}
	s.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ChildCallResult{ChildCallResult: frame}})
}

func (c *Orchestrator) watchChildCall(s *session, call *pb.ChildCallRequest, id string) {
	key := fmt.Sprintf("%s/%d/%d", call.ParentRequestId, call.ParentAttemptOrdinal, call.CallIndex)
	c.mu.Lock()
	if held := c.childWatches[key]; held == s {
		c.mu.Unlock()
		return
	}
	c.childWatches[key] = s
	c.mu.Unlock()
	go func() {
		defer func() {
			c.mu.Lock()
			if c.childWatches[key] == s {
				delete(c.childWatches, key)
			}
			c.mu.Unlock()
		}()
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			row, problem := c.opt.Store.RequestRow(id)
			if problem != nil || row == nil {
				c.sendChildResult(s, call, id, pb.ChildCallState_CHILD_CALL_STATE_FAILED, nil, exit.Internalf("recorded child request cannot be read"))
				return
			}
			if row.State == "succeeded" || (row.State == "finalizing" && row.ReusedFrom != "") {
				result, problem := c.childInlineResult(*row)
				if problem != nil && problem.Code == exit.Unavailable {
					select {
					case <-s.ctx.Done():
						return
					case <-c.done:
						return
					case <-ticker.C:
					}
					continue
				}
				if problem == nil && row.ReusedFrom == "" {
					artifacts, inspectProblem := c.childResultArtifacts(*row, result)
					problem = inspectProblem
					if problem == nil {
						problem = c.retainChildArtifacts(s.ctx, *row, "result", artifacts)
					}
					if problem != nil && problem.Code == exit.Unavailable {
						select {
						case <-s.ctx.Done():
							return
						case <-c.done:
							return
						case <-ticker.C:
						}
						continue
					}
				}
				if problem == nil {
					problem = c.releaseChildRetentions(context.Background(), row.ID, true)
				}
				if problem == nil && row.State == "finalizing" {
					problem = c.opt.Store.CompleteReusedChild(row.ID)
				}
				var byteGrants []*pb.ChildByteResultGrant
				var observation *pb.ExecutionObservation
				if problem == nil {
					byteGrants, observation, problem = c.childByteGrants(s, call, *row)
				}
				if problem != nil && problem.Code == exit.Unavailable {
					select {
					case <-s.ctx.Done():
						return
					case <-c.done:
						return
					case <-ticker.C:
					}
					continue
				}
				state := pb.ChildCallState_CHILD_CALL_STATE_SUCCEEDED
				if problem != nil {
					state = pb.ChildCallState_CHILD_CALL_STATE_FAILED
					if row.State == "finalizing" {
						_, _ = c.opt.Store.BlockRetainedWork(id, problem.ErrName(), problem.Message)
					}
				}
				c.sendChildResult(s, call, id, state, result, problem, &pb.ChildCallResult{ByteResultGrants: byteGrants, Observation: observation})
				return
			}
			switch row.State {
			case "blocked", "failed", "refused", "abandoned":
				code, detail, _ := c.opt.Store.RetainedFailure(id)
				if code == "" {
					code = "child.failed"
					detail = "the child operation failed"
				}
				c.sendChildResult(s, call, id, pb.ChildCallState_CHILD_CALL_STATE_FAILED, nil, exit.Named(exit.Failed, code, "%s", detail))
				return
			case "canceled", "paused", "pausing", "canceling", "releasing":
				c.sendChildResult(s, call, id, pb.ChildCallState_CHILD_CALL_STATE_CANCELED, nil, exit.Named(exit.Canceled, "child.stopped", "the child operation stopped"))
				return
			}
			parent, problem := c.opt.Store.RequestRow(call.ParentRequestId)
			if problem != nil || parent == nil || parent.State != "dispatching" || parent.Ordinal != int64(call.ParentAttemptOrdinal) {
				_ = c.PauseRequest(id, "parent attempt stopped")
				return
			}
			select {
			case <-s.ctx.Done():
				return
			case <-c.done:
				return
			case <-ticker.C:
			}
		}
	}()
}

func (c *Orchestrator) childInlineResult(row records.Request) ([]byte, *exit.Error) {
	id := row.ID
	if row.ReusedFrom != "" {
		id = row.ReusedFrom
	}
	attempts, problem := c.opt.Store.Attempts(id)
	if problem != nil || len(attempts) == 0 {
		return nil, exit.Internalf("successful child has no immutable result")
	}
	last := attempts[len(attempts)-1]
	if last.State != "closed" || last.TerminalStatus != "SUCCEEDED" {
		return nil, exit.Named(exit.Unavailable, "child.result_uncommitted", "child result is not committed")
	}
	doc, err := canonical.Read(last.TerminalBody, &pb.AttemptOutcomeBody{})
	if err != nil {
		return nil, exit.Internalf("child outcome is not its canonical document")
	}
	encoded := doc.Sub("result").Str("inline_result")
	if encoded == "" {
		return []byte(`{}`), nil
	}
	raw, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil || len(raw) > childCallMaxBytes {
		return nil, exit.New(exit.Validation, "child result exceeds its inline bound")
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return nil, exit.New(exit.Validation, "child result is not canonical JSON")
	}
	return raw, nil
}

func (c *Orchestrator) onChildCancel(s *session, call *pb.ChildCallCancel) {
	if call == nil || c.fenced(s, call.RecordOwnerEpoch, call.ControlStreamEpoch, call.WorkerBootId) {
		return
	}
	parent, problem := c.childParent(s, call.ParentRequestId, call.ParentAttemptOrdinal, call.ParentInvocationSpecDigest)
	if problem != nil {
		return
	}
	if c.cancelNativeEffect(s, call) {
		return
	}
	if c.cancelNativeSource(s, call) {
		return
	}
	children, problem := c.opt.Store.Children(call.ParentRequestId)
	if problem != nil {
		return
	}
	intent, err := canonical.Spell(call.IntentDigest)
	if err != nil {
		return
	}
	for _, child := range children {
		if child.ParentCallIndex == int64(call.CallIndex) && child.ChildIntentDigest == intent && !records.Settled(child.State) {
			if parent.State == "pausing" || parent.State == "paused" || parent.State == "blocked" {
				_ = c.PauseRequest(child.ID, "parent retained its child call")
			} else {
				_ = c.CancelRetainedRequest(child.ID, "parent canceled its child call")
			}
			return
		}
	}
}
