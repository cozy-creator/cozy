package orchestrator

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type privateEffectOwner interface {
	PreparePublicationEffect(context.Context, records.NativeCall) ([]byte, *exit.Error)
	ApplyPublicationEffect(context.Context, records.NativeCall) ([]byte, *exit.Error)
}

func (c *Orchestrator) onNativeEffect(s *session, parent records.Request, call *pb.ChildCallRequest) bool {
	if call.Module != publication.Module {
		return false
	}
	refuse := func(problem *exit.Error) {
		c.sendChildResult(s, call, "", pb.ChildCallState_CHILD_CALL_STATE_REFUSED, nil, problem)
	}
	iface, _ := canonical.Spell(call.InterfaceDigest)
	if iface != publication.InterfaceDigest() || (call.Export != "upload_checkpoint" && call.Export != "publish_release") {
		refuse(exit.New(exit.Validation, "publication effect differs from its fixed interface"))
		return true
	}
	if _, problem := publication.EffectDestination(call.Export, call.RequestCanonicalBytes); problem != nil {
		refuse(problem)
		return true
	}
	digest, _ := canonical.Spell(call.IntentDigest)
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s/%s/%d", recordOwnerID, parent.ID, call.CallIndex)))
	id := "effect-" + hex.EncodeToString(sum[:])
	parentSpec, _ := canonical.Spell(call.ParentInvocationSpecDigest)
	row, _, problem := c.opt.Store.AcceptNativeCall(records.NativeCall{ID: id, ParentRequestID: parent.ID, CallIndex: int64(call.CallIndex), Kind: "effect", Operation: call.Export, IntentDigest: digest, Request: call.RequestCanonicalBytes}, int64(call.ParentAttemptOrdinal), parentSpec, s.bootID)
	if problem != nil {
		refuse(problem)
		return true
	}
	c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_PENDING, nil, nil)
	c.watchNativeEffect(s, proto.Clone(call).(*pb.ChildCallRequest), row)
	return true
}
func (c *Orchestrator) watchNativeEffect(s *session, call *pb.ChildCallRequest, row records.NativeCall) {
	c.mu.Lock()
	if c.closing {
		c.mu.Unlock()
		return
	}
	active := c.transferRunning[row.ID]
	if !active {
		c.transferRunning[row.ID] = true
	}
	c.mu.Unlock()
	if !active {
		go func() {
			defer func() { c.mu.Lock(); delete(c.transferRunning, row.ID); c.mu.Unlock() }()
			c.runNativeEffect(row)
		}()
	}
	go func() {
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			observed, problem := c.opt.Store.NativeCall(row.ParentRequestID, row.CallIndex)
			if problem != nil || observed == nil {
				return
			}
			switch observed.State {
			case "succeeded":
				c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_SUCCEEDED, observed.Result, nil)
				return
			case "failed", "canceled":
				c.sendChildResult(s, call, row.ID, pb.ChildCallState_CHILD_CALL_STATE_FAILED, nil, exit.Named(exit.Failed, observed.SafeCode, "publication effect stopped"))
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
func (c *Orchestrator) runNativeEffect(row records.NativeCall) {
	defer func() {
		current, problem := c.opt.Store.NativeCall(row.ParentRequestID, row.CallIndex)
		if problem == nil && current != nil && (current.State == "succeeded" || current.State == "failed" || current.State == "canceled") {
			_ = c.releaseNativeEffectInputs(row.ID)
		}
	}()

	owner, ok := c.opt.ModelTransfers.(privateEffectOwner)
	if !ok {
		_ = c.opt.Store.StopNativeCall(row.ID, "failed", "publication.owner_absent")
		return
	}
	for {
		current, problem := c.opt.Store.NativeCall(row.ParentRequestID, row.CallIndex)
		if problem != nil || current == nil || current.State == "succeeded" || current.State == "failed" || current.State == "canceled" {
			return
		}
		row = *current
		parent, problem := c.opt.Store.RequestRow(row.ParentRequestID)
		if problem != nil || parent == nil {
			return
		}
		if (parent.State == "canceling" || parent.State == "canceled" || parent.State == "releasing") && row.State != "executing" {
			_ = c.opt.Store.StopNativeCall(row.ID, "canceled", "publication.parent_stopped")
			return
		}
		if parent.State != "dispatching" && row.State != "executing" {
			select {
			case <-c.done:
				return
			case <-time.After(time.Second):
			}
			continue
		}
		if row.Operation == "upload_checkpoint" {
			result, problem := c.runNativeUpload(context.Background(), row)
			if problem == nil {
				problem = c.opt.Store.CompleteNativeCall(row.ID, result, nil)
			}
			if problem == nil {
				return
			}
			if permanentEffectFailure(problem) {
				_ = c.opt.Store.StopNativeCall(row.ID, "failed", problem.ErrName())
				return
			}
			select {
			case <-c.done:
				return
			case <-time.After(time.Second):
			}
			continue
		}
		if len(row.Frozen) == 0 {
			frozen, problem := owner.PreparePublicationEffect(context.Background(), row)
			if problem == nil {
				problem = c.opt.Store.FreezeNativeCall(row.ID, frozen)
			}
			if problem != nil {
				if permanentEffectFailure(problem) {
					_ = c.opt.Store.StopNativeCall(row.ID, "failed", problem.ErrName())
					return
				}
			} else {
				continue
			}
		} else {
			result, problem := owner.ApplyPublicationEffect(context.Background(), row)
			if problem == nil {
				if problem = c.opt.Store.CompleteNativeCall(row.ID, result, nil); problem == nil {
					return
				}
			}
			if permanentEffectFailure(problem) {
				_ = c.opt.Store.StopNativeCall(row.ID, "failed", problem.ErrName())
				return
			}
		}
		select {
		case <-c.done:
			return
		case <-time.After(time.Second):
		}
	}
}

func permanentEffectFailure(problem *exit.Error) bool {
	// A revoked or refreshable credential cannot erase whether the remote write
	// may already have committed. Keep the frozen/executing obligation resumable.
	if problem != nil && problem.Code == exit.Credential {
		return false
	}
	return permanentTransferFailure(problem)
}

func (c *Orchestrator) releaseNativeEffectInputs(id string) *exit.Error {
	if problem := c.opt.Store.BeginNativeArtifactRelease(id, false); problem != nil {
		return problem
	}
	holds, problem := c.opt.Store.NativeArtifactRetentions(id)
	if problem != nil {
		return problem
	}
	for _, hold := range holds {
		if hold.State == "releasing" {
			if problem := c.changeNativeArtifactRetention(context.Background(), hold, true); problem != nil {
				return problem
			}
		}
	}
	return nil
}

func (c *Orchestrator) ResumeNativeEffects() *exit.Error {
	calls, problem := c.opt.Store.OwedNativeCalls("effect")
	if problem != nil {
		return problem
	}
	for _, call := range calls {
		c.mu.Lock()
		active := c.transferRunning[call.ID] || c.closing
		if !active {
			c.transferRunning[call.ID] = true
		}
		c.mu.Unlock()
		if !active {
			go func(row records.NativeCall) {
				defer func() { c.mu.Lock(); delete(c.transferRunning, row.ID); c.mu.Unlock() }()
				c.runNativeEffect(row)
			}(call)
		}
	}
	cleanup, problem := c.opt.Store.NativeEffectCleanupIDs()
	if problem != nil {
		return problem
	}
	for _, id := range cleanup {
		if problem := c.releaseNativeEffectInputs(id); problem != nil {
			return problem
		}
	}
	return nil
}
