package orchestrator

import (
	"context"
	"fmt"
	"sort"

	"google.golang.org/protobuf/proto"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func (c *Orchestrator) runModelPassThrough(req records.Request) {
	// This continuation owns only privileged source/destination I/O for the one
	// ordinary attempt-zero request. It is restartable from the sidecar and has no
	// scheduler, graph, retry budget, route, id namespace, or terminal of its own.
	c.mu.Lock()
	if c.transferRunning[req.ID] {
		c.mu.Unlock()
		return
	}
	c.transferRunning[req.ID] = true
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.transferRunning, req.ID)
		c.mu.Unlock()
	}()
	if c.opt.ModelTransfers == nil {
		c.logf("model transfer %s remains submitted: this daemon has no transfer owner", req.ID)
		return
	}
	transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
	if problem == nil && transfer != nil && (transfer.State == "completed" || transfer.State == "failed") {
		_, problem = c.opt.Store.SettleModelTransferRequest(req.ID, 0)
		if problem != nil {
			c.logf("pass-through model transfer %s settlement failed: %s", req.ID, problem.Message)
		}
		c.forgetTransferProgress(req.ID)
		return
	}
	if problem == nil && transfer != nil {
		problem = c.opt.ModelTransfers.PassThrough(context.Background(), req.ID,
			transfer.ModelTransferIntent)
	}
	if problem != nil {
		if permanentTransferFailure(problem) {
			_ = c.opt.Store.FailModelTransfer(req.ID, problem.ErrName(), problem.Message)
			_, _ = c.opt.Store.SettleModelTransferRequest(req.ID, 0)
			c.forgetTransferProgress(req.ID)
			c.signalClosed(requestWaitKey(req.ID), problem)
		} else {
			time.AfterFunc(2*time.Second, func() { c.runModelPassThrough(req) })
		}
		return
	}
	current, readProblem := c.opt.Store.RequestRow(req.ID)
	if readProblem != nil || current == nil || current.State == "canceled" {
		return
	}
	if _, problem := c.opt.Store.SettleModelTransferRequest(req.ID, 0); problem != nil {
		c.logf("pass-through model transfer %s settlement failed: %s", req.ID, problem.Message)
	}
	c.forgetTransferProgress(req.ID)
}

// weightsGrantExpired is the pod's name for the one weights-transfer refusal a freshly
// signed grant answers (tensorhub internal/workerhost/weights.go).
const weightsGrantExpired = "weights_grant_expired"

func (c *Orchestrator) moveModelTransferWeights(ctx context.Context,
	weights records.ModelTransferWeights, operationID string, mint WeightsGrantMinter,
) *exit.Error {
	request, problem := c.opt.Store.RequestRow(weights.RequestID)
	if problem != nil || request == nil {
		return problem
	}
	if request.Worker == "" {
		return exit.Named(exit.Structural, "model_transfer.remote_mover_not_needed",
			"local model transfer output is already in Creator TensorFS")
	}
	if mint == nil {
		return exit.Internalf("remote weights transfer has no grant minter")
	}
	if len(weights.Objects) == 0 {
		return exit.Named(exit.Conflict, "model_transfer.output_inventory_empty",
			"weights output %s adopted no objects", weights.OutputSlot)
	}
	known := make(map[string]int64, len(weights.Objects))
	for _, object := range weights.Objects {
		known[object.ObjectID] = object.Length
	}
	// ONE window for the whole walk. It outlives the loop below on purpose: a grant minted
	// on an earlier pass is reused while it is young and re-minted once it is not, which is
	// what stops a re-send from carrying a signature that has gone cold.
	window := NewWeightsGrantWindow(mint)
	for {
		objects, problem := c.opt.Store.ModelTransferObjects(weights.RequestID,
			weights.Attempt, weights.OutputSlot)
		if problem != nil {
			return problem
		}
		// The durable rows must cover the adopted inventory before an empty outstanding set
		// can mean "done". A projection that has not caught up is not a completed transfer.
		if len(objects) != len(known) {
			return exit.Named(exit.Unavailable, "model_transfer.object_rows_incomplete",
				"weights output %s has %d durable object rows for %d adopted objects",
				weights.OutputSlot, len(objects), len(known))
		}
		outstanding := make([]records.ModelTransferObject, 0, len(objects))
		for _, object := range objects {
			if object.State == "uploaded" || object.State == "already_present" ||
				object.State == "held" {
				continue
			}
			if object.State == "failed" {
				// AN EXPIRED GRANT IS NOT A VERDICT ABOUT THE OBJECT. The pod never spent
				// the signature it was handed, so nothing about these bytes was decided and
				// nothing about them has to change -- only the grant does. This used to
				// unwind the whole finalize as exit.Unavailable and rebuild the publication
				// from the top to get fresh URLs; the window is simply dropped instead, and
				// the send below carries a signature minted just now.
				if object.SafeCode != weightsGrantExpired {
					return exit.Named(exit.Failed, "model_transfer.object_failed",
						"worker weights transfer %s failed (%s): %s", object.ObjectID,
						object.SafeCode, object.SafeDetail)
				}
				window.Expire()
			}
			outstanding = append(outstanding, object)
		}
		if len(outstanding) == 0 {
			return nil
		}
		session, problem := c.rentalControl(request.Worker)
		if problem != nil {
			if wait := c.waitTransfer(ctx, weights.RequestID); wait != nil {
				return wait
			}
			continue
		}
		specDigest, err := canonical.Raw(weights.InvocationDigest)
		if err != nil {
			return exit.Internalf("persisted weights invocation digest is malformed: %s", err)
		}
		receiptDigest, err := canonical.Raw(weights.ReceiptDigest)
		if err != nil {
			return exit.Internalf("persisted weights receipt digest is malformed: %s", err)
		}
		ids := make([]string, 0, len(outstanding))
		for _, object := range outstanding {
			ids = append(ids, object.ObjectID)
		}
		for index, object := range outstanding {
			// The grant is obtained HERE, immediately before this object's bytes are asked
			// for, rather than for the whole batch before any of them moved.
			decision, problem := window.Spendable(ctx, object.ObjectID, ids[index:], time.Now())
			if problem != nil {
				return problem
			}
			if decision.Length <= 0 || known[object.ObjectID] != decision.Length {
				return exit.Named(exit.Validation, "model_transfer.transfer_decision_invalid",
					"the authorized grant for %s changed the adopted object inventory",
					object.ObjectID)
			}
			transfer := &pb.WeightsTransferRequest{RecordOwnerEpoch: recordOwnerEpoch,
				ControlStreamEpoch: session.epoch, WorkerBootId: session.bootID,
				RequestId: weights.RequestID, AttemptOrdinal: uint64(weights.Attempt),
				InvocationSpecDigest: specDigest, OutputSlot: weights.OutputSlot,
				WeightsTransactionId: weights.TransactionID, WeightsReceiptDigest: receiptDigest,
				OperationId: operationID, GrantRevision: uint64(object.GrantRevision + 1)}
			if decision.Held {
				transfer.Decision = &pb.WeightsTransferRequest_Held{Held: &pb.WeightsObjectRef{
					ObjectId: object.ObjectID, Length: uint64(decision.Length)}}
			} else {
				names := make([]string, 0, len(decision.Headers))
				for name := range decision.Headers {
					names = append(names, name)
				}
				sort.Strings(names)
				headers := make([]*pb.WeightsUploadHeader, 0, len(names))
				for _, name := range names {
					headers = append(headers, &pb.WeightsUploadHeader{Name: name,
						Value: decision.Headers[name]})
				}
				transfer.Decision = &pb.WeightsTransferRequest_UploadGrant{
					UploadGrant: &pb.WeightsUploadGrant{
						ObjectId: object.ObjectID, Length: uint64(decision.Length),
						Url: decision.URL, RequiredHeaders: headers,
						ExpiresAtUnix: decision.ExpiresAtUnix}}
			}
			session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_WeightsTransferRequest{
				WeightsTransferRequest: transfer}})
		}
		if wait := c.waitTransfer(ctx, weights.RequestID); wait != nil {
			return wait
		}
	}
}

func (c *Orchestrator) materializeModelTransfer(req records.Request, w *worker) (records.Request, *exit.Error) {
	transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
	if problem != nil || transfer == nil {
		return req, problem
	}
	if len(transfer.Models) > 0 {
		req.Models = append([]ModelRef(nil), transfer.Models...)
		return req, nil
	}
	if c.opt.ModelTransfers == nil {
		return req, exit.Named(exit.Unavailable, "model_transfer.owner_absent",
			"this daemon has no model transfer materializer")
	}
	if problem := c.opt.Store.BeginModelTransferMaterialization(req.ID); problem != nil {
		return req, problem
	}
	c.publishTransferProgress(req.ID, map[string]any{"stage": "source materialization",
		"transferred_bytes": int64(0), "total_bytes": transferSourceBytes(transfer.SourceFiles)})
	var models []ModelRef
	if w.spec.Connection == nil {
		models, problem = c.opt.ModelTransfers.MaterializeLocal(context.Background(), req.ID,
			transfer.ModelTransferIntent)
	} else {
		var capabilities []ModelSourceCapability
		capabilities, problem = c.opt.ModelTransfers.RefreshRemoteSource(context.Background(),
			transfer.ModelTransferIntent)
		if problem == nil {
			models, problem = c.prepareModelTransferRemote(context.Background(), req,
				transfer.ModelTransferIntent, capabilities)
		}
	}
	if problem != nil {
		if permanentTransferFailure(problem) {
			_ = c.opt.Store.FailModelTransfer(req.ID, problem.ErrName(), problem.Message)
		}
		if problem.Code == exit.Deadline || problem.Code == exit.Capacity {
			return req, exit.Named(exit.Unavailable, problem.ErrName(), "%s", problem.Message)
		}
		return req, problem
	}
	if problem := c.opt.Store.CompleteModelTransferMaterialization(req.ID, models); problem != nil {
		return req, problem
	}
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil || current == nil {
		return req, problem
	}
	if current.State == "canceled" {
		return req, exit.New(exit.Canceled, "model transfer %s was canceled", req.ID)
	}
	req.Models = append([]ModelRef(nil), models...)
	c.emit(req.ID, "request.inputs_materialized", 0, map[string]any{"models": len(models)})
	return req, nil
}

func transferSourceBytes(files []records.ModelTransferSourceFile) int64 {
	var total int64
	for _, file := range files {
		total += file.Length
	}
	return total
}

func (c *Orchestrator) publishTransferProgress(requestID string, value map[string]any) {
	c.mu.Lock()
	c.transferProgressSeq[requestID]++
	seq := c.transferProgressSeq[requestID]
	c.mu.Unlock()
	c.frames.publish(Frame{RequestID: requestID, Attempt: 0, Seq: seq,
		Type: "progress", Value: value})
}

func (c *Orchestrator) forgetTransferProgress(requestID string) {
	c.mu.Lock()
	delete(c.transferProgressSeq, requestID)
	delete(c.sourcePrepareReplies, requestID)
	c.mu.Unlock()
	c.frames.forget(requestID)
}

func (c *Orchestrator) prepareModelTransferRemote(ctx context.Context, req records.Request,
	intent records.ModelTransferIntent, capabilities []ModelSourceCapability,
) ([]ModelRef, *exit.Error) {
	selection, err := canonical.Raw(intent.SourceSelection)
	if err != nil || len(intent.SourceFiles) == 0 || len(intent.SourceProfiles) == 0 {
		return nil, exit.Named(exit.Validation, "model_transfer.source_plan_incomplete",
			"remote model transfer source plan is incomplete")
	}
	expected := make(map[string]records.ModelTransferSourceFile, len(intent.SourceFiles))
	for _, file := range intent.SourceFiles {
		expected[file.Member] = file
	}
	byMember := make(map[string]ModelSourceCapability, len(capabilities))
	for _, capability := range capabilities {
		file, ok := expected[capability.Member]
		if !ok || capability.ObjectID != "sha256:"+file.SHA256 ||
			capability.Length != file.Length || capability.URL == "" ||
			byMember[capability.Member].Member != "" {
			return nil, exit.Named(exit.Conflict, "model_transfer.source_capability_changed",
				"refreshed provider access changed the selected source files")
		}
		byMember[capability.Member] = capability
	}
	if len(byMember) != len(expected) {
		return nil, exit.Named(exit.Conflict, "model_transfer.source_capability_incomplete",
			"refreshed provider access returned %d of %d selected files", len(byMember), len(expected))
	}
	members := make([]string, 0, len(expected))
	for member := range expected {
		members = append(members, member)
	}
	sort.Strings(members)
	preparedStream, preparedRecovery := "", ""
	preparedCount := -1
	preparing := false
	var replyRevision uint64
	// restated names the control stream this owner has already stated its WHOLE selection
	// to. The pod's memory of a transfer is its TensorFS Store, not a journal: it indexes
	// what it proved by object, and a boot that lost that index answers a re-ask from the
	// store itself, `held`, moving nothing. So the selection is restated once per control
	// stream rather than only for members this owner has not yet seen verified — otherwise
	// a pod that restarted under a live rental is asked to prepare over a selection it was
	// never told about, and the transfer dies naming files nobody named to it.
	//
	// The key carries the BOOT, not the epoch alone: the pod mints control-stream epochs
	// from a counter that starts over with the process, so a restarted supervisor hands out
	// epoch 1 again — and that is precisely the pod whose index is empty.
	restated := ""
	for {
		transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
		if problem != nil {
			return nil, problem
		}
		if transfer != nil && len(transfer.Models) == len(intent.SourceProfiles) {
			return append([]ModelRef(nil), transfer.Models...), nil
		}
		if transfer != nil && transfer.State == "failed" {
			return nil, exit.Named(exit.Failed, transfer.ErrorCode, "%s", transfer.SafeError)
		}
		request, problem := c.opt.Store.RequestRow(req.ID)
		if problem != nil || request == nil {
			return nil, problem
		}
		if request.State == "canceled" {
			return nil, exit.New(exit.Canceled, "model transfer %s was canceled", req.ID)
		}
		session, problem := c.rentalControl(req.Worker)
		if problem != nil {
			if wait := c.waitTransfer(ctx, req.ID); wait != nil {
				return nil, wait
			}
			continue
		}
		statuses, problem := c.opt.Store.ModelTransferSourceStatuses(req.ID)
		if problem != nil {
			return nil, problem
		}
		verified := 0
		revisions := make(map[string]int64, len(statuses))
		for _, status := range statuses {
			revisions[status.Member] = status.CapabilityRevision
			if status.State == "verified" {
				verified++
			}
		}
		// The frames go out in ONE pass, in member order, and a preparation sent in the
		// same pass is read after all of them: the pod registers every member off its
		// control read loop before it considers a preparation for the same operation.
		//
		// ONE PASS PER CONTROL STREAM, and only one. This used to re-state every
		// non-verified member on every wake, and the pod's own ACCEPTED echo IS a wake --
		// a closed loop with the pod as its amplifier. On run 205 it turned over ~3,000
		// capability_revision bumps per member, each a fresh pod goroutine that took a
		// fetch lane and re-bought a fetch that was already doomed, and each carrying the
		// row further past the revision whose verdict was still in flight (cl-133). A
		// member this pod has been told about on this stream is in the pod's hands.
		stream := fmt.Sprintf("%s/%d", session.bootID, session.epoch)
		if restated != stream {
			for _, member := range members {
				file := expected[member]
				access := byMember[member]
				frame := &pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourceFileRequest{
					ModelSourceFileRequest: &pb.ModelSourceFileRequest{
						RecordOwnerEpoch:   recordOwnerEpoch,
						ControlStreamEpoch: session.epoch, WorkerBootId: session.bootID,
						OperationId: req.ID, SourceSelectionDigest: selection, Member: member,
						ObjectId: "sha256:" + file.SHA256, Length: uint64(file.Length),
						Provider: access.Provider, Url: access.URL, ExpiresAtUnix: access.ExpiresAtUnix,
						CapabilityRevision: uint64(revisions[member] + 1), Header: file.Header}}}
				if proto.Size(frame) > pb.MaxInlineControlBytes {
					return nil, exit.Named(exit.Validation, "model_transfer.source_header_too_large",
						"source header and capability for %s exceed the worker control bound", member)
				}
				session.send(frame)
			}
			restated = stream
			c.logf("model transfer %s: stating all %d selected source file(s) to %s on "+
				"control stream %d; a member this pod already holds is answered from its "+
				"TensorFS store and moves nothing",
				req.ID, len(members), session.instanceID, session.epoch)
		}
		// One call in flight. Its INCOMPLETE answer only opens the next call when
		// more source members have verified, so progress cannot echo into a loop.
		c.mu.Lock()
		replies := c.sourcePrepareReplies[req.ID]
		c.mu.Unlock()
		if preparedStream != stream || replies != replyRevision {
			preparing = false
			replyRevision = replies
		}
		checkpoints, recovery, problem := c.acknowledgedSourceCheckpoints(req.ID)
		if problem != nil {
			return nil, problem
		}
		if !preparing && (preparedStream != stream || preparedCount != verified || preparedRecovery != recovery) {
			profiles := make([]*pb.ModelSourceProfile, 0, len(intent.SourceProfiles))
			for slot, profile := range intent.SourceProfiles {
				profiles = append(profiles, &pb.ModelSourceProfile{Slot: slot, Profile: profile})
			}
			sort.Slice(profiles, func(i, j int) bool { return profiles[i].Slot < profiles[j].Slot })
			c.logf("model transfer %s: preparing %d profile(s) on %s with %d/%d source files verified",
				req.ID, len(profiles), session.instanceID, verified, len(expected))
			session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourcePrepareRequest{
				ModelSourcePrepareRequest: &pb.ModelSourcePrepareRequest{
					RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: session.epoch,
					WorkerBootId: session.bootID, OperationId: req.ID,
					SourceSelectionDigest: selection, Profiles: profiles, Checkpoints: checkpoints,
					SourceUri: intent.Source, DeclaredLicense: intent.SourceLicense}}})
			preparing = true
			preparedStream, preparedCount, preparedRecovery = stream, verified, recovery
		}
		if wait := c.waitTransfer(ctx, req.ID); wait != nil {
			return nil, wait
		}
	}
}

func (c *Orchestrator) finalizeModelTransfer(ctx context.Context, requestID string) *exit.Error {
	transfer, problem := c.opt.Store.ModelTransferOf(requestID)
	if problem != nil || transfer == nil || transfer.State == "completed" {
		return problem
	}
	if transfer.State == "canceled" {
		return exit.New(exit.Canceled, "model transfer %s finalization was canceled", requestID)
	}
	if c.opt.ModelTransfers == nil {
		problem = exit.Named(exit.Unavailable, "model_transfer.owner_absent",
			"this daemon has no model transfer finalizer")
	} else if problem = c.opt.Store.BeginModelTransferFinalization(requestID); problem == nil {
		problem = c.opt.ModelTransfers.Finalize(ctx, requestID,
			func(ctx context.Context, weights records.ModelTransferWeights, operationID string,
				mint WeightsGrantMinter,
			) *exit.Error {
				return c.moveModelTransferWeights(ctx, weights, operationID, mint)
			})
	}
	if problem != nil {
		if permanentTransferFailure(problem) {
			_ = c.opt.Store.FailModelTransfer(requestID, problem.ErrName(), problem.Message)
		}
		return problem
	}
	return nil
}

func permanentTransferFailure(problem *exit.Error) bool {
	if problem == nil {
		return false
	}
	switch problem.Code {
	case exit.Unavailable, exit.Deadline, exit.Capacity:
		return false
	default:
		return true
	}
}

func (c *Orchestrator) kickModelTransferFinalizer(s *session, requestID string, attempt int64) {
	c.mu.Lock()
	if c.transferRunning[requestID] {
		c.mu.Unlock()
		return
	}
	c.transferRunning[requestID] = true
	runCtx, cancel := context.WithCancel(context.Background())
	c.transferCancels[requestID] = cancel
	c.mu.Unlock()
	go func() {
		defer func() {
			c.mu.Lock()
			delete(c.transferRunning, requestID)
			delete(c.transferCancels, requestID)
			c.mu.Unlock()
			cancel()
		}()
		problem := c.finalizeModelTransfer(runCtx, requestID)
		if problem != nil {
			c.logf("model transfer %s finalization failed: %s", requestID, problem.Message)
			transfer, _ := c.opt.Store.ModelTransferOf(requestID)
			if transfer == nil || (transfer.State != "failed" && transfer.State != "canceled") {
				time.AfterFunc(2*time.Second, func() {
					c.kickModelTransferFinalizer(s, requestID, attempt)
				})
				return
			}
		}
		c.ackSettledOutcome(s, requestID, uint64(attempt))
	}()
}

func (c *Orchestrator) CancelModelTransferFinalization(requestID, actor string) *exit.Error {
	if actor == "" {
		actor = "an unnamed client"
	}
	if e := c.opt.Store.AppendEvent(requestID, "request.cancel_requested", 0,
		map[string]any{"actor": actor}); e != nil {
		return e
	}
	if problem := c.opt.Store.RequestModelTransferCancellation(requestID); problem != nil {
		return problem
	}
	c.mu.Lock()
	cancel := c.transferCancels[requestID]
	c.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	c.signalTransfer(requestID)
	return nil
}

func (c *Orchestrator) finishModelTransferRequest(requestID string, attempt int64) {
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || request == nil || request.State == "canceled" {
		return
	}
	if problem := c.releaseManagedNow(*request); problem != nil {
		c.logf("model transfer %s provider cleanup remains pending: %s", requestID, problem.Message)
		time.AfterFunc(2*time.Second, func() { c.finishModelTransferRequest(requestID, attempt) })
		return
	}
	if _, problem := c.opt.Store.SettleModelTransferRequest(requestID, attempt); problem != nil {
		c.logf("model transfer %s terminal settlement remains pending: %s", requestID, problem.Message)
		time.AfterFunc(2*time.Second, func() { c.finishModelTransferRequest(requestID, attempt) })
		return
	}
	request, problem = c.opt.Store.RequestRow(requestID)
	attemptRow, attemptProblem := c.opt.Store.AttemptRow(requestID, attempt)
	if problem == nil && attemptProblem == nil && request != nil && attemptRow != nil {
		request.Rental = false // provider absence was already proven above
		c.afterAck(*request, *attemptRow, nil)
	}
}

func (c *Orchestrator) ResumeModelTransfers() *exit.Error {
	owed, problem := c.opt.Store.ModelTransfersOwed()
	if problem != nil {
		return problem
	}
	for _, transfer := range owed {
		request, readProblem := c.opt.Store.RequestRow(transfer.RequestID)
		if readProblem != nil || request == nil || request.State == "canceled" {
			continue
		}
		if request.Package == "cozy/platform" && request.Entrypoint == "model-pass-through" {
			go c.runModelPassThrough(*request)
			continue
		}
		if request.State != "finalizing" {
			continue
		}
		attempt, readProblem := c.opt.Store.AttemptRow(request.ID, request.Ordinal)
		if readProblem != nil || attempt == nil {
			continue
		}
		if attempt.State == "closed" {
			go c.finishModelTransferRequest(request.ID, attempt.Attempt)
		} else if request.Worker == "" && attempt.State == "terminal" {
			c.kickRecoveredLocalTransfer(request.ID, attempt.Attempt)
		} else if request.Worker != "" {
			c.selectOrStart(*request)
		}
	}
	return nil
}

func (c *Orchestrator) kickRecoveredLocalTransfer(requestID string, attempt int64) {
	c.mu.Lock()
	if c.transferRunning[requestID] {
		c.mu.Unlock()
		return
	}
	c.transferRunning[requestID] = true
	runCtx, cancel := context.WithCancel(context.Background())
	c.transferCancels[requestID] = cancel
	c.mu.Unlock()
	go func() {
		defer func() {
			c.mu.Lock()
			delete(c.transferRunning, requestID)
			delete(c.transferCancels, requestID)
			c.mu.Unlock()
			cancel()
		}()
		transfer, _ := c.opt.Store.ModelTransferOf(requestID)
		if transfer == nil || (transfer.State != "failed" && transfer.State != "canceled") {
			if problem := c.finalizeModelTransfer(runCtx, requestID); problem != nil {
				transfer, _ = c.opt.Store.ModelTransferOf(requestID)
				if transfer != nil && (transfer.State == "failed" || transfer.State == "canceled") {
					// The ordinary terminal owns the verdict; only closure/teardown remains.
				} else {
					time.AfterFunc(2*time.Second, func() {
						c.kickRecoveredLocalTransfer(requestID, attempt)
					})
					return
				}
			}
		}
		if problem := c.opt.Store.Closed(requestID, attempt); problem != nil {
			return
		}
		c.finishModelTransferRequest(requestID, attempt)
	}()
}
