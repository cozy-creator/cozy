package orchestrator

import (
	"context"
	"sort"
	"strings"
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
			c.frames.forget(req.ID)
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
}

func (c *Orchestrator) moveModelTransferArtifact(ctx context.Context,
	artifact records.ModelTransferArtifact, operationID string,
	decisions []ArtifactTransferDecision,
) *exit.Error {
	request, problem := c.opt.Store.RequestRow(artifact.RequestID)
	if problem != nil || request == nil {
		return problem
	}
	if request.Worker == "" {
		return exit.Named(exit.Structural, "model_transfer.remote_mover_not_needed",
			"local model transfer output is already in Creator TensorFS")
	}
	byObject := make(map[string]ArtifactTransferDecision, len(decisions))
	known := make(map[string]int64, len(artifact.Objects))
	for _, object := range artifact.Objects {
		known[object.ObjectID] = object.Length
	}
	for _, decision := range decisions {
		if decision.ObjectID == "" || decision.Length <= 0 || known[decision.ObjectID] != decision.Length ||
			byObject[decision.ObjectID].ObjectID != "" {
			return exit.Named(exit.Validation, "model_transfer.transfer_decision_invalid",
				"artifact transfer decisions changed the adopted object inventory")
		}
		byObject[decision.ObjectID] = decision
	}
	for {
		objects, problem := c.opt.Store.ModelTransferObjects(artifact.RequestID,
			artifact.Attempt, artifact.OutputSlot)
		if problem != nil {
			return problem
		}
		complete := len(byObject) > 0
		for _, object := range objects {
			if _, selected := byObject[object.ObjectID]; !selected {
				continue
			}
			if object.State == "failed" {
				if strings.Contains(strings.ToLower(object.SafeCode), "expired") {
					return exit.Named(exit.Unavailable, object.SafeCode,
						"worker artifact transfer %s needs a refreshed grant: %s",
						object.ObjectID, object.SafeDetail)
				}
				return exit.Named(exit.Failed, "model_transfer.object_failed",
					"worker artifact transfer %s failed: %s", object.ObjectID,
					object.SafeDetail)
			}
			complete = complete && (object.State == "uploaded" ||
				object.State == "already_present" || object.State == "held")
		}
		if complete {
			return nil
		}
		session, problem := c.rentalControl(request.Worker)
		if problem != nil {
			if wait := c.waitTransfer(ctx, artifact.RequestID); wait != nil {
				return wait
			}
			continue
		}
		specDigest, err := canonical.Raw(artifact.InvocationDigest)
		if err != nil {
			return exit.Internalf("persisted artifact invocation digest is malformed: %s", err)
		}
		receiptDigest, err := canonical.Raw(artifact.ReceiptDigest)
		if err != nil {
			return exit.Internalf("persisted artifact receipt digest is malformed: %s", err)
		}
		for _, object := range objects {
			decision, selected := byObject[object.ObjectID]
			if !selected || object.State == "uploaded" || object.State == "already_present" ||
				object.State == "held" {
				continue
			}
			request := &pb.ArtifactTransferRequest{RecordOwnerEpoch: recordOwnerEpoch,
				ControlStreamGeneration: session.generation, WorkerBootId: session.bootID,
				RequestId: artifact.RequestID, AttemptOrdinal: uint64(artifact.Attempt),
				InvocationSpecDigest: specDigest, OutputSlot: artifact.OutputSlot,
				ArtifactTransactionId: artifact.TransactionID, ArtifactReceiptDigest: receiptDigest,
				OperationId: operationID, GrantRevision: uint64(object.GrantRevision + 1)}
			if decision.Held {
				request.Decision = &pb.ArtifactTransferRequest_Held{Held: &pb.ArtifactObjectRef{
					ObjectId: object.ObjectID, Length: uint64(decision.Length)}}
			} else {
				names := make([]string, 0, len(decision.Headers))
				for name := range decision.Headers {
					names = append(names, name)
				}
				sort.Strings(names)
				headers := make([]*pb.ArtifactUploadHeader, 0, len(names))
				for _, name := range names {
					headers = append(headers, &pb.ArtifactUploadHeader{Name: name, Value: decision.Headers[name]})
				}
				request.Decision = &pb.ArtifactTransferRequest_UploadGrant{UploadGrant: &pb.ArtifactUploadGrant{
					ObjectId: object.ObjectID, Length: uint64(decision.Length), Url: decision.URL,
					RequiredHeaders: headers, ExpiresAtUnix: decision.ExpiresAtUnix}}
			}
			session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ArtifactTransferRequest{
				ArtifactTransferRequest: request}})
		}
		if wait := c.waitTransfer(ctx, artifact.RequestID); wait != nil {
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
		allVerified := len(statuses) == len(expected)
		for _, status := range statuses {
			file := expected[status.Member]
			allVerified = allVerified && status.State == "verified"
			if status.State == "verified" {
				continue
			}
			access := byMember[status.Member]
			request := &pb.ModelSourceFileRequest{RecordOwnerEpoch: recordOwnerEpoch,
				ControlStreamGeneration: session.generation, WorkerBootId: session.bootID,
				OperationId: req.ID, SourceSelectionDigest: selection, Member: status.Member,
				ObjectId: "sha256:" + file.SHA256, Length: uint64(file.Length),
				Provider: access.Provider, Url: access.URL, ExpiresAtUnix: access.ExpiresAtUnix,
				CapabilityRevision: uint64(status.CapabilityRevision + 1)}
			session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourceFileRequest{
				ModelSourceFileRequest: request}})
		}
		if allVerified {
			profiles := make([]*pb.ModelSourceProfile, 0, len(intent.SourceProfiles))
			for slot, profile := range intent.SourceProfiles {
				profiles = append(profiles, &pb.ModelSourceProfile{Slot: slot, Profile: profile})
			}
			sort.Slice(profiles, func(i, j int) bool { return profiles[i].Slot < profiles[j].Slot })
			session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourcePrepareRequest{
				ModelSourcePrepareRequest: &pb.ModelSourcePrepareRequest{
					RecordOwnerEpoch: recordOwnerEpoch, ControlStreamGeneration: session.generation,
					WorkerBootId: session.bootID, OperationId: req.ID,
					SourceSelectionDigest: selection, Profiles: profiles,
					SourceUri: intent.Source, DeclaredLicense: intent.SourceLicense}}})
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
			func(ctx context.Context, artifact records.ModelTransferArtifact, operationID string,
				decisions []ArtifactTransferDecision,
			) *exit.Error {
				return c.moveModelTransferArtifact(ctx, artifact, operationID, decisions)
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

func (c *Orchestrator) CancelModelTransferFinalization(requestID string) *exit.Error {
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
