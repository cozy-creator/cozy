package orchestrator

import (
	"context"
	"encoding/hex"
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
	c.kickCheckpointUpload(req.ID)
}

// weightsGrantExpired is the pod's name for the one weights-transfer refusal a freshly
// signed grant answers (tensorhub internal/workerhost/weights.go).
const weightsGrantExpired = "weights_grant_expired"

func (c *Orchestrator) moveModelTransferWeights(ctx context.Context,
	weights records.ModelTransferWeights, mint WeightsGrantMinter,
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
			objectDigest, err := canonical.Raw(object.ObjectID)
			if err != nil {
				return exit.New(exit.Validation, "persisted weights object digest is malformed")
			}
			operationID := object.OperationID
			if operationID == "" {
				operationID = "weights-" + hex.EncodeToString(objectDigest)
			}
			// The Host binds one operation to one object within this transaction.
			// The Hub publication is a separate scope shared by all of its objects.
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
	if !transfer.HasAcquisition() {
		return req, nil
	}
	c.mu.Lock()
	bootID := w.bootID
	c.mu.Unlock()
	if len(transfer.Models) > 0 && (w.spec.Connection == nil || transfer.ModelsWorkerBootID == bootID) {
		if w.spec.Connection != nil {
			if problem := c.awaitSourceInputCustody(req, bootID); problem != nil {
				return req, problem
			}
		}
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
	if w.spec.Connection == nil {
		if problem := c.opt.Store.CompleteModelTransferMaterialization(req.ID, models, ""); problem != nil {
			return req, problem
		}
	} else if problem := c.awaitSourceInputCustody(req, bootID); problem != nil {
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
	delete(c.sourcePrepareBlocked, requestID)
	c.mu.Unlock()
	c.frames.forget(requestID)
	c.ForgetPhase(requestID)
}

type sourceCapabilityRetry struct {
	revision int64
	bytes    int64
	delay    time.Duration
	after    time.Time
	sent     bool
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
		if len(file.Header) == 0 {
			return nil, exit.Named(exit.Validation, "model_transfer.source_header_missing", "source header for %s was not retained before rental", file.Member)
		}
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
	stream, phase := "", ""
	declarations := map[string]int64{}
	retries := map[string]sourceCapabilityRetry{}
	retryWake := func(delay time.Duration) { time.AfterFunc(delay, func() { c.signalTransfer(req.ID) }) }
	preparedCount := -1
	preparedRecovery, preparedObservation := "", ""
	preparing := false
	var replyRevision uint64
	for {
		transfer, problem := c.opt.Store.ModelTransferOf(req.ID)
		if problem != nil {
			return nil, problem
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
		session, problem := c.rentalControl(request.Worker)
		if problem != nil {
			if wait := c.waitTransfer(ctx, req.ID); wait != nil {
				return nil, wait
			}
			continue
		}
		if transfer != nil && len(transfer.Models) == len(intent.SourceProfiles) && transfer.ModelsWorkerBootID == session.bootID {
			return append([]ModelRef(nil), transfer.Models...), nil
		}
		statuses, problem := c.opt.Store.ModelTransferSourceStatuses(req.ID)
		if problem != nil {
			return nil, problem
		}
		byStatus := make(map[string]records.ModelTransferSourceStatus, len(statuses))
		fulfilled := 0
		for _, status := range statuses {
			byStatus[status.Member] = status
			if status.WorkerBootID == session.bootID && (status.State == "verified" || status.State == "converted") {
				fulfilled++
			}
		}
		send := func(member, url string, revision int64) *exit.Error {
			file, access := expected[member], byMember[member]
			frame := &pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourceFileRequest{
				ModelSourceFileRequest: &pb.ModelSourceFileRequest{
					RecordOwnerEpoch: recordOwnerEpoch, ControlStreamEpoch: session.epoch, WorkerBootId: session.bootID,
					OperationId: req.ID, SourceSelectionDigest: selection, Member: member,
					ObjectId: "sha256:" + file.SHA256, Length: uint64(file.Length), Header: file.Header,
					Provider: access.Provider, Url: url, ExpiresAtUnix: access.ExpiresAtUnix,
					CapabilityRevision: uint64(revision)}}}
			if proto.Size(frame) > pb.MaxInlineControlBytes {
				return exit.Named(exit.Validation, "model_transfer.source_header_too_large", "source header and capability for %s exceed the worker control bound", member)
			}
			if !session.send(frame) {
				return exit.Unavailablef("source declaration control stream closed")
			}
			return nil
		}
		declare := func() *exit.Error {
			for _, member := range members {
				revision := byStatus[member].CapabilityRevision + 1
				if problem := send(member, "", revision); problem != nil {
					return problem
				}
				declarations[member] = revision
			}
			return nil
		}
		currentStream := fmt.Sprintf("%s/%d", session.bootID, session.epoch)
		c.mu.Lock()
		replies := c.sourcePrepareReplies[req.ID]
		backoff := c.sourcePrepareBlocked[req.ID]
		now := time.Now()
		blocked := backoff.stream == currentStream && now.Before(backoff.until) && fulfilled == preparedCount
		retryDue := backoff.stream == currentStream && !now.Before(backoff.until)
		c.mu.Unlock()
		if currentStream != stream {
			stream, phase = currentStream, "declaring"
			retries = map[string]sourceCapabilityRetry{}
			c.ForgetPhase(req.ID)
			preparedCount, preparing, replyRevision = -1, false, replies
			if problem := declare(); problem != nil {
				return nil, problem
			}
		} else if replies != replyRevision {
			preparing, replyRevision = false, replies
		}
		allDeclared := len(byStatus) == len(expected)
		for _, member := range members {
			status := byStatus[member]
			allDeclared = allDeclared && status.WorkerBootID == session.bootID && status.CapabilityRevision >= declarations[member]
		}
		checkpoints, recovery, observation, problem := c.sourceCheckpointHeads(req.ID, session.bootID)
		if problem != nil {
			return nil, problem
		}
		if phase == "declaring" && allDeclared {
			phase = "probe"
			if len(checkpoints) > 0 {
				host, problem := c.checkpointHost(*request)
				if problem != nil {
					return nil, problem
				}
				if problem := c.opt.ModelTransfers.RestoreSourceCheckpoints(ctx, req.ID, host); problem != nil {
					return nil, problem
				}
			}
		}
		// The restore probe lets TensorFS prove spent carriers. A metadata-only
		// query then observes CONVERTED for those members before any source grant.
		if phase == "probing" && !preparing && backoff.stream != currentStream {
			if problem := declare(); problem != nil {
				return nil, problem
			}
			phase, allDeclared = "sweeping", false
		}
		if phase == "sweeping" && allDeclared {
			phase = "download"
		}
		if phase == "download" {
			for _, member := range members {
				status := byStatus[member]
				if status.State == "verified" || status.State == "converted" {
					continue
				}
				if problem := send(member, byMember[member].URL, status.CapabilityRevision+1); problem != nil {
					return nil, problem
				}
			}
			phase = "running"
		}
		if phase == "running" {
			for _, member := range members {
				status := byStatus[member]
				if status.State != "accepted" || status.SafeCode == "" {
					continue
				}
				retry, known := retries[member]
				if !known || retry.revision != status.CapabilityRevision {
					delay := time.Second
					if known && status.Transferred <= retry.bytes {
						delay = min(30*time.Second, retry.delay*2)
					}
					retry = sourceCapabilityRetry{revision: status.CapabilityRevision, bytes: status.Transferred, delay: delay, after: time.Now().Add(delay)}
					retries[member] = retry
					retryWake(delay)
					continue
				}
				if retry.sent || time.Now().Before(retry.after) {
					continue
				}
				refreshed, problem := c.opt.ModelTransfers.RefreshRemoteSource(ctx, intent)
				if problem != nil {
					if permanentTransferFailure(problem) {
						return nil, problem
					}
					retry.delay = min(30*time.Second, retry.delay*2)
					retry.after = time.Now().Add(retry.delay)
					retries[member] = retry
					retryWake(retry.delay)
					continue
				}
				found := false
				for _, access := range refreshed {
					if access.Member == member {
						if access.ObjectID != "sha256:"+expected[member].SHA256 || access.Length != expected[member].Length || access.URL == "" {
							return nil, exit.Named(exit.Conflict, "model_transfer.source_capability_changed", "refreshed source retry changed the selected member")
						}
						byMember[member], found = access, true
					}
				}
				if !found {
					return nil, exit.Internalf("source retry lost its selected member")
				}
				if problem := send(member, byMember[member].URL, status.CapabilityRevision+1); problem != nil {
					return nil, problem
				}
				retry.sent = true
				retries[member] = retry
				c.logf("model transfer %s: refreshed source access for %s after %s", req.ID, member, status.SafeCode)
			}
		}
		if retryDue && !preparing {
			preparedCount = -1
			if phase == "probing" {
				phase = "probe"
			}
		}
		if !blocked && !preparing && (phase == "probe" || (phase == "running" && (preparedCount != fulfilled || preparedRecovery != recovery || preparedObservation != observation))) {
			profiles := make([]*pb.ModelSourceProfile, 0, len(intent.SourceProfiles))
			for slot, profile := range intent.SourceProfiles {
				profiles = append(profiles, &pb.ModelSourceProfile{Slot: slot, Profile: profile})
			}
			sort.Slice(profiles, func(i, j int) bool { return profiles[i].Slot < profiles[j].Slot })
			c.logf("model transfer %s: preparing %d profile(s) on %s with %d/%d source members fulfilled", req.ID, len(profiles), session.instanceID, fulfilled, len(expected))
			if !session.send(&pb.RecordOwnerFrame{Msg: &pb.RecordOwnerFrame_ModelSourcePrepareRequest{
				ModelSourcePrepareRequest: &pb.ModelSourcePrepareRequest{RecordOwnerEpoch: recordOwnerEpoch,
					ControlStreamEpoch: session.epoch, WorkerBootId: session.bootID, OperationId: req.ID,
					SourceSelectionDigest: selection, Profiles: profiles, Checkpoints: checkpoints,
					SourceUri: intent.Source, DeclaredLicense: intent.SourceLicense}}}) {
				return nil, exit.Unavailablef("source preparation control stream closed")
			}
			preparing, preparedCount, preparedRecovery, preparedObservation = true, fulfilled, recovery, observation
			if phase == "probe" {
				phase = "probing"
			}
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
		return nil
	}
	if transfer.State == "canceling" {
		if c.opt.ModelTransfers == nil {
			return exit.Unavailablef("publication cleanup owner is absent")
		}
		if problem := c.opt.ModelTransfers.AbandonModelTransferPublications(ctx, requestID); problem != nil {
			return problem
		}
		return c.opt.Store.CompleteModelTransferCancellation(requestID)
	}
	if c.opt.ModelTransfers == nil {
		problem = exit.Named(exit.Unavailable, "model_transfer.owner_absent",
			"this daemon has no model transfer finalizer")
	} else if problem = c.opt.Store.BeginModelTransferFinalization(requestID); problem == nil {
		problem = c.opt.ModelTransfers.Finalize(ctx, requestID,
			func(ctx context.Context, weights records.ModelTransferWeights,
				mint WeightsGrantMinter,
			) *exit.Error {
				return c.moveModelTransferWeights(ctx, weights, mint)
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

func (c *Orchestrator) kickModelTransferFinalizer(requestID string, attempt int64) {
	c.mu.Lock()
	if c.closing || c.transferRunning[requestID] {
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
			if transfer != nil && transfer.State == "failed" {
				return
			}
			if transfer == nil || (transfer.State != "failed" && transfer.State != "canceled") {
				time.AfterFunc(2*time.Second, func() {
					c.kickModelTransferFinalizer(requestID, attempt)
				})
				return
			}
		}
		request, readProblem := c.opt.Store.RequestRow(requestID)
		if readProblem != nil || request == nil {
			return
		}
		retainedAttempt, readProblem := c.opt.Store.AttemptRow(requestID, attempt)
		if readProblem != nil || retainedAttempt == nil {
			return
		}
		c.mu.Lock()
		var session *session
		if worker := c.workers[retainedAttempt.InstanceID]; worker != nil && worker.snapshotAcknowledged {
			session = c.sessions[worker.bootID]
		}
		c.mu.Unlock()
		if session == nil {
			time.AfterFunc(2*time.Second, func() { c.kickModelTransferFinalizer(requestID, attempt) })
			return
		}
		c.ackSettledOutcome(session, requestID, uint64(attempt))
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
	c.kickCheckpointUpload(requestID)
	return c.resumeModelTransferPublication(requestID)
}

func (c *Orchestrator) finishModelTransferRequest(requestID string, attempt int64) {
	c.kickCheckpointUpload(requestID)
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || request == nil || request.State == "canceled" {
		return
	}
	retainedAttempt, readProblem := c.opt.Store.AttemptRow(requestID, attempt)
	if readProblem != nil || retainedAttempt == nil || c.retainedPublication(*request, *retainedAttempt) {
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
	checkpointRequests, problem := c.opt.Store.CheckpointRequests()
	if problem != nil {
		return problem
	}
	for _, requestID := range checkpointRequests {
		c.kickCheckpointUpload(requestID)
	}
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
		if c.retainedPublication(*request, *attempt) {
			if transfer.State == "failed" {
				if request.Worker != "" {
					c.selectOrStart(*request)
				}
			} else {
				_ = c.resumeModelTransferPublication(request.ID)
			}
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
		request, readProblem := c.opt.Store.RequestRow(requestID)
		retainedAttempt, attemptProblem := c.opt.Store.AttemptRow(requestID, attempt)
		if readProblem != nil || attemptProblem != nil || request == nil || retainedAttempt == nil ||
			c.retainedPublication(*request, *retainedAttempt) {
			return
		}
		if problem := c.opt.Store.Closed(requestID, attempt); problem != nil {
			return
		}
		c.finishModelTransferRequest(requestID, attempt)
	}()
}
