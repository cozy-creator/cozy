package orchestrator

import (
	"context"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ModelSourceCapability is refreshed privileged access, never durable request meaning.
type ModelSourceCapability struct {
	Member, ObjectID, URL string
	Length                int64
	Provider              pb.ModelSourceProvider
	ExpiresAtUnix         uint64
}

func (c *Orchestrator) transferChannel(operationID string) chan struct{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := c.transferWake[operationID]
	if ch == nil {
		ch = make(chan struct{}, 1)
		c.transferWake[operationID] = ch
	}
	return ch
}

func (c *Orchestrator) signalTransfer(operationID string) {
	select {
	case c.transferChannel(operationID) <- struct{}{}:
	default:
	}
}

func (c *Orchestrator) signalAllTransfers() {
	c.mu.Lock()
	channels := make([]chan struct{}, 0, len(c.transferWake))
	for _, ch := range c.transferWake {
		channels = append(channels, ch)
	}
	c.mu.Unlock()
	for _, ch := range channels {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

func (c *Orchestrator) waitTransfer(ctx context.Context, operationID string) *exit.Error {
	select {
	case <-c.transferChannel(operationID):
		return nil
	case <-ctx.Done():
		return exit.New(exit.Canceled, "model transfer %s stopped: %s", operationID, ctx.Err())
	case <-c.done:
		return exit.Unavailablef("the daemon stopped while model transfer %s was active", operationID)
	}
}

func (c *Orchestrator) rentalControl(rentalID string) (*session, *exit.Error) {
	instanceID := rentalInstanceID(rentalID)
	c.mu.Lock()
	w := c.workers[instanceID]
	var s *session
	if w != nil && w.snapshotAcknowledged {
		s = c.sessions[w.bootID]
	}
	c.mu.Unlock()
	if w == nil || w.spec.Connection == nil || w.spec.Connection.RentalID != rentalID {
		return nil, exit.Unavailablef("rental %s has no attached worker", rentalID)
	}
	if s == nil {
		return nil, exit.Unavailablef("rental %s has no reconciled WorkerControl session", rentalID)
	}
	return s, nil
}

func (c *Orchestrator) onModelSourceFileStatus(s *session, frame *pb.ModelSourceFileStatus) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	transfer, problem := c.opt.Store.ModelTransferOf(frame.OperationId)
	if err != nil || problem != nil || transfer == nil || selection != transfer.SourceSelection {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.OperationId)
	if problem != nil || request == nil || request.Worker == "" ||
		rentalInstanceID(request.Worker) != s.instanceID {
		return
	}
	var expected *records.ModelTransferSourceFile
	for i := range transfer.SourceFiles {
		if transfer.SourceFiles[i].Member == frame.Member {
			expected = &transfer.SourceFiles[i]
			break
		}
	}
	if expected == nil || frame.ObjectId != "sha256:"+expected.SHA256 ||
		frame.Length != uint64(expected.Length) || frame.TransferredBytes > frame.Length {
		return
	}
	state := trimEnum(pb.ModelSourceFileState_name[int32(frame.State)], "MODEL_SOURCE_FILE_STATE_")
	state = map[string]string{"ACCEPTED": "accepted", "DOWNLOADING": "downloading",
		"VERIFIED": "verified", "CONVERTED": "converted", "FAILED": "failed"}[state]
	if state == "" {
		return
	}
	previous, previousCode := "", ""
	var previousRevision int64
	if statuses, problem := c.opt.Store.ModelTransferSourceStatuses(frame.OperationId); problem == nil {
		for _, status := range statuses {
			if status.Member == frame.Member {
				previous, previousCode, previousRevision = status.State, status.SafeCode, status.CapabilityRevision
			}
		}
	}
	if problem := c.opt.Store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
		RequestID: frame.OperationId, Member: frame.Member, ObjectID: frame.ObjectId, WorkerBootID: s.bootID,
		Length: int64(frame.Length), CapabilityRevision: int64(frame.CapabilityRevision),
		State: state, Transferred: int64(frame.TransferredBytes), SafeCode: frame.SafeCode,
		SafeDetail: frame.SafeDetail}); problem != nil {
		// The store keeps a terminal verdict that arrived under a superseded revision, so
		// what reaches here is a status the row genuinely did not need. It is still not
		// dropped in silence when it carried a verdict or an identity this owner did not
		// expect: the pod said something about a member, and the log is where a human
		// looks for it. An ordinary late byte frame is the one case worth staying quiet
		// about — it repeats by the thousand and decides nothing.
		if state == "failed" || problem.ErrName() != "model_transfer.source_status_superseded" {
			c.logf("model transfer %s: source %s reported %s under capability revision %d "+
				"(%s: %s) and the row did not take it: %s", frame.OperationId, frame.Member,
				state, frame.CapabilityRevision, frame.SafeCode, frame.SafeDetail, problem.Message)
		}
		return
	}
	// EVERY FAILED IS THE POD'S LAST WORD. This used to except `capability_expired`, on
	// the theory that a lapsed credential is answered by re-issuing under a fresh one.
	// Nothing emits that code — it appears in this repository once, in the line that read
	// it, and nowhere in tensorhub, cozy-runtime, tensorfs or the protocol. It is the same
	// retired exception host.go:264 already documents removing: the download delegation was
	// deleted (owner ruling 2026-09-03), so a fetch can no longer become unauthorized by
	// running long. Left in place it is worse than dead — `prepareModelTransferRemote`
	// holds a frozen capability set for its whole life, so the re-issue this excepted was
	// never built, and a FAILED that does not fail the transfer is now a member nobody
	// will ever state again.
	if state == "failed" {
		_ = c.opt.Store.FailModelTransfer(frame.OperationId, frame.SafeCode, frame.SafeDetail)
	}
	if statuses, problem := c.opt.Store.ModelTransferSourceStatuses(frame.OperationId); problem == nil {
		var transferred, total int64
		fulfilled := 0
		for _, status := range statuses {
			if status.State == "converted" && status.WorkerBootID == s.bootID {
				total += status.Transferred
			} else {
				total += status.Length
			}
			if status.WorkerBootID != s.bootID {
				continue
			}
			transferred += status.Transferred
			if status.State == "verified" || status.State == "converted" {
				fulfilled++
			}
		}
		value := map[string]any{"stage": "source download", "member": frame.Member,
			"state": state, "transferred_bytes": transferred, "total_bytes": total}
		// The delivery host, HTTP status and attempt reached the status row and the pod
		// log but never the operator's live view, so a throttled origin and a slow link
		// looked identical while a transfer was running. They ride every frame because
		// the row keeps one record per member and a later byte frame would blank them.
		if frame.SafeCode != "" {
			value["safe_code"] = frame.SafeCode
		}
		if frame.SafeDetail != "" {
			value["safe_detail"] = frame.SafeDetail
		}
		if total > 0 {
			value["fraction"] = float64(transferred) / float64(total)
		}
		c.publishTransferProgress(frame.OperationId, value)
		c.ObservePhase(frame.OperationId, PhaseSample{Name: PhaseDownloading, Detail: "source transfer", HasBytes: true, Moved: uint64(transferred), Total: uint64(total)})
		// A member's state change is one line in the daemon log; byte progress is the
		// live frame's alone. A queued transfer must be legible from the log (cl-099).
		if state != previous {
			detail := ""
			if state == "failed" {
				detail = " (" + frame.SafeCode + ": " + frame.SafeDetail + ")"
			}
			c.logf("model transfer %s: source %s %s on %s%s; %d of %d fulfilled, %d/%d B",
				frame.OperationId, frame.Member, state, s.instanceID, detail, fulfilled,
				len(transfer.SourceFiles), transferred, total)
		}
	}
	// THE WAKE IS FOR FACTS THE PREPARER CAN ACT ON. It used to fire on every accepted
	// frame, byte progress included, and `prepareModelTransferRemote` answered each one by
	// re-stating the member — the pod's own echo driving the owner to talk again. Byte
	// counts reach the live view above, which is where they were already going; the
	// preparer decides on a member's STATE. Session attach, session drop and cancellation
	// each signal on their own, so nothing it waits on is left without a wake.
	if state != previous || int64(frame.CapabilityRevision) != previousRevision || frame.SafeCode != previousCode {
		c.signalTransfer(frame.OperationId)
	}
}

type sourcePreparationBackoff struct {
	stream string
	until  time.Time
	delay  time.Duration
}

func (c *Orchestrator) onModelSourcePrepared(s *session, frame *pb.ModelSourcePrepared) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	transfer, problem := c.opt.Store.ModelTransferOf(frame.OperationId)
	if err != nil || problem != nil || transfer == nil || selection != transfer.SourceSelection {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.OperationId)
	if problem != nil || request == nil || request.Worker == "" ||
		rentalInstanceID(request.Worker) != s.instanceID {
		return
	}
	if frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED &&
		frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE &&
		frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED &&
		frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED {
		return
	}
	defer func() {
		c.mu.Lock()
		c.sourcePrepareReplies[frame.OperationId]++
		c.mu.Unlock()
		c.signalTransfer(frame.OperationId)
	}()
	if frame.Outcome == pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED && frame.SafeCode == "model_source_preparer_unavailable" {
		stream := fmt.Sprintf("%s/%d", s.bootID, s.epoch)
		c.mu.Lock()
		prior := c.sourcePrepareBlocked[frame.OperationId]
		delay := time.Second
		if prior.stream == stream {
			delay = min(30*time.Second, max(time.Second, prior.delay*2))
		}
		retry := sourcePreparationBackoff{stream: stream, delay: delay, until: time.Now().Add(delay)}
		c.sourcePrepareBlocked[frame.OperationId] = retry
		c.mu.Unlock()
		time.AfterFunc(delay, func() {
			c.mu.Lock()
			current := c.sourcePrepareBlocked[frame.OperationId]
			wake := !c.closing && current.until == retry.until
			c.mu.Unlock()
			if wake {
				c.signalTransfer(frame.OperationId)
			}
		})
		c.ObservePhase(frame.OperationId, PhaseSample{Name: PhasePreparing, Detail: "model_source_preparer_unavailable; retrying"})
		c.logf("model transfer %s: Runtime preparer unavailable; retrying in %s or on new source/worker progress", frame.OperationId, delay)
		return
	}
	c.mu.Lock()
	delete(c.sourcePrepareBlocked, frame.OperationId)
	c.mu.Unlock()
	checkpoints := make([]records.ModelSourceCheckpoint, 0, len(frame.Checkpoints))
	for _, checkpoint := range frame.Checkpoints {
		if checkpoint == nil || checkpoint.Head == nil || len(checkpoint.Head.Digest) != 32 ||
			len(checkpoint.PlanDigest) != 32 || checkpoint.Head.Length == 0 ||
			checkpoint.Head.Length > uint64(^uint64(0)>>1) || checkpoint.Index > uint64(^uint64(0)>>1) ||
			checkpoint.Bytes > uint64(^uint64(0)>>1) {
			_ = c.opt.Store.FailModelTransfer(frame.OperationId, "model_transfer.source_checkpoint_invalid",
				"worker returned malformed source checkpoint progress")
			return
		}
		checkpoints = append(checkpoints, records.ModelSourceCheckpoint{
			Slot: checkpoint.Slot, HeadID: "sha256:" + hex.EncodeToString(checkpoint.Head.Digest),
			HeadLength: int64(checkpoint.Head.Length), PlanDigest: "sha256:" + hex.EncodeToString(checkpoint.PlanDigest),
			Index: int64(checkpoint.Index), Bytes: int64(checkpoint.Bytes)})
	}
	if len(checkpoints) > 0 {
		if problem := c.opt.Store.ObserveModelSourceCheckpoints(frame.OperationId, selection, s.bootID, checkpoints); problem != nil {
			_ = c.opt.Store.FailModelTransfer(frame.OperationId, problem.ErrName(), problem.Message)
			return
		}
		var converted uint64
		for _, checkpoint := range checkpoints {
			converted += uint64(checkpoint.Bytes)
		}
		c.ObservePhase(frame.OperationId, PhaseSample{Name: PhasePreparing, Detail: "source conversion", HasBytes: true, Moved: converted})
		c.kickSourceCheckpointUpload(frame.OperationId)
	}
	c.logf("model transfer %s: source prepare %s on %s (%d source(s)) %s %s", frame.OperationId,
		trimEnum(pb.ModelSourcePrepareOutcome_name[int32(frame.Outcome)], "MODEL_SOURCE_PREPARE_OUTCOME_"),
		s.instanceID, len(frame.Sources), frame.SafeCode, frame.SafeDetail)
	if frame.Outcome == pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED {
		_ = c.opt.Store.FailModelTransfer(frame.OperationId, frame.SafeCode, frame.SafeDetail)
		return
	}
	if frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED &&
		frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED {
		return
	}
	rows := make([]records.ModelRef, 0, len(frame.Sources))
	for _, source := range frame.Sources {
		if source == nil || source.Manifest == nil || len(source.Manifest.Digest) != 32 ||
			source.Manifest.Length == 0 || source.Manifest.Length > uint64(^uint64(0)>>1) ||
			transfer.SourceProfiles[source.Slot] != source.Profile {
			return
		}
		rows = append(rows, records.ModelRef{Package: request.Package, Slot: source.Slot,
			Model:          frame.OperationId + "/" + source.Slot,
			Manifest:       "sha256:" + hex.EncodeToString(source.Manifest.Digest),
			ManifestLength: int64(source.Manifest.Length)})
	}
	if len(rows) != len(transfer.SourceProfiles) {
		return
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Slot < rows[j].Slot })
	for i := 1; i < len(rows); i++ {
		if rows[i-1].Slot >= rows[i].Slot {
			return
		}
	}
	if problem := c.opt.Store.CompleteModelTransferMaterialization(frame.OperationId, rows, s.bootID); problem != nil {
		return
	}
}

// Only remotely acknowledged heads travel back to a worker. Local observations
// may refer to bytes that vanished with the previous pod.
func (c *Orchestrator) sourceCheckpointHeads(requestID, bootID string) ([]*pb.ModelSourceCheckpoint, string, string, *exit.Error) {
	progress, problem := c.opt.Store.ModelSourceProgress(requestID)
	if problem != nil {
		return nil, "", "", problem
	}
	var out []*pb.ModelSourceCheckpoint
	var heads, observed []string
	for _, slot := range progress {
		if slot.WorkerBootID == bootID {
			observed = append(observed, slot.Observed.Slot+"="+slot.Observed.HeadID)
		}
		checkpoint := slot.Acknowledged
		if checkpoint == nil {
			continue
		}
		head, headErr := canonical.Raw(checkpoint.HeadID)
		plan, planErr := canonical.Raw(checkpoint.PlanDigest)
		if headErr != nil || planErr != nil {
			return nil, "", "", exit.Internalf("stored source checkpoint has malformed identity")
		}
		out = append(out, &pb.ModelSourceCheckpoint{Slot: checkpoint.Slot,
			Head: &pb.Ref{Digest: head, Length: uint64(checkpoint.HeadLength)}, PlanDigest: plan,
			Index: uint64(checkpoint.Index), Bytes: uint64(checkpoint.Bytes)})
		heads = append(heads, checkpoint.Slot+"="+checkpoint.HeadID)
	}
	return out, strings.Join(heads, ","), strings.Join(observed, ","), nil
}
