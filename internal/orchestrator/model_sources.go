package orchestrator

import (
	"context"
	"encoding/hex"
	"sort"

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
		return nil, exit.New(exit.NotFound, "rental %s has no attached worker", rentalID)
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
		"VERIFIED": "verified", "FAILED": "failed"}[state]
	if state == "" {
		return
	}
	previous := ""
	if statuses, problem := c.opt.Store.ModelTransferSourceStatuses(frame.OperationId); problem == nil {
		for _, status := range statuses {
			if status.Member == frame.Member {
				previous = status.State
			}
		}
	}
	if problem := c.opt.Store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
		RequestID: frame.OperationId, Member: frame.Member, ObjectID: frame.ObjectId,
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
	if state == "failed" && frame.SafeCode != "capability_expired" {
		_ = c.opt.Store.FailModelTransfer(frame.OperationId, frame.SafeCode, frame.SafeDetail)
	}
	if statuses, problem := c.opt.Store.ModelTransferSourceStatuses(frame.OperationId); problem == nil {
		var transferred, total int64
		verified := 0
		for _, status := range statuses {
			transferred += status.Transferred
			total += status.Length
			if status.State == "verified" {
				verified++
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
		// A member's state change is one line in the daemon log; byte progress is the
		// live frame's alone. A queued transfer must be legible from the log (cl-099).
		if state != previous {
			detail := ""
			if state == "failed" {
				detail = " (" + frame.SafeCode + ": " + frame.SafeDetail + ")"
			}
			c.logf("model transfer %s: source %s %s on %s%s; %d of %d verified, %d/%d B",
				frame.OperationId, frame.Member, state, s.instanceID, detail, verified,
				len(transfer.SourceFiles), transferred, total)
		}
	}
	c.signalTransfer(frame.OperationId)
}

func (c *Orchestrator) onModelSourcePrepared(s *session, frame *pb.ModelSourcePrepared) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	transfer, problem := c.opt.Store.ModelTransferOf(frame.OperationId)
	if err != nil || problem != nil || transfer == nil || selection != transfer.SourceSelection {
		return
	}
	c.logf("model transfer %s: source prepare %s on %s (%d source(s)) %s %s", frame.OperationId,
		trimEnum(pb.ModelSourcePrepareOutcome_name[int32(frame.Outcome)], "MODEL_SOURCE_PREPARE_OUTCOME_"),
		s.instanceID, len(frame.Sources), frame.SafeCode, frame.SafeDetail)
	if frame.Outcome == pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REFUSED {
		_ = c.opt.Store.FailModelTransfer(frame.OperationId, frame.SafeCode, frame.SafeDetail)
		c.signalTransfer(frame.OperationId)
		return
	}
	if frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_PREPARED &&
		frame.Outcome != pb.ModelSourcePrepareOutcome_MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED {
		return
	}
	request, problem := c.opt.Store.RequestRow(frame.OperationId)
	if problem != nil || request == nil || request.Worker == "" ||
		rentalInstanceID(request.Worker) != s.instanceID {
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
	if problem := c.opt.Store.CompleteModelTransferMaterialization(frame.OperationId, rows); problem != nil {
		return
	}
	c.signalTransfer(frame.OperationId)
}
