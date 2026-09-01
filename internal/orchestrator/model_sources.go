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

func (c *Orchestrator) onModelSourceFileStatus(_ *session, frame *pb.ModelSourceFileStatus) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	transfer, problem := c.opt.Store.ModelTransferOf(frame.OperationId)
	if err != nil || problem != nil || transfer == nil || selection != transfer.SourceSelection {
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
	if problem := c.opt.Store.RecordModelTransferSourceStatus(records.ModelTransferSourceStatus{
		RequestID: frame.OperationId, Member: frame.Member, ObjectID: frame.ObjectId,
		Length: int64(frame.Length), CapabilityRevision: int64(frame.CapabilityRevision),
		State: state, Transferred: int64(frame.TransferredBytes), SafeCode: frame.SafeCode,
		SafeDetail: frame.SafeDetail}); problem != nil {
		return
	}
	if state == "failed" && frame.SafeCode != "capability_expired" {
		_ = c.opt.Store.FailModelTransfer(frame.OperationId, frame.SafeCode, frame.SafeDetail)
	}
	c.signalTransfer(frame.OperationId)
}

func (c *Orchestrator) onModelSourcePrepared(_ *session, frame *pb.ModelSourcePrepared) {
	selection, err := canonical.Spell(frame.SourceSelectionDigest)
	transfer, problem := c.opt.Store.ModelTransferOf(frame.OperationId)
	if err != nil || problem != nil || transfer == nil || selection != transfer.SourceSelection {
		return
	}
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
	if problem != nil || request == nil {
		return
	}
	rows := make([]records.ModelRef, 0, len(frame.Sources))
	for _, source := range frame.Sources {
		if source == nil || source.Manifest == nil || len(source.Manifest.Digest) != 32 ||
			source.Manifest.Length == 0 || source.Manifest.Length > uint64(^uint64(0)>>1) ||
			len(source.CheckpointEvidenceCanonicalBytes) == 0 ||
			len(source.CheckpointEvidenceCanonicalBytes) > pb.MaxCheckpointEvidenceBytes ||
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
