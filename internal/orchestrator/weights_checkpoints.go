package orchestrator

import (
	"bytes"
	"context"
	"math"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func (c *Orchestrator) onWeightsCheckpoint(s *session, frame *pb.WeightsCheckpointFrame) {
	if c.fenced(s, frame.RecordOwnerEpoch, frame.ControlStreamEpoch, frame.WorkerBootId) {
		return
	}
	invocation, err := canonical.Spell(frame.InvocationSpecDigest)
	if err != nil {
		return
	}
	c.onWeightsTransaction(s, &pb.WeightsTransactionStatus{WeightsTransactionId: frame.WeightsTransactionId,
		RequestId: frame.RequestId, AttemptOrdinal: frame.AttemptOrdinal, InvocationSpecDigest: invocation, OutputSlot: frame.OutputSlot,
		WriterEpoch: frame.WriterEpoch, State: pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT,
		TensorfsDeclarationDigest: frame.TensorfsDeclarationDigest, Checkpoint: frame.Checkpoint, IntentReady: true})
}

// The event and the canonical Host snapshot feed the same current-attempt join.
func (c *Orchestrator) onWeightsTransaction(s *session, row *pb.WeightsTransactionStatus) {
	c.mu.Lock()
	current := !c.closing && s.bootID != "" && c.sessions[s.bootID] == s
	c.mu.Unlock()
	if !current {
		return
	}
	if row == nil || row.AttemptOrdinal == 0 || row.AttemptOrdinal > math.MaxInt64 || row.WriterEpoch == 0 || len(row.TensorfsDeclarationDigest) != 32 {
		return
	}
	if row.State != pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_INTENT && row.State != pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_RECEIPT {
		return
	}
	attempt, problem := c.opt.Store.AttemptRow(row.RequestId, int64(row.AttemptOrdinal))
	if problem != nil || attempt == nil || attempt.InstanceID != s.instanceID || attempt.InvocationDigest != row.InvocationSpecDigest {
		return
	}
	declared, problem := decodeWeightsOutputs(attempt.WeightsOutputs)
	if problem != nil {
		return
	}
	allowed := false
	for _, output := range declared {
		allowed = allowed || output.OutputID == row.OutputSlot
	}
	if !allowed {
		return
	}
	transfer, problem := c.opt.Store.ModelTransferOf(row.RequestId)
	if problem != nil {
		return
	}
	if transfer != nil {
		if problem := c.opt.Store.ObserveModelWeightsCheckpoint(s.instanceID, s.bootID, row); problem != nil {
			c.logf("weights checkpoint %s/%s refused (%s)", row.RequestId, row.OutputSlot, problem.ErrName())
			return
		}
	}
	copy := proto.Clone(row).(*pb.WeightsTransactionStatus)
	if row.State == pb.WeightsTransactionState_WEIGHTS_TRANSACTION_STATE_RECEIPT {
		copy.IntentReady = true
	}
	c.kickCheckpointUpload(row.RequestId, copy)
}

func (c *Orchestrator) readyWeightsCheckpoint(ctx context.Context, requestID string, row *pb.WeightsTransactionStatus) *exit.Error {
	req, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || req == nil {
		return problem
	}
	if req.State == "canceled" || req.Ordinal != int64(row.AttemptOrdinal) {
		return nil
	}
	attempt, problem := c.opt.Store.AttemptRow(requestID, int64(row.AttemptOrdinal))
	if problem != nil || attempt == nil {
		return problem
	}
	if attempt.State != "offered" && attempt.State != "accepted" && attempt.State != "recovered_open" {
		return nil
	}
	session, problem := c.rentalControl(req.Worker)
	if problem != nil {
		return problem
	}
	if session.instanceID != attempt.InstanceID || session.host == nil {
		return exit.Unavailablef("weights Ready has no current claimed Host")
	}
	invocation, err := canonical.Raw(row.InvocationSpecDigest)
	if err != nil {
		return exit.New(exit.Structural, "weights Ready has malformed invocation")
	}
	subject := &pb.WeightsCheckpointSubject{RequestId: requestID, InvocationSpecDigest: invocation, OutputSlot: row.OutputSlot,
		WeightsTransactionId: row.WeightsTransactionId, WriterEpoch: row.WriterEpoch, TensorfsDeclarationDigest: row.TensorfsDeclarationDigest}
	transfer, problem := c.opt.Store.ModelTransferOf(requestID)
	if problem != nil {
		return problem
	}
	var checkpoint *pb.CheckpointRef
	if transfer != nil {
		if transfer.State == "failed" || transfer.State == "canceling" || transfer.State == "canceled" || transfer.State == "completed" {
			return nil
		}
		if transfer.HasAcquisition() {
			// Producer dispatch already uses awaitSourceInputCustody. Recheck the exact same
			// authority before restoring dependent work; this pass never waits on its own uploader.
			progress, problem := c.opt.Store.ModelSourceProgress(requestID)
			if problem != nil {
				return problem
			}
			if len(progress) != len(transfer.SourceProfiles) {
				return exit.Unavailablef("source custody is incomplete before weights restore")
			}
			for _, slot := range progress {
				if slot.WorkerBootID != session.bootID || slot.Acknowledged == nil || *slot.Acknowledged != slot.Observed {
					return exit.Unavailablef("source custody is incomplete before weights restore")
				}
			}
		}
		if c.opt.ModelTransfers == nil {
			return exit.Unavailablef("weights checkpoint has no publication owner")
		}
		checkpoint, problem = c.opt.ModelTransfers.RestoreWeightsCheckpoint(ctx, requestID, checkpointAdapter(session.host, session.claim), subject)
		if problem != nil {
			return problem
		}
	}
	// Cancellation may win while native payload restoration is in flight.
	current, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil {
		return problem
	}
	if current == nil || current.State == "canceled" || current.Ordinal != req.Ordinal || current.Worker != req.Worker {
		return nil
	}
	answer, err := session.host.WeightsIntentReady(ctx, &pb.WeightsIntentReadyCall{Claim: session.claim, Request: &pb.WeightsIntentReadyRequest{
		RecordOwnerEpoch: session.claim.RecordOwnerEpoch, WorkerBootId: session.bootID, Weights: subject, AttemptOrdinal: row.AttemptOrdinal, Checkpoint: checkpoint}})
	if err != nil {
		return exit.Unavailablef("weights Ready awaits native checkpoint admission")
	}
	if answer == nil || answer.RecordOwnerEpoch != session.claim.RecordOwnerEpoch || answer.ControlStreamEpoch != 0 || answer.WorkerBootId != session.bootID ||
		answer.RequestId != requestID || answer.AttemptOrdinal != row.AttemptOrdinal || answer.OutputSlot != row.OutputSlot || answer.WeightsTransactionId != row.WeightsTransactionId ||
		answer.WriterEpoch != row.WriterEpoch || answer.Stage != pb.WeightsHostStage_WEIGHTS_HOST_STAGE_INTENT ||
		!bytes.Equal(answer.InvocationSpecDigest, subject.InvocationSpecDigest) || !bytes.Equal(answer.TensorfsDeclarationDigest, subject.TensorfsDeclarationDigest) ||
		(answer.Outcome != pb.WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_RECORDED && answer.Outcome != pb.WeightsHostOutcome_WEIGHTS_HOST_OUTCOME_REPLAYED) ||
		!proto.Equal(answer.Checkpoint, checkpoint) {
		return exit.New(exit.Structural, "weights Ready changed its exact current authority")
	}
	return nil
}
