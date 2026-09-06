package orchestrator

import (
	"bytes"
	"context"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// SourceCheckpointHost supplies the claimed pod's bounded byte-transfer lane.
// The publication owner supplies exact subjects and short-lived grants only.
type SourceCheckpointHost struct {
	BootID   string
	Page     func(context.Context, *pb.SourceCheckpointPageRequest) (*pb.SourceCheckpointPageResult, *exit.Error)
	Transfer func(context.Context, *pb.SourceCheckpointTransferRequest) (*pb.SourceCheckpointTransferStatus, *exit.Error)
}

func (c *Orchestrator) sourceCheckpointHost(request records.Request) (SourceCheckpointHost, *exit.Error) {
	s, problem := c.rentalControl(request.Worker)
	if problem != nil {
		return SourceCheckpointHost{}, problem
	}
	if s.host == nil {
		return SourceCheckpointHost{}, exit.Unavailablef("source checkpoint has no claimed PodHost lane")
	}
	return sourceCheckpointAdapter(s.host, s.claim), nil
}

func sourceCheckpointAdapter(host pb.PodHostClient, claim *pb.Claim) SourceCheckpointHost {
	return SourceCheckpointHost{BootID: claim.WorkerBootId,
		Page: func(ctx context.Context, request *pb.SourceCheckpointPageRequest) (*pb.SourceCheckpointPageResult, *exit.Error) {
			request.RecordOwnerEpoch, request.ControlStreamEpoch, request.WorkerBootId = recordOwnerEpoch, 0, claim.WorkerBootId
			answer, err := host.SourceCheckpointPage(ctx, &pb.SourceCheckpointPageCall{Claim: claim, Request: request})
			if err != nil {
				return nil, exit.Unavailablef("worker source checkpoint page is unavailable")
			}
			if answer == nil || answer.RecordOwnerEpoch != request.RecordOwnerEpoch || answer.ControlStreamEpoch != 0 ||
				answer.WorkerBootId != claim.WorkerBootId || answer.OperationId != request.OperationId || answer.Slot != request.Slot ||
				!bytes.Equal(answer.SourceSelectionDigest, request.SourceSelectionDigest) ||
				!bytes.Equal(answer.PlanDigest, request.PlanDigest) || !proto.Equal(answer.Head, request.Head) ||
				len(answer.Objects) > pb.MaxSourceCheckpointObjects || proto.Size(answer) > pb.MaxInlineControlBytes {
				return nil, exit.Named(exit.Structural, "model_transfer.source_checkpoint_page_invalid",
					"worker source checkpoint page changed its claimed subject or exceeded its bound")
			}
			if answer.SafeCode != "" {
				return nil, exit.Named(exit.Failed, answer.SafeCode, "%s", answer.SafeDetail)
			}
			return answer, nil
		},
		Transfer: func(ctx context.Context, request *pb.SourceCheckpointTransferRequest) (*pb.SourceCheckpointTransferStatus, *exit.Error) {
			request.RecordOwnerEpoch, request.ControlStreamEpoch, request.WorkerBootId = recordOwnerEpoch, 0, claim.WorkerBootId
			answer, err := host.SourceCheckpointTransfer(ctx, &pb.SourceCheckpointTransferCall{Claim: claim, Request: request})
			if err != nil {
				return nil, exit.Unavailablef("worker source checkpoint transfer is unavailable")
			}
			if answer == nil || answer.RecordOwnerEpoch != request.RecordOwnerEpoch || answer.ControlStreamEpoch != 0 ||
				answer.WorkerBootId != claim.WorkerBootId || answer.OperationId != request.OperationId || answer.Slot != request.Slot ||
				!bytes.Equal(answer.SourceSelectionDigest, request.SourceSelectionDigest) ||
				!bytes.Equal(answer.PlanDigest, request.PlanDigest) || !proto.Equal(answer.Head, request.Head) ||
				!proto.Equal(answer.Object, request.Object) || answer.TransferId != request.TransferId ||
				answer.GrantRevision != request.GrantRevision || answer.TransferredBytes > request.Object.Ref.Length {
				return nil, exit.Named(exit.Structural, "model_transfer.source_checkpoint_transfer_invalid",
					"worker source checkpoint transfer changed its claimed subject")
			}
			if answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_UPLOADED &&
				answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_ALREADY_PRESENT &&
				answer.State != pb.WeightsTransferState_WEIGHTS_TRANSFER_STATE_HELD {
				return nil, exit.Named(exit.Unavailable, "model_transfer.source_checkpoint_transfer_failed",
					"worker source checkpoint transfer failed (%s): %s", answer.SafeCode, answer.SafeDetail)
			}
			return answer, nil
		}}
}
