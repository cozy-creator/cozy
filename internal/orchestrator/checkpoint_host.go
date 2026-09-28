package orchestrator

import (
	"bytes"
	"context"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// CheckpointHost supplies the claimed pod's bounded byte-transfer lane.
// The publication owner supplies exact subjects and short-lived grants only.
type CheckpointHost struct {
	BootID   string
	Page     func(context.Context, *pb.CheckpointPageRequest) (*pb.CheckpointPageResult, *exit.Error)
	Transfer func(context.Context, *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error)
}

func checkpointAdapter(host pb.PodHostClient, claim *pb.Claim) CheckpointHost {
	return CheckpointHost{BootID: claim.WorkerBootId,
		Page: func(ctx context.Context, request *pb.CheckpointPageRequest) (*pb.CheckpointPageResult, *exit.Error) {
			request.RecordOwnerEpoch, request.ControlStreamEpoch, request.WorkerBootId = claim.RecordOwnerEpoch, 0, claim.WorkerBootId
			answer, err := host.CheckpointPage(ctx, &pb.CheckpointPageCall{Claim: claim, Request: request})
			if err != nil {
				return nil, exit.Unavailablef("worker source checkpoint page is unavailable")
			}
			if answer == nil || answer.RecordOwnerEpoch != request.RecordOwnerEpoch || answer.ControlStreamEpoch != 0 ||
				answer.WorkerBootId != claim.WorkerBootId || !proto.Equal(answer.Subject, request.Subject) ||
				!bytes.Equal(answer.PlanDigest, request.PlanDigest) || !proto.Equal(answer.Head, request.Head) ||
				len(answer.Objects) > pb.MaxCheckpointObjects || proto.Size(answer) > pb.MaxInlineControlBytes {
				return nil, exit.Named(exit.Structural, "model_transfer.source_checkpoint_page_invalid",
					"worker source checkpoint page changed its claimed subject or exceeded its bound")
			}
			if answer.SafeCode != "" {
				return nil, exit.Named(exit.Failed, answer.SafeCode, "%s", answer.SafeDetail)
			}
			return answer, nil
		},
		Transfer: func(ctx context.Context, request *pb.CheckpointTransferRequest) (*pb.CheckpointTransferStatus, *exit.Error) {
			request.RecordOwnerEpoch, request.ControlStreamEpoch, request.WorkerBootId = claim.RecordOwnerEpoch, 0, claim.WorkerBootId
			answer, err := host.CheckpointTransfer(ctx, &pb.CheckpointTransferCall{Claim: claim, Request: request})
			if err != nil {
				return nil, exit.Unavailablef("worker source checkpoint transfer is unavailable")
			}
			if answer == nil || answer.RecordOwnerEpoch != request.RecordOwnerEpoch || answer.ControlStreamEpoch != 0 ||
				answer.WorkerBootId != claim.WorkerBootId || !proto.Equal(answer.Subject, request.Subject) ||
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
