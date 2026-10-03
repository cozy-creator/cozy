package machinev1

import (
	"context"
	"fmt"

	pb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Progress is one preparation stage and its bytes so far (zero when it moves none).
type Progress struct {
	Stage                 string
	BytesDone, BytesTotal uint64
}

// Prepare runs `spec` as a warm run (install and download, nothing executed) under `id`,
// reporting each progress event, and answers once the machine holds everything the spec
// names. A repeated id attaches to the same preparation.
func Prepare(ctx context.Context, client pb.MachineClient, id string, spec *pb.RunSpec, report func(Progress)) error {
	spec.Kind = pb.RunKind_RUN_KIND_WARM
	stream, err := client.Run(ctx, &pb.RunRequest{Id: id, Spec: spec})
	if err != nil {
		return err
	}
	for {
		event, err := stream.Recv()
		if err != nil {
			return err
		}
		if progress := event.GetProgress(); progress != nil && report != nil {
			report(Progress{Stage: progress.GetStage(), BytesDone: progress.GetBytesDone(), BytesTotal: progress.GetBytesTotal()})
		}
		if outcome := event.GetOutcome(); outcome != nil {
			if outcome.GetStatus() == "succeeded" {
				return nil
			}
			reason := outcome.GetReason()
			return fmt.Errorf("preparation %s: %s", outcome.GetStatus(), reason.GetMessage())
		}
	}
}
