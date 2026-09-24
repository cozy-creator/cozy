package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// KeepRentalAlive uses the existing signed claim: a second owner claim would fence
// preparation. This is only called by the explicit manual command, never a loop.
func (c *Orchestrator) KeepRentalAlive(ctx context.Context, id, requestID string) (*pb.KeepRentalAliveResult, *exit.Error) {
	if requestID == "" || len(requestID) > pb.MaxRentalKeepaliveRequestIDBytes {
		return nil, exit.New(exit.Validation, "keepalive requires a bounded request ID")
	}
	instance, _, _, problem := c.ensureRentalContext(ctx, id)
	if problem != nil {
		return nil, problem
	}
	_, session, problem := c.localControlContext(ctx, instance)
	if problem != nil {
		return nil, problem
	}
	if session.host == nil {
		return nil, exit.New(exit.Conflict, "keepalive requires an attached rental Host")
	}
	info, err := session.host.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil {
		return nil, exit.Unavailablef("rental keepalive capability could not be confirmed: %s", err)
	}
	if !info.GetSupportsRentalKeepalive() {
		return nil, exit.New(exit.Conflict, "worker does not implement the required rental keepalive contract")
	}
	claim := proto.Clone(session.claim).(*pb.Claim)
	result, err := session.host.KeepRentalAlive(ctx, &pb.KeepRentalAliveRequest{Claim: claim, RequestId: requestID})
	if err != nil {
		return nil, exit.Unavailablef("rental keepalive was not acknowledged: %s", err)
	}
	if result.GetRequestId() != requestID || result.GetWorkerId() != claim.WorkerId || result.GetWorkerBootId() != claim.WorkerBootId {
		return nil, exit.New(exit.Conflict, "rental keepalive acknowledgment does not match the current owner request and worker boot")
	}
	return result, nil
}
