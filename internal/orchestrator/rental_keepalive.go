package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// KeepRentalAlive authenticates the recorded Host directly. Runtime protocol
// admission, availability and control ownership are irrelevant to this manual
// Host operation; opening a new Control claim would also fence preparation.
func (c *Orchestrator) KeepRentalAlive(ctx context.Context, id, requestID string) (*pb.KeepRentalAliveResult, *exit.Error) {
	if requestID == "" || len(requestID) > pb.MaxRentalKeepaliveRequestIDBytes {
		return nil, exit.New(exit.Validation, "keepalive requires a bounded request ID")
	}
	// This unary control acknowledgment uses the existing network-control bound.
	// It never cancels application work or retries the user action automatically.
	ctx, cancel := context.WithTimeout(ctx, hub.Timeout)
	defer cancel()
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing || c.opt.RentalClaimProof == nil {
		return nil, exit.Unavailablef("rental keepalive owner is unavailable")
	}
	row, problem := c.opt.Store.RentalRow(id)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.State != "ready" {
		return nil, exit.New(exit.Conflict, "keepalive requires a current ready rental")
	}
	remote := &WorkerConnection{RentalID: row.ID, Addr: row.Address, CACert: row.CertPath,
		WorkerID: row.ExpectedWorkerID, WorkerBootID: row.ExpectedWorkerBootID}
	if remote.RentalID != id || remote.Addr == "" || remote.CACert == "" || remote.WorkerID == "" || remote.WorkerBootID == "" {
		return nil, exit.New(exit.Credential, "keepalive requires the rental's pinned Host, worker and boot identity")
	}
	proof, problem := c.opt.RentalClaimProof(remote, recordOwnerEpoch)
	if problem != nil {
		return nil, problem
	}
	conn, err := dialWorker(remote.Addr, remote)
	if err != nil {
		return nil, exit.Unavailablef("rental keepalive could not open its pinned Host connection")
	}
	defer conn.Close()
	claim := &pb.Claim{RecordOwnerEpoch: recordOwnerEpoch, RecordOwnerId: recordOwnerID,
		WorkerId: remote.WorkerID, WorkerBootId: remote.WorkerBootID, WireMinor: pb.WireMinor, Proof: proof}
	// KeepRentalAlive itself is the Host-only capability boundary. ProtocolInfo
	// advertises the execution intersection and may fail while Runtime is down.
	result, err := pb.NewPodHostClient(conn).KeepRentalAlive(ctx, &pb.KeepRentalAliveRequest{Claim: claim, RequestId: requestID})
	if status.Code(err) == codes.Unimplemented {
		return nil, exit.New(exit.Conflict, "Host does not implement the required rental keepalive contract")
	}
	if err != nil {
		return nil, exit.Unavailablef("rental keepalive was not acknowledged: %s", err)
	}
	if result.GetRequestId() != requestID || result.GetWorkerId() != claim.WorkerId || result.GetWorkerBootId() != claim.WorkerBootId {
		return nil, exit.New(exit.Conflict, "rental keepalive acknowledgment does not match the current owner request and worker boot")
	}
	return result, nil
}
