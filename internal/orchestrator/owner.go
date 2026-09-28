package orchestrator

import (
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// THE RECORDOWNER SIDE of th-024's orientation (#436/#446/#454, renamed by #481 — bare
// "owner" is retired and the formal role is RecordOwner): the WORKER hosts `WorkerControl`
// and THIS side dials it, claims it, reconciles its ONE snapshot, and only then
// dispatches. One goroutine per worker owns the whole conversation; the `session` is that
// stream's sender half, fenced by the control stream epoch the worker minted.
//
// LAUNCH TIER (#437): record_owner_epoch is the constant 1 — this daemon is the one
// RecordOwner of every worker it spawns or connects to; the machinery that MINTS competing
// epochs is the hub's Wave-2 lease. Rented workers authenticate the channel with the
// provisioned Creator mTLS key.

const recordOwnerEpoch = 1

// recordOwnerID names this owner on Claim. Stable per daemon run is enough at launch:
// equal-epoch claims from the SAME RecordOwner are reconnects, anything else is refused.
const recordOwnerID = "cozy-local-client"

// RecordOwnerID and RecordOwnerEpoch are the owner every machine Claim names.
const (
	RecordOwnerID    = recordOwnerID
	RecordOwnerEpoch = recordOwnerEpoch
)

// dialWorker opens the channel: local sockets use the per-spawn proof; rented workers
// pin the exact readiness certificate and authenticate Claim with Creator's signature.
func dialWorker(addr string, remote *WorkerConnection) (*grpc.ClientConn, error) {
	if remote != nil && remote.CACert != "" {
		pin, err := workertls.LoadPin(remote.CACert)
		if err != nil {
			return nil, err
		}
		creds := credentials.NewTLS(pin.TLSConfig())
		return grpc.NewClient(addr, grpc.WithTransportCredentials(creds))
	}
	target := addr
	if strings.HasPrefix(addr, "unix:") || strings.HasPrefix(addr, "/") {
		target = "unix:" + strings.TrimPrefix(addr, "unix:")
	}
	return grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
}

// onClaimAck binds the claimed boot to the worker slot: identity checks, the durable
// binding, and the session registry (the ClaimAck is the flip's Register successor).
// ObservationFromClaimAck is what a rented worker's ClaimAck reads back about the pod.
func ObservationFromClaimAck(rentalID string, ack *pb.ClaimAck) RentalObservation {
	resources := ack.GetResources()
	return RentalObservation{
		RentalID: rentalID, Accelerator: resources.GetDeviceName(),
		DeviceCount: int(resources.GetDeviceCount()), Backend: resources.GetBackend(),
		WorkerInstance: ack.WorkerInstanceId, WorkerID: ack.WorkerId, WorkerBootID: ack.WorkerBootId,
	}
}
