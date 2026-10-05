package rental

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// KeepAlive resets a ready rental's idle deadline once, on whatever its pod runs: a machine
// serving cozy.machine.v1 (Status with keepalive), or one that predates it (worker.v1
// PodHost.KeepRentalAlive), such as the old agent or a dual-arm pod on that arm. An older
// peer is never refused. It uses only the rental's recorded address, pin and Creator key: it
// never starts this computer's machine, reads a Hub, retries or releases the rental.
func KeepAlive(ctx context.Context, l home.Layout, row *records.Rental) (records.RentalKeepalive, *exit.Error) {
	var out records.RentalKeepalive
	if row == nil || row.State != "ready" {
		return out, exit.New(exit.Conflict, "keepalive requires a current ready rental")
	}
	if row.Address == "" || row.CertPath == "" || row.ExpectedWorkerID == "" || row.ExpectedWorkerBootID == "" {
		return out, exit.New(exit.Credential, "keepalive requires the rental's pinned machine, worker and boot identity")
	}
	pin, err := workertls.LoadPin(row.CertPath)
	if err != nil {
		return out, exit.New(exit.Credential, "the rental's machine certificate pin is unreadable: %s", err)
	}
	key, problem := CreatorIdentityFor(l, row.ID)
	if problem != nil {
		return out, problem
	}
	client, err := machinev1.Dial(row.Address, pin.TLSConfig(), row.ExpectedWorkerID, key.Signer())
	if err != nil {
		return out, exit.Unavailablef("rental keepalive could not open its pinned machine connection")
	}
	defer client.Close()
	frame, err := client.Keepalive(ctx)
	if status.Code(err) == codes.Unimplemented {
		return keepAliveWorkerV1(ctx, l, row, pin)
	}
	if err != nil {
		return out, exit.Unavailablef("rental keepalive was not acknowledged: %s", status.Convert(err).Message())
	}
	return acknowledged(row, frame.GetWorkerId(), frame.GetBootId(), frame.GetIdleDeadlineUnixMs())
}

// keepAliveWorkerV1 is the same reset on a machine that predates cozy.machine.v1, claimed with
// the rental's ClaimProof as that machine requires.
func keepAliveWorkerV1(ctx context.Context, l home.Layout, row *records.Rental, pin *workertls.Pin) (records.RentalKeepalive, *exit.Error) {
	var out records.RentalKeepalive
	proof, problem := ClaimProof(l)(&orchestrator.WorkerConnection{RentalID: row.ID, Addr: row.Address, CACert: row.CertPath,
		WorkerID: row.ExpectedWorkerID, WorkerBootID: row.ExpectedWorkerBootID}, orchestrator.RecordOwnerEpoch)
	if problem != nil {
		return out, problem
	}
	conn, err := grpc.NewClient(row.Address, grpc.WithTransportCredentials(credentials.NewTLS(pin.TLSConfig())))
	if err != nil {
		return out, exit.Unavailablef("rental keepalive could not open its pinned machine connection")
	}
	defer conn.Close()
	nonce := make([]byte, 16)
	_, _ = rand.Read(nonce)
	request := "keepalive-" + hex.EncodeToString(nonce)
	result, err := pb.NewPodHostClient(conn).KeepRentalAlive(ctx, &pb.KeepRentalAliveRequest{RequestId: request,
		Claim: &pb.Claim{RecordOwnerEpoch: orchestrator.RecordOwnerEpoch, RecordOwnerId: orchestrator.RecordOwnerID,
			WorkerId: row.ExpectedWorkerID, WorkerBootId: row.ExpectedWorkerBootID, WireMinor: pb.WireMinor, Proof: proof}})
	if status.Code(err) == codes.Unimplemented {
		return out, exit.Named(exit.Unavailable, "rental.keepalive_unserved",
			"the rental's machine answers neither cozy.machine.v1 Status nor worker.v1 KeepRentalAlive")
	}
	if err != nil {
		return out, exit.Unavailablef("rental keepalive was not acknowledged: %s", status.Convert(err).Message())
	}
	if result.GetRequestId() != request {
		return out, exit.New(exit.Conflict, "rental keepalive acknowledged another request")
	}
	return acknowledged(row, result.GetWorkerId(), result.GetWorkerBootId(), result.GetIdleDeadlineUnixMs())
}

// acknowledged accepts an answer from the rental's own worker and boot that names a deadline
// still ahead; Creator schedules from its own clock, so the reset is stamped here.
func acknowledged(row *records.Rental, worker, boot string, deadline int64) (records.RentalKeepalive, *exit.Error) {
	if worker != row.ExpectedWorkerID || boot != row.ExpectedWorkerBootID {
		return records.RentalKeepalive{}, exit.New(exit.Conflict, "rental keepalive was answered by another worker or boot than the rental's")
	}
	now := time.Now().UnixMilli()
	if deadline <= now {
		return records.RentalKeepalive{}, exit.New(exit.Conflict, "the machine returned no future rental idle deadline")
	}
	return records.RentalKeepalive{WorkerID: worker, WorkerBootID: boot, AcknowledgedAtMS: now, IdleDeadlineMS: deadline}, nil
}
