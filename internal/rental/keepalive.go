package rental

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	"google.golang.org/grpc/status"
)

// KeepAlive resets a ready rental's idle deadline once, with Status's keepalive. It uses only
// the rental's recorded address, pin and Creator key: it
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
	if err != nil {
		return out, exit.Unavailablef("rental keepalive was not acknowledged: %s", status.Convert(err).Message())
	}
	return acknowledged(row, frame.GetWorkerId(), frame.GetBootId(), frame.GetIdleDeadlineUnixMs())
}

// acknowledged accepts the pinned worker's future deadline, or zero when the
// rental has been used and no longer expires automatically.
func acknowledged(row *records.Rental, worker, boot string, deadline int64) (records.RentalKeepalive, *exit.Error) {
	if worker != row.ExpectedWorkerID || boot != row.ExpectedWorkerBootID {
		return records.RentalKeepalive{}, exit.New(exit.Conflict, "rental keepalive was answered by another worker or boot than the rental's")
	}
	now := time.Now().UnixMilli()
	if deadline != 0 && deadline <= now {
		return records.RentalKeepalive{}, exit.New(exit.Conflict, "the machine returned no future rental idle deadline")
	}
	return records.RentalKeepalive{WorkerID: worker, WorkerBootID: boot, AcknowledgedAtMS: now, IdleDeadlineMS: deadline}, nil
}
