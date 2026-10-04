package machinev1

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	"google.golang.org/grpc/status"
)

// KeepRentalAlive is one explicit idle reset on a locally recorded ready rental. Both
// daemon and foreground maintenance use its same pinned machine identity and owner key.
// It never starts this computer's machine, reads a Hub, retries or releases a rental.
func KeepRentalAlive(ctx context.Context, row *records.Rental, signer Signer) (records.RentalKeepalive, *exit.Error) {
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
	client, err := Dial(row.Address, pin.TLSConfig(), row.ExpectedWorkerID, signer)
	if err != nil {
		return out, exit.Unavailablef("rental keepalive could not open its pinned machine connection")
	}
	defer client.Close()
	frame, err := client.Keepalive(ctx)
	if problem := Skew(err); problem != nil {
		return out, problem
	}
	if err != nil {
		return out, exit.Unavailablef("rental keepalive was not acknowledged: %s", status.Convert(err).Message())
	}
	if frame.GetWorkerId() != row.ExpectedWorkerID || frame.GetBootId() != row.ExpectedWorkerBootID {
		return out, exit.New(exit.Conflict, "rental keepalive was answered by another worker or boot than the rental's")
	}
	acknowledged := time.Now().UnixMilli()
	if frame.GetIdleDeadlineUnixMs() <= acknowledged {
		return out, exit.New(exit.Conflict, "the machine returned no future rental idle deadline")
	}
	return records.RentalKeepalive{WorkerID: frame.GetWorkerId(), WorkerBootID: frame.GetBootId(),
		AcknowledgedAtMS: acknowledged, IdleDeadlineMS: frame.GetIdleDeadlineUnixMs()}, nil
}
