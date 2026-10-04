package orchestrator

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/workertls"
	"google.golang.org/grpc/status"
)

// KeepRentalAlive asks the rental's machine to reset its idle deadline once (Status with
// keepalive) and answers what it now holds. Runtime admission and running work are
// irrelevant to this manual owner action, and nothing is retried.
func (c *Orchestrator) KeepRentalAlive(ctx context.Context, id string) (records.RentalKeepalive, *exit.Error) {
	var out records.RentalKeepalive
	ctx, cancel := context.WithTimeout(ctx, hub.Timeout)
	defer cancel()
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing || c.opt.Signer == nil {
		return out, exit.Unavailablef("rental keepalive owner is unavailable")
	}
	row, problem := c.opt.Store.RentalRow(id)
	if problem != nil {
		return out, problem
	}
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
	signer, problem := c.opt.Signer()
	if problem != nil {
		return out, problem
	}
	client, err := machinev1.Dial(row.Address, pin.TLSConfig(), row.ExpectedWorkerID, signer)
	if err != nil {
		return out, exit.Unavailablef("rental keepalive could not open its pinned machine connection")
	}
	defer client.Close()
	frame, err := client.Keepalive(ctx)
	if problem := machinev1.Skew(err); problem != nil {
		return out, problem
	}
	if err != nil {
		return out, exit.Unavailablef("rental keepalive was not acknowledged: %s", status.Convert(err).Message())
	}
	if frame.GetWorkerId() != row.ExpectedWorkerID || frame.GetBootId() != row.ExpectedWorkerBootID {
		return out, exit.New(exit.Conflict, "rental keepalive was answered by another worker or boot than the rental's")
	}
	return records.RentalKeepalive{WorkerID: frame.GetWorkerId(), WorkerBootID: frame.GetBootId(),
		AcknowledgedAtMS: time.Now().UnixMilli(), IdleDeadlineMS: frame.GetIdleDeadlineUnixMs()}, nil
}
