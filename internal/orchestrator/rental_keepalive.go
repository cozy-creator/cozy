package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/records"
)

// RentalSignerSource is the owner key that authorizes calls on one rental's machine.
type RentalSignerSource func(rentalID string) (machinev1.Signer, *exit.Error)

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
	if closing || c.opt.RentalSigner == nil {
		return out, exit.Unavailablef("rental keepalive owner is unavailable")
	}
	row, problem := c.opt.Store.RentalRow(id)
	if problem != nil {
		return out, problem
	}
	signer, problem := c.opt.RentalSigner(id)
	if problem != nil {
		return out, problem
	}
	return machinev1.KeepRentalAlive(ctx, row, signer)
}
