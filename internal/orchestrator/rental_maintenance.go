package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
)

// UseRental refuses a rental's use while its software update runs, so work waits and lands on
// the new software. The machine itself waits for its running work before it restarts, so a use
// already open is never fenced against an update.
func (c *Orchestrator) UseRental(id, _ string) (func(), *exit.Error) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing {
		return nil, exit.Named(exit.Unavailable, "daemon.stopping", "the daemon is stopping")
	}
	if problem := c.opt.Store.RuntimeUpdateHold(id); problem != nil {
		return nil, problem
	}
	return func() {}, nil
}

// MaintainRental runs a rental's software update over its current connection. The caller
// records the durable update, which holds the rental, before entering it.
func (c *Orchestrator) MaintainRental(ctx context.Context, id string,
	update func(context.Context, *WorkerConnection) *exit.Error,
) *exit.Error {
	if id == "" || update == nil || c.opt.Rentals == nil {
		return exit.New(exit.Validation, "maintenance requires one attached rental and an updater")
	}
	target, problem := c.opt.Rentals(id)
	if problem != nil {
		return problem
	}
	if target == nil || target.Connection == nil || target.Connection.RentalID != id {
		return exit.New(exit.Conflict, "maintenance rental identity changed")
	}
	return update(ctx, target.Connection)
}
