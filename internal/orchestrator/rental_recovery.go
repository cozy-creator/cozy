package orchestrator

import "github.com/cozy-creator/cozy/internal/exit"

// rentalCanServe answers whether the rental a request is pinned to could still take it.
// A row that is absent, terminally failed, or released cannot, and that is a durable fact
// this host can read without asking anything remote — which matters, because the thing it
// would have to ask is the thing that is gone.
func (c *Orchestrator) rentalCanServe(rentalID string) bool {
	if rentalID == "" {
		return false
	}
	row, problem := c.opt.Store.RentalRow(rentalID)
	if problem != nil {
		// An unreadable store is not an observation about the pod. Leave the attempt alone.
		return true
	}
	return row != nil && row.State != "failed" && row.State != "released" && row.State != "release_requested"
}

// UseRental refuses a rental's use once Calcifer is stopping. Work a software update holds
// waits before it is sent (machineRuns.stepV1), never here: reads and cancels go through.
func (c *Orchestrator) UseRental(id, _ string) (func(), *exit.Error) {
	c.mu.Lock()
	closing := c.closing
	c.mu.Unlock()
	if closing {
		return nil, exit.Named(exit.Unavailable, "calcifer.stopping", "Calcifer is stopping")
	}
	return func() {}, nil
}
