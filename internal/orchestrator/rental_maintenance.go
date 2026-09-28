package orchestrator

import (
	"context"
	"maps"
	"slices"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
)

// UseRental fences one client transport/preparation against maintenance. It is
// not an execution-idleness claim: Runtime must still refuse a restart while it
// owns queued or active execution. The release belongs to the actual connection.
// holder says who uses the rental, for a maintenance refusal to name.
func (c *Orchestrator) UseRental(id, holder string) (func(), *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closing || c.rentalMaintenance[id] {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its Runtime; preparation will resume afterward")
	}
	if problem := c.opt.Store.RuntimeUpdateHold(id); problem != nil {
		return nil, problem
	}
	if c.rentalUses[id] == nil {
		c.rentalUses[id] = map[uint64]string{}
	}
	c.rentalUseSeq++
	use := c.rentalUseSeq
	c.rentalUses[id][use] = holder
	var once sync.Once
	return func() {
		once.Do(func() {
			c.mu.Lock()
			delete(c.rentalUses[id], use)
			if len(c.rentalUses[id]) == 0 {
				delete(c.rentalUses, id)
			}
			c.mu.Unlock()
		})
	}, nil
}

// maintenanceBusyLocked names what a Runtime update of this rental would have to
// interrupt, or "". Callers hold c.mu.
func (c *Orchestrator) maintenanceBusyLocked(id string) string {
	switch {
	case c.closing:
		return "the daemon is stopping"
	case c.rentalMaintenance[id]:
		return "its Runtime is already updating"
	}
	holders := slices.Sorted(maps.Values(c.rentalUses[id]))
	return strings.Join(slices.Compact(holders), ", ")
}

// MaintainRental fences one rental against new transports and preparation while its
// Runtime is updated. Another rental remains independent. The caller records the durable
// update before entering this method.
func (c *Orchestrator) MaintainRental(ctx context.Context, id string,
	update func(context.Context, *WorkerConnection) *exit.Error,
) *exit.Error {
	if id == "" || update == nil || c.opt.Rentals == nil {
		return exit.New(exit.Validation, "maintenance requires one attached rental and an updater")
	}
	c.mu.Lock()
	if busy := c.maintenanceBusyLocked(id); busy != "" {
		c.mu.Unlock()
		return exit.Named(exit.Unavailable, "rental.maintenance_busy", "this rental is in use (%s); its Runtime was not changed; retry after that finishes", busy)
	}
	c.rentalMaintenance[id] = true
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.rentalMaintenance, id)
		c.mu.Unlock()
	}()
	target, problem := c.opt.Rentals(id)
	if problem != nil {
		return problem
	}
	if target == nil || target.Connection == nil || target.Connection.RentalID != id {
		return exit.New(exit.Conflict, "maintenance rental identity changed")
	}
	return update(ctx, target.Connection)
}
