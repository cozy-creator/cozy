package orchestrator

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Holder is what uses a rental: journaled work names its number, which `cozy run cancel`
// takes, and quick reads only what they do.
type Holder struct {
	Number int64
	What   string
}

func (h Holder) String() string {
	if h.Number > 0 {
		return fmt.Sprintf("#%d (%s)", h.Number, h.What)
	}
	return h.What
}

// UseRental fences one client transport/preparation against maintenance. It is
// not an execution-idleness claim: Runtime must still refuse a restart while it
// owns queued or active execution. The release belongs to the actual connection.
func (c *Orchestrator) UseRental(id string, holder Holder) (func(), *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closing || c.rentalMaintenance[id] {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its Runtime; preparation will resume afterward")
	}
	if problem := c.opt.Store.RuntimeUpdateHold(id); problem != nil {
		return nil, problem
	}
	if c.rentalUses[id] == nil {
		c.rentalUses[id] = map[uint64]Holder{}
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

// MaintenanceBlocker is why a Runtime update of this rental cannot start now, naming the
// work it would interrupt, or nil.
func (c *Orchestrator) MaintenanceBlocker(id string) *exit.Error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.maintenanceBlockerLocked(id)
}

func (c *Orchestrator) maintenanceBlockerLocked(id string) *exit.Error {
	switch {
	case c.closing:
		return exit.Named(exit.Unavailable, "rental.maintenance_busy", "the daemon is stopping; the Runtime was not changed")
	case c.rentalMaintenance[id]:
		return exit.Named(exit.Unavailable, "rental.maintenance_busy", "its Runtime is already updating")
	}
	if len(c.rentalUses[id]) == 0 {
		return nil
	}
	var names []string
	var numbers []int64
	for _, holder := range c.rentalUses[id] {
		names = append(names, holder.String())
		if holder.Number > 0 && !slices.Contains(numbers, holder.Number) {
			numbers = append(numbers, holder.Number)
		}
	}
	slices.Sort(names)
	slices.Sort(numbers)
	problem := exit.Named(exit.Unavailable, "rental.maintenance_busy", "blocked by %s; its Runtime was not changed", strings.Join(slices.Compact(names), ", "))
	if len(numbers) > 0 {
		cancels := make([]string, len(numbers))
		for i, number := range numbers {
			cancels[i] = fmt.Sprintf("cozy run cancel %d", number)
		}
		return problem.WithRemedy("%s, or retry after it finishes", strings.Join(cancels, "; "))
	}
	return problem.WithRemedy("retry after it finishes")
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
	if busy := c.maintenanceBlockerLocked(id); busy != nil {
		c.mu.Unlock()
		return busy
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
