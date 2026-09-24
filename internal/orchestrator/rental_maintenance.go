package orchestrator

import (
	"context"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// UseRental fences one client transport/preparation against maintenance. It is
// not an execution-idleness claim: Runtime must still refuse a restart while it
// owns queued or active execution. The release belongs to the actual connection.
func (c *Orchestrator) UseRental(id string) (func(), *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closing || c.rentalMaintenance[id] {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its Runtime; preparation will resume afterward")
	}
	if recorded, problem := c.opt.Store.RuntimeUpdate(id); problem != nil {
		return nil, problem
	} else if recorded != nil && recorded.Active() {
		return nil, exit.Named(exit.Unavailable, "rental.maintenance", "this rental has an unfinished Runtime update; preparation will resume after reconciliation")
	}
	c.rentalUses[id]++
	var once sync.Once
	return func() {
		once.Do(func() {
			c.mu.Lock()
			c.rentalUses[id]--
			if c.rentalUses[id] == 0 {
				delete(c.rentalUses, id)
			}
			c.mu.Unlock()
		})
	}, nil
}

// MaintainRental withdraws one rental from dispatch and gives its existing
// signed control authority to maintenance. Another rental remains independent.
// The caller records the durable update before entering this method and does
// not reopen dispatch until successful readback or confirmed rollback.
func (c *Orchestrator) MaintainRental(ctx context.Context, id string,
	update func(context.Context, *WorkerConnection) *exit.Error,
) *exit.Error {
	if id == "" || update == nil || c.opt.Rentals == nil {
		return exit.New(exit.Validation, "maintenance requires one attached rental and an updater")
	}
	c.mu.Lock()
	if c.closing || c.rentalMaintenance[id] || c.rentalUses[id] != 0 || c.ensuring[rentalInstanceID(id)] != nil {
		c.mu.Unlock()
		return exit.Named(exit.Unavailable, "rental.maintenance_busy", "this rental is still preparing or updating; retry after it finishes")
	}
	w := c.workers[rentalInstanceID(id)]
	if w != nil && !c.idleRentalWorkerLocked(w) {
		c.mu.Unlock()
		return exit.Named(exit.Unavailable, "rental.maintenance_busy", "this rental has active execution or uncollected results; its Runtime was not changed")
	}
	c.rentalMaintenance[id] = true
	if w != nil {
		w.stopping = true
	}
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.rentalMaintenance, id)
		c.mu.Unlock()
	}()
	if w != nil {
		// A remote worker has no local process to kill. This ends only its
		// previous controller connection, after atomically withdrawing it.
		c.stopClaimedWorker(w, StopGrace)
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

// RentalExecutionClaim reuses the rental's authenticated control authority for
// independent execution and preparation RPCs. A second Control Claim would fence
// the live session and cancel preparations on its connection.
func (c *Orchestrator) RentalExecutionClaim(ctx context.Context, id string) (*pb.Claim, *exit.Error) {
	instance, _, _, problem := c.ensureRentalContext(ctx, id)
	if problem != nil {
		return nil, problem
	}
	_, session, problem := c.localControl(instance)
	if problem != nil {
		return nil, problem
	}
	return proto.Clone(session.claim).(*pb.Claim), nil
}
