package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func (c *Orchestrator) activeParentFor(req records.Request) (*pb.JobDirective, *exit.Error) {
	if req.ParentRequestID == "" {
		return nil, nil
	}
	parent, problem := c.retainedOrchestrationParent(req)
	if problem != nil {
		return nil, problem
	}
	return c.jobDirective(parent), nil
}

// A private worker has one CPU parent and one ordinary execution slot. Preparing
// an unrelated root cannot displace either; an owned child may retain its parent.
// Callers hold c.mu, so an owner-side offer cannot cross this check unnoticed.
func (c *Orchestrator) rentalPreparationAllowedLocked(w *worker, req records.Request) *exit.Error {
	if w == nil || w.exited || w.stopping || !w.snapshotAcknowledged {
		return exit.Unavailablef("rental preparation awaits its reconciled worker")
	}
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil {
		return problem
	}
	if current == nil || (current.State != "submitted" && current.State != "queued") || current.Worker != req.Worker {
		return exit.Unavailablef("rental preparation request is no longer queued on this worker")
	}
	req = *current
	parent, problem := c.activeParentFor(req)
	if problem != nil {
		return problem
	}
	attempts, problem := c.opt.Store.OpenAttemptsOf(w.instanceID)
	if problem != nil {
		return problem
	}
	allowed := 0
	for _, attempt := range attempts {
		if parent != nil && attempt.RequestID == req.ParentRequestID {
			allowed++
			continue
		}
		return exit.Named(exit.Unavailable, "rental.execution_busy", "rental preparation waits for active request %s", attempt.RequestID)
	}
	if allowed > 1 || w.held > allowed || w.unacked != 0 || w.reservedJobs != 0 || w.seats.reserved != 0 {
		return exit.Unavailablef("rental preparation waits for its current execution and outcome obligations")
	}
	for _, offer := range c.offers {
		if offer.worker == w {
			return exit.Unavailablef("rental preparation waits for an outstanding execution offer")
		}
	}
	return nil
}

func (c *Orchestrator) claimRentalPreparation(instance string, req records.Request) *exit.Error {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[instance]
	if problem := c.rentalPreparationAllowedLocked(w, req); problem != nil {
		return problem
	}
	parent, problem := c.activeParentFor(req)
	if problem != nil {
		return problem
	}
	w.preparingRequest, w.preparingReady = req.ID, false
	w.orchestrationParent = parent
	return nil
}

// A parent awaiting its child cannot complete behind an unrelated root's FIFO
// claim. Only children of an exact still-active CPU parent receive this exception.
func (c *Orchestrator) activeChild(req records.Request) bool {
	if req.Worker == "" || req.ParentRequestID == "" {
		return false
	}
	_, problem := c.retainedOrchestrationParent(req)
	return problem == nil
}

// An unrelated root cannot consume an already-prepared ordinary lane while a
// CPU parent owns the machine's composition. This also covers unpinned claims.
func (c *Orchestrator) rentalParentAllows(w *worker, req records.Request) bool {
	if w.spec.Connection == nil {
		return true
	}
	attempts, problem := c.opt.Store.OpenAttemptsOf(w.instanceID)
	if problem != nil {
		return false
	}
	for _, attempt := range attempts {
		parent, problem := c.opt.Store.RequestRow(attempt.RequestID)
		if problem != nil {
			return false
		}
		if parent == nil || len(parent.OrchestrationDirective) == 0 {
			continue
		}
		probe := records.Request{Worker: w.spec.Connection.RentalID, ParentRequestID: parent.ID}
		if _, problem := c.retainedOrchestrationParent(probe); problem != nil {
			return false
		}
		if req.ID != parent.ID && req.ParentRequestID != parent.ID {
			return false
		}
	}
	return true
}
