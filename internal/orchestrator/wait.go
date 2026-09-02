package orchestrator

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/records"
)

// The wait vocabulary (cl-103). A queued/parked event names WHAT the queue is doing in a
// stable `wait` payload field beside the verbatim diagnostic `reason`, so a client can
// say "starting a worker" instead of parsing dispatcher vocabulary. Additive: `reason`
// is unchanged and stays the full diagnostic.
const (
	// WaitWorkerStart: no live worker serves this package yet; one is being started.
	WaitWorkerStart = "worker_start"
	// WaitWorkerWarming: a worker exists and is still loading toward dispatchable.
	WaitWorkerWarming = "worker_warming"
	// WaitSlotBusy: a dispatchable placement exists; every attempt slot is taken.
	WaitSlotBusy = "slot_busy"
	// WaitQueueAhead: capacity is spoken for by requests queued ahead of this one.
	WaitQueueAhead = "queue_ahead"
	// WaitRental: waiting for a rental machine to be acquired or assigned.
	WaitRental = "rental"
	// WaitModelTransfer: the model must land on the selected worker first.
	WaitModelTransfer = "model_transfer"
)

// waitFacts is the classification a queued/parked event carries beside the raw reason:
// the stable cause, and the machine word the wait is on when one is known.
type waitFacts struct {
	cause string
	on    string
}

// decorate adds the wait facts and the request's package to an event payload. The
// payload's `reason` is never touched here.
func (f waitFacts) decorate(payload map[string]any, req records.Request) map[string]any {
	if f.cause != "" {
		payload["wait"] = f.cause
	}
	if f.on != "" {
		payload["waiting_on"] = f.on
	}
	if req.Package != "" {
		payload["package"] = req.Package
	}
	return payload
}

// classifyCapacityWait reads the live routing and names the blocking condition. An empty
// answer means the routing is not the blocker (a pick exists; the refusal was elsewhere).
// Callers hold c.mu.
func (c *Orchestrator) classifyCapacityWait(req records.Request, r routing) waitFacts {
	if r.pick() != nil {
		return waitFacts{}
	}
	if len(r.claimed) > 0 {
		return waitFacts{cause: WaitQueueAhead}
	}
	if len(r.parked) > 0 {
		instance, _, _ := strings.Cut(r.parked[0], ":")
		return waitFacts{cause: WaitSlotBusy, on: c.machineWord(strings.TrimSpace(instance))}
	}
	// No eligible worker at all. A live worker already in the request's slot is loading
	// toward dispatchable; none at all means one is being started (selectOrStart).
	slot := pinnedPackage(req.Package, req.Worker)
	for _, w := range c.workers {
		if w.exited || w.stopping {
			continue
		}
		if req.Worker != "" && w.spec.Connection != nil {
			if w.instanceID == rentalInstanceID(req.Worker) {
				return waitFacts{cause: WaitWorkerWarming, on: c.machineWord(w.instanceID)}
			}
			continue
		}
		if w.spec.Connection == nil && w.spec.Placement.Package == slot {
			return waitFacts{cause: WaitWorkerWarming}
		}
	}
	return waitFacts{cause: WaitWorkerStart}
}

// waitOf classifies a request's blocking condition against the live routing.
func (c *Orchestrator) waitOf(req records.Request) waitFacts {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.classifyCapacityWait(req, c.route(req))
}

// machineWord names an attached worker's venue the way `run list` does: the rental's
// machine word. A local worker is this machine and needs no name. Callers hold c.mu.
func (c *Orchestrator) machineWord(instanceID string) string {
	w := c.workers[instanceID]
	if w == nil || w.spec.Connection == nil {
		return ""
	}
	id := w.spec.Connection.RentalID
	if rental, _ := c.opt.Store.RentalByMachine(id); rental != nil && rental.MachineName != "" {
		return rental.MachineName
	}
	return id
}
