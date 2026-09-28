package orchestrator

import (
	"fmt"
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Managing is the work this process holds in MEMORY that a stop would interrupt rather
// than drain — the half of "is there anything to manage" no row describes. Each entry is
// a transition in flight: a launch, a teardown, a transfer, a cleanup, an export, a worker
// that is busy or remote, an event stream a client is still reading.
//
// An idle local serving worker is deliberately absent. It is a warm cache, not an
// obligation: Close stops it cleanly and `cozy down` never refused on it. Counting it would keep the daemon up after every local run.
func (c *Orchestrator) Managing() ([]string, *exit.Error) {
	active, e := c.opt.Store.ActiveRequests()
	if e != nil {
		return nil, e
	}
	var held []string
	c.mu.Lock()
	for _, id := range c.pending {
		held = append(held, "queued "+id)
	}
	for slot := range c.starting {
		held = append(held, "launch "+slot)
	}
	for instance := range c.ensuring {
		held = append(held, "spawn "+instance)
	}
	for _, w := range c.workers {
		if c.idleLocalWorkerLocked(w, active) {
			continue
		}
		held = append(held, "worker "+w.instanceID+" ("+workerActivity(w)+")")
	}
	for id := range c.transferRunning {
		held = append(held, "model transfer "+id)
	}
	for id := range c.transferDispatching {
		held = append(held, "model transfer dispatch "+id)
	}
	for key := range c.mediaCleaning {
		held = append(held, "media cleanup "+key)
	}
	for id := range c.outputExporting {
		held = append(held, "output export "+id)
	}
	c.mu.Unlock()
	if n := c.frames.count(); n > 0 {
		held = append(held, fmt.Sprintf("%d open event stream(s)", n))
	}
	sort.Strings(held)
	return held, nil
}

func workerActivity(w *worker) string {
	switch {
	case w.exited || w.stopping:
		return "stopping"
	case w.spec.Connection != nil:
		return "attached to rental " + w.spec.Connection.RentalID
	case w.spec.IsJob():
		return "job"
	default:
		return "busy"
	}
}

// PrepareClientShutdown closes scheduling admission only when explicit client
// disconnect is safe, or when the caller explicitly overrides the dependency
// guard. Durable retention and idle policy keep their existing stronger rules.
func (c *Orchestrator) PrepareClientShutdown(force bool) ([]string, *exit.Error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closing {
		return nil, nil
	}
	var held []string
	if !force {
		obligations, problem := c.opt.Store.ClientShutdownObligations()
		if problem != nil {
			return nil, problem
		}
		for _, obligation := range obligations {
			held = append(held, obligation.String())
		}
		active, problem := c.opt.Store.ActiveRequests()
		if problem != nil {
			return nil, problem
		}
		for _, w := range c.workers {
			if w.spec.Connection == nil && !w.exited && (w.held > 0 || w.reservedJobs > 0 || !c.idleLocalProcessLocked(w, active, true)) {
				held = append(held, "local worker "+w.instanceID+" ("+workerActivity(w)+")")
			}
		}
		for id := range c.starting {
			held = append(held, "launch "+id)
		}
		for id := range c.ensuring {
			held = append(held, "preparation "+id)
		}
		for id := range c.offers {
			held = append(held, "dispatch "+id)
		}
		for id := range c.transferRunning {
			held = append(held, "transfer "+id)
		}
		for id := range c.transferDispatching {
			held = append(held, "transfer dispatch "+id)
		}
		for id := range c.sourcePauseRunning {
			held = append(held, "source control "+id)
		}
		for id := range c.checkpointUploads {
			held = append(held, "checkpoint upload "+id)
		}
		for id := range c.localTransfers {
			held = append(held, "input upload "+id)
		}
		for id := range c.outputExporting {
			held = append(held, "output export "+id)
		}
		for id := range c.rentalMaintenance {
			held = append(held, "rental maintenance "+id)
		}
		for id := range c.retainedCleaning {
			held = append(held, "retained cleanup "+id)
		}
		for id := range c.mediaCleaning {
			held = append(held, "media cleanup "+id)
		}
		if len(held) > 0 {
			sort.Strings(held)
			return held, nil
		}
	}
	c.closing = true
	return nil, nil
}
