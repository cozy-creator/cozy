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
// obligation: Close stops it cleanly, `cozy unload` reclaims exactly that set, and `cozy
// down` never refused on it. Counting it would keep the daemon up after every local run.
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
		if c.opt.PrivateExecution != nil && c.idleRentalWorkerLocked(w) {
			continue
		}
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
	if n := c.frames.count(); n > 0 && c.opt.PrivateExecution == nil {
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
