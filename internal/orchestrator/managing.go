package orchestrator

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Managing is the work this process holds in MEMORY that a stop would interrupt rather
// than drain — the half of "is there anything to manage" no row describes. Each entry is
// a transition in flight: a start, a transfer, a cleanup, an export, an event stream a
// client is still reading.
func (c *Orchestrator) Managing() ([]string, *exit.Error) {
	var held []string
	c.mu.Lock()
	for _, id := range c.pending {
		held = append(held, "queued "+id)
	}
	for slot := range c.starting {
		held = append(held, "launch "+slot)
	}
	for id := range c.transferRunning {
		held = append(held, "model transfer "+id)
	}
	for id := range c.outputExporting {
		held = append(held, "output export "+id)
	}
	c.mu.Unlock()
	sort.Strings(held)
	return held, nil
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
		for id := range c.starting {
			held = append(held, "launch "+id)
		}
		for id := range c.transferRunning {
			held = append(held, "transfer "+id)
		}
		for id := range c.outputExporting {
			held = append(held, "output export "+id)
		}
		for id := range c.rentalMaintenance {
			held = append(held, "rental maintenance "+id)
		}
		if len(held) > 0 {
			sort.Strings(held)
			return held, nil
		}
	}
	c.closing = true
	return nil, nil
}
