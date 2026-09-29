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

// CloseAdmission stops new work for a client down. Nothing in flight holds the daemon:
// machines run the work, and the next daemon resumes it from durable records.
func (c *Orchestrator) CloseAdmission() {
	c.mu.Lock()
	c.closing = true
	c.mu.Unlock()
}
