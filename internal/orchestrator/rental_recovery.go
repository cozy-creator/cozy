package orchestrator

import "github.com/cozy-creator/cozy/internal/exit"

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
