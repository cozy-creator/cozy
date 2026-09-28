package orchestrator

import (
	"context"
)

// transferWork is everything moving one request's model bytes right now: Hub fetches,
// provider downloads, conversion and publication. Each continuation joins it while it
// runs; a cancel stops all of them through ctx, and stopTransfer waits until they have.
type transferWork struct {
	ctx    context.Context
	cancel context.CancelFunc
	active int
	idle   chan struct{}
}

// joinTransfer returns the request's transfer context and the leave call that ends this
// continuation's membership.
func (c *Orchestrator) joinTransfer(requestID string) (context.Context, func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	work := c.transferWork[requestID]
	if work == nil {
		ctx, cancel := context.WithCancel(context.Background())
		work = &transferWork{ctx: ctx, cancel: cancel, idle: make(chan struct{})}
		c.transferWork[requestID] = work
	}
	work.active++
	left := false
	return work.ctx, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if left {
			return
		}
		left = true
		if work.active--; work.active == 0 {
			work.cancel()
			close(work.idle)
			if c.transferWork[requestID] == work {
				delete(c.transferWork, requestID)
			}
		}
	}
}

// cancelTransfer signals the request's transfer work to stop at its next safe point.
func (c *Orchestrator) cancelTransfer(requestID string) *transferWork {
	c.mu.Lock()
	work := c.transferWork[requestID]
	c.mu.Unlock()
	if work != nil {
		work.cancel()
	}
	c.signalTransfer(requestID)
	return work
}

// stopTransfer cancels the request's transfer work and returns once every continuation
// has returned. Bytes already admitted stay in the store for a later run to reuse.
func (c *Orchestrator) stopTransfer(requestID string) {
	if work := c.cancelTransfer(requestID); work != nil {
		<-work.idle
	}
}
