package orchestrator

import "context"

// This is only an in-flight call registry. Durable request and native-retention
// rows remain the authority for what cancellation still owes after a restart.
type retainedCleanup struct {
	done   chan struct{}
	cancel context.CancelFunc
}

func (c *Orchestrator) finishRetainedCancellation(id string) {
	if !c.startRetainedCleanup(id, c.runRetainedCancellation) {
		// Cancellation can arrive while a successful-root pass is still draining.
		// Its durable intent needs a new pass after that owner leaves the registry.
		c.retryRetainedCancellation(id)
	}
}

func (c *Orchestrator) startRetainedCleanup(id string, run func(context.Context, string)) bool {
	c.mu.Lock()
	if c.closing || c.retainedCleaning[id] != nil {
		c.mu.Unlock()
		return false
	}
	if c.retainedCleaning == nil {
		c.retainedCleaning = make(map[string]*retainedCleanup)
	}
	ctx, cancel := context.WithCancel(context.Background())
	call := &retainedCleanup{done: make(chan struct{}), cancel: cancel}
	c.retainedCleaning[id] = call
	c.mu.Unlock()
	go func() {
		defer func() {
			cancel()
			c.mu.Lock()
			delete(c.retainedCleaning, id)
			close(call.done)
			c.mu.Unlock()
		}()
		run(ctx, id)
	}()
	return true
}

// WaitRetainedCancellations waits for the cleanup passes already accepted by the
// owner. An unchanged cancellation request is not evidence that its I/O is idle.
// Expiry cancels those calls without erasing their durable disposal obligations.
func (c *Orchestrator) WaitRetainedCancellations(ctx context.Context) {
	for {
		c.mu.Lock()
		calls := make([]*retainedCleanup, 0, len(c.retainedCleaning))
		for _, call := range c.retainedCleaning {
			calls = append(calls, call)
		}
		c.mu.Unlock()
		if len(calls) == 0 {
			return
		}
		for _, call := range calls {
			select {
			case <-call.done:
			case <-ctx.Done():
				c.mu.Lock()
				for _, pending := range c.retainedCleaning {
					pending.cancel()
				}
				c.mu.Unlock()
				return
			}
		}
	}
}
