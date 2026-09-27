package orchestrator

import (
	"context"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
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

// releaseCanceledSource tells the pod to stop a canceled request's source download and
// conversion: its Host cancels the fetches and drains them before it answers. A pod that
// cannot answer yet is asked again until it does or its rental is gone. Retained work
// releases its source through its own cancellation instead.
func (c *Orchestrator) releaseCanceledSource(request records.Request) {
	if request.RetainWork || request.Worker == "" || !request.ModelTransfer.HasAcquisition() {
		return
	}
	go func() {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go func() {
			select {
			case <-c.done:
				cancel()
			case <-ctx.Done():
			}
		}()
		problem := c.releaseRetainedSource(ctx, request)
		if problem == nil {
			c.logf("canceled model transfer %s: rental %s stopped its source work", request.ID, request.Worker)
			return
		}
		c.logf("canceled model transfer %s: source release on rental %s pending: %s",
			request.ID, request.Worker, problem.Message)
		c.mu.Lock()
		closing := c.closing
		c.mu.Unlock()
		if problem.Code == exit.Unavailable && !closing {
			time.AfterFunc(ReportCadence, func() { c.releaseCanceledSource(request) })
		}
	}()
}
