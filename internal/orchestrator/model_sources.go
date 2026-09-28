package orchestrator

import (
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ModelSourceCapability is refreshed privileged access, never durable request meaning.
type ModelSourceCapability struct {
	Member, ObjectID, URL string
	Length                int64
	Provider              pb.ModelSourceProvider
	ExpiresAtUnix         uint64
}

func (c *Orchestrator) transferChannel(operationID string) chan struct{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := c.transferWake[operationID]
	if ch == nil {
		ch = make(chan struct{}, 1)
		c.transferWake[operationID] = ch
	}
	return ch
}

func (c *Orchestrator) signalTransfer(operationID string) {
	select {
	case c.transferChannel(operationID) <- struct{}{}:
	default:
	}
}
