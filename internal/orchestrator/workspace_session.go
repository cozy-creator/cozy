package orchestrator

import (
	"context"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Workspace effects belong to the configured TensorFS store, not a package
// worker's disposable home. An empty worker can service cleanup after all jobs exit.
func (c *Orchestrator) workspaceControl(rental string) (*session, *exit.Error) {
	return c.workspaceControlContext(context.Background(), rental)
}

func (c *Orchestrator) workspaceControlContext(ctx context.Context, rental string) (*session, *exit.Error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if ctx.Err() != nil {
		return nil, exit.Unavailablef("workspace cleanup canceled")
	}
	if rental != "" {
		if _, _, _, problem := c.ensureRentalContext(ctx, rental); problem != nil {
			return nil, problem
		}
		return c.rentalControl(rental)
	}
	if current := c.localWorkspace(); current != nil {
		return current, nil
	}
	resolver, ok := c.opt.Packages.(interface {
		ResolveWorkspace() (WorkerLaunchSpec, *exit.Error)
	})
	if !ok {
		return nil, exit.Unavailablef("this Creator cannot start the local Runtime workspace service")
	}
	spec, problem := resolver.ResolveWorkspace()
	if problem != nil {
		return nil, problem
	}
	if spec.Connection != nil || spec.Placement.Package != "" || spec.IsJob() || len(spec.Devices) != 0 ||
		spec.TensorFSRoot == "" || filepath.Clean(spec.TensorFSRoot) != filepath.Clean(c.opt.Cfg.TensorFSRoot) {
		return nil, exit.Internalf("local workspace service must use this store with no package or device allocation")
	}
	instance, _, problem := c.EnsureWorker(spec)
	if problem != nil {
		return nil, problem
	}
	if problem := c.ensureWorkerClaimedContext(ctx, instance); problem != nil {
		return nil, exit.Named(exit.Unavailable, "workspace.control_unavailable", "the local Runtime workspace service could not accept its claim: %s", problem.Message)
	}
	if current := c.localWorkspace(); current != nil {
		return current, nil
	}
	return nil, exit.Unavailablef("local Runtime workspace service lost its claim")
}

// localWorkspace is a claimed local worker on this store, if one is connected.
func (c *Orchestrator) localWorkspace() *session {
	c.mu.Lock()
	defer c.mu.Unlock()
	var selected *session
	for _, current := range c.sessions {
		worker := c.workers[current.instanceID]
		if current.host != nil || current.preparation == nil || current.claim == nil || current.ctx.Err() != nil ||
			worker == nil || worker.exited || worker.stopping || !worker.snapshotAcknowledged ||
			worker.spec.TensorFSRoot == "" || filepath.Clean(worker.spec.TensorFSRoot) != filepath.Clean(c.opt.Cfg.TensorFSRoot) {
			continue
		}
		if selected == nil || current.instanceID < selected.instanceID {
			selected = current
		}
	}
	return selected
}
