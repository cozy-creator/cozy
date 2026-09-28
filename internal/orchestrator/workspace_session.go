package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Workspace effects of classic work run only on a classic session that still exists; none
// is created any more, and classic work is ended at daemon start.
func (c *Orchestrator) workspaceControl(rental string) (*session, *exit.Error) {
	return c.workspaceControlContext(context.Background(), rental)
}

func (c *Orchestrator) workspaceControlContext(ctx context.Context, rental string) (*session, *exit.Error) {
	if rental != "" {
		return c.rentalControl(rental)
	}
	return nil, ClassicRetired()
}
