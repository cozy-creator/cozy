package orchestrator

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Workspace effects run on the rental that owns them. This computer's retired classic
// local worker had its own workspace service; its work is ended at daemon start.
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
	return nil, ClassicLocalRetired()
}
