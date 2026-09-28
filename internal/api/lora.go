package api

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// validateModelAdapters refuses an adapter stack: no Runtime writes a placement's adapters
// yet (tracker proto-062 R2b), and a machine never silently drops a requested overlay.
func validateModelAdapters(models []orchestrator.ModelRef) *exit.Error {
	for _, model := range models {
		if len(model.Adapters) > 0 {
			return exit.Named(exit.Unavailable, "model_adapters.not_applied",
				"no Runtime applies LoRA adapters yet (tracker proto-062 R2b); nothing was submitted")
		}
	}
	return nil
}
