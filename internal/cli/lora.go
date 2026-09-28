package cli

import "github.com/cozy-creator/cozy/internal/exit"

// loraNotApplied refuses --lora before anything is read or submitted: no Runtime writes a
// placement's adapter stack yet (tracker proto-062 R2b).
func loraNotApplied() *exit.Error {
	return exit.Named(exit.Unavailable, "model_adapters.not_applied",
		"no Runtime applies LoRA adapters yet (tracker proto-062 R2b); nothing was submitted")
}
