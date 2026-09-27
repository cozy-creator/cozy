package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

type runtimeObservation struct {
	Observed struct {
		Updating bool `json:"update_in_progress"`
		Durable  bool `json:"durable_updates"`
		Runtime  struct {
			Distribution string `json:"distribution"`
			Guarded      bool   `json:"supports_guarded_restart"`
		} `json:"runtime"`
		TensorFS string `json:"tensorfs"`
		Python   string `json:"python"`
	} `json:"observed"`
}

// Package SDK versions belong to isolated package environments. Ordinary
// dispatch only waits for explicit maintenance; worker protocol admission owns
// control compatibility and never upgrades a rental to satisfy package wheels.
func (u *rentalRuntimeUpdates) preflight(_ context.Context, _ records.Request, machine string) *exit.Error {
	current, problem := u.machines.store.RuntimeUpdate(machine)
	if problem != nil {
		return problem
	}
	if current != nil && current.Active() {
		return exit.Named(exit.Unavailable, "rental.maintenance", "this rental is updating its Runtime; this request remains queued")
	}
	return nil
}
