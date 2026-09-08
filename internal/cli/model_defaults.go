package cli

import (
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
)

// effectiveModelBindings projects owner overrides and immutable authored defaults
// into the existing ladder resolver's input. It does not write or synchronize Hub rows.
func effectiveModelBindings(slots []launch.Slot, overrides []hub.PackageBindingRow) map[string]hub.PackageBindingRow {
	current := make(map[string]hub.PackageBindingRow, len(overrides))
	for _, row := range overrides {
		current[row.Slot] = row
	}
	out := make(map[string]hub.PackageBindingRow, len(slots))
	for _, slot := range slots {
		if row, exists := current[slot.Path]; exists {
			out[slot.Path] = row
			continue
		}
		if slot.DefaultBinding != nil {
			out[slot.Path] = *slot.DefaultBinding
		}
	}
	return out
}

func declaredModelSlots(callables ...[]launch.Entrypoint) []launch.Slot {
	var slots []launch.Slot
	for _, group := range callables {
		for _, callable := range group {
			slots = append(slots, callable.Models...)
		}
	}
	return slots
}
