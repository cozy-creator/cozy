package cli

import (
	"regexp"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
)

var authoredLanePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}$`)
var digestShapedRelease = regexp.MustCompile(`^[0-9a-f]{64}$`)

// effectiveModelBindings projects owner overrides and immutable authored defaults
// into the existing ladder resolver's input. It does not write or synchronize Hub rows.
func effectiveModelBindings(slots []launch.Slot, overrides []hub.PackageBindingRow) (map[string]hub.PackageBindingRow, *exit.Error) {
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
		if len(slot.DefaultLadder) == 0 {
			continue
		}
		row := hub.PackageBindingRow{Slot: slot.Path}
		for _, rung := range slot.DefaultLadder {
			model, release, lane, manifest, problem := parseModelRef(rung.Lane)
			if problem != nil || manifest != "" || !modelReleaseLabelPattern.MatchString(release) || digestShapedRelease.MatchString(release) || !authoredLanePattern.MatchString(lane) {
				return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default lane %q must be org/model@release/lane", slot.Path, rung.Lane)
			}
			if row.Model != "" && (row.Model != model || row.Release != release) {
				return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default ladder must use the same model and release in every rung", slot.Path)
			}
			row.Model, row.Release = model, release
			row.Ladder = append(row.Ladder, hub.BindingRung{GPU: rung.GPU, Lane: lane})
		}
		if problem := hub.ValidateLadder(row.Ladder); problem != nil {
			return nil, exit.Named(exit.Validation, "package_model_default_invalid", "%s default ladder: %s", slot.Path, problem.Message)
		}
		out[slot.Path] = row
	}
	return out, nil
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
