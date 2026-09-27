package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// effectiveModelBindings projects owner overrides and immutable authored defaults
// into the existing ladder resolver's input. It does not write or synchronize Hub rows.
// owner is the package's account: org-relative defaults name its models.
func effectiveModelBindings(slots []launch.Slot, overrides []hub.PackageBindingRow, owner string) map[string]hub.PackageBindingRow {
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
		if authored := slot.Default(owner); authored != nil {
			out[slot.Path] = *authored
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

// missingModelDefaults describes deployment readiness, not permission to supply
// explicit model inputs to a local or rented invocation. Jobs have no default
// serving deployment and may take unpublished model inputs from their caller.
func missingModelDefaults(callable *launch.Entrypoint, defaults map[string]hub.PackageBindingRow) []string {
	if callable.Kind == "job" {
		return nil
	}
	var missing []string
	for _, slot := range callable.Models {
		if _, bound := defaults[slot.Path]; !bound {
			missing = append(missing, slot.Param)
		}
	}
	return missing
}

func modelDefaultAvailability(callable *launch.Entrypoint, defaults map[string]hub.PackageBindingRow) string {
	if missing := missingModelDefaults(callable, defaults); len(missing) > 0 {
		return "disabled: no default for " + strings.Join(missing, ", ")
	}
	return "available"
}

// packageOwner is the account an installed package's org-relative defaults name: its org
// once published, and the caller on this Tensorhub for editable `local/` source. It asks
// the Hub only when a slot is org-relative; an unknown caller leaves those defaults unset.
func packageOwner(pkg string, slots []launch.Slot, caller packagepublish.NamespaceSource) string {
	if org, _, published := strings.Cut(pkg, "/"); published && org != "local" {
		return org
	}
	for _, slot := range slots {
		if slot.RelativeDefault() && caller != nil {
			namespace, problem := caller()
			if problem != nil {
				return ""
			}
			return namespace.Account
		}
	}
	return ""
}
