package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// modelDefaultAvailability describes deployment readiness, not permission to supply explicit
// model inputs: a callable is available when every slot it constructs declares a default. A
// `Weights[M]` argument is the caller's raw input and needs none.
func modelDefaultAvailability(callable *launch.Entrypoint) string {
	var missing []string
	for _, slot := range callable.Models {
		if slot.Default == "" && !slot.Weights {
			missing = append(missing, slot.Param)
		}
	}
	if len(missing) > 0 {
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
