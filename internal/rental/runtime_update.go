package rental

import (
	"github.com/cozy-creator/cozy/internal/exit"
)

// RuntimeWire is the selected Runtime release's own declared cozy.worker.v1 range,
// read from its wheel. Nil means the release declares none.
type RuntimeWire struct {
	WireMinor        uint32 `json:"wire_minor"`
	MinimumWireMinor uint32 `json:"minimum_wire_minor"`
}

// RuntimeUpdateHost refuses installing a Runtime whose declared wire minimum the
// pinned PodHost does not reach. PodHost reports its intersection with the
// installed Runtime, so the refusal is conservative: a pod whose Host is newer than
// its Runtime still reads as the older minor, and gets the newer Runtime from a new
// worker image instead.
func RuntimeUpdateHost(version string, target *RuntimeWire, hostWireMinor uint32) *exit.Error {
	if target == nil || target.MinimumWireMinor == 0 || target.MinimumWireMinor > target.WireMinor {
		return exit.Named(exit.Structural, "rental.runtime_update_wire_unknown",
			"Runtime %s declares no readable worker protocol range — nothing was changed", version)
	}
	if hostWireMinor >= target.MinimumWireMinor {
		return nil
	}
	return exit.Named(exit.Conflict, "rental.runtime_update_host_too_old",
		"Runtime %s needs worker protocol %d or newer on the pod host; this rental reports %d — nothing was changed",
		version, target.MinimumWireMinor, hostWireMinor).
		WithRemedy("rent a machine on the current worker image")
}
