package machinev1

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/build"
	"github.com/cozy-creator/cozy/internal/exit"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A cozy and a machine a release apart can share no call for some verb. Each way round is one
// message naming the side to upgrade, never the transport's own words.

// NewerThanCozy names this cozy as the side to upgrade.
func NewerThanCozy(format string, args ...any) *exit.Error {
	return exit.Named(exit.Structural, "cozy.upgrade_required", "the machine is newer than this cozy (%s): %s", build.Version, fmt.Sprintf(format, args...)).
		WithRemedy("upgrade cozy: curl -fsSL https://github.com/cozy-creator/cozy/releases/latest/download/install.sh | sh")
}

// Skew reads an UNIMPLEMENTED call as a machine older than this cozy; nil for every other
// failure.
func Skew(err error) *exit.Error {
	// A machine from before live packages (TensorD `live-source/1`) reads their manifest as
	// one naming no code.
	if message := status.Convert(err).Message(); strings.Contains(message, "local_source_invalid") && strings.Contains(message, "source archive or root wheel") {
		return exit.Named(exit.Structural, "machine.upgrade_required",
			"the machine is older than this cozy (%s) and installs no live package", build.Version).
			WithRemedy("update the machine software: `cozy machine install` on this computer, or `cozy rental update <rental>` for a rental")
	}
	if status.Code(err) != codes.Unimplemented {
		return nil
	}
	return exit.Named(exit.Structural, "machine.upgrade_required",
		"the machine is older than this cozy (%s) and does not serve this call (%s)", build.Version, status.Convert(err).Message()).
		WithRemedy("update the machine software: `cozy machine install` on this computer, or `cozy rental update <rental>` for a rental")
}
