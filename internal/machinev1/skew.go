package machinev1

import (
	"github.com/cozy-creator/cozy/internal/build"
	"github.com/cozy-creator/cozy/internal/exit"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Older reads a failed call as a machine older than this cozy: it was reached and serves no
// such call. One message names the side to upgrade; nil for any other failure.
func Older(err error) *exit.Error {
	if status.Code(err) != codes.Unimplemented {
		return nil
	}
	return exit.Named(exit.Structural, "machine.upgrade_required",
		"the machine is older than this cozy (%s) and does not serve this call (%s)", build.Version, status.Convert(err).Message()).
		WithRemedy("update the machine: `cozy machine install` on this computer; a rental on an older image is replaced by a new one (`cozy rental new`)")
}
