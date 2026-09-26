package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
)

// CaptureMachineExecution reads only the immutable install/revision selected at
// intake. It never resolves a future child call or observes a mutable package pin.
func (r *Resolver) CaptureMachineExecution(request records.Request) (localpackage.ExecutionCapture, *exit.Error) {
	root, problem := r.LocalInstallation(request.InstallID, request.LocalInstallationID)
	if problem != nil {
		return localpackage.ExecutionCapture{}, problem
	}
	if root.Package != request.Package || root.Release != request.Release {
		return localpackage.ExecutionCapture{}, exit.New(exit.Conflict, "machine execution no longer names its captured package")
	}
	return localpackage.CaptureExecution(request.InstallID, root, r.store.ChildBindings, r.LocalInstallation)
}
