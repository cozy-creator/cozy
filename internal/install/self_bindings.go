package install

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// CaptureSelfBindings records one immutable child binding per `@invocable` job an
// install exposes to ITSELF.
//
// A binding is the frozen resolution of one callable — which install, which
// entrypoint, which carrier revision — and every consumer reads it: the child
// resolver submits from it, GC keeps the child install alive by it, and placement
// reads the composition's CPU orchestration role off it. A package awaiting its own
// sibling job needs exactly that resolution, so it belongs in the same table rather
// than in a special case at call time. The parent and the child are one install, so
// the row names no separate carrier revision: the calling request's own is the
// child's, which is also what keeps a replay byte-identical.
//
// The set is exactly what cozy-runtime advertises to a job attempt as self-callable
// (`Calls.bindings`): every job — never a serving entrypoint — carrying an invocable
// declaration, keyed by the install's own package interface digest.
func CaptureSelfBindings(st *records.Store, inst records.PackageInstall, surface *launch.PackageInterface) *exit.Error {
	if inst.ID == "" || inst.Dir == "" {
		return exit.Internalf("self binding capture needs one recorded install")
	}
	if surface == nil {
		read, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(inst.Dir), inst.PackageInterface)
		if problem != nil {
			return problem
		}
		surface = read
	}
	bindings := make([]records.ChildBinding, 0, len(surface.Jobs))
	for _, job := range surface.Jobs {
		if job.Invocable == nil {
			continue
		}
		bindings = append(bindings, records.ChildBinding{
			ParentInstallID: inst.ID,
			InterfaceDigest: surface.Digest,
			Module:          job.Invocable.Module,
			Export:          job.Invocable.Export,
			ChildInstallID:  inst.ID,
			Entrypoint:      job.Name,
		})
	}
	if len(bindings) == 0 {
		return nil
	}
	return st.RecordChildBindings(bindings)
}
