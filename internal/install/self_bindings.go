package install

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// CaptureSelfBindings records one immutable child binding per `@invocable`
// job or serving entrypoint an install exposes to itself.
//
// A binding is the frozen resolution of one callable — which install, which
// entrypoint, which carrier revision — and every consumer reads it: the child
// resolver submits from it, GC keeps the child install alive by it, and placement
// reads the composition's CPU orchestration role off it. A package awaiting its own
// sibling callable needs exactly that resolution, so it belongs in the same table rather
// than in a special case at call time. The parent and the child are one install, so
// the row names no separate carrier revision: the calling request's own is the
// child's, which is also what keeps a replay byte-identical.
//
// The set is exactly what cozy-runtime advertises to a job attempt as self-callable
// (`Calls.bindings`): every job and serving entrypoint carrying an invocable
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
	entries := append(append([]launch.Entrypoint(nil), surface.Jobs...), surface.Entrypoints...)
	bindings := make([]records.ChildBinding, 0, len(entries))
	for _, entry := range entries {
		if entry.Invocable == nil {
			continue
		}
		bindings = append(bindings, records.ChildBinding{
			ParentInstallID: inst.ID,
			InterfaceDigest: surface.Digest,
			Module:          entry.Invocable.Module,
			Export:          entry.Invocable.Export,
			ChildInstallID:  inst.ID,
			Entrypoint:      entry.Name,
		})
	}
	if len(bindings) == 0 {
		return nil
	}
	return st.RecordChildBindings(bindings)
}
