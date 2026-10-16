package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// A child's model slot is chosen by the machine that runs it: a retained ModelArtifact in the
// child's payload, else a choice the caller named, else the callee's authored default, at the
// lane the machine's rules pick for its GPUs. Before any machine exists, a composition's rental
// is sized by those defaults' lane bytes.

// childSizedDefault is a callee slot's default at the lane a rental is sized by.
func (r *Resolver) childSizedDefault(cards cardReader, pkg, entrypoint string, slot launch.Slot, bits uint32) (orchestrator.ModelRef, *exit.Error) {
	hctx, cancel := hub.Context()
	defer cancel()
	model, problem := sizedDefault(hctx, cards, pkg, packageOwner(pkg, []launch.Slot{slot}, r.namespace), slot, int(bits))
	model.Callable = pkg + "/" + entrypoint
	return model, problem
}

// UnpublishedChildModels supplies captured defaults for a CPU request's rental choice.
// A request sized by its own model slots keeps only those: install-wide callable
// capture does not mean those children are invoked or resident alongside it.
// As with PrivateRentalNeedsAccelerator, only CPU orchestration needs the traversal.
func (r *Resolver) UnpublishedChildModels(request records.Request) ([]records.ModelRef, *exit.Error) {
	if composes, problem := r.composesChildren(request); problem != nil || !composes {
		return nil, problem
	}
	var out []records.ModelRef
	queue, seen := []string{request.InstallID}, map[string]bool{}
	for len(queue) > 0 {
		installID := queue[0]
		queue = queue[1:]
		if seen[installID] {
			continue
		}
		seen[installID] = true
		bindings, problem := r.store.ChildBindings(installID)
		if problem != nil {
			return nil, problem
		}
		for _, binding := range bindings {
			if ownFunction(request, binding) {
				continue
			}
			child, problem := r.store.Install(binding.ChildInstallID)
			if problem != nil {
				return nil, problem
			}
			if child == nil {
				return nil, exit.Named(exit.Conflict, "child.install_absent",
					"captured child implementation is unavailable for model selection")
			}
			surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(child.Dir))
			if problem != nil {
				return nil, problem
			}
			fn, problem := surface.Function(binding.Entrypoint)
			if problem != nil {
				return nil, problem
			}
			for _, slot := range fn.Models {
				selected, problem := r.childSizedDefault(r.catalog(request.Hub), child.Package, binding.Entrypoint, slot, request.Bits)
				if problem != nil {
					// This is advisory sizing for an unknown future call. An
					// inaccessible default may be unused or explicitly overridden;
					// the frozen capture records its unavailable outcome and only
					// an actual omitted argument requires that source.
					continue
				}
				out = append(out, selected)
			}
			queue = append(queue, binding.ChildInstallID)
		}
	}
	return out, nil
}
