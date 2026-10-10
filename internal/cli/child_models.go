package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// A captured child's model slot has ONE authority and it is not the calling package.
//
// A retained ModelArtifact in the child's payload stays first: that is the private
// transaction lineage, where a parent acquired weights itself and hands them down. When
// the payload names none, the slot resolves the way a top-level `cozy run <package>/<callable>`
// resolves it — the owner's Hub binding, else the callee's authored default ladder — and
// the rung is chosen by the machine that will actually run the child. Package code never
// spells a model here; it only says which of its own callables to run.
//
// Without this a composition parent that holds no device cannot name a model at all:
// declaring a slot of its own would make it device-holding and cost it the CPU
// orchestration role (decision #601), and the runtime cannot inject one because the child
// payload is converted from the caller's own arguments.

// childModelLadder is the selection with its rung still open: the owner's binding or the
// callee's authored default, every rung resolved against the model card the way a rented
// top-level run resolves it. The machine decision reads this to keep a composition off a
// card no shot of it declares a lane for.
func (r *Resolver) childModelLadder(origin, pkg, entrypoint string, slot launch.Slot) (orchestrator.ModelRef, *exit.Error) {
	var empty orchestrator.ModelRef
	binding, problem := r.childSlotBinding(origin, pkg, entrypoint, slot)
	if problem != nil {
		return empty, problem
	}
	model, release, _, _, problem := hub.ParseModelRef(binding.Ref())
	if problem != nil {
		return empty, problem
	}
	ref, problem := hub.ParseRef(model)
	if problem != nil {
		return empty, problem
	}
	if ref.Org == "local" {
		return empty, exit.Named(exit.Unavailable, "child.model_local_selection",
			"%s selects the private local model %s for %s, which cannot be granted to a captured child",
			pkg, ref.String(), slot.Path).
			WithRemedy("publish the model under a non-local org, or hand the child a retained ModelArtifact")
	}
	hctx, cancel := hub.Context()
	defer cancel()
	_, selected, problem := modelReleaseCard(hctx, r.catalog(origin), ref, release)
	if problem != nil {
		return empty, problem
	}
	rungs := make([]records.ModelRung, 0, len(binding.Ladder))
	for _, rung := range binding.Ladder {
		if !slot.AllowsGPUCount(rung.GPUs) {
			return empty, exit.Named(exit.Validation, "package_model_default_invalid", "%s does not support %d GPUs", slot.Path, rung.GPUs)
		}
		lane, problem := laneOf(ref, selected, rung.Lane)
		if problem != nil {
			return empty, problem
		}
		if problem := requireCheckpointComponents(ref.String()+"@"+selected.Release+"/"+rung.Lane, slot, lane.Components); problem != nil {
			return empty, problem
		}
		if _, err := canonical.Raw(lane.ManifestID); err != nil {
			return empty, exit.Named(exit.Conflict, "child.model_manifest_invalid",
				"Tensorhub returned an invalid manifest for %s@%s/%s", ref.String(), selected.Release, rung.Lane)
		}
		rungs = append(rungs, records.ModelRung{GPU: rung.GPU, GPUs: rung.GPUs, Lane: rung.Lane,
			Manifest: lane.ManifestID, Bytes: lane.Bytes, ComponentBytes: lane.ComponentBytes})
	}
	return orchestrator.ModelRef{Package: pkg, Slot: slot.Param, BindingPath: slot.Path,
		Model: ref.String(), CatalogRepository: ref.String(), Release: selected.Release, ComponentUse: slot.ComponentUse, Ladder: rungs}, nil
}

// childSlotBinding is the selection order minus the run key: a run key belongs to the
// caller's own invocation and a captured child has none. An editable package has no hub
// row and stands on its authored default, exactly as a top-level run of it does.
func (r *Resolver) childSlotBinding(origin, pkg, entrypoint string, slot launch.Slot) (hub.PackageBindingRow, *exit.Error) {
	var rows []hub.PackageBindingRow
	if !strings.HasPrefix(pkg, "local/") {
		ref, problem := hub.ParseRef(pkg)
		if problem != nil {
			return hub.PackageBindingRow{}, problem
		}
		hctx, cancel := hub.Context()
		defer cancel()
		read, problem := r.catalog(origin).PackageBindings(hctx, ref)
		if problem != nil {
			return hub.PackageBindingRow{}, exit.Named(problem.Code, "child.model_binding_unreadable",
				"%s default bindings are not readable: %s", pkg, problem.Message)
		}
		rows = read
	}
	binding, ok := effectiveModelBindings([]launch.Slot{slot}, rows, packageOwner(pkg, []launch.Slot{slot}, r.namespace))[slot.Path]
	if !ok {
		return hub.PackageBindingRow{}, exit.Named(exit.NotFound, "child.model_unbound",
			"%s has no owner binding or authored default for %s model slot %s",
			pkg, entrypoint, slot.Path).
			WithRemedy("bind it: %s", bindRemedy(pkg, slot.Path))
	}
	if len(binding.Ladder) == 0 {
		return hub.PackageBindingRow{}, exit.Named(exit.Validation, "child.model_ladder_empty",
			"%s names no default lane for %s model slot %s", pkg, entrypoint, slot.Path).
			WithRemedy("declare a default lane: %s", bindRemedy(pkg, slot.Path))
	}
	return binding, nil
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
			job, problem := surface.Function(binding.Entrypoint)
			if problem != nil {
				return nil, problem
			}
			for _, slot := range job.Models {
				selected, problem := r.childModelLadder(request.Hub, child.Package, binding.Entrypoint, slot)
				if problem != nil {
					// This is advisory sizing for an unknown future call. An
					// inaccessible default may be unused or explicitly overridden;
					// the frozen capture records its unavailable outcome and only
					// an actual omitted argument requires that source.
					continue
				}
				selected.Slot = slot.Path
				selected.Callable = child.Package + "/" + binding.Entrypoint
				out = append(out, selected)
			}
			queue = append(queue, binding.ChildInstallID)
		}
	}
	return out, nil
}
