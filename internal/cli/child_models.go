package cli

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// A child's model slot is chosen by the machine that runs it: a retained ModelArtifact in the
// child's payload, else a choice the caller named, else the callee's authored default ladder,
// with the rung for that machine's GPUs. Before any machine exists, a composition's rental is
// sized by those same defaults.

// childModelLadder is a callee slot's authored default with its rung still open, every rung
// resolved against the model card. The machine decision reads it to keep a composition off a
// card no shot of it declares a lane for.
func (r *Resolver) childModelLadder(cards cardReader, pkg, entrypoint string, slot launch.Slot) (orchestrator.ModelRef, *exit.Error) {
	var empty orchestrator.ModelRef
	binding := slot.Default(packageOwner(pkg, []launch.Slot{slot}, r.namespace))
	if binding == nil || len(binding.Ladder) == 0 {
		return empty, exit.Named(exit.NotFound, "child.model_unbound",
			"%s declares no default for %s model slot %s", pkg, entrypoint, slot.Path)
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
			pkg, ref.String(), slot.Path)
	}
	hctx, cancel := hub.Context()
	defer cancel()
	_, selected, problem := modelReleaseCard(hctx, cards, ref, release)
	if problem != nil {
		return empty, problem
	}
	rungs := make([]records.ModelRung, 0, len(binding.Ladder))
	for _, rung := range binding.Ladder {
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
		Model: ref.String(), CatalogRepository: ref.String(), Release: selected.Release, ComponentUse: slot.ComponentUse,
		SupportedGPUs: slot.SupportedGPUCounts(), Ladder: rungs}, nil
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
				selected, problem := r.childModelLadder(r.catalog(request.Hub), child.Package, binding.Entrypoint, slot)
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
