package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
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

// childModelSelection resolves one declared slot of a captured callee, pinned to
// `accelerator` — the machine the child will run on, which for a rental composition is the
// parent's own pod. An empty accelerator is the host without an NVIDIA device and matches
// only a "*" rung, exactly as a local run of the callee would.
func (r *Resolver) childModelSelection(pkg, entrypoint string, slot launch.Slot, accelerator string, count int) (orchestrator.ModelRef, *exit.Error) {
	out, problem := r.childModelLadder(pkg, entrypoint, slot)
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	// The child runs on ONE known machine, so its rung is decided here rather than left
	// unpinned for a placement decision the child never enters — it inherits its parent's.
	fitted, _, ok := out.RungFor(accelerator, count)
	if !ok {
		return orchestrator.ModelRef{}, exit.Named(exit.Conflict, "child.model_rung_absent",
			"%s model %s has no lane declared for %s (ladder: %s)",
			pkg, slot.Param, orNone(accelerator), rungText(out.Ladder)).
			WithRemedy("declare a rung for this accelerator, or run the composition on a machine its ladder names")
	}
	out = out.Pin(fitted)
	// A JOB's model is an invocation INPUT, so it carries exact manifest bytes the way
	// `jobManifestInputs` grants them for a top-level job — not a bare model identity.
	ref, problem := hub.ParseRef(out.Model)
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	raw, problem := r.catalog.ReleaseManifest(hctx, ref, out.Release, out.Lane)
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	if !manifestBytes(raw, out.Manifest) {
		// The lane moved since selection; read the selected checkpoint itself.
		if checkpoint, fallback := r.catalog.CheckpointManifest(hctx, ref, out.Manifest); fallback == nil {
			raw = checkpoint
		}
	}
	if !manifestBytes(raw, out.Manifest) {
		return orchestrator.ModelRef{}, exit.Named(exit.Conflict, "child.model_manifest_changed",
			"Tensorhub serves no bytes for the selected checkpoint %s", out.Manifest)
	}
	out.ManifestLength = int64(len(raw))
	return out, nil
}

// childModelLadder is the selection with its rung still open: the owner's binding or the
// callee's authored default, every rung resolved against the model card the way a rented
// top-level run resolves it. The machine decision reads this to keep a composition off a
// card no shot of it declares a lane for.
func (r *Resolver) childModelLadder(pkg, entrypoint string, slot launch.Slot) (orchestrator.ModelRef, *exit.Error) {
	var empty orchestrator.ModelRef
	binding, problem := r.childSlotBinding(pkg, entrypoint, slot)
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
	_, selected, problem := modelReleaseCard(hctx, r.catalog, ref, release)
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
func (r *Resolver) childSlotBinding(pkg, entrypoint string, slot launch.Slot) (hub.PackageBindingRow, *exit.Error) {
	var rows []hub.PackageBindingRow
	if !strings.HasPrefix(pkg, "local/") {
		ref, problem := hub.ParseRef(pkg)
		if problem != nil {
			return hub.PackageBindingRow{}, problem
		}
		hctx, cancel := hub.Context()
		defer cancel()
		read, problem := r.catalog.PackageBindings(hctx, ref)
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

// childAccelerator is the machine a captured child of this parent will run on. A rental
// composition keeps parent and child on one pod, so the parent's own paid accelerator is
// the child's; a local parent's child runs on this host's device.
func (r *Resolver) childAccelerator(parent records.Request) (string, int, *exit.Error) {
	if parent.Worker == "" {
		inventory := hostgpu.Probe(r.cfg)
		if len(inventory.GPUs) == 0 {
			return "", 0, nil
		}
		return inventory.GPUs[0].Model, len(inventory.GPUs), nil
	}
	row, problem := r.store.RentalRow(parent.Worker)
	if problem != nil {
		return "", 0, problem
	}
	if row == nil {
		return "", 0, exit.Named(exit.Conflict, "child.parent_machine_absent",
			"the parent's machine is no longer recorded")
	}
	if records.CPUAccelerator(row.AcceleratorModel) {
		return "", 0, nil
	}
	return row.AcceleratorModel, row.AcceleratorCount, nil
}

// UnpublishedChildModels supplies captured defaults for a CPU request's rental choice.
// A request sized by its own model slots keeps only those: install-wide callable
// capture does not mean those children are invoked or resident alongside it.
// As with PrivateRentalNeedsAccelerator, only CPU orchestration needs the traversal.
func (r *Resolver) UnpublishedChildModels(request records.Request) ([]records.ModelRef, *exit.Error) {
	if request.SizedByOwnModels() || request.InstallID == "" {
		return nil, nil
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
				selected, problem := r.childModelLadder(child.Package, binding.Entrypoint, slot)
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

// rungText is one selection's ladder as a readable line, the ModelRung spelling of
// hub.LadderText: `H100=fp8-adaln-pruned > *=bf16`.
func rungText(ladder []records.ModelRung) string {
	parts := make([]string, 0, len(ladder))
	for _, rung := range ladder {
		parts = append(parts, rung.String())
	}
	return strings.Join(parts, " > ")
}

// EnsureLocalModels runs after child admission, on the ordinary activation path.
// The control-frame reader must not block on a model download before recording the child.
func (r *Resolver) EnsureLocalModels(models []orchestrator.ModelRef) *exit.Error {
	for _, model := range models {
		if model.Downloadable() {
			if problem := r.ensureLocalChildModel(model); problem != nil {
				return problem
			}
		}
	}
	return nil
}

// A local child skips the top-level CLI's model intake. Use that same acquisition
// owner here; code publication cannot make cold model bytes appear in the store.
func (r *Resolver) ensureLocalChildModel(model orchestrator.ModelRef) *exit.Error {
	tool, problem := tfs.Open(r.cfg)
	if problem != nil {
		return problem
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return problem
	}
	work, problem := scratch.Temp(layout.Tmp, "child-model-")
	if problem != nil {
		return problem
	}
	defer work.Release()
	spec := model.Model
	if model.Release != "" {
		spec += "@" + model.Release
	}
	spec += "@" + model.Manifest
	fetch := transfer.Fetch{Tool: tool, Hub: r.catalog, Spec: spec, Lane: model.Lane,
		Scratch: work.Path, Locks: layout.AcquisitionLocks()}
	ctx, cancel := hub.LongContext()
	defer cancel()
	resolved, problem := fetch.Resolve(ctx)
	if problem != nil {
		return problem
	}
	if fetch.Ref.String() != model.Model || resolved.ManifestID != model.Manifest || resolved.Release != model.Release || resolved.Lane != model.Lane {
		return exit.Named(exit.Conflict, "child.model_manifest_changed", "local child acquisition differs from its exact model selection")
	}
	fetched, problem := fetch.Acquire(ctx, resolved)
	if problem != nil {
		return problem
	}
	if fetched.ManifestID != model.Manifest || fetched.ManifestLength != model.ManifestLength {
		return exit.Named(exit.Conflict, "child.model_manifest_changed", "local child acquisition differs from its exact manifest")
	}
	return nil
}
