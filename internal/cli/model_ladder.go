package cli

import (
	"context"
	"fmt"
	"sort"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// The binding ladder is a FIT MAP (cl-166): the hub binding says which lane of a model
// release belongs on which GPU class, and the lane a run is pinned to is a function of the
// machine that runs it — the host's own accelerator locally, the winning rental remotely.
// Everything here reads the model card ONCE and fails with what the card actually offers,
// so a binding naming a release or lane the card does not carry is refused with the real
// list rather than a bare "not found" hours later on a paid pod.

// modelReleaseCard reads the card and selects `release` (the newest unyanked one when
// empty). A release named explicitly is honoured even when yanked. A miss names the
// releases the card offers.
func modelReleaseCard(ctx context.Context, c *hub.Client, ref hub.Ref, release string) (
	hub.ModelCard, *hub.ModelReleaseSummary, *exit.Error,
) {
	return modelReleaseCardForLane(ctx, c, ref, release, "")
}

func modelReleaseCardForLane(ctx context.Context, c *hub.Client, ref hub.Ref, release, lane string) (hub.ModelCard, *hub.ModelReleaseSummary, *exit.Error) {
	card, problem := c.ModelCard(ctx, ref)
	if problem != nil {
		return card, nil, problem
	}
	if card.Model.Ref() != ref.String() {
		return card, nil, exit.Named(exit.Conflict, "rental.model_catalog_changed",
			"Tensorhub returned model %s while resolving %s", card.Model.Ref(), ref.String())
	}
	explicit := release != ""
	if !explicit {
		for _, candidate := range card.Releases {
			if candidate.Yanked || candidate.YankedAt != "" {
				continue
			}
			containsLane := lane == ""
			for _, row := range candidate.Lanes {
				if row.Lane == lane {
					containsLane = true
				}
			}
			if !containsLane {
				continue
			}
			if release == "" || newerModelRelease(candidate.Release, release) {
				release = candidate.Release
			}
		}
	}
	var yanked *hub.ModelReleaseSummary
	for i := range card.Releases {
		candidate := &card.Releases[i]
		if candidate.Release != release {
			continue
		}
		if !releaseYanked(candidate) {
			return card, candidate, nil
		}
		if explicit && yanked == nil {
			yanked = candidate
		}
	}
	if yanked != nil {
		return card, yanked, nil
	}
	return card, nil, exit.Named(exit.NotFound, "model.release_not_found",
		"model %s has no available release %q (releases: %s)",
		ref.String(), release, orNone(strings.Join(availableReleases(card), ", ")))
}

func releaseYanked(release *hub.ModelReleaseSummary) bool {
	return release.Yanked || release.YankedAt != ""
}

// warnYanked says that an explicitly named release is yanked and is used anyway.
func warnYanked(ctx *Context, ref hub.Ref, release *hub.ModelReleaseSummary) {
	if releaseYanked(release) {
		fmt.Fprintf(ctx.Err, "warning: %s@%s is yanked; using it because it was named explicitly\n", ref.String(), release.Release)
	}
}

// Model cards do not carry creation times for native TensorFS releases. Use
// the existing version parser for numeric and prerelease precedence; model labels
// outside its version grammar retain their historical lexical ordering.
func newerModelRelease(candidate, current string) bool {
	next, nextErr := pep440.Parse(candidate)
	prior, priorErr := pep440.Parse(current)
	if nextErr == nil && priorErr == nil {
		return next.GreaterThan(prior)
	}
	return candidate > current
}

func availableReleases(card hub.ModelCard) []string {
	var out []string
	for _, candidate := range card.Releases {
		if !candidate.Yanked && candidate.YankedAt == "" {
			out = append(out, candidate.Release)
		}
	}
	sort.Strings(out)
	return out
}

// laneOf selects one lane of a release by name, or names the lanes the release carries.
func laneOf(ref hub.Ref, selected *hub.ModelReleaseSummary, lane string) (hub.ModelLaneSummary, *exit.Error) {
	names := make([]string, 0, len(selected.Lanes))
	for _, candidate := range selected.Lanes {
		if candidate.Lane == lane {
			return candidate, nil
		}
		names = append(names, candidate.Lane)
	}
	sort.Strings(names)
	return hub.ModelLaneSummary{}, exit.Named(exit.NotFound, "model.lane_not_found",
		"model %s@%s has no lane %q (lanes: %s)", ref.String(), selected.Release, lane,
		orNone(strings.Join(names, ", ")))
}

// localRungs picks the invocation's slots on this host as one group, the way a rented
// machine's are picked (rental.Pin). A slot whose ladder names no GPU here still takes its
// first declared lane when that rung is uncounted: the ladder is not a local hardware
// allowlist, and Runtime still checks encoding support, construction compatibility and
// memory capacity. An explicit override is its own rung.
func localRungs(ctx *Context, selected []invocationModelSpec) ([]records.ModelRef, *exit.Error) {
	inventory := hostgpu.Probe(ctx.Cfg)
	accelerator := ""
	if len(inventory.GPUs) > 0 {
		accelerator = inventory.GPUs[0].Model
	}
	refs := make([]records.ModelRef, len(selected))
	for i, spec := range selected {
		if spec.Explicit {
			refs[i].Manifest, refs[i].Lane = spec.Ref, spec.Lane
			continue
		}
		if len(spec.Binding.Ladder) == 0 {
			return nil, exit.Named(exit.Validation, "model_ladder_empty",
				"the model binding has no default lanes").
				WithRemedy("declare a default lane or supply model.<param>=org/model@release/lane")
		}
		for _, rung := range spec.Binding.Ladder {
			refs[i].Ladder = append(refs[i].Ladder, records.ModelRung{GPU: rung.GPU, GPUs: rung.GPUs, Lane: rung.Lane})
		}
		if first := refs[i].Ladder[0]; first.GPUs == 0 {
			refs[i].Ladder = append(refs[i].Ladder, records.ModelRung{GPU: "*", Lane: first.Lane})
		}
	}
	pinned, _, ok := rental.Pin(refs, accelerator, len(inventory.GPUs))
	if !ok {
		return nil, exit.Named(exit.Validation, "model_gpu_group_unavailable", "no authored group fits the local inventory of %d GPUs (%s)", len(inventory.GPUs), orNone(accelerator))
	}
	return pinned, nil
}

// bindRemedy is the exact command that gives a slot its hub binding.
func bindRemedy(packageName, slot string) string {
	return fmt.Sprintf("cozy package bind %s %s org/model@release --gpu <GPU>=<lane> --gpu '*'=<lane>",
		packageName, slot)
}

// resolveRemoteLadder resolves a configured ladder for a rented run WITHOUT choosing a lane: every
// rung is bound to the exact manifest the card publishes for its lane, and the machine
// decision pins one of them once the machine exists.
func resolveRemoteLadder(ctx *Context, packageName string, slot launch.Slot,
	spec invocationModelSpec) (orchestrator.ModelRef, *exit.Error) {
	var empty orchestrator.ModelRef
	modelName, release, _, _, problem := hub.ParseModelRef(spec.Ref)
	if problem != nil {
		return empty, problem
	}
	ref, problem := hub.ParseRef(modelName)
	if problem != nil {
		return empty, problem
	}
	if ref.Org == "local" {
		return empty, localModelOnRental(ref)
	}
	hctx, cancel := hub.Context()
	defer cancel()
	_, selected, problem := modelReleaseCard(hctx, client(ctx), ref, release)
	if problem != nil {
		return empty, rebind(problem, packageName, slot.Path)
	}
	warnYanked(ctx, ref, selected)
	rungs, problem := ladderRungs(ctx, ref, selected, slot, spec.Ref, spec.Binding.Ladder)
	if problem != nil {
		return empty, rebind(problem, packageName, slot.Path)
	}
	return orchestrator.ModelRef{Package: packageName, Slot: slot.Path, Model: ref.String(), CatalogRepository: ref.String(),
		Release: selected.Release, ComponentUse: slot.ComponentUse, Ladder: rungs}, nil
}

// ladderRungs binds each rung to the release's manifest for its lane. A rung the release
// cannot serve is skipped with a warning; only a ladder with no usable rung refuses.
func ladderRungs(ctx *Context, ref hub.Ref, selected *hub.ModelReleaseSummary, slot launch.Slot,
	spec string, ladder []hub.BindingRung) ([]records.ModelRung, *exit.Error) {
	rungs := make([]records.ModelRung, 0, len(ladder))
	var first *exit.Error
	for _, rung := range ladder {
		lane, problem := laneOf(ref, selected, rung.Lane)
		if problem == nil && !slot.AllowsGPUCount(rung.GPUs) {
			problem = exit.Named(exit.Validation, "package_model_default_invalid", "%s does not support %d GPUs", slot.Path, rung.GPUs)
		}
		if problem == nil {
			problem = requireCheckpointComponents(spec+"/"+rung.Lane, slot, lane.Components)
		}
		if _, err := canonical.Raw(lane.ManifestID); problem == nil && err != nil {
			problem = exit.Named(exit.Conflict, "rental.model_manifest_invalid",
				"Tensorhub returned an invalid manifest for %s@%s/%s", ref.String(), selected.Release, rung.Lane)
		}
		if problem != nil {
			if first == nil {
				first = problem
			}
			fmt.Fprintf(ctx.Err, "warning: %s skips rung %s: %s\n", slot.Path, rung.String(), problem.Message)
			continue
		}
		rungs = append(rungs, records.ModelRung{GPU: rung.GPU, GPUs: rung.GPUs, Lane: rung.Lane,
			Manifest: lane.ManifestID, Bytes: lane.Bytes, ComponentBytes: lane.ComponentBytes})
	}
	if len(rungs) == 0 && first != nil {
		return nil, first
	}
	return rungs, nil
}

// rebind turns a card miss into the early refusal that names its fix: the binding points
// at something the card does not offer, and `cozy package bind` is the one write path.
func rebind(problem *exit.Error, packageName, slot string) *exit.Error {
	return problem.WithRemedy("the model selection for %s slot %s names it; rebind: %s",
		packageName, slot, bindRemedy(packageName, slot))
}

func localModelOnRental(ref hub.Ref) *exit.Error {
	return exit.Named(exit.Unavailable, "rental_local_model_sync_required",
		"%s is a private local model and cannot be granted to a rented worker by path", ref.String()).
		WithRemedy("upload it under a non-local org, or explicitly sync it through the model upload workflow")
}
