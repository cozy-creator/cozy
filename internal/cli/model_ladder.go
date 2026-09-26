package cli

import (
	"context"
	"fmt"
	"sort"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostgpu"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// The binding ladder is a FIT MAP (cl-166): the hub binding says which lane of a model
// release belongs on which GPU class, and the lane a run is pinned to is a function of the
// machine that runs it — the host's own accelerator locally, the winning rental remotely.
// Everything here reads the model card ONCE and fails with what the card actually offers,
// so a binding naming a release or lane the card does not carry is refused with the real
// list rather than a bare "not found" hours later on a paid pod.

// modelReleaseCard reads the card and selects `release` (the newest unyanked one when
// empty). A miss names the releases the card offers.
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
	if release == "" {
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
	for i := range card.Releases {
		candidate := &card.Releases[i]
		if candidate.Release == release && !candidate.Yanked && candidate.YankedAt == "" {
			return card, candidate, nil
		}
	}
	return card, nil, exit.Named(exit.NotFound, "model.release_not_found",
		"model %s has no available release %q (releases: %s)",
		ref.String(), release, orNone(strings.Join(availableReleases(card), ", ")))
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

// localAccelerator is the host's NVIDIA device as the ladder matches it, "" without one.
func localAccelerator(cfg config.Config) string {
	inventory := hostgpu.Probe(cfg)
	if len(inventory.GPUs) == 0 {
		return ""
	}
	return inventory.GPUs[0].Model
}

// localRung chooses a GPU-specific preference when present, otherwise the first
// declared lane. The ladder is not a local hardware allowlist: Runtime still
// checks actual encoding support, construction compatibility and memory capacity.
func localRung(ctx *Context, ladder []hub.BindingRung) (hub.BindingRung, *exit.Error) {
	accelerator := localAccelerator(ctx.Cfg)
	for _, rung := range ladder {
		if records.RungMatches(rung.GPU, accelerator) {
			return rung, nil
		}
	}
	if len(ladder) > 0 {
		return ladder[0], nil
	}
	return hub.BindingRung{}, exit.Named(exit.Validation, "model_ladder_empty",
		"the model binding has no default lanes").
		WithRemedy("declare a default lane or supply model.<param>=org/model@release/lane")
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
	rungs := make([]records.ModelRung, 0, len(spec.Binding.Ladder))
	for _, rung := range spec.Binding.Ladder {
		lane, problem := laneOf(ref, selected, rung.Lane)
		if problem != nil {
			return empty, rebind(problem, packageName, slot.Path)
		}
		if problem := requireCheckpointComponents(spec.Ref+"/"+rung.Lane, slot, lane.Components); problem != nil {
			return empty, problem
		}
		if _, err := canonical.Raw(lane.ManifestID); err != nil {
			return empty, exit.Named(exit.Conflict, "rental.model_manifest_invalid",
				"Tensorhub returned an invalid manifest for %s@%s/%s", ref.String(), selected.Release, rung.Lane)
		}
		rungs = append(rungs, records.ModelRung{GPU: rung.GPU, Lane: rung.Lane,
			Manifest: lane.ManifestID, Bytes: lane.Bytes, ComponentBytes: lane.ComponentBytes})
	}
	return orchestrator.ModelRef{Package: packageName, Slot: slot.Path, Model: ref.String(), CatalogRepository: ref.String(),
		Release: selected.Release, ComponentUse: slot.ComponentUse, Ladder: rungs}, nil
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
