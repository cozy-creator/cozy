package cli

import (
	"cmp"
	"context"
	"slices"
	"sort"
	"strings"
	"sync"

	pep440 "github.com/aquasecurity/go-pep440-version"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// A machine picks every lane it serves (xs-042): this side only sizes a rental it is about to
// choose, from the bytes of the lane the machine's rules would serve. Everything here reads the
// model card ONCE and fails with what the card actually offers, so a choice naming a release
// or lane the card does not carry is refused with the real list rather than a bare "not
// found" hours later on a paid pod.

// cardReader reads a model card, a Hub or the cards one command already read, and resolves a
// checkpoint named by hash at the Hub.
type cardReader interface {
	ModelCard(context.Context, hub.Ref) (hub.ModelCard, *exit.Error)
	ResolveModel(ctx context.Context, spec, lane string) (hub.ModelResolution, *exit.Error)
}

// readCards reads each distinct model card refs name at once, so a run with several cold slots
// waits on one round trip rather than one per slot. A card it could not read is read again,
// and refused, where it is used.
func readCards(ctx *Context, refs []string) cardReader {
	cards := &readCardSet{hub: client(ctx), cards: map[string]hub.ModelCard{}}
	var wait sync.WaitGroup
	var mu sync.Mutex
	for _, raw := range refs {
		model, _, _, _, problem := hub.ParseModelRef(raw)
		ref, refProblem := hub.ParseRef(model)
		if problem != nil || refProblem != nil || ref.Org == "local" {
			continue
		}
		if _, seen := cards.cards[ref.String()]; seen {
			continue
		}
		cards.cards[ref.String()] = hub.ModelCard{}
		wait.Add(1)
		go func() {
			defer wait.Done()
			hctx, cancel := hub.Context()
			defer cancel()
			card, problem := cards.hub.ModelCard(hctx, ref)
			mu.Lock()
			defer mu.Unlock()
			if problem != nil {
				delete(cards.cards, ref.String())
			} else {
				cards.cards[ref.String()] = card
			}
		}()
	}
	wait.Wait()
	return cards
}

type readCardSet struct {
	hub   *hub.Client
	cards map[string]hub.ModelCard
}

func (s *readCardSet) ModelCard(ctx context.Context, ref hub.Ref) (hub.ModelCard, *exit.Error) {
	if card, read := s.cards[ref.String()]; read {
		return card, nil
	}
	return s.hub.ModelCard(ctx, ref)
}

func (s *readCardSet) ResolveModel(ctx context.Context, spec, lane string) (hub.ModelResolution, *exit.Error) {
	return s.hub.ResolveModel(ctx, spec, lane)
}

// modelReleaseCardForLane reads the card and selects `release` (the newest unyanked one
// publishing lane when empty). A release named explicitly is honoured even when yanked. A
// miss names the releases the card offers.
func modelReleaseCardForLane(ctx context.Context, c cardReader, ref hub.Ref, release, lane string) (hub.ModelCard, *hub.ModelReleaseSummary, *exit.Error) {
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

// laneOrder is the lane vocabulary (xs-042 C7) in the order a machine whose GPUs run every
// format serves a width: the width's own format, then the 8-bit ones, then 16 bits.
func laneOrder(bits int) []string {
	switch bits {
	case 16:
		return []string{"bf16", "fp16"}
	case 4:
		return []string{"nvfp4", "int8", "fp8", "bf16", "fp16"}
	}
	return []string{"int8", "fp8", "bf16", "fp16"}
}

// sizingLane is the lane a rental is sized by at bits: the first of the width's order the
// release publishes, else its largest lane. int8 and fp8 weigh the same, so the GPU's row
// does not change the size.
func sizingLane(release *hub.ModelReleaseSummary, bits int) (hub.ModelLaneSummary, bool) {
	for _, name := range laneOrder(bits) {
		for _, lane := range release.Lanes {
			if lane.Lane == name {
				return lane, true
			}
		}
	}
	var largest hub.ModelLaneSummary
	for _, lane := range release.Lanes {
		if lane.Bytes >= largest.Bytes {
			largest = lane
		}
	}
	return largest, len(release.Lanes) > 0
}

// sizedModel is model (a slot's default or the caller's choice) with the bytes a rental for
// slot is sized by: its named lane, else the lane the machine would serve at bits. Only a
// named lane or manifest is kept: an unnamed one is the machine's to pick, so a choice
// carries the sized bytes and no lane, and a default (no Choice) carries its sized lane,
// which no machine is sent.
func sizedModel(ctx context.Context, cards cardReader, slot launch.Slot, model orchestrator.ModelRef, bits int) (orchestrator.ModelRef, *exit.Error) {
	ref, problem := hub.ParseRef(model.Model)
	if problem != nil {
		return model, problem
	}
	if ref.Org == "local" {
		return model, localModelOnMachine(ref)
	}
	if model.Release == "" && model.Manifest != "" {
		spelled := ref.String() + "@" + model.Manifest
		resolved, problem := cards.ResolveModel(ctx, spelled, "")
		if problem != nil {
			return model, problem
		}
		if resolved.Model != ref.String() || resolved.ManifestID != model.Manifest || resolved.ManifestLength <= 0 || resolved.Bytes <= 0 {
			return model, exit.Named(exit.Conflict, "rental.model_resolution_changed",
				"Tensorhub returned different or incomplete checkpoint facts for %s", spelled)
		}
		model.HubCheckpoint, model.ManifestLength = true, resolved.ManifestLength
		model.Bytes, model.ComponentBytes, model.ComponentUse = resolved.Bytes, resolved.ComponentBytes, slot.ComponentUse
		return model, requireCheckpointComponents(spelled, slot, resolved.Components)
	}
	_, selected, problem := modelReleaseCardForLane(ctx, cards, ref, model.Release, model.Lane)
	if problem != nil {
		return model, problem
	}
	model.Release = selected.Release
	spelled := ref.String() + "@" + selected.Release
	lane, found := sizingLane(selected, bits)
	switch {
	case model.Lane != "":
		lane, problem = laneOf(ref, selected, model.Lane)
	case model.Manifest != "":
		if i := slices.IndexFunc(selected.Lanes, func(candidate hub.ModelLaneSummary) bool { return candidate.ManifestID == model.Manifest }); i >= 0 {
			lane = selected.Lanes[i]
		}
	case !found:
		problem = exit.New(exit.NotFound, "model %s publishes no lane", spelled)
	}
	if problem == nil && model.Manifest != "" && lane.ManifestID != model.Manifest {
		problem = exit.New(exit.NotFound, "model %s does not contain manifest %s", spelled, model.Manifest)
	}
	if problem != nil {
		return model, problem
	}
	if _, err := canonical.Raw(lane.ManifestID); err != nil {
		return model, exit.Named(exit.Conflict, "rental.model_manifest_invalid",
			"Tensorhub returned an invalid manifest for %s/%s", spelled, lane.Lane)
	}
	if problem := requireCheckpointComponents(spelled+"/"+lane.Lane, slot, lane.Components); problem != nil {
		return model, problem
	}
	if !model.Choice || model.Lane != "" || model.Manifest != "" {
		model.Lane, model.Manifest = lane.Lane, lane.ManifestID
	}
	model.Bytes, model.ComponentBytes, model.ComponentUse = lane.Bytes, lane.ComponentBytes, slot.ComponentUse
	return model, nil
}

// sizedDefault is slot's default release, owner qualifying an org-relative one, at the lane
// a rental for it is sized by. No machine is sent it: the machine loads the default itself.
func sizedDefault(ctx context.Context, cards cardReader, pkg, owner string, slot launch.Slot, bits int) (orchestrator.ModelRef, *exit.Error) {
	spelled := slot.DefaultRef(owner)
	if spelled == "" {
		return orchestrator.ModelRef{}, noDefault(pkg, slot)
	}
	model, release, _, _, problem := hub.ParseModelRef(spelled)
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	return sizedModel(ctx, cards, slot, orchestrator.ModelRef{Package: pkg, Slot: slot.Path,
		Model: model, CatalogRepository: model, Release: release, SupportedGPUs: slot.SupportedGPUCounts()}, cmp.Or(bits, slot.Bits()))
}

// exactModel is the one checkpoint raw names for a model operation (download, upload,
// quantize), which moves exact bytes: its lane when named, else the release's only one.
func exactModel(ctx *Context, raw, lane string) (orchestrator.ModelRef, *exit.Error) {
	name, release, refLane, manifest, problem := hub.ParseModelRef(raw)
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	if refLane != "" && lane != "" && refLane != lane {
		return orchestrator.ModelRef{}, exit.Usagef("%q asks for lane %q while lane %q was already selected", raw, refLane, lane)
	}
	model := orchestrator.ModelRef{Model: name, CatalogRepository: name, Release: release, Lane: cmp.Or(refLane, lane), Manifest: manifest}
	if release == "" && manifest != "" && model.Lane != "" {
		return model, exit.Usagef("a checkpoint digest without a release cannot select a lane")
	}
	ref, problem := hub.ParseRef(name)
	if problem != nil {
		return model, problem
	}
	var cards cardReader = client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	if model.Lane == "" && manifest == "" && ref.Org != "local" {
		card, selected, problem := modelReleaseCardForLane(hctx, cards, ref, release, "")
		if problem != nil {
			return model, problem
		}
		if len(selected.Lanes) != 1 {
			return model, exit.Usagef("model %s@%s has %d lanes", ref.String(), selected.Release, len(selected.Lanes)).
				WithRemedy("append /<lane> or #sha256:<digest> to select one exact manifest")
		}
		model.Lane = selected.Lanes[0].Lane
		cards = &readCardSet{hub: client(ctx), cards: map[string]hub.ModelCard{ref.String(): card}}
	}
	return sizedModel(hctx, cards, launch.Slot{}, model, 8)
}

// noDefault is a slot with no model to load: no default, or an org-relative one whose
// owner is unknown here.
func noDefault(pkg string, slot launch.Slot) *exit.Error {
	if slot.Default != "" {
		return exit.Named(exit.Unavailable, "package_model_default_owner_unknown",
			"%s default %s for model slot %s names its owner's model, and the owner is unknown here", pkg, slot.Default, slot.Path).
			WithRemedy("sign in, or choose one for this run: model.%s=org/model@release[/lane]", slot.Param)
	}
	return exit.Named(exit.NotFound, "model.default_absent", "%s declares no default for model slot %s", pkg, slot.Path).
		WithRemedy("choose one for this run: model.%s=org/model@release[/lane]", slot.Param)
}

// A machine's Host fetches every model it runs from the hub; a local/ model's bytes are in
// this client's store, which no machine reads by path (proto-062 PR 7 moves them).
func localModelOnMachine(ref hub.Ref) *exit.Error {
	return exit.Named(exit.Unavailable, "machine_local_model_sync_required",
		"%s is a private local model and cannot be granted to a machine by path", ref.String()).
		WithRemedy("upload it under a non-local org, or explicitly sync it through the model upload workflow")
}
