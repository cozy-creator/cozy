package cli

import (
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/tfs"
)

func modelLabel(flag, value string) (string, *exit.Error) {
	if value == "" || value != strings.TrimSpace(value) || len(value) > 128 ||
		!utf8.ValidString(value) || strings.ContainsRune(value, '/') {
		return "", exit.Usagef("%s must be one non-empty path-safe label of at most 128 bytes", flag)
	}
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
			return "", exit.Usagef("%s contains a control character", flag)
		}
	}
	return value, nil
}

func releaseLaneChanges(setSpecs, removeSpecs []string) (map[string]string, []string, *exit.Error) {
	set := make(map[string]string, len(setSpecs))
	for _, spec := range setSpecs {
		name, checkpoint, ok := strings.Cut(spec, "=")
		if !ok {
			return nil, nil, exit.Usagef("--lane %q must be name=checkpoint-id", spec)
		}
		name, problem := modelLabel("--lane name", name)
		if problem != nil {
			return nil, nil, problem
		}
		checkpoint, problem = tfs.ManifestID(checkpoint)
		if problem != nil {
			return nil, nil, problem.WithRemedy("use --lane name=sha256:<64 lowercase hex>")
		}
		if _, exists := set[name]; exists {
			return nil, nil, exit.Usagef("--lane names %q more than once", name)
		}
		set[name] = checkpoint
	}
	remove := make([]string, 0, len(removeSpecs))
	removed := make(map[string]bool, len(removeSpecs))
	for _, raw := range removeSpecs {
		name, problem := modelLabel("--remove-lane", raw)
		if problem != nil {
			return nil, nil, problem
		}
		if removed[name] {
			return nil, nil, exit.Usagef("--remove-lane names %q more than once", name)
		}
		if _, alsoSet := set[name]; alsoSet {
			return nil, nil, exit.Usagef("lane %q cannot be set and removed in one update", name)
		}
		removed[name] = true
		remove = append(remove, name)
	}
	if len(set) == 0 && len(remove) == 0 {
		return nil, nil, exit.Usagef("model publish requires --lane name=checkpoint-id or --remove-lane name")
	}
	sort.Strings(remove)
	return set, remove, nil
}

func modelReleaseLaneMap(release hub.ModelRelease) (map[string]string, *exit.Error) {
	lanes := make(map[string]string, len(release.Lanes))
	for _, lane := range release.Lanes {
		name, problem := modelLabel("release lane", lane.Lane)
		if problem != nil {
			return nil, exit.Internalf("Tensorhub returned an invalid model release lane")
		}
		checkpoint, problem := tfs.ManifestID(lane.CheckpointID)
		if problem != nil || lanes[name] != "" {
			return nil, exit.Internalf("Tensorhub returned an invalid model release lane")
		}
		lanes[name] = checkpoint
	}
	return lanes, nil
}

func handleModelPublish(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	if ref.Org == "local" {
		return exit.Usagef("local/ is reserved for private aliases and cannot be published")
	}
	release, problem := modelLabel("--release", ctx.Inv.Value("--release"))
	if problem != nil {
		return problem
	}
	set, remove, problem := releaseLaneChanges(ctx.Inv.Values["--lane"],
		ctx.Inv.Values["--remove-lane"])
	if problem != nil {
		return problem
	}
	c, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	current, lookup := c.ModelRelease(hctx, ref, release)
	if lookup != nil && lookup.Code != exit.NotFound {
		return lookup
	}
	expectedRevision := int64(0)
	expectedLanes := map[string]string{}
	if lookup == nil {
		if current.Release != release || current.Revision < 1 || current.Yanked {
			return exit.Internalf("Tensorhub returned an invalid current model release")
		}
		expectedRevision = current.Revision
		expectedLanes, problem = modelReleaseLaneMap(current)
		if problem != nil {
			return problem
		}
	}
	for lane, checkpoint := range set {
		expectedLanes[lane] = checkpoint
	}
	for _, lane := range remove {
		delete(expectedLanes, lane)
	}
	reason := "cozy model publish " + ref.String() + "@" + release
	updated, problem := c.UpdateModelRelease(hctx, ref, release, expectedRevision, set, remove, reason)
	if problem != nil {
		return problem
	}
	if updated.Release != release || updated.Revision < 1 || updated.Yanked {
		return exit.Internalf("Tensorhub returned an invalid model release update")
	}
	if updated.Changed {
		if updated.Revision <= expectedRevision {
			return exit.Internalf("Tensorhub returned an invalid model release update")
		}
	} else if updated.Revision != expectedRevision {
		return exit.Internalf("Tensorhub returned an invalid model release update")
	}
	lanes, problem := modelReleaseLaneMap(updated)
	if problem != nil {
		return problem
	}
	if len(lanes) != len(expectedLanes) {
		return exit.Internalf("Tensorhub returned a different model release lane map")
	}
	for lane, checkpoint := range expectedLanes {
		if lanes[lane] != checkpoint {
			return exit.Internalf("Tensorhub returned a different model release lane map")
		}
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "model", V: ref.String()}, {K: "release", V: release},
		{K: "revision", V: updated.Revision}, {K: "lanes", V: lanes},
		{K: "status", V: "published"}, {K: "changed", V: updated.Changed},
	}, "model", "release", "revision", "lanes", "status", "changed"))
}

func handleModelYank(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	release, problem := modelLabel("--release", ctx.Inv.Value("--release"))
	if problem != nil {
		return problem
	}
	c, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	yanked, problem := c.YankModelRelease(hctx, ref, release,
		"cozy model yank "+ref.String()+"@"+release)
	if problem != nil {
		return problem
	}
	if yanked.Release != release || yanked.Revision < 1 || !yanked.Yanked {
		return exit.Internalf("Tensorhub returned an invalid yanked model release")
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "model", V: ref.String()}, {K: "release", V: release},
		{K: "revision", V: yanked.Revision}, {K: "status", V: "yanked"},
		{K: "changed", V: yanked.Changed},
	}, "model", "release", "revision", "status", "changed"))
}
