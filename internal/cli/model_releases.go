package cli

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/tfs"
)

func modelLabel(flag, value string) (string, *exit.Error) {
	if !hub.ValidModelLabel(value) {
		return "", exit.Usagef("%s must start alphanumeric and use at most 64 bytes of letters, digits, ., _, +, !, or -", flag)
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
	if len(expectedLanes) == 0 {
		return exit.Usagef("a model release must retain at least one lane").
			WithRemedy("yank %s@%s to hide the entire release", ref.String(), release).
			WithNext("cozy model yank " + ref.String() + " --release " + release)
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

func handleModelRetarget(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	release, problem := modelLabel("--release", ctx.Inv.Value("--release"))
	if problem != nil {
		return problem
	}
	lane, problem := modelLabel("--lane", ctx.Inv.Value("--lane"))
	if problem != nil {
		return problem
	}
	checkpoint, problem := tfs.ManifestID(ctx.Inv.Value("--to"))
	if problem != nil {
		return problem.WithRemedy("use --to sha256:<64 lowercase hex>")
	}
	c, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	updated, problem := c.RetargetModelLane(hctx, ref, release, lane, checkpoint,
		"cozy model retarget "+ref.String()+"@"+release+"/"+lane)
	if problem != nil {
		return problem
	}
	if updated.Release != release || updated.Revision < 1 || updated.Yanked {
		return exit.Internalf("Tensorhub returned an invalid model release update")
	}
	lanes, problem := modelReleaseLaneMap(updated)
	if problem != nil {
		return problem
	}
	if lanes[lane] != checkpoint {
		return exit.Internalf("Tensorhub returned a different lane pointer")
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "model", V: ref.String()}, {K: "release", V: release},
		{K: "lane", V: lane}, {K: "checkpoint", V: checkpoint},
		{K: "revision", V: updated.Revision}, {K: "status", V: "retargeted"},
		{K: "changed", V: updated.Changed},
	}, "model", "release", "lane", "checkpoint", "revision", "status", "changed"))
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

// handleModelDelete deletes a Tensorhub model or one unreleased checkpoint. Cozy never
// prompts, so --yes is the confirmation. Objects stay until the Hub's GC finds them
// unreferenced; content another model shares is never reclaimed.
func handleModelDelete(ctx *Context) *exit.Error {
	target := strings.TrimSpace(ctx.Inv.Args[0])
	name, checkpoint, one := strings.Cut(target, "#")
	ref, problem := hub.ParseRef(name)
	if problem != nil {
		return problem
	}
	if _, err := canonical.Raw(checkpoint); one && err != nil {
		return exit.Usagef("%q names no checkpoint", target).WithRemedy("write org/name#sha256:<64 hex>")
	}
	c := client(ctx)
	if !ctx.Inv.Bool("--yes") {
		return exit.Named(exit.Usage, "model.delete_unconfirmed",
			"deleting %s from %s cannot be undone", target, c.Base()).
			WithRemedy("rerun with --yes to delete it")
	}
	c, problem = ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	reason := "cozy model delete " + target
	gc := "Tensorhub's next GC reclaims objects no other model references"
	if one {
		if problem := c.RemoveCheckpoint(hctx, ref, checkpoint, reason); problem != nil {
			return problem
		}
		return emit(ctx, output.Record{Fields: []output.Field{{K: "model", V: ref.String()},
			{K: "checkpoint", V: checkpoint}, {K: "status", V: "deleted"}}, Notes: []string{gc}})
	}
	deleted, problem := c.DeleteModel(hctx, ref, reason)
	if problem != nil {
		return problem
	}
	return emit(ctx, output.Record{Fields: []output.Field{{K: "model", V: deleted.Model},
		{K: "checkpoints", V: deleted.Checkpoints}, {K: "reclaimable_objects", V: deleted.ReclaimableObjects},
		{K: "reclaimable_bytes", V: deleted.ReclaimableBytes}, {K: "status", V: "deleted"}},
		Notes: []string{gc + ": about " + output.Bytes(deleted.ReclaimableBytes) + " here"}})
}
