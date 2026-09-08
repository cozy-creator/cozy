package cli

import (
	"context"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
)

// A package's default bindings are MUTABLE HUB ROWS — one per (package, slot path) — and
// `cozy package bind` sets them; Tensorhub removes defaults whose slots disappear.
// package.toml carries no bindings and publish seeds none. A row names a model release and its ladder, the fit
// map from GPU class to lane. Bind verifies both against the hub before writing — the
// slot against the package's latest published interface, the release and every lane
// against the model card — so a binding can never name what the card does not offer.

func handlePackageBindings(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	rows, problem := client(ctx).PackageBindings(hctx, ref)
	if problem != nil {
		return problem
	}
	l := output.List{
		Name:      "bindings",
		Fields:    []string{"slot", "model", "release", "ladder", "revision"},
		AllFields: []string{"slot", "model", "release", "ladder", "revision", "updated"},
	}
	for _, row := range rows {
		l.Rows = append(l.Rows, map[string]string{
			"slot": row.Slot, "model": row.Model, "release": row.Release,
			"ladder": hub.LadderText(row.Ladder), "revision": output.Int(row.Revision),
			"updated": row.UpdatedAt,
		})
	}
	if len(l.Rows) == 0 {
		l.Notes = []string{"No owner overrides."}
		l.Next = []string{"cozy run " + ref.String() + " --describe"}
	}
	return emit(ctx, l)
}

func handlePackageBind(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	slot := strings.TrimSpace(ctx.Inv.Args[1])
	if slot == "" || strings.ContainsAny(slot, " \t") {
		return exit.Usagef("%q is not a slot path such as generate.models.model", ctx.Inv.Args[1])
	}
	model, release, problem := parseBindingTarget(ctx.Inv.Args[2])
	if problem != nil {
		return problem
	}
	ladder, problem := parseLadder(ctx.Inv.Values["--gpu"])
	if problem != nil {
		return problem
	}
	c, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	if problem := verifyPackageSlot(hctx, c, ref, slot); problem != nil {
		return problem
	}
	modelRef, problem := hub.ParseRef(model)
	if problem != nil {
		return problem
	}
	if problem := verifyLadderOnCard(hctx, c, modelRef, release, ladder); problem != nil {
		return problem
	}
	// CAS: read the row's current revision (0 for an unbound slot), then move. An exact
	// replay is a hub-side revision-keeping no-op, so this pair never invents a conflict
	// for the same intent sent twice.
	expected, problem := packageBindingRevision(hctx, c, ref, slot)
	if problem != nil {
		return problem
	}
	written, problem := c.BindPackageSlot(hctx, ref, slot, model, release, ladder, expected,
		"cozy package bind "+ref.String()+" "+slot)
	if problem != nil {
		return problem
	}
	if written.Binding.Slot != slot || written.Binding.Model != model ||
		written.Binding.Release != release || written.Binding.Revision < 1 ||
		hub.LadderText(written.Binding.Ladder) != hub.LadderText(ladder) {
		return exit.Internalf("Tensorhub returned a different binding row")
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "slot", V: slot},
		{K: "model", V: model}, {K: "release", V: release},
		{K: "ladder", V: hub.LadderText(ladder)},
		{K: "revision", V: written.Binding.Revision}, {K: "status", V: "bound"},
		{K: "changed", V: written.Changed},
	}, "package", "slot", "model", "release", "ladder", "revision", "status", "changed"))
}

func handlePackageUnbind(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	slot := strings.TrimSpace(ctx.Inv.Args[1])
	if slot == "" || strings.ContainsAny(slot, " \t") {
		return exit.Usagef("%q is not a slot path such as generate.models.model", slot)
	}
	c, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	expected, problem := packageBindingRevision(hctx, c, ref, slot)
	if problem != nil {
		return problem
	}
	result, problem := c.UnbindPackageSlot(hctx, ref, slot, expected, "cozy package unbind "+ref.String()+" "+slot)
	if problem != nil {
		return problem
	}
	if result.Slot != slot {
		return exit.Internalf("Tensorhub reset a different binding slot")
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "slot", V: slot},
		{K: "status", V: "unbound"}, {K: "changed", V: result.Changed},
	}, "package", "slot", "status", "changed"))
}

func packageBindingRevision(ctx context.Context, c *hub.Client, ref hub.Ref, slot string) (int64, *exit.Error) {
	rows, problem := c.PackageBindings(ctx, ref)
	if problem != nil {
		return 0, problem
	}
	for _, row := range rows {
		if row.Slot == slot {
			return row.Revision, nil
		}
	}
	return 0, nil
}

// verifyPackageSlot refuses a slot path the package's latest published interface does
// not declare, naming the slots it does.
func verifyPackageSlot(ctx context.Context, c *hub.Client, ref hub.Ref, slot string) *exit.Error {
	card, problem := c.PackageCard(ctx, ref)
	if problem != nil {
		return problem
	}
	release, problem := newestPackageRelease(card.Releases)
	if problem != nil {
		return problem
	}
	detail, problem := c.PackageRelease(ctx, ref, release)
	if problem != nil {
		return problem
	}
	packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return problem
	}
	var declared []string
	for _, callables := range [][]launch.Entrypoint{packageInterface.Entrypoints, packageInterface.Jobs} {
		for i := range callables {
			for _, candidate := range callables[i].Models {
				if candidate.Path == slot {
					return nil
				}
				declared = append(declared, candidate.Path)
			}
		}
	}
	sort.Strings(declared)
	return exit.Named(exit.NotFound, "binding.slot_not_found",
		"%s@%s declares no model slot %q (slots: %s)", ref.String(), release, slot,
		orNone(strings.Join(declared, ", ")))
}

// verifyLadderOnCard refuses a release the card does not offer unyanked, or a rung lane
// the release does not carry, naming what it does.
func verifyLadderOnCard(ctx context.Context, c *hub.Client, ref hub.Ref, release string,
	ladder []hub.BindingRung) *exit.Error {
	_, selected, problem := modelReleaseCard(ctx, c, ref, release)
	if problem != nil {
		return problem
	}
	for _, rung := range ladder {
		lane, problem := laneOf(ref, selected, rung.Lane)
		if problem != nil {
			return problem
		}
		if _, err := canonical.Raw(lane.ManifestID); err != nil {
			return exit.Named(exit.Conflict, "rental.model_manifest_invalid",
				"Tensorhub returned an invalid manifest for %s@%s/%s", ref.String(), release, rung.Lane)
		}
	}
	return nil
}

// parseLadder reads the ordered `--gpu <GPU>=<lane>` rungs.
func parseLadder(raw []string) ([]hub.BindingRung, *exit.Error) {
	ladder := make([]hub.BindingRung, 0, len(raw))
	for _, spec := range raw {
		gpu, lane, ok := strings.Cut(strings.TrimSpace(spec), "=")
		if !ok {
			return nil, exit.Usagef("--gpu %q is not <GPU>=<lane>", spec)
		}
		ladder = append(ladder, hub.BindingRung{GPU: gpu, Lane: lane})
	}
	if problem := hub.ValidateLadder(ladder); problem != nil {
		return nil, problem
	}
	return ladder, nil
}

// parseBindingTarget reads the verb's slice of the one ref grammar: a binding row pins
// org/model@release, its lanes ride the ladder, and an exact manifest is per-run
// narrowing on the `model.<param>=` run key.
func parseBindingTarget(raw string) (model, release string, problem *exit.Error) {
	model, release, lane, manifest, problem := hub.ParseModelRef(raw)
	if problem != nil {
		return "", "", problem
	}
	switch {
	case manifest != "":
		return "", "", exit.Usagef("%q pins an exact manifest and a binding cannot hold one", raw).
			WithRemedy("bind org/model@release; narrow to one manifest per run with model.<param>=…#%s", manifest)
	case lane != "":
		return "", "", exit.Usagef("%q names a lane; a binding's lanes ride its ladder", raw).
			WithRemedy("bind org/model@release --gpu <GPU>=%s", lane)
	case release == "":
		return "", "", exit.Usagef("%q names no release; a binding pins org/model@release", raw)
	}
	return model, release, nil
}
