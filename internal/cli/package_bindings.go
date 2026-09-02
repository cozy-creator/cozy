package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

// th-116: a package's default bindings are MUTABLE HUB POINTERS — one row per
// (package, slot path), seeded from the shipped package.toml at release commit
// and retargeted here arbitrarily afterwards. `bind` never judges
// compatibility: an incompatible retarget is the author's own foot, flagged by
// computed preflight, and a `model.<param>=` run key still overrides any default per
// invocation.

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
		Fields:    []string{"slot", "model", "release", "lane", "revision"},
		AllFields: []string{"slot", "model", "release", "lane", "revision", "updated"},
	}
	for _, row := range rows {
		l.Rows = append(l.Rows, map[string]string{
			"slot": row.Slot, "model": row.Model, "release": orNone(row.Release),
			"lane": orNone(row.Lane), "revision": output.Int(row.Revision),
			"updated": row.UpdatedAt,
		})
	}
	if len(l.Rows) == 0 {
		l.Next = []string{"cozy package bind " + ref.String() + " <slot-path> org/model@release"}
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
	model, release, lane, problem := parseBindingTarget(ctx.Inv.Args[2])
	if problem != nil {
		return problem
	}
	c, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	// CAS: read the row's current revision (0 for an unseeded slot), then move.
	// An exact replay is a hub-side revision-keeping no-op, so this pair never
	// invents a conflict for the same intent sent twice.
	current, problem := c.PackageBindings(hctx, ref)
	if problem != nil {
		return problem
	}
	expected := int64(0)
	for _, row := range current {
		if row.Slot == slot {
			expected = row.Revision
		}
	}
	written, problem := c.BindPackageSlot(hctx, ref, slot, model, release, lane, expected,
		"cozy package bind "+ref.String()+" "+slot)
	if problem != nil {
		return problem
	}
	if written.Binding.Slot != slot || written.Binding.Model != model ||
		written.Binding.Release != release || written.Binding.Lane != lane ||
		written.Binding.Revision < 1 {
		return exit.Internalf("Tensorhub returned a different binding row")
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "slot", V: slot},
		{K: "model", V: model}, {K: "release", V: orNone(release)},
		{K: "lane", V: orNone(lane)},
		{K: "revision", V: written.Binding.Revision}, {K: "status", V: "bound"},
		{K: "changed", V: written.Changed},
	}, "package", "slot", "model", "release", "lane", "revision", "status", "changed"))
}

// parseModelRef reads the ONE model-ref grammar (cl-109):
//
//	org/model[@release[/lane]][#sha256:<hex>]
//
// It serves the `model.<param>=` run key and `package bind` alike; a lane narrows to
// one encoding and a manifest to one exact release artifact.
func parseModelRef(raw string) (model, release, lane, manifest string, problem *exit.Error) {
	spec := strings.TrimSpace(raw)
	if strings.Count(spec, "#") > 1 {
		return "", "", "", "", exit.Usagef("%q carries more than one manifest", raw)
	}
	rest, digest, hasManifest := strings.Cut(spec, "#")
	if hasManifest {
		if digest == "" {
			return "", "", "", "", exit.Usagef("%q carries an empty manifest", raw)
		}
		if _, err := canonical.Raw(digest); err != nil {
			return "", "", "", "", exit.Usagef("%q is not a sha256 model manifest", digest)
		}
		manifest = digest
	}
	if strings.Count(rest, "@") > 1 {
		return "", "", "", "", exit.Usagef("%q carries more than one release", rest)
	}
	model, versioned, pinned := strings.Cut(rest, "@")
	if _, e := hub.ParseRef(model); e != nil {
		return "", "", "", "", e
	}
	if pinned {
		var sliced bool
		release, lane, sliced = strings.Cut(versioned, "/")
		if release == "" || sliced && lane == "" || strings.ContainsAny(lane, " \t") {
			return "", "", "", "", exit.Usagef("%q is not org/model[@release[/lane]][#sha256:<hex>]", raw)
		}
	}
	return model, release, lane, manifest, nil
}

// parseBindingTarget reads the verb's slice of the one ref grammar. A binding row pins
// org/model[@release[/lane]]; an exact manifest is per-run narrowing and rides the
// `model.<param>=` run key instead.
func parseBindingTarget(raw string) (model, release, lane string, problem *exit.Error) {
	model, release, lane, manifest, problem := parseModelRef(raw)
	if problem != nil {
		return "", "", "", problem
	}
	if manifest != "" {
		return "", "", "", exit.Usagef("%q pins an exact manifest and a binding cannot hold one", raw).
			WithRemedy("bind org/model[@release[/lane]]; narrow to one manifest per run with model.<param>=…#%s", manifest)
	}
	return model, release, lane, nil
}
