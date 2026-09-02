package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

// th-116: a package's default bindings are MUTABLE HUB POINTERS — one row per
// (package, slot path), seeded from the shipped package.toml at release commit
// and retargeted here arbitrarily afterwards. `bind` never judges
// compatibility: an incompatible retarget is the author's own foot, flagged by
// computed preflight, and `--model` still overrides any default per invocation.

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

// parseBindingTarget reads the verb's org/model[@release[/lane]] spelling.
func parseBindingTarget(raw string) (model, release, lane string, problem *exit.Error) {
	spec := strings.TrimSpace(raw)
	model, rest, pinned := strings.Cut(spec, "@")
	if _, problem = hub.ParseRef(model); problem != nil {
		return "", "", "", problem
	}
	if !pinned {
		return model, "", "", nil
	}
	release, lane, _ = strings.Cut(rest, "/")
	if release == "" || strings.ContainsAny(lane, " \t") {
		return "", "", "", exit.Usagef("%q is not org/model[@release[/lane]]", raw)
	}
	return model, release, lane, nil
}
