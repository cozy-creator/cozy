package cli

import (
	"context"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// Owner overrides are mutable Hub rows, one per (package, slot path). Authored
// defaults stay in the published function interface; publishing seeds no rows.
// Bind verifies the slot against the latest package interface and every lane
// against the model release. Unbind removes the override so authored defaults apply.

// handlePackageBindings shows every model slot one package release declares: the
// release's authored default ladder beside the owner's override, and which one a run
// uses. An override for a slot the release does not declare is listed too.
func handlePackageBindings(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	c := client(ctx)
	rows, problem := c.PackageBindings(hctx, ref)
	if problem != nil {
		return packageHubProblem(ctx, ref.String(), problem)
	}
	release := ""
	if v := ctx.Inv.Values["--version"]; len(v) > 0 {
		release = strings.TrimSpace(v[len(v)-1])
	}
	if release == "" {
		card, problem := c.PackageCard(hctx, ref)
		if problem != nil {
			return packageHubProblem(ctx, ref.String(), problem)
		}
		if release, problem = newestPackageRelease(card.Releases); problem != nil {
			return problem
		}
	}
	detail, problem := c.PackageRelease(hctx, ref, release)
	if problem != nil {
		return problem
	}
	packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return problem
	}
	overrides := make(map[string]hub.PackageBindingRow, len(rows))
	for _, row := range rows {
		overrides[row.Slot] = row
	}
	l := output.List{
		Name:      "bindings",
		Fields:    []string{"slot", "source", "model", "release", "ladder", "default"},
		AllFields: []string{"slot", "source", "model", "release", "ladder", "default", "revision", "updated"},
		Machine:   []string{"package_release"},
		Lead:      []string{"Model slots of " + ref.String() + "@" + release + ":"},
	}
	add := func(slot, source string, effective *hub.PackageBindingRow, authored *hub.PackageBindingRow) {
		row := map[string]string{"slot": slot, "source": source, "package_release": release}
		if effective != nil {
			row["model"], row["release"] = effective.Model, effective.Release
			row["ladder"] = hub.LadderText(effective.Ladder)
			if effective.Revision > 0 {
				row["revision"], row["updated"] = output.Int(effective.Revision), effective.UpdatedAt
			}
		}
		if authored != nil {
			row["default"] = authored.Ref() + " " + hub.LadderText(authored.Ladder)
		}
		l.Rows = append(l.Rows, row)
	}
	declared, serving := map[string]bool{}, map[string]bool{}
	for _, entrypoint := range packageInterface.Entrypoints {
		for _, slot := range entrypoint.Models {
			serving[slot.Path] = true
		}
	}
	slots := declaredModelSlots(packageInterface.Entrypoints, packageInterface.Jobs)
	sort.Slice(slots, func(i, j int) bool { return slots[i].Path < slots[j].Path })
	for _, slot := range slots {
		declared[slot.Path] = true
		authored := slot.Default(ref.Org)
		if override, bound := overrides[slot.Path]; bound {
			add(slot.Path, "override", &override, authored)
			continue
		}
		if authored != nil {
			add(slot.Path, "default", authored, authored)
			continue
		}
		add(slot.Path, "none", nil, nil)
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Slot < rows[j].Slot })
	for _, row := range rows {
		if !declared[row.Slot] {
			add(row.Slot, "override (undeclared)", &row, nil)
		}
	}
	if len(l.Rows) == 0 {
		l.Notes = []string{ref.String() + "@" + release + " declares no model slots."}
	}
	for _, row := range l.Rows {
		// A job's model input may come from its caller; only a serving slot needs one.
		if row["source"] == "none" && serving[row["slot"]] {
			l.Next = []string{"cozy package bind " + ref.String() + " " + row["slot"] + " <org/model@release> --gpu <GPU>=<lane>"}
			break
		}
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
	selected, problem := verifyLadderOnCard(hctx, c, modelRef, release, ladder)
	if problem != nil {
		return problem
	}
	warnYanked(ctx, modelRef, selected)
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
	if written.Binding.Slot != slot || written.Binding.Revision < 1 {
		return exit.Internalf("Tensorhub returned a different binding row")
	}
	record := compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "slot", V: slot},
		{K: "model", V: written.Binding.Model}, {K: "release", V: written.Binding.Release},
		{K: "ladder", V: hub.LadderText(written.Binding.Ladder)},
		{K: "revision", V: written.Binding.Revision}, {K: "status", V: "bound"},
		{K: "changed", V: written.Changed},
	}, "package", "slot", "model", "release", "ladder", "revision", "status", "changed")
	record.Notes = changed(ctx, ref)
	return emit(ctx, record)
}

// changed records that this command changed ref's releases or bindings: every later run
// carries the new binding revision, and a machine holding what it resolved before re-resolves.
func changed(ctx *Context, ref hub.Ref) []string {
	store, problem := records.Open(home.Paths(ctx.Cfg.Home).DB)
	if problem == nil {
		defer store.Close()
		_, problem = store.ChangeBindingRevision()
	}
	if problem != nil {
		return []string{"machines keep what they resolved of " + ref.String() + ": " + problem.Message}
	}
	return nil
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
	record := compactRecord([]output.Field{
		{K: "package", V: ref.String()}, {K: "slot", V: slot},
		{K: "status", V: "unbound"}, {K: "changed", V: result.Changed},
	}, "package", "slot", "status", "changed")
	record.Notes = changed(ctx, ref)
	return emit(ctx, record)
}

// packageBindingRevision reads the rows as written, so an unusable row can still be
// rebound or unbound: that is its repair.
func packageBindingRevision(ctx context.Context, c *hub.Client, ref hub.Ref, slot string) (int64, *exit.Error) {
	rows, problem := c.PackageBindingRows(ctx, ref)
	if problem != nil {
		return 0, problem
	}
	var revision int64
	for _, row := range rows {
		if row.Slot == slot {
			revision = max(revision, row.Revision)
		}
	}
	return revision, nil
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

// verifyLadderOnCard refuses a release the card does not carry, or a rung lane the release
// does not carry, naming what it does.
func verifyLadderOnCard(ctx context.Context, c *hub.Client, ref hub.Ref, release string,
	ladder []hub.BindingRung) (*hub.ModelReleaseSummary, *exit.Error) {
	_, selected, problem := modelReleaseCard(ctx, c, ref, release)
	if problem != nil {
		return nil, problem
	}
	for _, rung := range ladder {
		lane, problem := laneOf(ref, selected, rung.Lane)
		if problem != nil {
			return nil, problem
		}
		if _, err := canonical.Raw(lane.ManifestID); err != nil {
			return nil, exit.Named(exit.Conflict, "rental.model_manifest_invalid",
				"Tensorhub returned an invalid manifest for %s@%s/%s", ref.String(), release, rung.Lane)
		}
	}
	return selected, nil
}

// parseLadder reads the ordered `--gpu <GPU>=<lane>` rungs.
func parseLadder(raw []string) ([]hub.BindingRung, *exit.Error) {
	ladder := make([]hub.BindingRung, 0, len(raw))
	for _, spec := range raw {
		gpu, lane, ok := strings.Cut(strings.TrimSpace(spec), "=")
		if !ok {
			return nil, exit.Usagef("--gpu %q is not <GPU>=<lane>", spec)
		}
		count := 0
		if prefix, model, found := strings.Cut(gpu, "x"); found {
			if parsed, err := strconv.Atoi(prefix); err == nil {
				if parsed < 1 {
					return nil, exit.Usagef("GPU count must be positive")
				}
				count, gpu = parsed, model
			}
		}
		ladder = append(ladder, hub.BindingRung{GPU: gpu, GPUs: count, Lane: lane})
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
