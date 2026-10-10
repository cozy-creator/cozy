package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// open resolves the local layout and the ONE lifecycle database. Mutating verbs
// additionally take the single-writer lock, waiting while a live command holds it.
func open(cfg config.Config, write bool) (home.Layout, *records.Store, *install.Writer, *exit.Error) {
	l, e := home.Open(cfg.Home)
	if e != nil {
		return l, nil, nil, e
	}
	var w *install.Writer
	if write {
		if w, e = home.WaitWriter(l, false, os.Stderr); e != nil {
			return l, nil, nil, e
		}
	}
	st, e := records.Open(l.DB)
	if e != nil {
		if w != nil {
			w.Unlock()
		}
		return l, nil, nil, e
	}
	return l, st, w, nil
}

func handleInstall(ctx *Context) *exit.Error {
	if explicitPackageDirectory(ctx.Inv.Args[0]) {
		return handleDirectoryInstall(ctx)
	}
	if ctx.Inv.Bool("--editable") {
		return exit.Usagef("--editable requires an explicit package directory")
	}
	if ctx.Inv.Value("--rental") != "" {
		return handleRentalPackageInstall(ctx)
	}
	return handleRegistryInstall(ctx)
}

func explicitPackageDirectory(value string) bool {
	value = strings.TrimSpace(value)
	return value == "." || value == ".." || filepath.IsAbs(value) ||
		strings.HasPrefix(value, "./") || strings.HasPrefix(value, "../") ||
		strings.HasPrefix(value, `.\`) || strings.HasPrefix(value, `..\`)
}

// handleDirectoryInstall installs local/<name> from a directory on this computer: as it is now,
// or --editable, following its files. Unchanged since its install, it is already installed.
// With --rental the same code is installed there too, only the objects it lacks written.
func handleDirectoryInstall(ctx *Context) *exit.Error {
	path := strings.TrimSpace(ctx.Inv.Args[0])
	if ctx.Inv.Value("--version") != "" {
		return exit.Usagef("an explicit package directory does not take --version").
			WithRemedy("use `cozy package install %s` by itself", path)
	}
	editable := ctx.Inv.Bool("--editable")
	pack, problem := packagepublish.PrepareLocalFrom(path)
	if problem != nil {
		return problem
	}
	defer pack.Close()
	files, bytes, problem := pack.SourceInventory()
	if problem != nil {
		return problem
	}
	stats, _, problem := pack.SourceStats()
	if problem != nil {
		return problem
	}
	ref, problem := install.ParseRef("local/" + pack.Name)
	if problem != nil {
		return problem
	}
	l, st, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return problem
	}
	defer st.Close()
	defer writer.Unlock()
	result := &install.Result{Idempotent: true}
	if prior, _ := activeInstall(st, "", ref.Package); prior != nil && prior.SourceKind == "local" && prior.Version == pack.Release &&
		editable == !prior.Captured() && (!editable || prior.SourceRef == pack.Tree) && packagepublish.SourceStatsUnchanged(prior.Dir, stats) {
		result.Install = *prior
	} else if problem = packagePublishStage(ctx, "Creating local package environment", func() *exit.Error {
		var installProblem *exit.Error
		result, installProblem = install.Run(l, st, install.Request{Ref: ref, Force: true,
			Local: &install.LocalSource{Bytes: bytes, Files: files, Frozen: !editable,
				Package: ref.Package, Release: pack.Release, Tree: pack.Tree, Namespace: commandNamespace(ctx)}})
		return installProblem
	}); problem != nil {
		return problem
	}
	cleanup := reclaimInstallResult(l, st, result)
	rental := ctx.Inv.Value("--rental")
	if rental == "" {
		return emitInstallResult(ctx, result, cleanup...)
	}
	writer.Unlock()
	selection := records.RentalInstallSelection{Package: ref.Package, Release: pack.Release, Hub: ctx.Cfg.HubURL, Local: result.Install.ID}
	return enqueueRentalInstall(ctx, rental, selection, !ctx.Inv.Bool("--no-wait"))
}

func handlePackageRecover(ctx *Context) *exit.Error {
	layout, st, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return problem
	}
	defer st.Close()
	defer writer.Unlock()
	count, problem := st.RecoverPackageInventory(ctx.Inv.Args[0], layout.Installs)
	if problem != nil {
		return problem
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "status", V: "recovered"}, {K: "installs", V: count},
		{K: "source", V: ctx.Inv.Args[0]}, {K: "models_changed", V: false},
	}, "status", "installs", "models_changed"))
}

// runExample renders one concrete invoke command for the install's first
// callable so "what now?" is answered by the install itself. Best effort: a
// package whose PackageInterface cannot be read still gets the bare form.
func runExample(inst records.PackageInstall) string {
	example := "cozy run " + inst.Package
	raw, err := os.ReadFile(launch.PackageInterfacePath(inst.Dir))
	if err != nil {
		return example
	}
	d, problem := launch.DecodePackageInterface(raw)
	if problem != nil {
		return example
	}
	names := d.PublicNames()
	if len(names) == 0 {
		return example
	}
	callable, problem := d.Function(names[0])
	if problem != nil {
		return example
	}
	// The ONE contract printer (launch.UsageLine) renders the hint, the `--describe`
	// contract, and the submit refusal's remedy from the same PackageInterface facts.
	return launch.UsageLine(inst.Package+"/"+callable.Name, callable)
}

// Reclaim while the caller still owns the install writer. Reporting after model
// prefetch must not acquire another writer or turn an active install into failure.
func reclaimInstallResult(l home.Layout, st *records.Store, res *install.Result) []output.Field {
	if res.Idempotent || res.Superseded == "" {
		return nil
	}
	reclaimed, problem := install.Reclaim(l, st, res.Superseded)
	if problem != nil {
		res.Warnings = append(res.Warnings,
			"the new version is active; cleanup of the prior version was deferred: "+problem.Message)
		return nil
	}
	return []output.Field{{K: "superseded", V: res.Superseded}, {K: "reclaimed", V: output.Bytes(reclaimed)}}
}

func emitInstallResult(ctx *Context, res *install.Result, cleanup ...output.Field) *exit.Error {
	inst := res.Install
	status := "installed"
	if res.Idempotent {
		status = "already installed"
	}
	fields := []output.Field{
		{K: "package", V: inst.Package}, {K: "major", V: inst.Major},
		{K: "version", V: inst.Version}, {K: "status", V: status},
		{K: "disk", V: diskText(inst)}, {K: "changed", V: !res.Idempotent},
		{K: "install_id", V: inst.ID}, {K: "source", V: inst.SourceKind + " " + inst.SourceRef},
		{K: "verified", V: inst.Verified},
		{K: "python", V: inst.Python}, {K: "uv", V: inst.UV},
		{K: "platform", V: inst.Platform}, {K: "cuda_extra", V: orNone(inst.Extra)},
		{K: "packages", V: inst.Packages},
		{K: "closure", V: strings.ReplaceAll(inst.Closure, "\n", " ")},
		{K: "placement_set", V: inst.PlacementSetDigest},
	}
	keys := []string{"package", "version", "status"}
	if hub := installHub(ctx, inst); hub != "" {
		fields, keys = append(fields, output.Field{K: "hub", V: hub}), append(keys, "hub")
	}
	if res.Idempotent {
		rec := compactRecord(fields, keys...)
		rec.Next = []string{runExample(inst)}
		return emit(ctx, rec)
	}
	fields = append(fields,
		output.Field{K: "staged", V: fmt.Sprintf("%d files, %s", res.Files, output.Bytes(res.Bytes))},
		output.Field{K: "timings", V: timingsText(res.Timings)},
	)
	fields = append(fields, cleanup...)
	rec := compactRecord(fields, append(keys, "disk")...)
	rec.Notes = append(rec.Notes, res.Warnings...)
	rec.Next = []string{runExample(inst), "cozy package list"}
	return emit(ctx, rec)
}

// rentalPackages is `cozy package list --rental=<name>`: the packages that rental's machine holds,
// each with the level it holds it at and the hub it came from.
func rentalPackages(ctx *Context) *exit.Error {
	_, envs, problem := rentalEnvironments(ctx)
	if problem != nil {
		return problem
	}
	l := output.List{Name: "packages", Fields: []string{"package", "release", "level", "hub"}, AllFields: []string{"package", "release", "level", "hub", "installation"}}
	for _, env := range envs {
		hub := ""
		if env.Hub != "" {
			hub = ctx.Cfg.HubLabel(env.Hub)
		}
		l.Rows = append(l.Rows, map[string]string{"package": env.Package, "release": env.Release, "level": env.Level, "hub": hub, "installation": env.Installation})
	}
	return emit(ctx, l)
}

func handleLs(ctx *Context) *exit.Error {
	if ctx.Inv.Value("--rental") != "" {
		return rentalPackages(ctx)
	}
	_, st, _, e := open(ctx.Cfg, false)
	if e != nil {
		return e
	}
	defer st.Close()
	rows, e := st.Installed()
	if e != nil {
		return e
	}
	l := output.List{
		Name:      "packages",
		Fields:    []string{"package", "version", "python", "python_status", "size", "dependencies"},
		AllFields: []string{"package", "major", "version", "python", "python_status", "size", "dependencies", "placement_set", "install_id", "source", "hub", "scope", "synced", "verified", "installed"},
		Bytes:     []string{"size", "dependencies"},
		Notes:     []string{"org/name references resolve at hub " + ctx.Cfg.HubText(ctx.Cfg.HubURL) + "; another hub's need --tensorhub=<hub>."},
	}
	inventory, pythonProblem := hostruntime.PythonExecutors(context.Background())
	rows = slices.DeleteFunc(rows, func(inst records.PackageInstall) bool { return !inHubScope(ctx, inst) })
	// Published installations first: a page of local captures must not hide them.
	captured := func(inst records.PackageInstall) int {
		if inst.SourceKind == "tensorhub" {
			return 0
		}
		return 1
	}
	slices.SortStableFunc(rows, func(a, b records.PackageInstall) int { return captured(a) - captured(b) })
	for _, inst := range rows {
		pythonStatus := "supported"
		if pythonProblem != nil {
			pythonStatus = "unknown: " + pythonProblem.Message
		} else if _, problem := inventory.Select("", inst.Python); problem != nil {
			pythonStatus = "unusable: " + problem.Message
			if launch.ProvisionablePython(inventory.ProvisionableMinors, "", inst.Python, inventory.SupportedMinors) {
				pythonStatus = "provisionable: installed on demand"
			}
		}
		synced, e := syncedText(st, inst)
		if e != nil {
			return e
		}
		l.Rows = append(l.Rows, map[string]string{
			"package":       inst.Package,
			"major":         fmt.Sprintf("v%d", inst.Major),
			"version":       inst.Version,
			"install_id":    inst.ID,
			"size":          output.Int(inst.BytesExcl),
			"dependencies":  output.Int(inst.BytesShared),
			"python":        inst.Python,
			"python_status": pythonStatus,
			"uv":            inst.UV,
			"cuda_extra":    orNone(inst.Extra),
			"packages":      fmt.Sprintf("%d", inst.Packages),
			"closure":       strings.ReplaceAll(inst.Closure, "\n", " "),
			"placement_set": inst.PlacementSetDigest,
			"source":        inst.SourceKind + " " + inst.SourceRef,
			"hub":           installHub(ctx, inst),
			"scope":         installHubScope(ctx, inst),
			"synced":        synced,
			"verified":      fmt.Sprintf("%t", inst.Verified),
			"installed":     inst.CreatedAt,
		})
	}
	l.Fields = append(l.Fields, "hub", "scope")
	if !everyHub(ctx) {
		l.Notes = append(l.Notes, "Showing that hub's and local installations; omit --tensorhub to list every hub's.")
	}
	if len(l.Rows) == 0 {
		l.Next = []string{"cozy package search"}
	}
	return emit(ctx, l)
}

// installHub names the hub a published install came from; a local install has none.
func installHub(ctx *Context, inst records.PackageInstall) string {
	if inst.SourceKind != "tensorhub" || inst.Hub == "" {
		return ""
	}
	return ctx.Cfg.HubLabel(ctx.forHub(inst.Hub).Cfg.HubURL)
}

// syncedText is what the daemon's source watcher last recorded about an editable install's
// tree (cl-097): `synced` once its rebuild activated this install, `stale <error>` when the
// last rebuild was refused, nothing for a published install or a tree no daemon has read.
func syncedText(st *records.Store, inst records.PackageInstall) (string, *exit.Error) {
	if inst.SourceKind != "local" {
		return "", nil
	}
	event, e := st.LastPackageEvent(inst.Package)
	if e != nil || event == nil {
		return "", e
	}
	switch event.Type {
	case "package.refreshed":
		if install, _ := event.Payload["install"].(string); install == inst.ID {
			return "synced", nil
		}
	case "package.refresh_failed":
		cause, _ := event.Payload["error"].(string)
		return "stale " + cause, nil
	}
	return "", nil
}

func handleRm(ctx *Context) *exit.Error {
	if ctx.Inv.Value("--rental") != "" {
		return rentalPackageRemove(ctx)
	}
	l, st, w, e := open(ctx.Cfg, true)
	if e != nil {
		return e
	}
	defer st.Close()
	defer w.Unlock()

	removed := output.List{
		Name:      "packages",
		Fields:    []string{"package", "reclaimed"},
		AllFields: []string{"package", "major", "reclaimed", "install_id"},
	}
	var freed int64
	unreferenced, e := st.Unreferenced()
	if e != nil {
		return e
	}
	for _, arg := range ctx.Inv.Args {
		pins, e := st.Pins(strings.TrimSpace(arg))
		if e != nil {
			return e
		}
		if len(pins) == 0 && !slices.ContainsFunc(unreferenced, func(u records.PackageInstall) bool { return u.Package == strings.TrimSpace(arg) }) {
			return exit.Named(exit.NotFound, "package.not_installed", "%s is not installed on this computer", arg).
				WithNext("cozy package list")
		}
	}
	for _, arg := range ctx.Inv.Args {
		ref, e := install.ParseRef(arg)
		if e != nil {
			return e
		}
		if ref.HasMajor {
			return exit.Usagef("a package removal does not take a version").
				WithRemedy("use `cozy package remove %s`; only one version can be installed", ref.Package)
		}
		targets, e := st.Pins(ref.Package)
		if e != nil {
			return e
		}
		// One pin per hub: every hub's installation of the name, unless --tensorhub narrows.
		targets = slices.DeleteFunc(targets, func(p records.Pin) bool {
			return !everyHub(ctx) && p.Hub != records.ReferenceHub(ctx.Cfg.HubURL, p.Package)
		})
		if len(targets) == 0 {
			if other, _ := st.Pins(ref.Package); len(other) > 0 {
				return foreignPackageRemoval(ctx, other[0].Hub, ref.Package)
			}
		}
		for _, p := range targets {
			n, e := install.Remove(l, st, p.Hub, p.Package, p.Major)
			if e != nil {
				return e
			}
			if n == 0 {
				removed.Notes = append(removed.Notes,
					"package files are retained until accepted work finishes and its worker exits")
			}
			freed += n
			removed.Rows = append(removed.Rows, map[string]string{
				"package":    p.Package,
				"major":      fmt.Sprintf("v%d", p.Major),
				"install_id": p.InstallID,
				"reclaimed":  output.Bytes(n),
			})
		}
	}
	// Removing a package also clears superseded installs for the same selected
	// major. Active requests/workers were fenced above and the database claim rechecks.
	if unreferenced, e = st.Unreferenced(); e != nil {
		return e
	}
	for _, superseded := range unreferenced {
		selected := false
		for _, arg := range ctx.Inv.Args {
			ref, problem := install.ParseRef(arg)
			if problem != nil {
				return problem
			}
			selected = ref.Package == superseded.Package
			if selected {
				break
			}
		}
		if !selected {
			continue
		}
		// Unreferenced rows omit their origin; read the complete install before
		// reclaiming so --tensorhub leaves a same-name package from another hub untouched.
		full, problem := st.Install(superseded.ID)
		if problem != nil {
			return problem
		}
		if full == nil || !inHubScope(ctx, *full) {
			continue
		}
		n, problem := install.Reclaim(l, st, superseded.ID)
		if problem != nil {
			return problem
		}
		freed += n
		// A reclaimed unpinned install is a removal the human sees too; without its row
		// the verb reported "No packages found." and changed: false while deleting it.
		removed.Rows = append(removed.Rows, map[string]string{
			"package":    superseded.Package,
			"major":      fmt.Sprintf("v%d", superseded.Major),
			"install_id": superseded.ID,
			"reclaimed":  output.Bytes(n),
		})
	}
	unlockLocal := localpackage.Guard()
	localProblem := localpackage.Sweep(l, st)
	unlockLocal()
	if localProblem != nil {
		return localProblem
	}
	if len(removed.Rows) == 0 {
		removed.Aggregates = []output.Field{{K: "changed", V: false}}
		return emit(ctx, removed)
	}
	w.Unlock()
	st.Close()
	removed.Aggregates = []output.Field{
		{K: "changed", V: true},
		{K: "reclaimed", V: output.Bytes(freed)},
	}
	removed.Notes = []string{"exclusive package bytes were removed; shared TensorFS model bytes were untouched"}
	removed.Next = []string{"cozy package list"}
	return emit(ctx, removed)
}

func diskText(inst records.PackageInstall) string {
	if inst.BytesShared == 0 {
		return output.Bytes(inst.BytesExcl)
	}
	return fmt.Sprintf("%s (+%s shared)", output.Bytes(inst.BytesExcl), output.Bytes(inst.BytesShared))
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

func timingsText(t []install.Timing) string {
	parts := make([]string, 0, len(t))
	for _, x := range t {
		parts = append(parts, fmt.Sprintf("%s %dms", x.Stage, x.Took.Milliseconds()))
	}
	return strings.Join(parts, " · ")
}
