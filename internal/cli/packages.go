package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
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
// additionally take the single-writer lock.
func open(cfg config.Config, write bool) (home.Layout, *records.Store, *install.Writer, *exit.Error) {
	l, e := home.Open(cfg.Home)
	if e != nil {
		return l, nil, nil, e
	}
	var w *install.Writer
	if write {
		if w, e = install.Lock(l); e != nil {
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
	if ctx.Inv.Value("--rental") != "" {
		if explicitPackageDirectory(ctx.Inv.Args[0]) || ctx.Inv.Bool("--editable") {
			return exit.Usagef("package install --rental requires a published org/name; editable directories use private execution").
				WithRemedy("use `cozy run ./project/<function> --rental=<rental>` to upload local code privately")
		}
		return handleRentalPackageInstall(ctx)
	}
	if explicitPackageDirectory(ctx.Inv.Args[0]) {
		if !ctx.Inv.Bool("--editable") {
			return exit.Usagef("an explicit package directory requires --editable")
		}
		return handleDirectoryInstall(ctx)
	}
	if ctx.Inv.Bool("--editable") {
		return exit.Usagef("--editable requires an explicit package directory")
	}
	return handleRegistryInstall(ctx)
}

func explicitPackageDirectory(value string) bool {
	value = strings.TrimSpace(value)
	return value == "." || value == ".." || filepath.IsAbs(value) ||
		strings.HasPrefix(value, "./") || strings.HasPrefix(value, "../") ||
		strings.HasPrefix(value, `.\`) || strings.HasPrefix(value, `..\`)
}

func handleDirectoryInstall(ctx *Context) *exit.Error {
	path := strings.TrimSpace(ctx.Inv.Args[0])
	if ctx.Inv.Value("--version") != "" {
		return exit.Usagef("an explicit package directory does not take registry or legacy source options").
			WithRemedy("use `cozy package install %s` by itself", path)
	}
	pack, problem := packagepublish.PrepareLocalFrom(path)
	if problem != nil {
		return problem
	}
	defer pack.Close()
	files, bytes, problem := pack.SourceInventory()
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
	var result *install.Result
	problem = packagePublishStage(ctx, "Creating local package environment", func() *exit.Error {
		var installProblem *exit.Error
		result, installProblem = install.Run(l, st, install.Request{Ref: ref, Force: true,
			Local: &install.LocalSource{Bytes: bytes, Files: files,
				Package: ref.Package, Release: pack.Release, Tree: pack.Tree}})
		return installProblem
	})
	if problem != nil {
		return problem
	}
	return emitInstallResult(ctx, result, reclaimInstallResult(l, st, result)...)
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
	if res.Idempotent {
		keys := []string{"package", "version", "status"}
		rec := compactRecord(fields, keys...)
		rec.Next = []string{runExample(inst)}
		return emit(ctx, rec)
	}
	fields = append(fields,
		output.Field{K: "staged", V: fmt.Sprintf("%d files, %s", res.Files, output.Bytes(res.Bytes))},
		output.Field{K: "timings", V: timingsText(res.Timings)},
	)
	fields = append(fields, cleanup...)
	rec := compactRecord(fields, "package", "version", "status", "disk")
	rec.Notes = append(rec.Notes, res.Warnings...)
	rec.Next = []string{runExample(inst), "cozy package list"}
	return emit(ctx, rec)
}

func handleLs(ctx *Context) *exit.Error {
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
		AllFields: []string{"package", "major", "version", "python", "python_status", "size", "dependencies", "placement_set", "install_id", "source", "synced", "verified", "installed"},
		Bytes:     []string{"size", "dependencies"},
	}
	inventory, pythonProblem := hostruntime.PythonExecutors(context.Background())
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
			"synced":        synced,
			"verified":      fmt.Sprintf("%t", inst.Verified),
			"installed":     inst.CreatedAt,
		})
	}
	if len(l.Rows) == 0 {
		l.Next = []string{"cozy package search"}
		return emit(ctx, l)
	}
	return emit(ctx, l)
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
		for _, p := range targets {
			n, e := install.Remove(l, st, p.Package, p.Major)
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
	unreferenced, e := st.Unreferenced()
	if e != nil {
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
	// The pin is gone before residency changes. Accepted work keeps its exact install;
	// the daemon retires only workers it can prove idle.
	w.Unlock()
	st.Close()
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
	client, problem := dial(ctx)
	if problem == nil {
		unloaded, unloadProblem := client.Unload()
		if unloadProblem != nil {
			removed.Notes = append(removed.Notes,
				"idle worker cleanup was deferred: "+unloadProblem.Message)
		} else if unloaded.Count > 0 {
			removed.Notes = append(removed.Notes,
				fmt.Sprintf("retired %d idle package worker(s)", unloaded.Count))
		}
	} else {
		removed.Notes = append(removed.Notes, "idle worker cleanup was deferred: "+problem.Message)
	}
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
