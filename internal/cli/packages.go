package cli

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
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
	if explicitPackageDirectory(ctx.Inv.Args[0]) {
		return handleDirectoryInstall(ctx)
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
	pack, problem := packagepublish.PrepareFrom(path)
	if problem != nil {
		return problem
	}
	defer pack.Close()
	sourceDigest, files, bytes, problem := pack.SourceIdentity()
	if problem != nil {
		return problem
	}
	ref, problem := install.ParseRef(pack.Organization + "/" + pack.Name)
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
			Local: &install.LocalSource{SourceDigest: sourceDigest, Bytes: bytes, Files: files,
				Package: ref.Package, Release: pack.Release, Tree: pack.Tree}})
		return installProblem
	})
	if problem != nil {
		return problem
	}
	return emitInstallResult(ctx, st, result)
}

func emitInstallResult(ctx *Context, st *records.Store, res *install.Result) *exit.Error {
	g := res.Gen
	fields := []output.Field{
		{K: "package", V: g.Package}, {K: "major", V: g.Major},
		{K: "version", V: g.Version}, {K: "status", V: "installed"},
		{K: "disk", V: diskText(g)}, {K: "changed", V: !res.Idempotent},
		{K: "generation", V: g.ID}, {K: "source", V: g.SourceKind + " " + g.SourceRef},
		{K: "source_digest", V: g.SourceDigest}, {K: "verified", V: g.Verified},
		{K: "python", V: g.Python}, {K: "uv", V: g.UV}, {K: "lock", V: g.LockDigest},
		{K: "platform", V: g.Platform}, {K: "cuda_extra", V: orNone(g.Extra)},
		{K: "link_mode", V: g.LinkMode}, {K: "packages", V: g.Packages},
		{K: "closure", V: strings.ReplaceAll(g.Closure, "\n", " ")},
		{K: "package_descriptor", V: g.PackageDescriptor},
		{K: "profile", V: g.SelectionProfile},
		{K: "placement_set", V: g.PlacementSetDigest},
	}
	if res.Idempotent {
		return emit(ctx, compactRecord(fields, "package", "version", "status", "changed"))
	}
	fields = append(fields,
		output.Field{K: "staged", V: fmt.Sprintf("%d files, %s", res.Files, output.Bytes(res.Bytes))},
		output.Field{K: "timings", V: timingsText(res.Timings)},
	)
	if res.Superseded != "" {
		reclaimed, problem := install.Reclaim(st, res.Superseded)
		if problem != nil {
			res.Warnings = append(res.Warnings,
				"the new version is active; cleanup of the prior version was deferred: "+problem.Message)
		} else {
			fields = append(fields,
				output.Field{K: "superseded", V: res.Superseded},
				output.Field{K: "reclaimed", V: output.Bytes(reclaimed)})
		}
	}
	rec := compactRecord(fields, "package", "version", "status", "disk", "changed")
	rec.Notes = append(rec.Notes, res.Warnings...)
	rec.Next = []string{"cozy package list"}
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
		Fields:    []string{"package", "version", "disk"},
		AllFields: []string{"package", "major", "version", "disk", "profile", "placement_set", "generation", "source", "verified", "installed", "exclusive", "shared"},
	}
	for _, g := range rows {
		l.Rows = append(l.Rows, map[string]string{
			"package":            g.Package,
			"major":              fmt.Sprintf("v%d", g.Major),
			"version":            g.Version,
			"generation":         g.ID,
			"disk":               diskText(g),
			"exclusive":          output.Bytes(g.BytesExcl),
			"shared":             output.Bytes(g.BytesShared),
			"python":             g.Python,
			"uv":                 g.UV,
			"cuda_extra":         orNone(g.Extra),
			"link_mode":          g.LinkMode,
			"packages":           fmt.Sprintf("%d", g.Packages),
			"closure":            strings.ReplaceAll(g.Closure, "\n", " "),
			"package_descriptor": g.PackageDescriptor,
			"profile":            g.SelectionProfile,
			"placement_set":      g.PlacementSetDigest,
			"source":             g.SourceKind + " " + g.SourceRef,
			"verified":           fmt.Sprintf("%t", g.Verified),
			"installed":          g.CreatedAt,
		})
	}
	if len(l.Rows) == 0 {
		l.Next = []string{"cozy package search"}
		return emit(ctx, l)
	}
	return emit(ctx, l)
}

func handleRm(ctx *Context) *exit.Error {
	_, st, w, e := open(ctx.Cfg, true)
	if e != nil {
		return e
	}
	defer st.Close()
	defer w.Unlock()

	removed := output.List{
		Name:      "packages",
		Fields:    []string{"package", "reclaimed"},
		AllFields: []string{"package", "major", "reclaimed", "generation"},
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
			if ref.HasMajor && p.Major != ref.Major {
				continue
			}
			n, e := install.Remove(st, p.Package, p.Major)
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
				"generation": p.InstallID,
				"reclaimed":  output.Bytes(n),
			})
		}
	}
	// Removing a package also clears superseded generations for the same selected
	// major. Active requests/workers were fenced above and the database claim rechecks.
	unreferenced, e := st.Unreferenced()
	if e != nil {
		return e
	}
	for _, generation := range unreferenced {
		selected := false
		for _, arg := range ctx.Inv.Args {
			ref, problem := install.ParseRef(arg)
			if problem != nil {
				return problem
			}
			selected = ref.Package == generation.Package && (!ref.HasMajor || ref.Major == generation.Major)
			if selected {
				break
			}
		}
		if !selected {
			continue
		}
		n, problem := install.Reclaim(st, generation.ID)
		if problem != nil {
			return problem
		}
		freed += n
	}
	if len(removed.Rows) == 0 {
		removed.Aggregates = []output.Field{{K: "changed", V: false}}
		return emit(ctx, removed)
	}
	// The pin is gone before residency changes. Accepted work keeps its exact generation;
	// the daemon retires only workers it can prove idle.
	w.Unlock()
	st.Close()
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

func diskText(g records.PackageInstall) string {
	if g.BytesShared == 0 {
		return output.Bytes(g.BytesExcl)
	}
	return fmt.Sprintf("%s (+%s shared)", output.Bytes(g.BytesExcl), output.Bytes(g.BytesShared))
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
