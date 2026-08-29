package cli

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/install"
	"github.com/cozy-creator/cozy-creator/internal/managedinstall"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/packageprofile"
	"github.com/cozy-creator/cozy-creator/internal/records"
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
	if profile := strings.TrimSpace(ctx.Inv.Value("--profile")); profile != "" {
		return handleManagedInstall(ctx, profile)
	}
	ref, e := install.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	l, st, w, e := open(ctx.Cfg, true)
	if e != nil {
		return e
	}
	defer st.Close()
	defer w.Unlock()

	res, e := install.Run(l, st, install.Request{
		Ref:           ref,
		Archive:       ctx.Inv.Value("--from"),
		ExpectDigest:  ctx.Inv.Value("--digest"),
		Dir:           ctx.Inv.Value("--dir"),
		AllowUnsigned: ctx.Inv.Bool("--allow-unsigned"),
		Force:         ctx.Inv.Bool("--force"),
	})
	if e != nil {
		return e
	}
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
	}
	if res.Idempotent {
		return emit(ctx, compactRecord(fields, "package", "major", "version", "status", "changed"))
	}
	fields = append(fields,
		output.Field{K: "staged", V: fmt.Sprintf("%d files, %s expanded, %s compressed", res.Files, output.Bytes(res.Bytes), output.Bytes(res.Compressed))},
		output.Field{K: "timings", V: timingsText(res.Timings)},
	)
	if res.Superseded != "" {
		reclaimed, problem := install.Reclaim(st, res.Superseded)
		if problem != nil {
			return problem
		}
		fields = append(fields,
			output.Field{K: "superseded", V: res.Superseded},
			output.Field{K: "reclaimed", V: output.Bytes(reclaimed)})
	}
	rec := compactRecord(fields, "package", "major", "version", "status", "disk", "changed")
	rec.Notes = append(rec.Notes, res.Warnings...)
	rec.Next = []string{"cozy package list"}
	return emit(ctx, rec)
}

func handleManagedInstall(ctx *Context, profile string) *exit.Error {
	if ctx.Inv.Value("--from") != "" || ctx.Inv.Value("--digest") != "" ||
		ctx.Inv.Bool("--allow-unsigned") || ctx.Inv.Value("--dir") != "" {
		return exit.Usagef("published --profile install cannot be combined with --from, --digest, --allow-unsigned, or editable --dir").
			WithRemedy("a qualified release uses exact Tensorhub documents and prebuilt wheels; development source uses the separate --from/--dir lane")
	}
	ref, release, e := packageReleaseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	profiles, e := packageprofile.NormalizeSet([]string{profile})
	if e != nil {
		return e
	}
	majorText := strings.TrimSpace(ctx.Inv.Value("--major"))
	if !strings.HasPrefix(majorText, "v") {
		return exit.Usagef("published --profile install needs --major vN")
	}
	major, err := strconv.Atoi(strings.TrimPrefix(majorText, "v"))
	if err != nil || major <= 0 {
		return exit.Usagef("--major %q is not vN with N greater than zero", majorText)
	}
	ttlText := strings.TrimSpace(ctx.Inv.Value("--grant-ttl"))
	if ttlText == "" {
		ttlText = "10m"
	}
	ttl, err := time.ParseDuration(ttlText)
	if err != nil || ttl <= 0 || ttl > time.Hour {
		return exit.Usagef("--grant-ttl %q is not a positive duration at or below 1h", ttlText)
	}
	reason := "cozy package install " + ref.String() + "@" + release + " for " + profiles[0]
	l, st, writer, e := open(ctx.Cfg, true)
	if e != nil {
		return e
	}
	defer st.Close()
	defer writer.Unlock()
	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	grant, e := c.PackageLocalQualificationMaterials(hctx, ref, release, profiles[0], int64(ttl/time.Second), reason)
	if e != nil {
		return e
	}
	installed, e := managedinstall.Run(hctx, l, st, managedinstall.Request{
		Package: ref.String(), Release: release, Major: major, Profile: profiles[0],
		Force: ctx.Inv.Bool("--force"), Grant: grant, Config: ctx.Cfg,
	})
	if e != nil {
		return e
	}
	g, facts := installed.Install, installed.Facts
	fields := []output.Field{
		{K: "package", V: g.Package}, {K: "major", V: g.Major}, {K: "release", V: g.Version},
		{K: "status", V: "installed"}, {K: "changed", V: !installed.Idempotent},
		{K: "profile", V: facts.Profile}, {K: "candidate", V: facts.CandidateID},
		{K: "generation", V: g.ID}, {K: "base_realization", V: facts.BaseRealizationDigest},
		{K: "wheelhouse", V: facts.WheelhouseManifestDigest},
		{K: "environment", V: facts.EnvironmentSpecDigest},
		{K: "receipt", V: facts.InstalledReceiptDigest}, {K: "host_evidence", V: facts.HostEvidenceDigest},
		{K: "lease", V: facts.LeaseID + " until " + facts.LeaseExpiresAt},
		{K: "disk", V: diskText(g)},
	}
	if installed.Superseded != "" {
		reclaimed, problem := install.Reclaim(st, installed.Superseded)
		if problem != nil {
			return problem
		}
		fields = append(fields,
			output.Field{K: "superseded", V: installed.Superseded},
			output.Field{K: "reclaimed", V: output.Bytes(reclaimed)})
	}
	return emit(ctx, compactRecord(fields, "package", "major", "release", "profile", "status", "disk", "changed"))
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
		Fields:    []string{"package", "major", "version", "disk"},
		AllFields: []string{"package", "major", "version", "disk", "generation", "source", "verified", "installed", "exclusive", "shared"},
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
	live, e := st.LiveWorkers()
	if e != nil {
		return e
	}
	for _, worker := range live {
		if worker.WorkerID == "remote" {
			continue
		}
		for _, arg := range ctx.Inv.Args {
			ref, problem := install.ParseRef(arg)
			if problem != nil {
				return problem
			}
			if ref.Package == worker.Package {
				return exit.New(exit.Conflict, "%s is still resident in local worker %s", ref.Package, worker.InstanceID).
					WithRemedy("run `cozy unload`, then remove the package").
					WithNext("cozy unload")
			}
		}
	}
	active, e := st.ActiveRequests()
	if e != nil {
		return e
	}
	for _, request := range active {
		for _, arg := range ctx.Inv.Args {
			ref, problem := install.ParseRef(arg)
			if problem != nil {
				return problem
			}
			if request.Worker == "" && ref.Package == request.Package {
				return exit.New(exit.Conflict, "%s still has active invocation %s", ref.Package, request.ID).
					WithRemedy("cancel the invocation before removing its package").
					WithNext("cozy invoke cancel " + request.ID)
			}
		}
	}

	removed := output.List{
		Name:      "packages",
		Fields:    []string{"package", "major", "reclaimed"},
		AllFields: []string{"package", "major", "reclaimed", "generation"},
	}
	var freed int64
	for _, arg := range ctx.Inv.Args {
		ref, e := install.ParseRef(arg)
		if e != nil {
			return e
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
