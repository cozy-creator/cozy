package app

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/install"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

// open resolves the local layout and the ONE lifecycle database. Mutating verbs
// additionally take the single-writer lock.
func open(write bool) (home.Layout, *records.Store, *install.Writer, *exit.Error) {
	l, e := home.Open()
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
	ref, e := install.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	l, st, w, e := open(true)
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
		CrashAfter:    ctx.Inv.Value("--crash-after"),
	})
	if e != nil {
		return e
	}
	g := res.Gen
	rec := render.Record{
		Kind: "install",
		Fields: []render.Field{
			{K: "endpoint", V: g.Endpoint},
			{K: "major", V: g.Major},
			{K: "version", V: g.Version},
			{K: "generation", V: g.ID},
			{K: "source", V: g.SourceKind + " " + g.SourceRef},
			{K: "source_digest", V: g.SourceDigest},
			{K: "verified", V: g.Verified},
			{K: "python", V: g.Python},
			{K: "uv", V: g.UV},
			{K: "lock", V: g.LockDigest},
			{K: "platform", V: g.Platform},
			{K: "cuda_extra", V: orNone(g.Extra)},
			{K: "link_mode", V: g.LinkMode},
			{K: "packages", V: g.Packages},
			{K: "closure", V: strings.ReplaceAll(g.Closure, "\n", " ")},
			{K: "descriptor", V: g.Descriptor},
			{K: "disk", V: diskText(g)},
		},
	}
	if res.Idempotent {
		rec.Fields = append(rec.Fields, render.Field{K: "result", V: "already pinned — nothing changed"})
		rec.Next = []string{"cozy ls"}
		return emit(ctx, rec)
	}
	rec.Fields = append(rec.Fields,
		render.Field{K: "staged", V: fmt.Sprintf("%d files, %s expanded, %s compressed", res.Files, render.Bytes(res.Bytes), render.Bytes(res.Compressed))},
		render.Field{K: "timings", V: timingsText(res.Timings)},
	)
	if res.Superseded != "" {
		rec.Fields = append(rec.Fields, render.Field{K: "superseded", V: res.Superseded})
		rec.Notes = append(rec.Notes,
			"the superseded generation is untouched on disk until `cozy gc` reclaims it")
	}
	rec.Notes = append(rec.Notes, res.Warnings...)
	rec.Next = []string{"cozy ls"}
	return emit(ctx, rec)
}

func handleLs(ctx *Context) *exit.Error {
	_, st, _, e := open(false)
	if e != nil {
		return e
	}
	defer st.Close()
	rows, e := st.Installed()
	if e != nil {
		return e
	}
	l := render.List{
		Kind:      "ls",
		Fields:    []string{"endpoint", "major", "version", "disk"},
		AllFields: []string{"endpoint", "major", "version", "generation", "disk", "exclusive", "shared", "python", "uv", "cuda_extra", "link_mode", "packages", "closure", "descriptor", "source", "verified", "installed"},
		Empty:     "0 endpoints installed",
	}
	var excl, shared int64
	for _, g := range rows {
		excl += g.BytesExcl
		shared += g.BytesShared
		l.Rows = append(l.Rows, map[string]string{
			"endpoint":   g.Endpoint,
			"major":      fmt.Sprintf("v%d", g.Major),
			"version":    g.Version,
			"generation": g.ID,
			"disk":       diskText(g),
			"exclusive":  render.Bytes(g.BytesExcl),
			"shared":     render.Bytes(g.BytesShared),
			"python":     g.Python,
			"uv":         g.UV,
			"cuda_extra": orNone(g.Extra),
			"link_mode":  g.LinkMode,
			"packages":   fmt.Sprintf("%d", g.Packages),
			"closure":    strings.ReplaceAll(g.Closure, "\n", " "),
			"descriptor": g.Descriptor,
			"source":     g.SourceKind + " " + g.SourceRef,
			"verified":   fmt.Sprintf("%t", g.Verified),
			"installed":  g.CreatedAt,
		})
	}
	if len(l.Rows) == 0 {
		l.Next = []string{"cozy install org/endpoint"}
		return emit(ctx, l)
	}
	l.Aggregates = []render.Field{
		{K: "endpoints", V: len(l.Rows)},
		{K: "exclusive", V: render.Bytes(excl)},
		{K: "hardlink-shared", V: render.Bytes(shared)},
	}
	l.Notes = []string{"read from the install records; no directory was walked"}
	return emit(ctx, l)
}

func handleRm(ctx *Context) *exit.Error {
	_, st, w, e := open(true)
	if e != nil {
		return e
	}
	defer st.Close()
	defer w.Unlock()

	removed := render.List{
		Kind:      "rm",
		Fields:    []string{"endpoint", "major", "generation"},
		AllFields: []string{"endpoint", "major", "generation", "reclaimed"},
		Empty:     "0 installs removed",
	}
	var freed int64
	for _, arg := range ctx.Inv.Args {
		ref, e := install.ParseRef(arg)
		if e != nil {
			return e
		}
		targets, e := st.Pins(ref.Endpoint)
		if e != nil {
			return e
		}
		for _, p := range targets {
			if ref.HasMajor && p.Major != ref.Major {
				continue
			}
			n, e := install.Remove(st, p.Endpoint, p.Major)
			if e != nil {
				return e
			}
			freed += n
			removed.Rows = append(removed.Rows, map[string]string{
				"endpoint":   p.Endpoint,
				"major":      fmt.Sprintf("v%d", p.Major),
				"generation": p.Generation,
				"reclaimed":  render.Bytes(n),
			})
		}
	}
	if len(removed.Rows) == 0 {
		removed.Empty = fmt.Sprintf("0 installs removed — %s is not installed", strings.Join(ctx.Inv.Args, ", "))
		removed.Next = []string{"cozy ls"}
		return emit(ctx, removed)
	}
	removed.Aggregates = []render.Field{
		{K: "removed", V: len(removed.Rows)},
		{K: "reclaimed", V: render.Bytes(freed)},
	}
	removed.Notes = []string{"shared-CAS weights are untouched; `cozy gc` reclaims what nothing references"}
	removed.Next = []string{"cozy gc"}
	return emit(ctx, removed)
}

func handleGC(ctx *Context) *exit.Error {
	write := ctx.Inv.Bool("--yes")
	l, st, w, e := open(write)
	if e != nil {
		return e
	}
	defer st.Close()
	if w != nil {
		defer w.Unlock()
	}
	plan, e := install.Plan(l, st)
	if e != nil {
		return e
	}
	out := render.List{
		Kind:      "gc",
		Fields:    []string{"kind", "id", "endpoint", "bytes"},
		AllFields: []string{"kind", "id", "endpoint", "version", "bytes", "reason"},
		Empty:     "0 bytes reclaimable",
	}
	var total int64
	for _, r := range plan {
		total += r.Bytes
		out.Rows = append(out.Rows, map[string]string{
			"kind": r.Kind, "id": r.ID, "endpoint": r.Endpoint, "version": r.Version,
			"bytes": render.Bytes(r.Bytes), "reason": r.Reason,
		})
	}
	if !write {
		out.Aggregates = []render.Field{
			{K: "reclaimable", V: render.Bytes(total)},
			{K: "items", V: len(plan)},
			{K: "cas", V: "0 objects (the shared weights CAS lands with cl-012)"},
		}
		if len(plan) == 0 {
			out.Notes = []string{"nothing is unreferenced — this is the answer, not an empty listing"}
		} else {
			out.Notes = []string{"this is the plan; nothing was removed"}
			out.Next = []string{"cozy gc --yes"}
		}
		return emit(ctx, out)
	}
	freed, e := install.Collect(l, st, plan)
	if e != nil {
		return e
	}
	out.Aggregates = []render.Field{
		{K: "freed", V: render.Bytes(freed)},
		{K: "items", V: len(plan)},
	}
	if len(plan) > 0 {
		out.Notes = []string{"only generations nothing references were reclaimed; every pin still resolves"}
	}
	return emit(ctx, out)
}

func diskText(g records.Generation) string {
	if g.BytesShared == 0 {
		return render.Bytes(g.BytesExcl)
	}
	return fmt.Sprintf("%s (+%s shared)", render.Bytes(g.BytesExcl), render.Bytes(g.BytesShared))
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
