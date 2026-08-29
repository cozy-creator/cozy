package app

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/endpointprofile"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/install"
	"github.com/cozy-creator/cozy-creator/internal/managedinstall"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/retention"
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

func handleManagedInstall(ctx *Context, profile string) *exit.Error {
	if ctx.Inv.Value("--from") != "" || ctx.Inv.Value("--digest") != "" ||
		ctx.Inv.Bool("--allow-unsigned") || ctx.Inv.Value("--dir") != "" {
		return exit.Usagef("published --profile install cannot be combined with --from, --digest, --allow-unsigned, or editable --dir").
			WithRemedy("a qualified release uses exact Tensorhub documents and prebuilt wheels; development source uses the separate --from/--dir lane")
	}
	ref, release, e := endpointReleaseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	profiles, e := endpointprofile.NormalizeSet([]string{profile})
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
	deviceText := strings.TrimSpace(ctx.Inv.Value("--device"))
	if deviceText == "" {
		deviceText = "0"
	}
	device, err := strconv.Atoi(deviceText)
	if err != nil || device < 0 || device > 63 {
		return exit.Usagef("--device %q is not an index from 0 through 63", deviceText)
	}
	reason, e := mutationReason(ctx, "install --profile")
	if e != nil {
		return e
	}
	l, st, writer, e := open(ctx.Cfg, true)
	if e != nil {
		return e
	}
	defer st.Close()
	defer writer.Unlock()
	c := client(ctx)
	hctx, cancel := hub.LongContext()
	defer cancel()
	grant, e := c.EndpointLocalQualificationMaterials(hctx, ref, release, profiles[0], int64(ttl/time.Second), reason)
	if e != nil {
		return e
	}
	installed, e := managedinstall.Run(hctx, l, st, managedinstall.Request{
		Endpoint: ref.String(), Release: release, Major: major, Profile: profiles[0],
		Force: ctx.Inv.Bool("--force"), DeviceIndex: &device, Grant: grant, Config: ctx.Cfg,
	})
	if e != nil {
		return e
	}
	g, facts := installed.Install, installed.Facts
	rec := render.Record{Kind: "install", Fields: []render.Field{
		{K: "endpoint", V: g.Endpoint}, {K: "major", V: g.Major}, {K: "release", V: g.Version},
		{K: "profile", V: facts.Profile}, {K: "candidate", V: facts.CandidateID},
		{K: "generation", V: g.ID}, {K: "base_realization", V: facts.BaseRealizationDigest},
		{K: "wheelhouse", V: facts.WheelhouseManifestDigest},
		{K: "environment", V: facts.EnvironmentSpecDigest},
		{K: "receipt", V: facts.InstalledReceiptDigest}, {K: "host_evidence", V: facts.HostEvidenceDigest},
		{K: "lease", V: facts.LeaseID + " until " + facts.LeaseExpiresAt},
		{K: "disk", V: diskText(g)},
	}, Notes: []string{
		"installed only after independent local hardware qualification; no cloud qualification was borrowed",
		"no dependency resolution, Torch/CUDA install, or native build ran",
		"control install, portable Runtime receipt, local-base fingerprint, host evidence, and lease are separate recorded facts",
	}, Next: []string{"cozy start " + g.Endpoint + "@v" + strconv.Itoa(g.Major), "cozy run <org/endpoint/vN/function>"}}
	if facts.NativeEvidenceDigest != "" {
		rec.Fields = append(rec.Fields, render.Field{K: "native_evidence", V: facts.NativeEvidenceDigest})
	}
	if installed.Idempotent {
		rec.Fields = append(rec.Fields, render.Field{K: "result", V: "already pinned — nothing changed"})
	}
	if installed.Superseded != "" {
		rec.Fields = append(rec.Fields, render.Field{K: "superseded", V: installed.Superseded})
	}
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
		l.Next = []string{"cozy install <org/endpoint>", "cozy endpoint search"}
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
	_, st, w, e := open(ctx.Cfg, true)
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
				"generation": p.InstallID,
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
	horizon, e := retention.ParseHorizon(ctx.Inv.Value("--keep-media"))
	if e != nil {
		return e
	}
	l, st, w, e := open(ctx.Cfg, write)
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
	// THE SECOND PLANE (cl-033). Unreferenced generations are garbage; retained outputs
	// are the user's work, kept on purpose after the pod that made them was destroyed.
	// They share `gc` because they share the question — what is this root storing, and
	// what may go — but not the rule: one is reclaimed because nothing points at it, the
	// other only because it is older than the declared horizon.
	media, e := retention.Build(l, st, horizon, time.Now())
	if e != nil {
		return e
	}
	publicationItems, publicationBytes, e := st.PublicationStats()
	if e != nil {
		return e
	}
	out := render.List{
		Kind:      "gc",
		Fields:    []string{"kind", "id", "endpoint", "bytes"},
		AllFields: []string{"kind", "id", "endpoint", "version", "age", "bytes", "reason"},
		Empty:     "0 bytes reclaimable",
	}
	var total int64
	for _, r := range plan {
		total += r.Bytes
		out.Rows = append(out.Rows, map[string]string{
			"kind": r.Kind, "id": r.ID, "endpoint": r.Endpoint, "version": r.Version,
			"age": "-", "bytes": render.Bytes(r.Bytes), "reason": r.Reason,
		})
	}
	for _, m := range media.Items {
		total += m.Bytes
		out.Rows = append(out.Rows, map[string]string{
			"kind": "media", "id": m.MediaID, "endpoint": m.Endpoint, "version": "-",
			"age": retention.Age(m.Age), "bytes": render.Bytes(m.Bytes), "reason": m.Reason,
		})
	}
	items := len(plan) + len(media.Items)
	if media.Foreign > 0 {
		out.Notes = append(out.Notes, fmt.Sprintf(
			"%d output row(s) record a path outside the local output namespace and were NOT planned",
			media.Foreign))
	}
	if publicationItems > 0 {
		out.Notes = append(out.Notes, fmt.Sprintf(
			"%d job publication(s), %s are retained indefinitely; gc reports but never reclaims the durable result plane",
			publicationItems, render.Bytes(publicationBytes)))
	}
	if !write {
		out.Aggregates = []render.Field{
			{K: "reclaimable", V: render.Bytes(total)},
			{K: "items", V: items},
			{K: "media", V: fmt.Sprintf("%d output(s), %s beyond the %s horizon",
				len(media.Items), render.Bytes(media.Bytes), retention.Short(horizon))},
			{K: "retained", V: fmt.Sprintf("%d output(s), %s kept",
				media.RetainedItems, render.Bytes(media.RetainedBytes))},
			{K: "publications", V: fmt.Sprintf("%d publication(s), %s retained indefinitely",
				publicationItems, render.Bytes(publicationBytes))},
			{K: "cas", V: "0 objects (the shared weights CAS lands with cl-012)"},
		}
		if items == 0 {
			out.Notes = append(out.Notes, "nothing is unreferenced and no output is past the "+
				retention.Short(horizon)+" horizon — this is the answer, not an empty listing")
			out.Next = []string{"cozy media ls"}
		} else {
			out.Notes = append(out.Notes, "this is the plan; nothing was removed")
			out.Next = []string{"cozy gc --yes"}
		}
		return emit(ctx, out)
	}
	freed, e := install.Collect(l, st, plan)
	if e != nil {
		return e
	}
	reclaimed, e := retention.Collect(l, st, media)
	if e != nil {
		return e
	}
	out.Aggregates = []render.Field{
		{K: "freed", V: render.Bytes(freed + reclaimed)},
		{K: "items", V: items},
		{K: "media", V: fmt.Sprintf("%d output(s), %s", len(media.Items), render.Bytes(reclaimed))},
		{K: "retained", V: fmt.Sprintf("%d output(s), %s kept",
			media.RetainedItems, render.Bytes(media.RetainedBytes))},
		{K: "publications", V: fmt.Sprintf("%d publication(s), %s retained indefinitely",
			publicationItems, render.Bytes(publicationBytes))},
	}
	if items > 0 {
		out.Notes = append(out.Notes,
			"only generations nothing references and outputs past the "+retention.Short(horizon)+
				" horizon were reclaimed; a reclaimed media id answers 410, never 404")
		out.Next = []string{"cozy media ls"}
	}
	return emit(ctx, out)
}

func diskText(g records.EndpointInstall) string {
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
