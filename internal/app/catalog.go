package app

import (
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/render"
)

// The catalog verbs (cl-011). Launch-1 tensorhub has no identity plane (decisions
// #229), so these split exactly two ways: a public read that carries no credential,
// and a first-party write that carries the ONE static admin token. Nothing here
// stores a credential, refreshes one, or knows what an account is.
//
// The line every one of them holds: a hub refusal reaches the user verbatim — the
// hub's code, message and remedy — under a code from the shared exit matrix.

// identityNote is the one sentence that keeps the deferral visible instead of
// leaving a user to guess why there is no `cozy login`.
const identityNote = "no accounts at Launch 1: catalog reads are public, first-party writes carry TENSORHUB_TOKEN (th-031 arms identity)"

func client(ctx *Context) *hub.Client {
	rev, _ := buildStamp()
	return hub.New(ctx.Cfg, "cozy/"+tag+"+"+rev)
}

// stamp trims a hub timestamp to whole seconds — the catalog's own RFC3339 with
// nanoseconds is nine digits of noise in a listing column.
func stamp(ts string) string {
	if i := strings.IndexByte(ts, '.'); i >= 0 {
		if z := strings.IndexAny(ts[i:], "Z+-"); z >= 0 {
			return ts[:i] + ts[i+z:]
		}
	}
	return ts
}

func handleSearch(ctx *Context) *exit.Error {
	kind := strings.TrimSpace(ctx.Inv.Value("--kind"))
	if kind != "" {
		if e := hub.CheckKind(kind); e != nil {
			return e.WithNext("cozy help search")
		}
	}
	query := strings.ToLower(strings.TrimSpace(strings.Join(ctx.Inv.Args, " ")))

	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	repos, e := c.Repos(hctx)
	if e != nil {
		return e
	}

	l := render.List{
		Kind:      "search",
		Fields:    []string{"ref", "kind", "created"},
		AllFields: []string{"ref", "kind", "created", "org", "name"},
	}
	for _, r := range repos {
		if kind != "" && r.Kind != kind {
			continue
		}
		if query != "" && !strings.Contains(strings.ToLower(r.Ref()), query) {
			continue
		}
		l.Rows = append(l.Rows, map[string]string{
			"ref": r.Ref(), "kind": r.Kind, "created": stamp(r.CreatedAt),
			"org": r.Org, "name": r.Name,
		})
	}

	l.Aggregates = []render.Field{
		{K: "results", V: len(l.Rows)},
		{K: "catalog", V: len(repos)},
		{K: "hub", V: c.Base()},
	}
	// The public listing is a whole-catalog read filtered here. That is honest at a
	// catalog this size and dishonest at scale, so it says so at the hub's own cap
	// rather than silently returning a prefix as if it were the answer.
	if len(repos) >= 200 {
		l.Notes = append(l.Notes,
			"the hub's public listing is capped at 200 rows and this filter runs client-side; server-side search lands with th-003")
	}
	switch {
	case len(l.Rows) == 0 && query == "" && kind == "":
		l.Empty = "0 repos in the catalog"
		l.Next = []string{"cozy repo create <org/name> --kind model --reason <why>"}
	case len(l.Rows) == 0:
		l.Empty = "0 results for " + describeQuery(query, kind)
		l.Next = []string{"cozy search"}
	default:
		l.Next = []string{"cozy repo show " + l.Rows[0]["ref"]}
	}
	return emit(ctx, l)
}

func describeQuery(query, kind string) string {
	parts := []string{}
	if query != "" {
		parts = append(parts, "\""+query+"\"")
	}
	if kind != "" {
		parts = append(parts, "kind "+kind)
	}
	return strings.Join(parts, " · ")
}

func handleRepoShow(ctx *Context) *exit.Error {
	ref, e := hub.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	repos, e := c.Repos(hctx)
	if e != nil {
		return e
	}
	for _, r := range repos {
		if r.Org != ref.Org || r.Name != ref.Name {
			continue
		}
		return emit(ctx, render.Record{
			Kind: "repo",
			Fields: []render.Field{
				{K: "ref", V: r.Ref()},
				{K: "repo_kind", V: r.Kind},
				{K: "created", V: stamp(r.CreatedAt)},
				{K: "hub", V: c.Base()},
			},
			Notes: []string{
				"resolved from the public listing; the per-repo route with releases, lanes and sizing lands with th-003",
			},
			Next: []string{"cozy search"},
		})
	}
	return exit.New(exit.NotFound, "no repo %q in the catalog at %s", ref, c.Base()).
		WithRemedy("`cozy search %s` lists what the hub does hold", ref.Name).
		WithNext("cozy search "+ref.Name, "cozy repo create "+ref.String()+" --kind model --reason <why>")
}

func handleRepoCreate(ctx *Context) *exit.Error {
	ref, e := hub.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	kind := strings.TrimSpace(ctx.Inv.Value("--kind"))
	if kind == "" {
		return exit.Usagef("`cozy repo create` needs --kind <model|endpoint>").
			WithRemedy("the hub locks a repo's kind at creation; there is no default").
			WithNext("cozy help repo create")
	}
	if e := hub.CheckKind(kind); e != nil {
		return e.WithNext("cozy help repo create")
	}
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return exit.Usagef("`cozy repo create` needs --reason <why>").
			WithRemedy("the hub records why every first-party write happened, before it acts").
			WithNext("cozy help repo create")
	}

	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	r, e := c.CreateRepo(hctx, ref.Org, ref.Name, kind, reason)
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{
		Kind: "repo",
		Fields: []render.Field{
			{K: "ref", V: r.Ref()},
			{K: "repo_kind", V: r.Kind},
			{K: "created", V: stamp(r.CreatedAt)},
			{K: "hub", V: c.Base()},
			{K: "reason", V: reason},
		},
		Notes: []string{"the hub recorded this write in its admin audit before performing it"},
		Next:  []string{"cozy repo show " + r.Ref()},
	})
}

func handleHubStatus(ctx *Context) *exit.Error {
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()

	rec := render.Record{
		Kind: "hub",
		Fields: []render.Field{
			{K: "url", V: c.Base() + " (" + ctx.Cfg.HubURLSource + ")"},
			{K: "token", V: c.Token().Digest() + " (" + ctx.Cfg.HubTokenSource + ")"},
		},
		Notes: []string{identityNote},
	}

	health, e := c.Health(hctx)
	if e != nil {
		// An unreachable hub is a STATE this verb reports, not a refusal it raises:
		// the answer to "what hub am I pointed at" is exactly what a user needs when
		// it is down. Content-first, exit 0 — the same rule as bare `cozy`.
		rec.Fields = append(rec.Fields, render.Field{K: "reachable", V: false})
		rec.Notes = append([]string{e.Message, e.Remedy}, rec.Notes...)
		rec.Next = []string{"TENSORHUB_URL=<url> cozy hub status"}
		return emit(ctx, rec)
	}
	rec.Fields = append(rec.Fields,
		render.Field{K: "reachable", V: true},
		render.Field{K: "status", V: health.Status},
		render.Field{K: "env", V: health.Env},
	)
	if repos, e := c.Repos(hctx); e == nil {
		rec.Fields = append(rec.Fields, render.Field{K: "repos", V: len(repos)})
	} else {
		rec.Notes = append(rec.Notes, "the catalog listing refused: "+e.Message)
	}
	if !c.Token().Present() {
		rec.Notes = append(rec.Notes, "reads work as they are; first-party writes need TENSORHUB_TOKEN")
	}
	rec.Next = []string{"cozy search"}
	return emit(ctx, rec)
}

func handleHubConfig(ctx *Context) *exit.Error {
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	env, keys, e := c.EffectiveConfig(hctx)
	if e != nil {
		return e
	}

	l := render.List{
		Kind:      "hub config",
		Fields:    []string{"key", "value", "source"},
		AllFields: []string{"key", "value", "source", "secret", "doc"},
		Empty:     "0 keys",
	}
	secrets := 0
	for _, k := range keys {
		if k.Secret {
			secrets++
		}
		l.Rows = append(l.Rows, map[string]string{
			"key": k.Key, "value": k.Value, "source": k.Source,
			"secret": boolText(k.Secret), "doc": k.Doc,
		})
	}
	l.Aggregates = []render.Field{
		{K: "keys", V: len(keys)},
		{K: "secrets", V: secrets},
		{K: "env", V: env},
		{K: "hub", V: c.Base()},
	}
	l.Notes = []string{"a secret key renders as sha256:<12 hex>; its value never leaves the hub"}
	l.Next = []string{"cozy hub status"}
	return emit(ctx, l)
}

func boolText(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
