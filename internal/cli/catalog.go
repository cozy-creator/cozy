package cli

import (
	"fmt"
	"slices"
	"strings"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
)

// The catalog verbs (cl-011). Reads are public; writes obtain a short bearer from
// the account machine key unless the operator explicitly configured a static token.
//
// The line every one of them holds: a hub refusal reaches the user verbatim — the
// hub's code, message and remedy — under a code from the shared exit matrix.

func client(ctx *Context) *hub.Client {
	rev, _ := buildStamp()
	auth := ctx.AccountAuth
	if auth == nil {
		auth = accountauth.New(ctx.Cfg)
	}
	return hub.New(ctx.Cfg, "cozy/"+version()+"+"+rev).
		WithTokenSource(auth)
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

func handlePackageSearch(ctx *Context) *exit.Error { return handleResourceSearch(ctx, "package") }
func handleModelSearch(ctx *Context) *exit.Error   { return handleResourceSearch(ctx, "model") }

func handleResourceSearch(ctx *Context, kind string) *exit.Error {
	family := strings.TrimSpace(ctx.Inv.Value("--family"))
	if len(ctx.Inv.Args) == 1 && family == "" {
		if ref, problem := hub.ParseRef(ctx.Inv.Args[0]); problem == nil {
			if kind == "model" {
				return handleModelSearchRef(ctx, ref)
			}
			return handlePackageShow(ctx, ref)
		}
	}
	query := strings.ToLower(strings.TrimSpace(strings.Join(ctx.Inv.Args, " ")))
	limit := 20
	if raw := ctx.Inv.Value("--limit"); raw != "" {
		var err error
		if _, err = fmt.Sscan(raw, &limit); err != nil || limit < 1 || limit > 100 {
			return exit.Usagef("--limit %q is not between 1 and 100", raw)
		}
	}

	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	var resources []hub.Resource
	var search hub.ResourceSearch
	var e *exit.Error
	if kind == "package" {
		resources, search, e = c.Packages(hctx, query)
	} else {
		resources, search, e = c.Models(hctx, query, family)
	}
	if e != nil {
		return e
	}
	matching := max(search.Total, len(resources))
	more := searchMore(len(resources), limit)
	if len(resources) > limit {
		resources = resources[:limit]
	}

	if kind == "model" {
		cards := make([]hub.ModelCard, 0, len(resources))
		for _, resource := range resources {
			card, problem := c.ModelCard(hctx, hub.Ref{Org: resource.Org, Name: resource.Name})
			if problem != nil {
				return problem
			}
			cards = append(cards, card)
		}
		return emit(ctx, modelSearchView{Cards: cards, Matching: matching, More: more})
	}
	l := output.List{Name: kind + "s", Fields: []string{"ref", "latest"},
		AllFields: []string{"ref", "latest", "created", "org", "name"}, Total: matching, More: more, Uncapped: true}
	for _, r := range resources {
		l.Rows = append(l.Rows, map[string]string{"ref": r.Ref(), "latest": r.LatestRelease,
			"created": stamp(r.CreatedAt), "org": r.Org, "name": r.Name})
	}

	// The public listing is a whole-catalog read filtered here. That is honest at a
	// catalog this size and dishonest at scale, so it says so at the hub's own cap
	// rather than silently returning a prefix as if it were the answer.
	if search.Capped {
		l.Notes = append(l.Notes,
			fmt.Sprintf("Tensorhub returned the first %d of %d matching %ss", search.Limit, search.Total, kind))
	}
	switch {
	case len(l.Rows) == 0 && query == "":
		l.Next = []string{"cozy help " + kind + " publish"}
	case len(l.Rows) == 0:
		l.Next = []string{"cozy " + kind + " search"}
	}
	return emit(ctx, l)
}

// searchMore is how to see the matches a search left out: a higher --limit while the Hub
// returned more than it showed, else a narrower search.
func searchMore(returned, limit int) string {
	switch {
	case returned > limit && returned <= 100:
		return fmt.Sprintf("Use --limit %d to show all.", returned)
	case returned > limit:
		return "Use --limit 100 to show more, or narrow the search."
	}
	return "Narrow the search to see the rest."
}

// handlePackageInfo lists one published package's releases, newest first, from the
// package card the Hub already serves.
func handlePackageInfo(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(strings.TrimSpace(ctx.Inv.Args[0]))
	if problem != nil {
		return problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx).PackageCard(hctx, ref)
	if problem != nil {
		return problem
	}
	l := output.List{Name: "releases", Fields: []string{"release", "published", "yanked"},
		AllFields: []string{"release", "published", "yanked", "yanked_at"},
		Lead:      []string{ref.String() + ", latest " + card.Package.LatestRelease + ":"}}
	releases := slices.Clone(card.Releases)
	slices.SortFunc(releases, func(a, b hub.ReleaseSummary) int {
		left, errLeft := pep440.Parse(a.Release)
		right, errRight := pep440.Parse(b.Release)
		if errLeft != nil || errRight != nil {
			return strings.Compare(b.Release, a.Release)
		}
		return right.Compare(left)
	})
	for _, row := range releases {
		yanked := "no"
		if row.Yanked || row.YankedAt != "" {
			yanked = "yes"
		}
		l.Rows = append(l.Rows, map[string]string{"release": row.Release, "published": stamp(row.CutAt),
			"yanked": yanked, "yanked_at": stamp(row.YankedAt)})
	}
	l.Total = len(l.Rows)
	return emit(ctx, l)
}

func handleModelSearchRef(ctx *Context, ref hub.Ref) *exit.Error {
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx).ModelCard(hctx, ref)
	if problem != nil {
		return problem
	}
	return emit(ctx, modelSearchView{Cards: []hub.ModelCard{card}})
}

func handleModelFamily(ctx *Context) *exit.Error {
	ref, problem := hub.ParseRef(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	family := strings.TrimSpace(ctx.Inv.Value("--family"))
	clear := ctx.Inv.Bool("--clear")
	if clear && family != "" {
		return exit.Usagef("provide a family or --clear, not both")
	}
	if !clear && family == "" {
		return exit.Usagef("provide one recognized family or --clear")
	}
	if clear {
		family = ""
	}
	hctx, cancel := hub.Context()
	defer cancel()
	model, problem := client(ctx).SetModelFamily(hctx, ref, family)
	if problem != nil {
		return problem
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "model", V: model.Ref()}, {K: "family", V: model.Family}, {K: "status", V: "updated"},
	}, "model", "family", "status"))
}

func handlePackageShow(ctx *Context, ref hub.Ref) *exit.Error {
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	r, e := c.Package(hctx, ref)
	if e != nil {
		return e
	}
	return emit(ctx, output.Record{
		Fields: []output.Field{
			{K: "ref", V: r.Ref()},
			{K: "created", V: stamp(r.CreatedAt)},
		},
	})
}
