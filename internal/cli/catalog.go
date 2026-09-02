package cli

import (
	"fmt"
	"sort"
	"strings"

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
	return hub.New(ctx.Cfg, "cozy/"+tag+"+"+rev).
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
				return handleModelShow(ctx, ref)
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
	if len(resources) > limit {
		resources = resources[:limit]
	}

	l := output.List{Name: kind + "s", Fields: []string{"ref", "latest"},
		AllFields: []string{"ref", "latest", "created", "org", "name"}, Total: search.Total}
	if kind == "model" {
		l.Fields = []string{"model", "family", "lanes"}
		l.AllFields = []string{"model", "family", "lanes", "created", "org", "name"}
	}
	for _, r := range resources {
		row := map[string]string{
			"ref": r.Ref(), "latest": r.LatestRelease,
			"created": stamp(r.CreatedAt), "org": r.Org, "name": r.Name,
		}
		if kind == "model" {
			card, problem := c.ModelCard(hctx, hub.Ref{Org: r.Org, Name: r.Name})
			if problem != nil {
				return problem
			}
			row = modelSearchRow(card)
		}
		l.Rows = append(l.Rows, row)
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

func handleModelShow(ctx *Context, ref hub.Ref) *exit.Error {
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := c.ModelCard(hctx, ref)
	if problem != nil {
		return problem
	}
	return emit(ctx, output.List{
		Name: "models", Fields: []string{"model", "family", "lanes"},
		AllFields: []string{"model", "family", "lanes", "created", "org", "name"},
		Rows:      []map[string]string{modelSearchRow(card)}, Total: 1,
	})
}

func modelSearchRow(card hub.ModelCard) map[string]string {
	seen := map[string]bool{}
	for _, release := range card.Releases {
		if release.Yanked {
			continue
		}
		releaseName := strings.TrimSpace(release.Release)
		if releaseName == "" {
			continue
		}
		for _, lane := range release.Lanes {
			if name := strings.TrimSpace(lane.Lane); name != "" {
				seen[releaseName+"/"+name] = true
			}
		}
	}
	coordinates := make([]string, 0, len(seen))
	for coordinate := range seen {
		coordinates = append(coordinates, coordinate)
	}
	sort.Strings(coordinates)
	return map[string]string{
		"model": card.Model.Ref(), "family": card.Model.Family,
		"lanes":   strings.Join(coordinates, ", "),
		"created": stamp(card.Model.CreatedAt), "org": card.Model.Org, "name": card.Model.Name,
	}
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
