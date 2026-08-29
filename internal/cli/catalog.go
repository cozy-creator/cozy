package cli

import (
	"fmt"
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
	return hub.New(ctx.Cfg, "cozy/"+tag+"+"+rev).
		WithTokenSource(accountauth.New(ctx.Cfg))
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
	if len(ctx.Inv.Args) == 1 {
		if _, problem := hub.ParseRef(ctx.Inv.Args[0]); problem == nil {
			return handleResourceShow(ctx, kind)
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
		resources, search, e = c.Models(hctx, query)
	}
	if e != nil {
		return e
	}
	if len(resources) > limit {
		resources = resources[:limit]
	}

	l := output.List{
		Name:      kind + "s",
		Fields:    []string{"ref"},
		AllFields: []string{"ref", "created", "org", "name"},
		Total:     search.Total,
	}
	for _, r := range resources {
		l.Rows = append(l.Rows, map[string]string{
			"ref": r.Ref(), "created": stamp(r.CreatedAt), "org": r.Org, "name": r.Name,
		})
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

func handleResourceShow(ctx *Context, kind string) *exit.Error {
	ref, e := hub.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	var r hub.Resource
	if kind == "package" {
		r, e = c.Package(hctx, ref)
	} else {
		r, e = c.Model(hctx, ref)
	}
	if e != nil {
		return e
	}
	rec := output.Record{
		Fields: []output.Field{
			{K: "ref", V: r.Ref()},
			{K: "created", V: stamp(r.CreatedAt)},
		},
	}
	if kind == "model" {
		rec.Next = []string{"cozy model download " + r.Ref()}
	}
	return emit(ctx, rec)
}
