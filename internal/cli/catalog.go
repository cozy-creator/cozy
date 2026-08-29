package cli

import (
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/hub"
	"github.com/cozy-creator/cozy-creator/internal/output"
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

func handleEndpointSearch(ctx *Context) *exit.Error { return handleResourceSearch(ctx, "endpoint") }
func handleModelSearch(ctx *Context) *exit.Error    { return handleResourceSearch(ctx, "model") }

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
	if kind == "endpoint" {
		resources, search, e = c.Endpoints(hctx, query)
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
		Kind:      kind + " search",
		Fields:    []string{"ref", "created"},
		AllFields: []string{"ref", "created", "org", "name"},
	}
	for _, r := range resources {
		l.Rows = append(l.Rows, map[string]string{
			"ref": r.Ref(), "created": stamp(r.CreatedAt), "org": r.Org, "name": r.Name,
		})
	}

	l.Aggregates = []output.Field{
		{K: "results", V: len(l.Rows)},
		{K: kind + "s", V: search.Total},
		{K: "hub", V: c.Base()},
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
		l.Empty = "0 " + kind + "s in the catalog"
		l.Next = []string{"cozy " + kind + " publish <org/name>"}
	case len(l.Rows) == 0:
		l.Empty = "0 results for \"" + query + "\""
		l.Next = []string{"cozy " + kind + " search"}
	default:
		verb := "install"
		if kind == "model" {
			verb = "download"
		}
		l.Next = []string{"cozy " + kind + " " + verb + " <org/name>"}
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
	if kind == "endpoint" {
		r, e = c.Endpoint(hctx, ref)
	} else {
		r, e = c.Model(hctx, ref)
	}
	if e != nil {
		return e
	}
	return emit(ctx, output.Record{
		Kind: kind,
		Fields: []output.Field{
			{K: "ref", V: r.Ref()},
			{K: "created", V: stamp(r.CreatedAt)},
			{K: "hub", V: c.Base()},
		},
		Next: []string{"cozy " + kind + " search"},
	})
}
