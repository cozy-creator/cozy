package app

import (
	"fmt"
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

func handleEndpointSearch(ctx *Context) *exit.Error { return handleResourceSearch(ctx, "endpoint") }
func handleModelSearch(ctx *Context) *exit.Error    { return handleResourceSearch(ctx, "model") }

func handleResourceSearch(ctx *Context, kind string) *exit.Error {
	query := strings.ToLower(strings.TrimSpace(strings.Join(ctx.Inv.Args, " ")))

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

	l := render.List{
		Kind:      kind + " search",
		Fields:    []string{"ref", "created"},
		AllFields: []string{"ref", "created", "org", "name"},
	}
	for _, r := range resources {
		l.Rows = append(l.Rows, map[string]string{
			"ref": r.Ref(), "created": stamp(r.CreatedAt), "org": r.Org, "name": r.Name,
		})
	}

	l.Aggregates = []render.Field{
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
		l.Next = []string{"cozy " + kind + " create <org/name> --reason <why>"}
	case len(l.Rows) == 0:
		l.Empty = "0 results for \"" + query + "\""
		l.Next = []string{"cozy " + kind + " search"}
	default:
		l.Next = []string{"cozy " + kind + " show <org/name>"}
	}
	return emit(ctx, l)
}

func handleEndpointShow(ctx *Context) *exit.Error { return handleResourceShow(ctx, "endpoint") }
func handleModelShow(ctx *Context) *exit.Error    { return handleResourceShow(ctx, "model") }

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
	return emit(ctx, render.Record{
		Kind: kind,
		Fields: []render.Field{
			{K: "ref", V: r.Ref()},
			{K: "created", V: stamp(r.CreatedAt)},
			{K: "hub", V: c.Base()},
		},
		Next: []string{"cozy " + kind + " search"},
	})
}

func handleEndpointCreate(ctx *Context) *exit.Error { return handleResourceCreate(ctx, "endpoint") }
func handleModelCreate(ctx *Context) *exit.Error    { return handleResourceCreate(ctx, "model") }

func handleResourceCreate(ctx *Context, kind string) *exit.Error {
	ref, e := hub.ParseRef(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	reason := strings.TrimSpace(ctx.Inv.Value("--reason"))
	if reason == "" {
		return exit.Usagef("`cozy %s create` needs --reason <why>", kind).
			WithRemedy("the hub records why every first-party write happened, before it acts").
			WithNext("cozy help " + kind + " create")
	}

	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	var r hub.Resource
	if kind == "endpoint" {
		r, e = c.CreateEndpoint(hctx, ref.Org, ref.Name, reason)
	} else {
		r, e = c.CreateModel(hctx, ref.Org, ref.Name, reason)
	}
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{
		Kind: kind,
		Fields: []render.Field{
			{K: "ref", V: r.Ref()},
			{K: "created", V: stamp(r.CreatedAt)},
			{K: "hub", V: c.Base()},
			{K: "reason", V: reason},
		},
		Notes: []string{"the hub recorded this write in its admin audit before performing it"},
		Next:  []string{"cozy " + kind + " show " + r.Ref()},
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
	if _, search, e := c.Endpoints(hctx, ""); e == nil {
		rec.Fields = append(rec.Fields, render.Field{K: "endpoints", V: search.Total})
	} else {
		rec.Notes = append(rec.Notes, "the endpoint listing refused: "+e.Message)
	}
	if _, search, e := c.Models(hctx, ""); e == nil {
		rec.Fields = append(rec.Fields, render.Field{K: "models", V: search.Total})
	} else {
		rec.Notes = append(rec.Notes, "the model listing refused: "+e.Message)
	}
	if !c.Token().Present() {
		rec.Notes = append(rec.Notes, "reads work as they are; first-party writes need TENSORHUB_TOKEN")
	}
	rec.Next = []string{"cozy endpoint search", "cozy model search"}
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
