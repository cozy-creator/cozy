package app

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

// `describe` / `doctor` / `fit` — the inspection verbs. THE RUNTIME DERIVES, cozy-creator
// renders (cozy-creator.md). No verdict, no schema and no host fact is recomputed here.
//
// Where they read from differs, and deliberately:
//
//	describe  the RECORDED surface. cl-009's install ran the release's own
//	          `cozy-runtime describe --check` and recorded what it vouched for, so the
//	          committed descriptor is a proven document and reading it back is free.
//	          Re-running `describe` per invocation would import the endpoint's whole
//	          module graph to learn a fact already on the record. **Divergence from
//	          cozy-creator.md's "delegates to `cozy-runtime describe`": the delegation
//	          happened at install and this renders its result.** The recorded digest is
//	          checked on every read, so an edited tree refuses.
//	doctor    the LocalService's own /v1/local/doctor, plus the pinned generation's
//	          `cozy-runtime doctor` for device/driver/CUDA/encoder/CAS facts.
//	fit       always delegated and never cached: a verdict prices against the card's
//	          MEASURED free bytes right now, and yesterday's answer is not an answer.

func handleDescribe(ctx *Context) *exit.Error {
	// `org/endpoint[@vN][/function]` — cut the FUNCTION off the END, not the org off the
	// front: the ref itself contains a slash.
	ref, function := splitFunction(ctx.Inv.Args[0])
	endpoint, major, _ := splitMajor(ref)
	facts, e := generationFacts(ctx, endpoint, major)
	if e != nil {
		return e
	}
	d := facts.Descriptor
	if function != "" {
		ep, e := d.Function(function)
		if e != nil {
			return e
		}
		return emit(ctx, render.Record{Kind: "function", Fields: []render.Field{
			{K: "endpoint", V: endpoint},
			{K: "function", V: ep.Name},
			{K: "kind", V: ep.Kind},
			{K: "gpu", V: ep.GPU},
			{K: "request", V: fieldLines(ep.Request)},
			{K: "result", V: fieldLines(ep.Result)},
			{K: "outputs", V: launch.AssetPaths(ep.Result)},
			{K: "models", V: slotLines(ep)},
			{K: "capabilities", V: ep.Caps},
		}, Notes: []string{"the surface this release's OWN runtime derived and the install verified"}})
	}
	rows := []map[string]string{}
	for i := range d.Entrypoints {
		ep := &d.Entrypoints[i]
		rows = append(rows, map[string]string{
			"function": ep.Name, "kind": ep.Kind,
			"gpu":     fmt.Sprintf("%t", ep.GPU),
			"request": ep.Request.Name,
			"result":  ep.Result.Name,
			"outputs": strings.Join(launch.AssetPaths(ep.Result), ", "),
			"models":  strings.Join(slotLines(ep), ", "),
		})
	}
	for i := range d.Jobs {
		job := &d.Jobs[i]
		rows = append(rows, map[string]string{
			"function": job.Name, "kind": "job",
			"request": job.Request.Name, "result": job.Result.Name,
		})
	}
	return emit(ctx, render.List{Kind: "describe",
		Fields:    []string{"function", "kind", "request", "result"},
		AllFields: []string{"function", "kind", "gpu", "request", "result", "outputs", "models"},
		Rows:      rows,
		Empty:     "this release registers 0 functions",
		Aggregates: []render.Field{
			{K: "endpoint", V: endpoint},
			{K: "generation", V: short12(facts.Generation.ID)},
			{K: "surface_digest", V: d.Digest},
		},
		Next: []string{"cozy describe " + ctx.Inv.Args[0] + "/<function>"}})
}

func splitFunction(raw string) (ref, function string) {
	parts := strings.Split(raw, "/")
	if len(parts) >= 3 {
		return strings.Join(parts[:len(parts)-1], "/"), parts[len(parts)-1]
	}
	return raw, ""
}

func fieldLines(s launch.Struct) []string {
	out := make([]string, 0, len(s.Fields))
	for _, f := range s.Fields {
		out = append(out, f.Name+": "+renderType(f.Type)+" ("+f.Wire+")")
	}
	return out
}

// renderType prints the descriptor's own rendered type verbatim — a scalar as its name,
// anything structured as the document it is. This host does not paraphrase a schema.
func renderType(raw json.RawMessage) string {
	var scalar string
	if json.Unmarshal(raw, &scalar) == nil {
		return scalar
	}
	return string(raw)
}

func slotLines(ep *launch.Entrypoint) []string {
	out := make([]string, 0, len(ep.Models))
	for _, m := range ep.Models {
		components := strings.Join(m.ComponentUse[ep.Name], "+")
		if components == "" {
			components = "-"
		}
		out = append(out, m.Path+" "+m.Class+" ["+components+"]")
	}
	return out
}

func short12(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// ------------------------------------------------------------------------- doctor

func handleDoctor(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	doc, e := c.Doctor()
	if e != nil {
		return e
	}
	fields := []render.Field{}
	for _, key := range []string{"service", "host", "counts", "event_head"} {
		if v, ok := doc[key]; ok {
			fields = append(fields, render.Field{K: key, V: v})
		}
	}
	workers, e := c.Workers()
	if e != nil {
		return e
	}
	live := make([]string, 0, len(workers))
	for _, w := range workers {
		live = append(live, fmt.Sprintf("%s %s pid=%d %s plans=%d",
			w.Endpoint, w.InstanceID, w.PID, w.Intake, len(w.Plans)))
	}
	fields = append(fields, render.Field{K: "workers", V: live})

	// The DEVICE facts are the runtime's, read from a pinned generation's own venv. An
	// empty host is not an error: with nothing installed there is no runtime to ask, and
	// saying so is a definitive answer.
	endpoints := ctx.Inv.Args
	_ = endpoints
	notes := []string{}
	if facts, e := anyGeneration(ctx); e == nil && facts != nil {
		host, e := facts.Runtime.HostFacts()
		if e != nil {
			notes = append(notes, "device facts unreadable: "+e.Message)
		} else {
			fields = append(fields, render.Field{K: "device", V: hostSummary(host)})
			notes = append(notes, "device/driver/CUDA facts derived by "+
				facts.Generation.Endpoint+"'s own cozy-runtime")
		}
	} else {
		notes = append(notes, "no endpoint is installed, so no runtime is present to report device facts")
	}
	return emit(ctx, render.Record{Kind: "doctor", Fields: fields, Notes: notes,
		Next: []string{"cozy fit <org/endpoint>"}})
}

// hostSummary keeps the runtime's own words. A tri-state fact (present/absent/UNREADABLE)
// is carried through as-is: flattening "unreadable" into "absent" is exactly the lie the
// tri-state exists to prevent.
func hostSummary(doc map[string]any) map[string]any {
	if rows, ok := doc["host"].(map[string]any); ok {
		return rows
	}
	return doc
}

// anyGeneration picks one pinned generation to ask host questions of. Every generation
// carries a cozy-runtime and they all measure the same host.
func anyGeneration(ctx *Context) (*launch.Facts, *exit.Error) {
	c, e := dial(ctx)
	if e != nil {
		return nil, e
	}
	rows, e := c.Endpoints()
	if e != nil {
		return nil, e
	}
	for _, row := range rows {
		facts, e := generationFacts(ctx, row.Endpoint, 0)
		if e == nil {
			return facts, nil
		}
	}
	return nil, exit.New(exit.NotFound, "no installed endpoint")
}

// --------------------------------------------------------------------------- fit

func handleFit(ctx *Context) *exit.Error {
	for _, flag := range []string{"--model", "--lane"} {
		if v := ctx.Inv.Value(flag); v != "" {
			return exit.Named(exit.Usage, "override_unresolved",
				"%s is admissible on this host and nothing resolves one yet", flag).
				WithRemedy("the local binding resolver lands with cl-005")
		}
	}
	ref, function := splitFunction(ctx.Inv.Args[0])
	endpoint, major, _ := splitMajor(ref)
	facts, e := generationFacts(ctx, endpoint, major)
	if e != nil {
		return e
	}
	hostFacts, verdicts, e := facts.Runtime.Fit(function, nil)
	if e != nil {
		return e
	}
	// THE RUNTIME'S OWN DOCUMENT, rendered. There is no parser here and no second
	// arithmetic: `verdict` is the word the runtime chose, `rendered` is its own line, and
	// the only thing this side derives is the CLI's exit code over them.
	host := []render.Field{}
	for _, key := range []string{"device", "allocatable_bytes", "free_bytes", "total_bytes"} {
		if v, ok := hostFacts[key]; ok {
			host = append(host, render.Field{K: key, V: v})
		}
	}
	rows, worst := []map[string]string{}, exit.OK
	exact := false
	for _, v := range verdicts {
		rows = append(rows, map[string]string{
			"function": v.Function, "verdict": v.Verdict, "rendered": v.Rendered,
		})
		exact = exact || v.Exact
		if code := fitExit(v.Verdict); code > worst {
			worst = code
		}
	}
	notes := []string{"verdicts are the runtime's own — the same arithmetic `run` prices with"}
	if !exact {
		notes = append(notes,
			"no payload given — structural/range information only, not an exact fit")
	}
	l := render.List{Kind: "fit",
		Fields:     []string{"function", "verdict", "rendered"},
		AllFields:  []string{"function", "verdict", "rendered"},
		Rows:       rows,
		Empty:      "0 fit verdicts",
		Aggregates: append([]render.Field{{K: "endpoint", V: endpoint}}, host...),
		Notes:      notes,
		Next:       []string{"cozy run " + endpoint + "/vN/<function>"},
	}
	if worst == exit.OK {
		return emit(ctx, l)
	}
	// A fit verdict is ADVISORY above the physical floor: a degradable shortfall degrades
	// and exits 0. Below it there is no rung, and that is exit 14 with the shortfall the
	// runtime quantified; a structural incompatibility is 6. cr-016's own `fit` exits 0
	// for every verdict — it is an inspection verb in a bare venv — so this mapping is
	// cozy-creator's CLI contract over the runtime's unchanged verdict, never a second
	// arithmetic.
	if err := l.Emit(ctx.Out, ctx.Mode()); err != nil && !ctx.Mode().JSON {
		return exit.As(err)
	}
	return exit.Named(worst, "fit_"+worst.Name(),
		"%s has no fitting plan on this host", endpoint).
		WithRemedy("the rows above carry the runtime's quantified verdict").
		WithNext("cozy doctor")
}

func fitExit(verdict string) exit.Code {
	switch verdict {
	case "capacity":
		return exit.Capacity
	case "structurally-incompatible":
		return exit.Structural
	case "unbound":
		return exit.NotFound
	}
	return exit.OK
}
