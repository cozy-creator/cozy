package app

import (
	"fmt"
	"os"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/render"
	"github.com/cozy-creator/cozy-creator/internal/retention"
)

// `cozy media ls` (cl-033) — WHAT IS THIS HOST STORING.
//
// Retained outputs were the one thing on this root nothing could enumerate: the bytes are
// mirrored here before a terminal is acked and kept after the pod is destroyed, so a user
// could fetch one by an id they already had and could not ask what they had at all.
//
// It reads the ONE local database DIRECTLY rather than through the client API, which is
// deliberate: the whole point of this plane is that it survives the pod, and it should
// survive `cozy down` too. A user who has shut everything off can still ask what is on
// their disk, and `--json` gives cl-007's gallery the same rows over a pipe until that
// UI's own route lands with it.
func handleMediaLs(ctx *Context) *exit.Error {
	horizon, e := retention.ParseHorizon(ctx.Inv.Value("--keep-media"))
	if e != nil {
		return e
	}
	l, st, _, e := open(ctx.Cfg, false)
	if e != nil {
		return e
	}
	defer st.Close()

	rows, e := st.RetainedOutputs()
	if e != nil {
		return e
	}
	plan, e := retention.Build(l, st, horizon, time.Now())
	if e != nil {
		return e
	}
	reclaimable := map[string]bool{}
	for _, it := range plan.Items {
		reclaimable[it.MediaID] = true
	}

	out := render.List{
		Kind:      "media",
		Fields:    []string{"media_id", "output", "bytes", "age", "state"},
		AllFields: []string{"media_id", "output", "endpoint", "request", "attempt", "kind", "type", "bytes", "age", "state", "path"},
		Empty:     "0 retained outputs — nothing this host published is on this disk",
	}
	now := time.Now().UTC()
	var onDisk int64
	live, gone := 0, 0
	for _, r := range rows {
		age := time.Duration(0)
		if t, err := time.Parse(time.RFC3339Nano, r.VisibleAt); err == nil {
			age = now.Sub(t)
		}
		// The state is read from the DISK, not inferred from the row: a file that went
		// missing without `cozy gc` taking it is the one case a listing must not lie about.
		_, err := os.Stat(r.Path)
		state := "on disk"
		switch {
		case r.ReclaimedAt != "":
			state = "reclaimed"
			gone++
		case err != nil:
			state = "absent"
			gone++
		case reclaimable[r.MediaID]:
			state = "reclaimable"
			live++
			onDisk += r.Length
		default:
			live++
			onDisk += r.Length
		}
		out.Rows = append(out.Rows, map[string]string{
			"media_id": r.MediaID, "output": r.OutputID, "endpoint": r.Endpoint,
			"request": r.RequestID, "attempt": fmt.Sprint(r.Attempt), "kind": r.Kind,
			"type": r.MimeType, "bytes": render.Bytes(r.Length), "age": retention.Age(age),
			"state": state, "path": r.Path,
		})
	}
	if len(out.Rows) == 0 {
		out.Next = []string{"cozy run <org/endpoint/vN/function>"}
		return emit(ctx, out)
	}
	out.Aggregates = []render.Field{
		{K: "outputs", V: len(out.Rows)},
		{K: "on_disk", V: fmt.Sprintf("%d output(s), %s", live, render.Bytes(onDisk))},
		{K: "reclaimable", V: fmt.Sprintf("%d output(s), %s past the %s horizon",
			len(plan.Items), render.Bytes(plan.Bytes), retention.Short(horizon))},
		{K: "reclaimed", V: gone},
	}
	out.Notes = []string{
		"a media id is fetched with `GET /v1/media/{id}` on the LocalService; a reclaimed id answers 410",
	}
	out.Next = []string{"cozy gc"}
	return emit(ctx, out)
}
