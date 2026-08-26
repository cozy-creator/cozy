package app

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	localapi "github.com/cozy-creator/cozy-creator-v2/internal/client"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

// THE LIFECYCLE AND REQUEST VERBS (cl-010), every one of them a CLIENT of the local
// client API. There is no direct-Go path from a verb to the orchestrator: `dial` is the
// only way into any of them, and what it returns speaks HTTP to a separate process.
//
// ONE PRODUCT EXECUTION PATH. `cozy run` is
//
//	POST /v1/requests            durable request, idempotency key + body digest
//	GET  /v1/requests/{id}/events  the attempt's own lifecycle, terminal-stop
//	GET  /v1/media/{id}          the accepted output's bytes, by opaque id
//
// and a cold run traverses exactly the states a warm one does — a warm worker changes
// latency, never the path, the history, or who owns cancellation. `start` is the
// lifecycle/prewarm verb and is NEVER a second invocation mechanism: it makes a worker
// resident and returns.

// dial builds the API client. Every verb here has already passed the shared exit-9 gate,
// so this is the credential read and nothing else.
func dial(ctx *Context) (*localapi.Client, *exit.Error) {
	return localapi.Open(ctx.Cfg, ctx.Service)
}

// ---------------------------------------------------------------------- start / stop

func handleStart(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	ref := ctx.Inv.Args[0]
	began := time.Now()
	res, e := c.EnsureWorker(ref, !ctx.Inv.Bool("--no-warm"))
	if e != nil {
		return e
	}
	fields := []render.Field{
		{K: "endpoint", V: res.Endpoint},
		{K: "instance", V: res.InstanceID},
	}
	fields = append(fields, render.Field{K: "change", V: res.Change})
	if res.Change == "none" {
		// Already serving = idempotent 0 (cozy-creator.md). It is a STATE this verb
		// reports, not a refusal it raises.
		return emit(ctx, render.Record{Kind: "worker",
			Fields: append(fields, render.Field{K: "state", V: "resident"}),
			Notes:  []string{res.Note}, Next: []string{"cozy run " + ref + "/<function>"}})
	}
	// `-d` returns as soon as the process is spawned; the default WAITS for the worker to
	// advertise a dispatchable plan, because "started" and "warm" are different facts and
	// a prewarm verb that returned on the first is useless.
	if ctx.Inv.Bool("--detach") {
		return emit(ctx, render.Record{Kind: "worker",
			Fields: append(fields, render.Field{K: "state", V: "spawned"}),
			Notes:  []string{res.Note}, Next: []string{"cozy status"}})
	}
	worker, e := waitReady(c, res.InstanceID)
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{Kind: "worker", Fields: append(fields,
		render.Field{K: "state", V: "ready"},
		render.Field{K: "pid", V: worker.PID},
		render.Field{K: "devices", V: worker.Devices},
		render.Field{K: "ready_plans", V: len(worker.Plans)},
		render.Field{K: "took", V: took(began)}),
		Next: []string{"cozy run " + ref + "/<function>"}})
}

// waitReady polls the workers listing until the protocol says dispatchable. The fact is
// the WORKER's own — the placement's SERVING AXIS at DISPATCHABLE plus at least one
// advertised plan — read through the API, never a clock and never "the process is alive".
//
// It said "never a clock" while holding one: a 10-minute ceiling, which is a statement
// about how large a model may be rather than about anything having gone wrong. The ways
// this fails are all the worker's own and all visible through the listing — it EXITS, it
// goes SILENT, it holds a FAULT, or this owner REFUSED it — which is the same set
// `orchestrator.EnsurePlacementReady` decides on, so the two sides of the same wait cannot
// disagree.
func waitReady(c *localapi.Client, instance string) (localapi.Worker, *exit.Error) {
	silent := (orchestrator.SilentReports * orchestrator.ReportCadence).Milliseconds()
	errorGrace := orchestrator.ErrorGrace.Milliseconds()
	for {
		// THIS OWNER'S OWN VERDICT COMES FIRST, exactly as it does inside the orchestrator:
		// a worker whose claim was refused here is not slow and not silent, and waiting out
		// eight missed report periods to call it stalled would report a network symptom for
		// an identity fact this process already established.
		workers, e := c.Workers()
		if e != nil {
			return localapi.Worker{}, e
		}
		found := false
		for _, w := range workers {
			if w.InstanceID != instance {
				continue
			}
			found = true
			if w.Exited {
				return localapi.Worker{}, exit.New(exit.Failed,
					"the endpoint worker exited before advertising a plan").
					WithRemedy("its log is under the local root's workers/%s", instance).
					WithNext("cozy logs <org/endpoint>")
			}
			if w.Dispatchable() {
				return w, nil
			}
			if w.Fault != "" && w.ErrorForMS == 0 {
				// A refusal this owner recorded at claim time carries no error clock: it is
				// a settled verdict, not a state the worker might leave.
				return localapi.Worker{}, exit.New(exit.Conflict,
					"this host refused the worker at that address: %s", w.Fault).
					WithNext("cozy logs <org/endpoint>")
			}
			if w.QuietMS > silent {
				return localapi.Worker{}, exit.Named(exit.Failed, "worker_silent",
					"the endpoint worker has reported no observed state for %d ms, which is %d missed "+
						"periods of %s: it is stalled, not slow", w.QuietMS,
					orchestrator.SilentReports, orchestrator.ReportCadence).
					WithNext("cozy logs <org/endpoint>")
			}
			if w.ErrorForMS > errorGrace {
				return localapi.Worker{}, exit.New(exit.Failed,
					"the endpoint worker's placement has held a fault for %d ms and never "+
						"became dispatchable: %s", w.ErrorForMS, w.Fault).
					WithNext("cozy logs <org/endpoint>")
			}
		}
		if !found {
			return localapi.Worker{}, exit.New(exit.Failed,
				"worker %s is no longer registered with the orchestrator", instance)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func handleStop(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	all := ctx.Inv.Bool("--all")
	if !all && len(ctx.Inv.Args) == 0 {
		return exit.Usagef("`cozy stop` needs <org/endpoint> or --all").
			WithNext("cozy help stop")
	}
	workers, e := c.Workers()
	if e != nil {
		return e
	}
	target := ""
	if !all {
		target = ctx.Inv.Args[0]
	}
	stopped, rows := 0, []map[string]string{}
	for _, w := range workers {
		if !all && w.Endpoint != target {
			continue
		}
		// The DRAIN happens on the orchestrator's side of this call and it BLOCKS: the
		// route returns after the whole process group has gone. A remembered pid is
		// never signalled from here — the orchestrator stops what the orchestrator started.
		res, e := c.ShutdownWorker(w.InstanceID)
		if e != nil {
			return e
		}
		if res.Stopped {
			stopped++
		}
		rows = append(rows, map[string]string{
			"endpoint": w.Endpoint, "instance": w.InstanceID,
			"stopped": fmt.Sprintf("%t", res.Stopped), "pid": itoa(w.PID),
		})
	}
	l := render.List{Kind: "stop",
		Fields:     []string{"endpoint", "instance", "stopped"},
		AllFields:  []string{"endpoint", "instance", "stopped", "pid"},
		Rows:       rows,
		Empty:      "0 workers to stop",
		Aggregates: []render.Field{{K: "stopped", V: stopped}},
	}
	if len(rows) == 0 {
		// Not running is an idempotent 0, and it says which fact it is answering.
		l.Notes = []string{"nothing was running: `cozy stop` is idempotent"}
	}
	return emit(ctx, l)
}

// ---------------------------------------------------------------------------- logs

func handleLogs(ctx *Context) *exit.Error {
	subject := ctx.Inv.Args[0]
	// A PATH-SHAPED attempt input refuses (cozy-creator.md). Triage is addressed by the
	// orchestrator's OPAQUE attempt key; there is no path form, and the route this verb
	// calls takes no path parameter at all.
	if strings.ContainsAny(subject, "/\\") && !looksLikeEndpoint(subject) {
		return exit.Usagef("%q is a path, and an attempt is named by its opaque key", subject).
			WithRemedy("`cozy run` prints the attempt key; `cozy status --full` lists them").
			WithNext("cozy help logs")
	}
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	if looksLikeEndpoint(subject) {
		return endpointLog(ctx, c, subject)
	}
	t, e := c.Triage(subject)
	if e != nil {
		return e
	}
	// `explain` is the SERVER's projection of the bundle, verified against the accepted
	// terminal before it was rendered. This prints it; it re-explains nothing.
	if ctx.Mode().JSON {
		return emit(ctx, render.Record{Kind: "triage", Fields: []render.Field{
			{K: "attempt_key", V: t.AttemptKey}, {K: "subject_id", V: t.SubjectID},
			{K: "length", V: t.Length}, {K: "digest", V: t.Digest},
			{K: "verified", V: t.Verified}, {K: "explain", V: t.Explain},
			{K: "bundle", V: t.Bundle},
		}})
	}
	return emit(ctx, render.Lines{Kind: "triage", Key: "explain", Items: t.Explain,
		Empty: "the bundle carries no explainable section",
		Extra: []render.Field{
			{K: "attempt_key", V: t.AttemptKey},
			{K: "subject_id", V: t.SubjectID},
			{K: "bytes", V: render.Bytes(t.Length)},
			{K: "verified", V: t.Verified},
		},
		Notes: []string{"verified on read against what the accepted terminal declared"},
		Next:  []string{"cozy logs " + t.AttemptKey + " --json"}})
}

func looksLikeEndpoint(s string) bool {
	org, name, ok := strings.Cut(s, "/")
	return ok && org != "" && name != "" && !strings.Contains(name, "/") && !strings.HasPrefix(s, ".")
}

// endpointLog tails one endpoint worker's process log. The log is a FILE in the local
// root the orchestrator owns; the worker listing is what says which instance owns it, so
// the path is derived from an identity the server issued and never from user input.
func endpointLog(ctx *Context, c *localapi.Client, endpoint string) *exit.Error {
	workers, e := c.Workers()
	if e != nil {
		return e
	}
	instance := ""
	for _, w := range workers {
		if w.Endpoint == endpoint {
			instance = w.InstanceID
		}
	}
	if instance == "" {
		return exit.New(exit.NotFound, "no worker of %s is running on this host", endpoint).
			WithRemedy("an endpoint's process log exists while its worker does; an attempt's triage bundle outlives it").
			WithNext("cozy start "+endpoint, "cozy status")
	}
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return e
	}
	path := filepath.Join(l.WorkerDir(instance), "worker.log")
	lines := 100
	if v := ctx.Inv.Value("--lines"); v != "" {
		n := 0
		if _, err := fmt.Sscanf(v, "%d", &n); err != nil || n <= 0 {
			return exit.Usagef("--lines %q is not a positive count", v)
		}
		lines = n
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return exit.New(exit.NotFound, "worker %s has no log at %s", instance, path)
	}
	all := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(all) > lines {
		all = all[len(all)-lines:]
	}
	return emit(ctx, render.Lines{Kind: "log", Key: "lines", Items: all,
		Empty: "the worker log is empty",
		Extra: []render.Field{{K: "endpoint", V: endpoint}, {K: "instance", V: instance}},
		Notes: []string{"the orchestrator-owned process log; an attempt's triage bundle is `cozy logs <attempt>`"}})
}

// ----------------------------------------------------------------------------- run

func handleRun(ctx *Context) *exit.Error {
	if ctx.Inv.Bool("--cloud") {
		// PLACEMENT IS A CLIENT-BOUNDARY CHOICE, and the cloud host does not exist yet:
		// Launch-1 tensorhub is a headless distribution hub with no request plane
		// (decisions #229). Refusing by name beats a flag that silently runs locally.
		return exit.Named(exit.Usage, "not_implemented",
			"--cloud names a host this build cannot submit to").
			WithRemedy("Launch-1 tensorhub carries no request plane; the serving routes land with th-004+").
			WithNext("cozy run " + ctx.Inv.Args[0] + " --local")
	}
	target, e := parseTarget(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	// Overrides refuse HERE as well as at the API, and for the same reason: nothing
	// resolves one until cl-005, and an override that is silently ignored is the worse
	// bug. The refusal is the CLIENT's so it costs no round trip.
	for _, flag := range []string{"--model", "--lane", "--adapter"} {
		if v := ctx.Inv.Value(flag); v != "" {
			return exit.Named(exit.Usage, "override_unresolved",
				"%s is admissible on this host and nothing resolves one yet", flag).
				WithRemedy("the local binding resolver lands with cl-005; this build refuses rather than ignoring it")
		}
	}
	// TWO RUNTIME FLAGS HAVE NO WIRE FIELD, and they refuse rather than being dropped.
	// `--seed` and `--offline` are cozy-runtime's own (a deterministic RNG at construction,
	// a CAS-only acquisition), and the submission this host makes carries neither — the
	// ExecutionSpec's key set is closed and adding a member is a th-024 change. A flag
	// silently ignored is the failure mode `override_unresolved` exists to prevent, and
	// these are the same shape.
	if v := ctx.Inv.Value("--seed"); v != "" {
		return exit.Named(exit.Usage, "not_implemented",
			"--seed names the runtime's construction RNG and no wire field carries it").
			WithRemedy("if the endpoint declares a seed field, pass it as payload: `seed=%s`", v).
			WithNext("cozy describe <org/endpoint>/<function>")
	}
	if ctx.Inv.Bool("--offline") {
		return exit.Named(exit.Usage, "not_implemented",
			"--offline is the runtime's CAS-only acquisition mode and no wire field carries it").
			WithRemedy("this host serves what its local artifact index already holds; a miss refuses at start").
			WithNext("cozy-runtime run <fn> --offline (the standalone door)")
	}
	deadline, e := runDeadline(ctx)
	if e != nil {
		return e
	}

	// THE PAYLOAD IS TYPED AGAINST THE RECORDED SCHEMA — the surface the release's own
	// runtime vouched for at install — so a typo costs a millisecond instead of a model
	// load, and `steps=2` is an int because the schema says int.
	ep, e := entrypointOf(ctx, target)
	if e != nil {
		return e
	}
	input, e := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}

	c, e := dial(ctx)
	if e != nil {
		return e
	}
	key := ctx.Inv.Value("--idempotency-key")
	if key == "" {
		key = mintKey()
	}
	began := time.Now()
	handle, e := c.Submit(api.Submission{
		Endpoint: target.Endpoint, Function: target.Function, Input: input,
		Worker: strings.TrimSpace(ctx.Inv.Value("--worker")),
	}, key)
	if e != nil {
		return e
	}
	submitted := time.Since(began)

	stream := ctx.Inv.Bool("--stream")
	if !stream && !ctx.Mode().JSON {
		fmt.Fprintf(ctx.Out, "request %s · attempt %d · %s\n",
			handle.RequestID, handle.Attempt, handle.Status)
		if handle.Replay {
			fmt.Fprintln(ctx.Out, "note: this key was already recorded — the SAME request answered, nothing new started")
		}
	}

	terminal, stopped, e := watch(ctx, c, handle.RequestID, stream, deadline)
	if e != nil {
		return e
	}
	life, e := c.Request(handle.RequestID)
	if e != nil {
		return e
	}
	saved, e := saveOutputs(ctx, c, life)
	if e != nil {
		return e
	}
	return renderRun(ctx, life, terminal, stopped, saved, submitted, began)
}

// watch consumes the request's own event stream to its terminal, rendering progress as
// it goes. SIGINT does not kill this process: it CANCELS the request through the
// orchestrator and keeps watching, because the attempt's own journaled terminal is what
// settles it and a client that walked away would leave the card held.
// watch returns the terminal event and WHY the client stopped waiting, which is not the
// same question as what the terminal says: a canceled terminal caused by `--timeout` is
// exit 10, because a caller that set a deadline wants to know the deadline is what
// happened.
func watch(ctx *Context, c *localapi.Client, requestID string, stream bool,
	deadline time.Duration) (*localapi.Event, string, *exit.Error) {
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)

	stopped := ""
	cancel := func(why string) {
		fmt.Fprintf(ctx.Err, "\n%s — the attempt's own terminal still settles it\n", why)
		if e := c.Cancel(requestID); e != nil {
			fmt.Fprintf(ctx.Err, "cancel: %s\n", e.Message)
		}
	}
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case _, ok := <-interrupt:
			if !ok {
				return
			}
			stopped = "canceled"
			cancel("cancel requested")
		case <-deadlineC(deadline):
			// `--timeout` is a REQUEST DEADLINE the client enforces the only way a client
			// honestly can: by asking the orchestrator to cancel. It is not the
			// supervisor's watchdog deadline (that one is on the attempt, and this host
			// has no wire field for it) — walking away instead would leave the card held.
			stopped = "deadline"
			cancel(fmt.Sprintf("--timeout %s expired", deadline))
		case <-done:
		}
	}()

	lines := newProgress(ctx, stream)
	terminal, e := c.Watch(requestID, 0, lines.on)
	lines.done()
	return terminal, stopped, e
}

// deadlineC is a timer channel, or one that never fires when no deadline was set.
func deadlineC(d time.Duration) <-chan time.Time {
	if d <= 0 {
		return nil
	}
	return time.After(d)
}

// runDeadline reads `--timeout <dur>`.
func runDeadline(ctx *Context) (time.Duration, *exit.Error) {
	v := ctx.Inv.Value("--timeout")
	if v == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil || d <= 0 {
		return 0, exit.Usagef("--timeout %q is not a positive duration", v).
			WithRemedy("durations are Go-spelled: 30s, 5m, 1h30m")
	}
	return d, nil
}

// progress renders the live lane. Two shapes, and they are not the same surface:
// `--stream` is NDJSON of the typed envelope for a machine, and the default is one
// rewritten line for a person.
type runProgress struct {
	ctx    *Context
	stream bool
	last   string
	dirty  bool
}

func newProgress(ctx *Context, stream bool) *runProgress {
	return &runProgress{ctx: ctx, stream: stream}
}

func (p *runProgress) on(e localapi.Event) bool {
	if p.stream {
		data, err := json.Marshal(e)
		if err == nil {
			fmt.Fprintln(p.ctx.Out, string(data))
		}
		return true
	}
	if p.ctx.Mode().JSON {
		return true // one JSON document on stdout: the run's own, at the end
	}
	line := progressLine(e)
	if line == "" || line == p.last {
		return true
	}
	p.last, p.dirty = line, true
	// stderr, deliberately: stdout carries the RESULT, so a piped `cozy run` is not
	// polluted by the progress of producing it.
	fmt.Fprintf(p.ctx.Err, "\r\033[K%s", line)
	return true
}

func (p *runProgress) done() {
	if p.dirty {
		fmt.Fprintln(p.ctx.Err)
	}
}

// progressLine renders one event. The runtime's own frame vocabulary is carried through
// (`progress`, `stage`, `metric`) rather than translated into a second one.
func progressLine(e localapi.Event) string {
	kind := strings.TrimPrefix(e.Type, "request.")
	switch kind {
	case "progress":
		if v, ok := e.Payload["value"].(map[string]any); ok {
			return "  " + strings.TrimSpace(fmt.Sprintf("%s %s", kind, compactValue(v)))
		}
	case "stage", "metric":
		if v, ok := e.Payload["value"].(map[string]any); ok {
			return "  " + kind + " " + compactValue(v)
		}
	case "queued":
		if reason, ok := e.Payload["reason"].(string); ok {
			return "  queued — " + reason
		}
		return "  queued"
	case "dispatched", "accepted", "submitted", "requeued", "attempt_failed":
		return "  " + kind + " " + compactValue(e.Payload)
	}
	return ""
}

func compactValue(v map[string]any) string {
	keys := make([]string, 0, len(v))
	for k := range v {
		switch k {
		case "live", "seq":
			continue
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, v[k]))
	}
	return strings.Join(parts, " ")
}

// saveOutputs fetches every accepted output by its OPAQUE id and writes it under --out.
// The filename comes from the output's FIELD PATH plus the mime type the manifest
// declared — the CLI composes no path a server did not name.
func saveOutputs(ctx *Context, c *localapi.Client, life api.Lifecycle) ([]map[string]string, *exit.Error) {
	dir := ctx.Inv.Value("--out")
	if dir == "" || len(life.Outputs) == 0 {
		return nil, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, exit.Internalf("cannot create %s: %s", dir, err)
	}
	saved := []map[string]string{}
	for _, out := range life.Outputs {
		name := strings.ReplaceAll(out.OutputID, "/", "_") + extensionOf(out.MimeType)
		path := filepath.Join(dir, name)
		n, digest, e := c.Media(out.MediaID, func(body io.Reader) (int64, *exit.Error) {
			f, err := os.Create(path)
			if err != nil {
				return 0, exit.Internalf("cannot write %s: %s", path, err)
			}
			defer f.Close()
			n, err := io.Copy(f, body)
			if err != nil {
				return n, exit.Internalf("cannot write %s: %s", path, err)
			}
			return n, nil
		})
		if e != nil {
			return nil, e
		}
		row := map[string]string{
			"output": out.OutputID, "path": path, "bytes": render.Bytes(n),
			"media_id": out.MediaID, "mime": out.MimeType, "digest": out.Digest,
		}
		// The server declares a digest on the response; a mismatch would mean the bytes
		// changed between the manifest and the wire, which is worth saying out loud.
		if digest != "" && out.Digest != "" && digest != out.Digest {
			return nil, exit.Named(exit.Validation, "media_digest_mismatch",
				"output %s served %s where its manifest declared %s", out.OutputID, digest, out.Digest)
		}
		saved = append(saved, row)
	}
	return saved, nil
}

// extensionOf names a file from the type the OUTPUT MANIFEST declared. An unknown or
// opaque type gets NO invented extension: the mime is on the record, and a guessed suffix
// would be this client making a claim about bytes it never looked at.
//
// TODAY EVERY LOCAL OUTPUT IS OPAQUE, and that is a real defect one layer down rather than
// a gap here: `cozy_runtime.internal.worker.grants` writes
// `mime_type="application/octet-stream"` on every OutputEntry, although the asset it just
// encoded knows its own `media_type` (cr-016's own record says the file extension comes
// from it). So the terminal declares no type, `/v1/media` serves every output as an
// attachment, and `--out` writes `image` rather than `image.png`. Named in cl-010's record
// as a cr-016/cr-017 seam; the day the manifest carries the real type, this table names
// the file and nothing else changes.
func extensionOf(mime string) string {
	switch mime {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "video/mp4":
		return ".mp4"
	case "audio/wav":
		return ".wav"
	case "application/json":
		return ".json"
	}
	return ""
}

// opaqueType is the type an output carries when nobody declared one.
const opaqueType = "application/octet-stream"

// renderRun prints the run's answer and maps the terminal onto the SHARED matrix:
// succeeded 0 · failed 11 · canceled 12 · deadline 10, from `exit.JobTerminal`.
func renderRun(ctx *Context, life api.Lifecycle, terminal *localapi.Event, stopped string,
	saved []map[string]string, submitted time.Duration, began time.Time) *exit.Error {
	status := localapi.StreamStatus(terminal)
	if status == "" {
		status = life.Status
	}
	if stopped == "deadline" && status == "canceled" {
		// The DEADLINE is why this ended, and the shared matrix has a code for it.
		status = "deadline"
	}
	fields := []render.Field{
		{K: "request", V: life.RequestID},
		{K: "endpoint", V: life.Endpoint},
		{K: "function", V: life.Function},
		{K: "status", V: life.Status},
		{K: "attempts", V: life.Attempts},
	}
	if life.Result != nil {
		fields = append(fields, render.Field{K: "result", V: life.Result})
	}
	outs := make([]string, 0, len(life.Outputs))
	for _, o := range life.Outputs {
		outs = append(outs, o.OutputID+" "+o.MediaID+" "+render.Bytes(o.Length))
	}
	if len(outs) > 0 {
		fields = append(fields, render.Field{K: "outputs", V: outs})
	}
	notes := []string{}
	if len(saved) > 0 {
		paths := make([]string, 0, len(saved))
		opaque := false
		for _, s := range saved {
			paths = append(paths, s["path"]+" ("+s["bytes"]+")")
			opaque = opaque || s["mime"] == opaqueType || s["mime"] == ""
		}
		fields = append(fields, render.Field{K: "saved", V: paths})
		if opaque {
			// DEGRADE LOUDLY. The file is exactly the bytes the manifest declared and its
			// digest matched; what is missing is the TYPE, and the runtime is the only
			// thing that ever knew it.
			notes = append(notes,
				"an output declares no media type, so it is written without an extension — "+
					"the worker's manifest hard-codes application/octet-stream (cr-016 seam)")
		}
	}
	if life.Metrics != nil {
		fields = append(fields, render.Field{K: "metrics", V: life.Metrics})
	}
	if life.Triage != nil {
		fields = append(fields, render.Field{K: "attempt_key", V: life.Triage.AttemptKey})
	}
	fields = append(fields,
		render.Field{K: "submit_ms", V: submitted.Milliseconds()},
		render.Field{K: "wall_ms", V: time.Since(began).Milliseconds()})

	rec := render.Record{Kind: "run", Fields: fields, Notes: notes}
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
		if life.Triage != nil {
			rec.Next = []string{"cozy logs " + life.Triage.AttemptKey}
		}
		return emit(ctx, rec)
	}
	// A FAILING terminal is still an ANSWER: the whole record is printed before the
	// typed refusal, so a failure carries its metrics, its attempt key and its outputs.
	if err := rec.Emit(ctx.Out, ctx.Mode()); err != nil && !ctx.Mode().JSON {
		return exit.As(err)
	}
	e := exit.Named(code, status, "request %s ended %s", life.RequestID, status)
	errType, why := life.ErrorType, life.Error
	if why == "" && terminal != nil {
		// A request that failed BEFORE ANY ATTEMPT has no attempt row to carry a cause —
		// an unplaceable pin, a credential the orchestrator refused to read, a worker that
		// could not be started. Its reason exists on the terminal EVENT and nowhere else,
		// and dropping it left the client with "ended failed" and no way to learn why.
		errType, why = eventText(terminal, "error_type"), eventText(terminal, "error")
	}
	if why != "" {
		e.Message = fmt.Sprintf("request %s ended %s: %s — %s",
			life.RequestID, status, errType, why)
	}
	if life.Triage != nil {
		e.WithRemedy("the retained triage bundle explains it").
			WithNext("cozy logs " + life.Triage.AttemptKey)
	}
	return e
}

func mapTerminal(status string) string {
	switch status {
	case "completed", "succeeded":
		return "succeeded"
	case "canceled":
		return "canceled"
	case "failed":
		return "failed"
	}
	return status
}

func took(began time.Time) string {
	return fmt.Sprintf("%.2fs", time.Since(began).Seconds())
}

// mintKey mints this invocation's idempotency key. One key names one request forever, so
// a fresh invocation gets a fresh one and a caller that wants retry safety across process
// restarts passes its own.
func mintKey() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "idem-" + fmt.Sprint(time.Now().UnixNano())
	}
	return "idem-" + hex.EncodeToString(b[:])
}

// ------------------------------------------------------------------- target parsing

// Target is one invocation subject: `org/endpoint/vN/function`.
type Target struct {
	Endpoint string
	Major    int
	Function string
	Ref      string // the endpoint ref as the resolver takes it: `org/endpoint@vN`
}

// parseTarget reads `org/endpoint/vN/function`. The semver-major is a REQUIRED path
// segment (README §1): it is resolved through that (endpoint, major)'s serving pointer,
// which locally is the install pin, and a majorless target is a usage refusal rather
// than a default.
func parseTarget(raw string) (Target, *exit.Error) {
	parts := strings.Split(strings.TrimSpace(raw), "/")
	usage := exit.Usagef("%q is not org/endpoint/vN/function", raw).
		WithRemedy("the semver-major is a required path segment — it resolves through that endpoint's serving pointer").
		WithNext("cozy ls", "cozy help run")
	if len(parts) != 4 {
		return Target{}, usage
	}
	major, ok := majorOf(parts[2])
	if !ok || parts[0] == "" || parts[1] == "" || parts[3] == "" {
		return Target{}, usage
	}
	endpoint := parts[0] + "/" + parts[1]
	return Target{
		Endpoint: endpoint, Major: major, Function: parts[3],
		Ref: endpoint + "@" + parts[2],
	}, nil
}

func majorOf(segment string) (int, bool) {
	digits, ok := strings.CutPrefix(segment, "v")
	if !ok || digits == "" {
		return 0, false
	}
	n := 0
	for _, c := range digits {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

// entrypointOf reads one function's declared surface out of the install records. It is a
// LOCAL read of a fact the release's own runtime already proved, which is why it costs no
// subprocess and no round trip.
func entrypointOf(ctx *Context, t Target) (*launch.Entrypoint, *exit.Error) {
	facts, e := generationFacts(ctx, t.Endpoint, t.Major)
	if e != nil {
		return nil, e
	}
	return facts.Descriptor.Function(t.Function)
}

// generationFacts resolves an endpoint ref to its pinned generation's facts.
func generationFacts(ctx *Context, endpoint string, major int) (*launch.Facts, *exit.Error) {
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return nil, e
	}
	store, e := records.Open(l.DB)
	if e != nil {
		return nil, e
	}
	defer store.Close()
	pins, e := store.Pins(endpoint)
	if e != nil {
		return nil, e
	}
	if len(pins) == 0 {
		return nil, exit.New(exit.NotFound, "%s is not installed on this host", endpoint).
			WithRemedy("`cozy ls` lists what is").
			WithNext("cozy install "+endpoint, "cozy ls")
	}
	chosen, found := pins[0], major == 0
	for _, p := range pins {
		if p.Major == major {
			chosen, found = p, true
		}
	}
	if !found {
		return nil, exit.New(exit.NotFound, "%s is installed, but not at v%d", endpoint, major).
			WithRemedy("installed majors: %s", majorsOf(pins)).
			WithNext("cozy ls")
	}
	gen, e := store.Install(chosen.InstallID)
	if e != nil {
		return nil, e
	}
	if gen == nil {
		return nil, exit.Internalf("%s is pinned to generation %s and that row is gone",
			endpoint, chosen.InstallID)
	}
	return launch.Read(*gen, ctx.Cfg.Home, ctx.Cfg.Tool())
}

// eventText reads one string field out of an event's payload. It is how a pre-attempt
// failure's reason reaches the client: the event is where that reason lives.
func eventText(e *localapi.Event, key string) string {
	if e == nil {
		return ""
	}
	s, _ := e.Payload[key].(string)
	return s
}
