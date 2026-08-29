package cli

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/api"
	localapi "github.com/cozy-creator/cozy-creator/internal/client"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/output"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/rental"
)

// THE LIFECYCLE AND REQUEST VERBS (cl-010), every one of them a CLIENT of the local
// client API. There is no direct-Go path from a verb to the orchestrator: `dial` is the
// only way into any of them, and what it returns speaks HTTP to a separate process.
//
// ONE PRODUCT EXECUTION PATH. `cozy invoke run` is
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
	return localapi.Open(ctx.Cfg, ctx.Daemon)
}

// ----------------------------------------------------------------------------- run

func handleInvokeRun(ctx *Context) *exit.Error {
	job, problem := invocationIsJob(ctx)
	if problem != nil {
		return problem
	}
	if !job {
		if len(ctx.Inv.Values["--input"]) > 0 {
			return exit.Usagef("--input-tree applies only to a job callable")
		}
		if ctx.Inv.Value("--org") != "" {
			return exit.Usagef("--org applies only to a job callable")
		}
		return handleRun(ctx)
	}
	if ctx.Inv.Value("--worker") != "" {
		return exit.Named(exit.Usage, "rental_job_unsupported",
			"private rental dispatch for job callables is not implemented").
			WithRemedy("run this job locally, or choose a serving entrypoint on the rental")
	}
	if ctx.Inv.Bool("--stream") || len(ctx.Inv.Values["--asset"]) > 0 ||
		ctx.Inv.Value("--out") != "" || ctx.Inv.Value("--timeout") != "" {
		return exit.Usagef("the selected callable is a job and received a serving-only flag").
			WithRemedy("jobs accept payload values, --in, --input-tree, --org, --detach, and local execution")
	}
	if !ctx.Inv.Bool("--detach") {
		ctx.Inv.Bools["--follow"] = true
	}
	return handleJobSubmit(ctx)
}

func invocationIsJob(ctx *Context) (bool, *exit.Error) {
	target, problem := parseTarget(ctx.Inv.Args[0])
	if problem != nil {
		return false, problem
	}
	var descriptor *launch.PackageDescriptor
	if worker := strings.TrimSpace(ctx.Inv.Value("--worker")); worker != "" {
		layout, problem := home.Open(ctx.Cfg.Home)
		if problem != nil {
			return false, problem
		}
		store, problem := records.Open(layout.DB)
		if problem != nil {
			return false, problem
		}
		defer store.Close()
		descriptor, problem = rental.PackageDescriptor(store, worker)
		if problem != nil {
			return false, problem
		}
	} else {
		facts, problem := generationFacts(ctx, target.Package, target.Major)
		if problem != nil {
			return false, problem
		}
		descriptor = facts.PackageDescriptor
	}
	for _, job := range descriptor.Jobs {
		if job.Name == target.Function {
			return true, nil
		}
	}
	if _, problem := descriptor.Function(target.Function); problem != nil {
		return false, problem
	}
	return false, nil
}

func handleRun(ctx *Context) *exit.Error {
	target, e := parseTarget(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	deadline, e := runDeadline(ctx)
	if e != nil {
		return e
	}

	// THE PAYLOAD IS TYPED AGAINST THE RECORDED SCHEMA — the surface the release's own
	// runtime vouched for at install — so a typo costs a millisecond instead of a model
	// load, and `steps=2` is an int because the schema says int.
	worker := strings.TrimSpace(ctx.Inv.Value("--worker"))
	var ep *launch.Entrypoint
	if worker != "" {
		ep, e = remoteEntrypointOf(ctx, worker, target.Function)
	} else {
		ep, e = entrypointOf(ctx, target)
	}
	if e != nil {
		return e
	}
	if legacy := launch.LegacyFileTerm(ctx.Inv.Args[1:]); worker != "" && legacy != "" {
		return exit.Named(exit.Usage, "remote_file_input_ambiguous",
			"%s embeds file bytes into a JSON string and cannot name a remote input grant", legacy).
			WithRemedy("use `--asset <field-path>=<file>`; the field path becomes the exact worker-protocol input id")
	}
	input, e := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	input, assets, e := launch.ParseAssets(ep, input, ctx.Inv.Values["--asset"])
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
		Package: target.Package, Function: target.Function, Input: input,
		Worker: worker, LocalAssets: assets,
	}, key)
	if e != nil {
		return e
	}
	submitted := time.Since(began)
	if ctx.Inv.Bool("--detach") {
		fields := []output.Field{
			{K: "id", V: handle.RequestID}, {K: "kind", V: "invocation"},
			{K: "target", V: target.Package + "/" + target.Function},
			{K: "status", V: handle.Status}, {K: "attempt", V: handle.Attempt},
			{K: "changed", V: !handle.Replay},
		}
		rec := compactRecord(fields, "id", "target", "status", "changed")
		rec.Next = []string{"cozy invoke cancel " + handle.RequestID}
		return emit(ctx, rec)
	}

	stream := ctx.Inv.Bool("--stream")
	if !stream {
		fmt.Fprintf(ctx.Err, "request %s · attempt %d · %s\n",
			handle.RequestID, handle.Attempt, handle.Status)
		if handle.Replay {
			fmt.Fprintln(ctx.Err, "note: this key returned the existing invocation")
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

func handleInvokeCancel(ctx *Context) *exit.Error {
	id := ctx.Inv.Args[0]
	if strings.HasPrefix(id, "job-") {
		return handleJobCancel(ctx)
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	before, problem := client.Request(id)
	if problem != nil {
		return problem
	}
	if invocationSettled(before.Status) {
		fields := append(invocationFields(before), output.Field{K: "changed", V: false})
		return emit(ctx, compactRecord(fields, "id", "target", "status", "changed"))
	}
	if problem := client.Cancel(id); problem != nil {
		return problem
	}
	if _, problem := client.Watch(id, 0, func(localapi.Event) bool { return true }); problem != nil {
		return problem
	}
	after, problem := client.Request(id)
	if problem != nil {
		return problem
	}
	fields := append(invocationFields(after), output.Field{K: "changed", V: true})
	return emit(ctx, compactRecord(fields, "id", "target", "status", "changed"))
}

func handleInvokeList(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	limit := 50
	if raw := ctx.Inv.Value("--limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 || parsed > 500 {
			return exit.Usagef("--limit %q is not between 1 and 500", raw)
		}
		limit = parsed
	}
	rows, problem := client.Requests(ctx.Inv.Value("--state"), limit)
	if problem != nil {
		return problem
	}
	pkg := strings.TrimSpace(ctx.Inv.Value("--package"))
	list := output.List{
		Name: "invocations", Fields: []string{"id", "kind", "target", "status"},
		AllFields: []string{"id", "kind", "target", "status", "attempts", "created"},
	}
	states := map[string]int{}
	for _, life := range rows {
		if pkg != "" && life.Package != pkg {
			continue
		}
		kind := life.Kind
		if kind == "" {
			kind = "invocation"
		}
		list.Rows = append(list.Rows, map[string]string{
			"id": life.RequestID, "kind": kind,
			"target": life.Package + "/" + life.Function, "status": life.Status,
			"attempts": strconv.Itoa(life.Attempts), "created": life.CreatedAt,
		})
		states[life.Status]++
	}
	keys := make([]string, 0, len(states))
	for state := range states {
		keys = append(keys, state)
	}
	sort.Strings(keys)
	for _, state := range keys {
		list.Aggregates = append(list.Aggregates, output.Field{K: state, V: states[state]})
	}
	return emit(ctx, list)
}

func invocationFields(life api.Lifecycle) []output.Field {
	kind := life.Kind
	if kind == "" {
		kind = "invocation"
	}
	return []output.Field{
		{K: "id", V: life.RequestID}, {K: "kind", V: kind},
		{K: "target", V: life.Package + "/" + life.Function},
		{K: "status", V: life.Status}, {K: "attempts", V: life.Attempts},
	}
}

func invocationSettled(status string) bool {
	switch status {
	case "completed", "failed", "canceled":
		return true
	default:
		return false
	}
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
			fmt.Fprintln(p.ctx.Err, string(data))
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
	// stderr, deliberately: stdout carries the RESULT, so a piped `cozy invoke run` is not
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
	return saveOutputsAt(c, life, dir)
}

// saveOutputsAt is the one verified local-download path for ordinary requests and
// accepted result downloads: every output is received through the opaque media API, hashed
// independently, checked against the request manifest, and published as one set.
func saveOutputsAt(c *localapi.Client, life api.Lifecycle, dir string) ([]map[string]string, *exit.Error) {
	if dir == "" || len(life.Outputs) == 0 {
		return nil, nil
	}
	names, e := outputNames(life.Outputs)
	if e != nil {
		return nil, e
	}
	return publishOutputSet(dir, life.Outputs, names, c.Media)
}

// outputNames names each saved file by the output's own FIELD-PATH id plus the extension
// its declared MIME type earns — `image` -> `image.png` — so a two-output result is
// addressable by name and a caller never has to know the manifest's order (decisions #248
// and #375). The id is the same single path element the grant was fenced to; it is fenced
// again here, because a client writing into the caller's own directory verifies rather
// than trusts. An untyped output keeps its bare field path and no extension.
func outputNames(outputs []api.MediaRef) ([]string, *exit.Error) {
	names := make([]string, len(outputs))
	seenIDs, seenNames := map[string]bool{}, map[string]bool{}
	for index, out := range outputs {
		if out.OutputID == "" || seenIDs[out.OutputID] {
			return nil, exit.Named(exit.Validation, "output_name_collision",
				"the output manifest repeats or omits output id %q", out.OutputID)
		}
		seenIDs[out.OutputID] = true
		if e := orchestrator.FenceOutputID(out.OutputID); e != nil {
			return nil, e
		}
		names[index] = out.OutputID + extensionOf(out.MimeType)
		if seenNames[names[index]] {
			return nil, exit.Named(exit.Validation, "output_name_collision",
				"two output manifest rows resolve to %s", names[index])
		}
		seenNames[names[index]] = true
	}
	return names, nil
}

type mediaFetch func(mediaID string, receive func(localapi.MediaResponse) *exit.Error) *exit.Error

// publishOutputSet stages every output into a temporary sibling of dir, verifies each
// against the MANIFEST (length and digest; the response headers must agree but prove
// nothing), makes the set durable, then publishes it. A dir that does not yet exist
// appears in one rename; an existing dir receives the already-verified files. Any
// failure removes the staging dir and publishes nothing.
func publishOutputSet(dir string, outputs []api.MediaRef, names []string, fetch mediaFetch) ([]map[string]string, *exit.Error) {
	absolute, err := filepath.Abs(dir)
	if err != nil {
		return nil, exit.Internalf("cannot resolve %s: %s", dir, err)
	}
	parent := filepath.Dir(absolute)
	if info, err := os.Lstat(absolute); err == nil && info.IsDir() {
		parent = absolute
	} else if err := os.MkdirAll(parent, 0o755); err != nil {
		return nil, exit.Internalf("cannot create %s: %s", parent, err)
	}
	staging, err := os.MkdirTemp(parent, "."+filepath.Base(absolute)+".staging-*")
	if err != nil {
		return nil, exit.Internalf("cannot stage outputs for %s: %s", dir, err)
	}
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(staging)
		}
	}()
	saved := make([]map[string]string, 0, len(outputs))
	for index, out := range outputs {
		staged := filepath.Join(staging, names[index])
		var n int64
		var digest string
		e := fetch(out.MediaID, func(res localapi.MediaResponse) *exit.Error {
			if res.ContentLength >= 0 && res.ContentLength != out.Length {
				return exit.Named(exit.Validation, "media_length_mismatch",
					"output %s declared Content-Length %d where its manifest declared %d B",
					out.OutputID, res.ContentLength, out.Length)
			}
			if res.Digest != "" && res.Digest != out.Digest {
				return exit.Named(exit.Validation, "media_digest_mismatch",
					"output %s served header %s where its manifest declared %s",
					out.OutputID, res.Digest, out.Digest)
			}
			var e *exit.Error
			n, digest, e = receiveVerified(staged, out.Length, out.Digest, res.Body)
			return e
		})
		if e != nil {
			return nil, e
		}
		saved = append(saved, map[string]string{
			"output": out.OutputID, "path": filepath.Join(dir, names[index]), "bytes": output.Bytes(n),
			"media_id": out.MediaID, "mime": out.MimeType, "digest": digest,
		})
	}
	if e := syncDirectory(staging); e != nil {
		return nil, e
	}
	if _, err := os.Lstat(absolute); os.IsNotExist(err) {
		if err := os.Rename(staging, absolute); err != nil {
			return nil, exit.Internalf("cannot publish verified outputs to %s: %s", dir, err)
		}
		published = true
		return saved, syncDirectory(parent)
	}
	for _, name := range names {
		if err := os.Rename(filepath.Join(staging, name), filepath.Join(absolute, name)); err != nil {
			return nil, exit.Internalf("cannot publish verified output %s: %s", name, err)
		}
	}
	if e := syncDirectory(absolute); e != nil {
		return nil, e
	}
	published = true
	_ = os.Remove(staging)
	return saved, nil
}

// receiveVerified writes at most expectedLength+1 bytes to path, fsyncs, and checks the
// exact length and independent sha256 against the manifest.
func receiveVerified(path string, expectedLength int64, expectedDigest string,
	body io.Reader) (int64, string, *exit.Error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return 0, "", exit.Internalf("cannot stage %s: %s", filepath.Base(path), err)
	}
	hash := sha256.New()
	length, err := io.Copy(io.MultiWriter(f, hash), io.LimitReader(body, expectedLength+1))
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return length, "", exit.Internalf("cannot stage %s: %s", filepath.Base(path), err)
	}
	digest := "sha256:" + hex.EncodeToString(hash.Sum(nil))
	if length != expectedLength {
		return length, digest, exit.Named(exit.Validation, "media_length_mismatch",
			"output %s received %d B where its manifest declared %d B",
			filepath.Base(path), length, expectedLength)
	}
	if digest != expectedDigest {
		return length, digest, exit.Named(exit.Validation, "media_digest_mismatch",
			"output %s received %s where its manifest declared %s",
			filepath.Base(path), digest, expectedDigest)
	}
	return length, digest, nil
}

// syncTree fsyncs every directory under root, children before parents.
func syncDirectory(path string) *exit.Error {
	if runtime.GOOS == "windows" {
		return nil
	}
	dir, err := os.Open(path)
	if err != nil {
		return exit.Internalf("cannot open directory for sync: %s", err)
	}
	err = dir.Sync()
	_ = dir.Close()
	if err != nil {
		return exit.Internalf("cannot make directory %s durable: %s", path, err)
	}
	return nil
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
	fields := []output.Field{
		{K: "id", V: life.RequestID},
		{K: "target", V: life.Package + "/" + life.Function},
		{K: "package", V: life.Package},
		{K: "function", V: life.Function},
		{K: "status", V: life.Status},
		{K: "attempts", V: life.Attempts},
	}
	if life.Result != nil {
		fields = append(fields, output.Field{K: "result", V: life.Result})
	}
	outs := make([]string, 0, len(life.Outputs))
	for _, o := range life.Outputs {
		outs = append(outs, o.OutputID+" "+o.MediaID+" "+output.Bytes(o.Length))
	}
	if len(outs) > 0 {
		fields = append(fields, output.Field{K: "outputs", V: outs})
	}
	notes := []string{}
	if len(saved) > 0 {
		paths := make([]string, 0, len(saved))
		opaque := false
		for _, s := range saved {
			paths = append(paths, s["path"]+" ("+s["bytes"]+")")
			opaque = opaque || s["mime"] == opaqueType || s["mime"] == ""
		}
		fields = append(fields, output.Field{K: "saved", V: paths})
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
		fields = append(fields, output.Field{K: "metrics", V: life.Metrics})
	}
	if life.Triage != nil {
		fields = append(fields, output.Field{K: "attempt_key", V: life.Triage.AttemptKey})
	}
	fields = append(fields,
		output.Field{K: "submit_ms", V: submitted.Milliseconds()},
		output.Field{K: "wall_ms", V: time.Since(began).Milliseconds()})

	defaults := []string{"id", "target", "status"}
	if life.Result != nil {
		defaults = append(defaults, "result")
	}
	if len(saved) > 0 {
		defaults = append(defaults, "saved")
	}
	defaults = append(defaults, "wall_ms")
	rec := compactRecord(fields, defaults...)
	rec.Notes = notes
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
		if life.Triage != nil {
			rec.Next = []string{"cozy invoke list --full"}
		}
		return emit(ctx, rec)
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
			WithNext("cozy invoke list --full")
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

// Target is one invocation subject: `org/package/vN/function`.
type Target struct {
	Package  string
	Major    int
	Function string
	Ref      string // the package ref as the resolver takes it: `org/package@vN`
}

// parseTarget reads `org/package/vN/function`. The semver-major is a REQUIRED path
// segment: it is resolved through that (package, major)'s serving pointer,
// which locally is the install pin, and a majorless target is a usage refusal rather
// than a default.
func parseTarget(raw string) (Target, *exit.Error) {
	parts := strings.Split(strings.TrimSpace(raw), "/")
	usage := exit.Usagef("%q is not org/package/vN/function", raw).
		WithRemedy("the semver-major is a required path segment — it resolves through that package's serving pointer").
		WithNext("cozy package list", "cozy help run")
	if len(parts) != 4 {
		return Target{}, usage
	}
	major, ok := majorOf(parts[2])
	if !ok || parts[0] == "" || parts[1] == "" || parts[3] == "" {
		return Target{}, usage
	}
	pkg := parts[0] + "/" + parts[1]
	return Target{
		Package: pkg, Major: major, Function: parts[3],
		Ref: pkg + "@" + parts[2],
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
	facts, e := generationFacts(ctx, t.Package, t.Major)
	if e != nil {
		return nil, e
	}
	return facts.PackageDescriptor.Function(t.Function)
}

func remoteEntrypointOf(ctx *Context, worker, function string) (*launch.Entrypoint, *exit.Error) {
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return nil, e
	}
	store, e := records.Open(l.DB)
	if e != nil {
		return nil, e
	}
	defer store.Close()
	descriptor, e := rental.PackageDescriptor(store, worker)
	if e != nil {
		return nil, e
	}
	return descriptor.Function(function)
}

// generationFacts resolves a package ref to its pinned generation's facts.
func generationFacts(ctx *Context, pkg string, major int) (*launch.Facts, *exit.Error) {
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return nil, e
	}
	store, e := records.Open(l.DB)
	if e != nil {
		return nil, e
	}
	defer store.Close()
	pins, e := store.Pins(pkg)
	if e != nil {
		return nil, e
	}
	if len(pins) == 0 {
		return nil, exit.New(exit.NotFound, "%s is not installed on this host", pkg).
			WithRemedy("`cozy package list` lists what is").
			WithNext("cozy package install "+pkg, "cozy package list")
	}
	chosen, found := pins[0], major == 0
	for _, p := range pins {
		if p.Major == major {
			chosen, found = p, true
		}
	}
	if !found {
		return nil, exit.New(exit.NotFound, "%s is installed, but not at v%d", pkg, major).
			WithRemedy("installed majors: %s", majorsOf(pins)).
			WithNext("cozy package list")
	}
	gen, e := store.Install(chosen.InstallID)
	if e != nil {
		return nil, e
	}
	if gen == nil {
		return nil, exit.Internalf("%s is pinned to generation %s and that row is gone",
			pkg, chosen.InstallID)
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
