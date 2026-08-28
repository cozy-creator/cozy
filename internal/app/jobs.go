package app

import (
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/api"
	localapi "github.com/cozy-creator/cozy-creator/internal/client"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/render"
)

// THE JOB VERBS (cl-004): `submit` · `status` · `ls` · `follow` · `cancel`. They are
// SIBLINGS of the request verbs, not a second surface — `internal/client` is the one way
// each of them reaches the service, the durable event stream they follow is the same one
// `cozy run` follows, and the terminal mapping is the same shared matrix
// (0 · 11 · 12 · 10, through `exit.JobTerminal`).
//
// What the job verbs say that `run` does not:
//
//   - the PUBLICATION. A job's landed writes are a durable publication root, and every
//     verb that reports on a job reports it — repo, root, entries, bytes, and the local
//     CAS root holding whatever checkpoints it declared.
//   - the QUEUE. Many jobs may be submitted at once against one worker; `status` and `ls`
//     show the position each one is waiting at, and they drain in submission order
//     (owner directive, decisions #394 / cr-019).
//   - the BILL, and only where a RATE EXISTS. A rateless local job renders no cost line
//     at all. `$0.00` would be a claim about money nobody measured.

// ---------------------------------------------------------------------- job submit

func handleJobSubmit(ctx *Context) *exit.Error {
	if ctx.Inv.Bool("--cloud") {
		return exit.Named(exit.Usage, "not_implemented",
			"--cloud names a host this build cannot submit jobs to").
			WithRemedy("the hub's job plane lands with th-008; cl-004 is the LOCAL job surface").
			WithNext("cozy job submit " + ctx.Inv.Args[0] + " --local")
	}
	target, e := parseTarget(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	if v := ctx.Inv.Value("--model"); v != "" {
		return exit.Named(exit.Usage, "override_unresolved",
			"--model is admissible on this host and nothing resolves one yet").
			WithRemedy("the local binding resolver lands with cl-005; a job's typed model input " +
				"arrives as a materialized tree today (`--input <ref>=<dir>`)")
	}
	job, e := jobFactsOf(ctx, target)
	if e != nil {
		return e
	}
	// THE PAYLOAD IS TYPED AGAINST THE RECORDED SCHEMA before a job exists — the same
	// client-side check `cozy run` makes, over the job's own declared request struct.
	input, e := launch.ParsePayload(&launch.Entrypoint{
		Name: job.Name, Request: job.Request, Result: job.Result,
	}, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	trees, e := parseTrees(ctx.Inv.Values["--input"])
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
	handle, e := c.SubmitJob(api.JobSubmission{
		Endpoint: target.Endpoint, Function: target.Function, Input: input,
		Org: ctx.Inv.Value("--org"), Trees: trees,
	}, key)
	if e != nil {
		return e
	}
	if ctx.Inv.Bool("--follow") {
		return followJob(ctx, c, handle.JobID, began)
	}
	fields := []render.Field{
		{K: "job", V: handle.JobID},
		{K: "endpoint", V: handle.Endpoint},
		{K: "function", V: handle.Function},
		{K: "status", V: handle.Status},
		{K: "publication", V: handle.Repo},
	}
	notes := []string{}
	if handle.Replay {
		notes = append(notes,
			"this key was already recorded — the SAME job answered, nothing new started")
	}
	return emit(ctx, render.Record{Kind: "job", Fields: fields, Notes: notes,
		Next: []string{"cozy job follow " + handle.JobID, "cozy job status " + handle.JobID}})
}

// parseTrees reads `--input <ref>=<dir>`, the typed input TREES a job's fields hydrate
// from. The ref is the job's own request-field value; the directory is a read capability
// the orchestrator grants. A field naming a ref that is not here never hydrates.
func parseTrees(values []string) ([]string, *exit.Error) {
	out := []string{}
	for _, v := range values {
		ref, dir, ok := strings.Cut(v, "=")
		if !ok || ref == "" || dir == "" {
			return nil, exit.Usagef("--input %q is not <ref>=<directory>", v).
				WithRemedy("a job's typed model/dataset input arrives as a materialized tree").
				WithNext("cozy job submit <target> --input cozy/sdxl@lane=/path/to/store")
		}
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			return nil, exit.New(exit.NotFound, "--input %s: %s is not a directory", ref, dir).
				WithRemedy("an input tree is a MATERIALIZED directory the orchestrator grants a read of")
		}
		out = append(out, v)
	}
	return out, nil
}

// jobFactsOf reads one job's declared surface out of the install records — the same
// LOCAL read `cozy run` makes of an entrypoint's, and for the same reason.
func jobFactsOf(ctx *Context, t Target) (*jobSurface, *exit.Error) {
	facts, e := generationFacts(ctx, t.Endpoint, t.Major)
	if e != nil {
		return nil, e
	}
	for i := range facts.Descriptor.Jobs {
		job := &facts.Descriptor.Jobs[i]
		if job.Name != t.Function {
			continue
		}
		return &jobSurface{Name: job.Name, Request: job.Request, Result: job.Result}, nil
	}
	names := []string{}
	for _, job := range facts.Descriptor.Jobs {
		names = append(names, job.Name)
	}
	known := strings.Join(names, ", ")
	if known == "" {
		known = "no jobs"
	}
	// A NAME THAT IS AN ENTRYPOINT is worth saying out loud: it is the commonest way to
	// reach here, and "no such job" would send a reader looking for a typo.
	if _, e := facts.Descriptor.Function(t.Function); e == nil {
		return nil, exit.Named(exit.Usage, "not_a_job",
			"%s registers %q as an ENTRYPOINT, not a @job", t.Endpoint, t.Function).
			WithRemedy("an entrypoint is invoked with `cozy run`; a job is run to completion").
			WithNext("cozy run " + ctx.Inv.Args[0])
	}
	return nil, exit.Named(exit.NotFound, "unknown_job",
		"%s registers no job named %q", t.Endpoint, t.Function).
		WithRemedy("it registers: %s", known).
		WithNext("cozy describe " + t.Endpoint)
}

type jobSurface struct {
	Name    string
	Request launch.Struct
	Result  launch.Struct
}

// ---------------------------------------------------------------------- job status

func handleJobStatus(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	state, e := c.Job(ctx.Inv.Args[0])
	if e != nil {
		return e
	}
	// EXIT 0: the READ succeeded. The job's own outcome is DATA here, not this command's
	// verdict (cozy-creator.md) — a failed job read successfully is exit 0 with `failed`
	// in the document, and `cozy job follow` is the verb whose exit code is the terminal.
	return emit(ctx, render.Record{Kind: "job", Fields: jobFields(state, true),
		Notes: jobNotes(state),
		Next:  jobNext(state)})
}

// jobFields renders one job. The BILL LINE IS ABSENT unless a rate exists — the field is
// not emitted at all, rather than emitted as zero.
func jobFields(state api.JobState, full bool) []render.Field {
	fields := []render.Field{
		{K: "job", V: state.JobID},
		{K: "endpoint", V: state.Endpoint},
		{K: "function", V: state.Function},
		{K: "status", V: state.Status},
		{K: "elapsed", V: fmt.Sprintf("%.1fs", float64(state.ElapsedMS)/1000)},
	}
	if state.QueuePosition != nil {
		fields = append(fields, render.Field{K: "queue_position", V: *state.QueuePosition})
	}
	if state.Stage != "" {
		fields = append(fields, render.Field{K: "stage", V: state.Stage})
	}
	if state.Progress != nil {
		fields = append(fields, render.Field{K: "progress", V: compactValue(state.Progress)})
	}
	fields = append(fields, render.Field{
		K: "retries", V: fmt.Sprintf("%d/%d", state.Requeues, state.RetryBudget)})
	if state.Bill != nil {
		fields = append(fields, render.Field{K: "bill", V: microUSD(state.Bill.MicroUSD)})
	}
	if !full {
		return fields
	}
	if p := state.Publication; p != nil {
		fields = append(fields,
			render.Field{K: "publication", V: p.Repo},
			render.Field{K: "publication_root", V: p.Root},
			render.Field{K: "published", V: fmt.Sprintf("%d entr(y|ies), %s, verdict %s",
				p.Entries, render.Bytes(p.Bytes), strings.ToLower(p.Status))})
	}
	if len(state.Checkpoints) > 0 {
		rows := make([]string, 0, len(state.Checkpoints))
		for _, c := range state.Checkpoints {
			rows = append(rows, fmt.Sprintf("#%d %s/%s %s", c.Attempt, c.OperationKey,
				c.LogicalKey, c.Outcome))
		}
		fields = append(fields, render.Field{K: "checkpoints_declared", V: rows})
	}
	if len(state.Outputs) > 0 {
		outs := make([]string, 0, len(state.Outputs))
		for _, o := range state.Outputs {
			outs = append(outs, o.OutputID+" "+o.MediaID+" "+render.Bytes(o.Length))
		}
		fields = append(fields, render.Field{K: "outputs", V: outs})
	}
	if len(state.Artifacts) > 0 {
		artifacts := make([]string, 0, len(state.Artifacts))
		for _, artifact := range state.Artifacts {
			state := strings.ToLower(artifact.Disposition)
			if artifact.Outcome != "" {
				state = strings.ToLower(artifact.Outcome)
			}
			artifacts = append(artifacts, fmt.Sprintf("#%d %s %s receipt=%s root=%s",
				artifact.Attempt, artifact.OutputSlot, state,
				artifact.ReceiptDigest, artifact.ScratchRootID))
		}
		fields = append(fields, render.Field{K: "artifacts", V: artifacts})
	}
	if state.Result != nil {
		fields = append(fields, render.Field{K: "result", V: state.Result})
	}
	if state.Metrics != nil {
		fields = append(fields, render.Field{K: "metrics", V: state.Metrics})
	}
	if state.ErrorType != "" {
		fields = append(fields,
			render.Field{K: "error_type", V: state.ErrorType},
			render.Field{K: "error", V: state.Error})
	}
	return fields
}

func jobNotes(state api.JobState) []string {
	notes := []string{}
	if state.Bill == nil {
		// ABSENT IS ABSENT, and it says WHY — a reader who expected a number needs to
		// know nothing was measured rather than that the job was free.
		notes = append(notes,
			"no cost: this host has no configured local rate, so there is no bill to render "+
				"(COZY_LOCAL_RATE_MICRO_USD_PER_HOUR)")
	}
	return notes
}

func jobNext(state api.JobState) []string {
	switch state.Status {
	case "queued", "in_progress":
		return []string{"cozy job follow " + state.JobID, "cozy job cancel " + state.JobID}
	}
	if state.Triage != nil {
		return []string{"cozy logs " + state.Triage.AttemptKey}
	}
	return nil
}

func microUSD(n int64) string {
	return fmt.Sprintf("$%d.%06d", n/1_000_000, n%1_000_000)
}

// ---------------------------------------------------------------------- job ls

func handleJobLs(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	jobs, counts, e := c.Jobs(ctx.Inv.Value("--state"), ctx.Inv.Value("--endpoint"), 200)
	if e != nil {
		return e
	}
	rows := make([]map[string]string, 0, len(jobs))
	for _, job := range jobs {
		row := map[string]string{
			"job": job.JobID, "target": job.Endpoint + "/" + job.Function,
			"state": job.Status, "age": fmt.Sprintf("%.1fs", float64(job.ElapsedMS)/1000),
			"publication": "-", "bill": "-",
		}
		if job.Publication != nil {
			row["publication"] = job.Publication.Repo
		}
		if job.Bill != nil {
			row["bill"] = microUSD(job.Bill.MicroUSD)
		}
		if job.QueuePosition != nil {
			row["state"] = fmt.Sprintf("queued(%d)", *job.QueuePosition)
		}
		rows = append(rows, row)
	}
	aggregates := make([]render.Field, 0, len(counts))
	states := make([]string, 0, len(counts))
	for state := range counts {
		states = append(states, state)
	}
	sort.Strings(states)
	for _, state := range states {
		aggregates = append(aggregates, render.Field{K: state, V: counts[state]})
	}
	return emit(ctx, render.List{Kind: "job",
		Fields:     []string{"job", "target", "state", "age"},
		AllFields:  []string{"job", "target", "state", "age", "publication", "bill"},
		Rows:       rows,
		Empty:      "0 jobs",
		Aggregates: aggregates,
	})
}

// ---------------------------------------------------------------------- job follow

func handleJobFollow(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	return followJob(ctx, c, ctx.Inv.Args[0], time.Now())
}

// followJob attaches to the durable lifecycle stream and exits with the TERMINAL mapping.
//
// A job that finished while detached still yields its terminal: the durable lane is
// replayed from cursor 0, so the terminal event is delivered even though it was appended
// before this process existed. That is the whole reason `follow` reads the durable stream
// rather than watching for a live frame.
func followJob(ctx *Context, c *localapi.Client, jobID string, began time.Time) *exit.Error {
	stream := ctx.Inv.Bool("--json")
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case _, ok := <-interrupt:
			if !ok {
				return
			}
			// SIGINT CANCELS, it does not abandon: the job's own journaled terminal is
			// what settles it, and walking away would leave the worker running.
			fmt.Fprintln(ctx.Err, "\ncancel requested — the job's own terminal still settles it")
			if e := c.CancelJob(jobID); e != nil {
				fmt.Fprintf(ctx.Err, "cancel: %s\n", e.Message)
			}
		case <-done:
		}
	}()

	lines := newJobProgress(ctx, stream)
	terminal, e := c.Watch(jobID, 0, lines.on)
	lines.done()
	if e != nil {
		return e
	}
	state, e := c.Job(jobID)
	if e != nil {
		return e
	}
	status := localapi.StreamStatus(terminal)
	if status == "" {
		status = state.Status
	}
	// A REQUEST THAT ENDED BEFORE ANY ATTEMPT has its reason only in the terminal EVENT:
	// there is no attempt row, so the state document has no terminal to read a cause off.
	// Without this the client printed `failed` and nothing else — which is exactly the
	// case cl-004's publication-escape arm produces, and the reason it exists.
	if state.ErrorType == "" && terminal != nil {
		state.ErrorType, _ = terminal.Payload["error_type"].(string)
		state.Error, _ = terminal.Payload["error"].(string)
	}
	fields := append(jobFields(state, true),
		render.Field{K: "wall_ms", V: time.Since(began).Milliseconds()})
	rec := render.Record{Kind: "job", Fields: fields, Notes: jobNotes(state)}
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
		if state.Publication != nil {
			rec.Next = []string{"cozy job status " + jobID}
		}
		return emit(ctx, rec)
	}
	// A FAILING terminal is still an ANSWER, and for a job it is an answer that may carry
	// a publication: a failed run's landed writes still land, with the verdict stamped as
	// metadata (jobs.md). The whole record prints before the typed refusal.
	if err := rec.Emit(ctx.Out, ctx.Mode()); err != nil && !ctx.Mode().JSON {
		return exit.As(err)
	}
	err := exit.Named(code, status, "job %s ended %s", jobID, status)
	if state.Error != "" {
		err.Message = fmt.Sprintf("job %s ended %s: %s — %s",
			jobID, status, state.ErrorType, state.Error)
	}
	if state.Requeues > 0 {
		err.WithRemedy("the orchestrator's retry projection spent %d of a %d-attempt budget "+
			"on neutral outcomes before settling", state.Requeues, state.RetryBudget)
	}
	if state.Triage != nil {
		err.WithNext("cozy logs " + state.Triage.AttemptKey)
	}
	return err
}

// jobProgress renders the stream. `--json` is NDJSON of the typed envelope for a machine;
// the default is one rewritten line for a person, on stderr so a piped `job follow` is not
// polluted by the progress of producing its answer.
type jobProgress struct {
	ctx    *Context
	stream bool
	last   string
	dirty  bool
}

func newJobProgress(ctx *Context, stream bool) *jobProgress {
	return &jobProgress{ctx: ctx, stream: stream}
}

func (p *jobProgress) on(e localapi.Event) bool {
	if p.stream {
		if data, err := json.Marshal(e); err == nil {
			fmt.Fprintln(p.ctx.Out, string(data))
		}
		return true
	}
	line := progressLine(e)
	if line == "" || line == p.last {
		return true
	}
	p.last, p.dirty = line, true
	fmt.Fprintf(p.ctx.Err, "\r\033[K%s", line)
	return true
}

func (p *jobProgress) done() {
	if p.dirty {
		fmt.Fprintln(p.ctx.Err)
	}
}

// ---------------------------------------------------------------------- job cancel

func handleJobCancel(ctx *Context) *exit.Error {
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	jobID := ctx.Inv.Args[0]
	state, e := c.Job(jobID)
	if e != nil {
		return e
	}
	// ALREADY TERMINAL = IDEMPOTENT 0 printing the terminal. A cancel that arrives after
	// the terminal is late, not wrong.
	if settled(state.Status) {
		return emit(ctx, render.Record{Kind: "job", Fields: jobFields(state, false),
			Notes: []string{"already terminal: `cozy job cancel` is idempotent"}})
	}
	if e := c.CancelJob(jobID); e != nil {
		return e
	}
	// BLOCK UNTIL THE CANCELED TERMINAL. The request was made; the attempt's own
	// journaled terminal is what settles it, so this watches the durable stream to it.
	if _, e := c.Watch(jobID, 0, func(localapi.Event) bool { return true }); e != nil {
		return e
	}
	final, e := c.Job(jobID)
	if e != nil {
		return e
	}
	return emit(ctx, render.Record{Kind: "job", Fields: jobFields(final, false),
		Notes: []string{"the terminal the worker journaled is what settled it, not this request"}})
}

func settled(status string) bool {
	switch status {
	case "completed", "failed", "canceled":
		return true
	}
	return false
}
