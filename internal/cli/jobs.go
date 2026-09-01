package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
)

// THE JOB VERBS (cl-004): `submit` · `status` · `ls` · `follow` · `cancel`. They are
// SIBLINGS of the request verbs, not a second surface — `internal/client` is the one way
// each of them reaches the daemon, the durable event stream they follow is the same one
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

func handleJobSubmit(ctx *Context, target Target, job *launch.Entrypoint) *exit.Error {
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
		Package: target.Package, Function: target.Function, Input: input,
		Org: ctx.Inv.Value("--org"), Trees: trees, InstallID: target.InstallID,
		Release: target.Release, ReleaseDigest: target.ReleaseDigest, Rental: rentalRequested(ctx),
		RentalRequired: ctx.Inv.Bool("--rental-only"),
	}, key)
	if e != nil {
		return e
	}
	if !ctx.Mode().JSON {
		fmt.Fprintf(ctx.Err, "Invoking %s/%s...\n", handle.Package, handle.Function)
	}
	if ctx.Inv.Bool("--follow") {
		return followJob(ctx, c, handle.JobID, began)
	}
	terminal, problem := observe(ctx, c, handle.JobID, false, optimisticObservation, began)
	if problem != nil {
		return problem
	}
	state, problem := c.Job(handle.JobID)
	if problem != nil {
		return problem
	}
	if terminal != nil || settled(state.Status) {
		return renderJobTerminal(ctx, state, terminal, began)
	}
	return renderSubmittedJob(ctx, state, !handle.Replay)
}

func renderSubmittedJob(ctx *Context, state api.JobState, changed bool) *exit.Error {
	fields := []output.Field{
		{K: "run", V: state.JobID},
		{K: "target", V: state.Package + "/" + state.Function},
		{K: "status", V: runStatus(state.Status)},
	}
	defaults := []string{"target", "status"}
	if state.QueuePosition != nil {
		queue := fmt.Sprint(*state.QueuePosition)
		if state.QueueDepth != nil && *state.QueueDepth >= *state.QueuePosition {
			queue += "/" + fmt.Sprint(*state.QueueDepth)
		}
		fields = append(fields, output.Field{K: "queue_position", V: queue})
		defaults = append(defaults, "queue_position")
	}
	fields = append(fields, output.Field{K: "changed", V: changed})
	defaults = append(defaults, "run")
	rec := compactRecord(fields, defaults...)
	rec.Next = []string{
		"cozy run watch " + state.JobID,
		"cozy run cancel " + state.JobID,
	}
	return emit(ctx, rec)
}

// parseTrees reads `--input-tree <ref>=<dir>`, the typed input TREES a job's fields hydrate
// from. The ref is the job's own request-field value; the directory is a read capability
// the orchestrator grants. A field naming a ref that is not here never hydrates.
func parseTrees(values []string) ([]string, *exit.Error) {
	out := []string{}
	for _, v := range values {
		ref, dir, ok := strings.Cut(v, "=")
		if !ok || ref == "" || dir == "" {
			return nil, exit.Usagef("--input-tree %q is not <ref>=<directory>", v).
				WithRemedy("a job's typed model/dataset input arrives as a materialized tree").
				WithNext("cozy run <target> --input-tree cozy/sdxl@lane=/path/to/store")
		}
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			return nil, exit.New(exit.NotFound, "--input-tree %s: %s is not a directory", ref, dir).
				WithRemedy("an input tree is a MATERIALIZED directory the orchestrator grants a read of")
		}
		out = append(out, v)
	}
	return out, nil
}

// ---------------------------------------------------------------------- job status

func jobFields(state api.JobState, full bool) []output.Field {
	fields := []output.Field{
		{K: "job", V: state.JobID},
		{K: "package", V: state.Package},
		{K: "function", V: state.Function},
		{K: "status", V: state.Status},
		{K: "elapsed", V: fmt.Sprintf("%.1fs", float64(state.ElapsedMS)/1000)},
	}
	if state.QueuePosition != nil {
		fields = append(fields, output.Field{K: "queue_position", V: *state.QueuePosition})
	}
	if state.Stage != "" {
		fields = append(fields, output.Field{K: "stage", V: state.Stage})
	}
	if state.Progress != nil {
		fields = append(fields, output.Field{K: "progress", V: compactValue(state.Progress)})
	}
	fields = append(fields, output.Field{
		K: "retries", V: fmt.Sprintf("%d/%d", state.Requeues, state.RetryBudget)})
	if state.Bill != nil {
		fields = append(fields, output.Field{K: "bill", V: microUSD(state.Bill.MicroUSD)})
	}
	if !full {
		return fields
	}
	if p := state.Publication; p != nil {
		fields = append(fields,
			output.Field{K: "publication", V: p.Repo},
			output.Field{K: "publication_root", V: p.Root},
			output.Field{K: "published", V: fmt.Sprintf("%d entr(y|ies), %s, verdict %s",
				p.Entries, output.Bytes(p.Bytes), strings.ToLower(p.Status))})
	}
	if len(state.Checkpoints) > 0 {
		rows := make([]string, 0, len(state.Checkpoints))
		for _, c := range state.Checkpoints {
			rows = append(rows, fmt.Sprintf("#%d %s/%s %s", c.Attempt, c.OperationKey,
				c.LogicalKey, c.Outcome))
		}
		fields = append(fields, output.Field{K: "checkpoints_declared", V: rows})
	}
	if len(state.Outputs) > 0 {
		outs := make([]string, 0, len(state.Outputs))
		for _, o := range state.Outputs {
			outs = append(outs, o.OutputID+" "+o.MediaID+" "+output.Bytes(o.Length))
		}
		fields = append(fields, output.Field{K: "outputs", V: outs})
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
		fields = append(fields, output.Field{K: "artifacts", V: artifacts})
	}
	if state.Result != nil {
		fields = append(fields, output.Field{K: "result", V: state.Result})
	}
	if state.Metrics != nil {
		fields = append(fields, output.Field{K: "metrics", V: state.Metrics})
	}
	if state.ErrorType != "" {
		fields = append(fields,
			output.Field{K: "error_type", V: state.ErrorType},
			output.Field{K: "error", V: state.Error})
	}
	return fields
}

func microUSD(n int64) string {
	return fmt.Sprintf("$%d.%06d", n/1_000_000, n%1_000_000)
}

// ---------------------------------------------------------------------- job ls

// ---------------------------------------------------------------------- job follow

// followJob attaches to the durable lifecycle stream and exits with the TERMINAL mapping.
//
// A job that finished before --await attached still yields its terminal: the durable lane is
// replayed from cursor 0, so the terminal event is delivered even though it was appended
// before this process existed. That is the whole reason `follow` reads the durable stream
// rather than watching for a live frame.
func followJob(ctx *Context, c *localapi.Client, jobID string, began time.Time) *exit.Error {
	interrupt := make(chan os.Signal, 2)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)
	watchCtx, stopWatch := context.WithCancel(context.Background())
	defer stopWatch()
	forced := make(chan struct{}, 1)
	cancelFailed := make(chan *exit.Error, 1)
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
		case <-done:
			return
		}
		cancelResult := make(chan *exit.Error, 1)
		go func() { cancelResult <- c.CancelJob(jobID) }()
		select {
		case <-interrupt:
			fmt.Fprintln(ctx.Err, "second interrupt — stopped waiting; the job remains recorded")
			forced <- struct{}{}
			stopWatch()
			return
		case problem := <-cancelResult:
			if problem != nil {
				fmt.Fprintf(ctx.Err, "cancel: %s\n", problem.Message)
				cancelFailed <- problem
				stopWatch()
				return
			}
		case <-done:
			return
		}
		select {
		case <-interrupt:
			fmt.Fprintln(ctx.Err, "second interrupt — stopped waiting; the job remains recorded")
			forced <- struct{}{}
			stopWatch()
		case <-done:
		}
	}()

	lines := newProgress(ctx, false, began)
	terminal, e := c.WatchContext(watchCtx, jobID, 0, lines.on)
	lines.done()
	select {
	case problem := <-cancelFailed:
		return problem
	default:
	}
	select {
	case <-forced:
		return exit.New(exit.Canceled,
			"stopped waiting for %s; it remains visible in `cozy run list`", jobID).
			WithNext("cozy run list")
	default:
	}
	if e != nil {
		return e
	}
	state, e := c.Job(jobID)
	if e != nil {
		return e
	}
	return renderJobTerminal(ctx, state, terminal, began)
}

func renderJobTerminal(ctx *Context, state api.JobState, terminal *localapi.Event,
	began time.Time,
) *exit.Error {
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
		output.Field{K: "wall_ms", V: time.Since(began).Milliseconds()})
	defaults := []string{"job", "status"}
	if state.Result != nil {
		defaults = append(defaults, "result")
	}
	if state.Publication != nil {
		defaults = append(defaults, "publication")
	}
	defaults = append(defaults, "wall_ms")
	rec := compactRecord(fields, defaults...)
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
		if state.Publication != nil {
			rec.Next = []string{"cozy run list --full"}
		}
		return emit(ctx, rec)
	}
	err := exit.Named(code, status, "job %s ended %s", state.JobID, status)
	if state.Error != "" {
		err.Message = fmt.Sprintf("job %s ended %s: %s — %s",
			state.JobID, status, state.ErrorType, state.Error)
	}
	if state.Requeues > 0 {
		err.WithRemedy("the orchestrator's retry projection spent %d of a %d-attempt budget "+
			"on neutral outcomes before settling", state.Requeues, state.RetryBudget)
	}
	if state.Triage != nil {
		err.WithNext("cozy run list --full")
	}
	return err
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
		fields := append(jobFields(state, true), output.Field{K: "changed", V: false})
		return emit(ctx, compactRecord(fields, "job", "status", "changed"))
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
	fields := append(jobFields(final, true), output.Field{K: "changed", V: true})
	return emit(ctx, compactRecord(fields, "job", "status", "changed"))
}

func settled(status string) bool {
	switch status {
	case "completed", "failed", "canceled":
		return true
	}
	return false
}
