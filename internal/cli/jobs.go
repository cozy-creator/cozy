package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sort"
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
	input, overrides, e := launch.ParsePayload(&launch.Entrypoint{
		Name: job.Name, Request: job.Request, Result: job.Result, Models: job.Models,
	}, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	if len(overrides) > 0 {
		return exit.Usagef("model.<param>= applies to serving callables; remote modeled jobs are not supported yet")
	}
	input, assets, e := launch.ParseAssets(job, input, ctx.Inv.Values["--asset"])
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
	key := requestKey(ctx.Inv.Value("--idempotency-key"))
	began := time.Now()
	handle, e := c.SubmitJob(api.JobSubmission{
		Package: target.Package, Function: target.Function, Input: input, LocalAssets: assets,
		Org: ctx.Inv.Value("--org"), Trees: trees, InstallID: target.InstallID,
		Release: target.Release, Rental: rentalRequested(ctx),
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
	reference := runReference(state.Number, state.JobID)
	fields := []output.Field{
		{K: "run", V: reference}, {K: "id", V: state.JobID},
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
	// A job still in flight whose transfer has already refused a member says so HERE, on
	// the reattachment surface, rather than only in a terminal nobody has reached yet.
	if rows := modelSourceLines(state.ModelSources); modelSourceVerdict(state.ModelSources) {
		fields = append(fields, output.Field{K: "model_sources", V: rows})
		defaults = append(defaults, "model_sources")
	}
	fields = append(fields, output.Field{K: "changed", V: changed})
	defaults = append(defaults, "run")
	rec := compactRecord(fields, defaults...)
	rec.Next = []string{
		"cozy run watch " + reference,
		"cozy run cancel " + reference,
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

func jobFields(mode output.Mode, state api.JobState, full bool) []output.Field {
	fields := []output.Field{
		{K: "job", V: runReference(state.Number, state.JobID)},
		{K: "id", V: state.JobID},
		{K: "package", V: state.Package},
		{K: "function", V: state.Function},
		{K: "status", V: state.Status},
		{K: "queued", V: seconds(state.QueuedMS)},
		{K: "execution", V: seconds(state.ExecutionMS)},
	}
	if state.CanceledBy != "" {
		fields = append(fields, output.Field{K: "canceled_by", V: state.CanceledBy})
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
	// A MEMBER THAT DID NOT VERIFY IS SAID OUT LOUD, not folded into a percentage. This is
	// the surface run 205 did not have: 44 of 48 verified for 2h55m, four members
	// permanently failed, and every client read showed a fraction.
	if rows := modelSourceLines(state.ModelSources); len(rows) > 0 {
		fields = append(fields, output.Field{K: "model_sources", V: rows})
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
			output.Field{K: "publication_root", V: mode.Hyperlink(p.Root)},
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
	if len(state.ModelOutputs) > 0 {
		fields = append(fields, output.Field{K: "model_outputs", V: state.ModelOutputs})
	}
	if len(state.Outputs) > 0 {
		outs := make([]string, 0, len(state.Outputs))
		for _, o := range state.Outputs {
			outs = append(outs, o.OutputID+" "+o.MediaID+" "+output.Bytes(o.Length))
		}
		fields = append(fields, output.Field{K: "outputs", V: outs})
	}
	if len(state.Weights) > 0 {
		weights := make([]string, 0, len(state.Weights))
		for _, row := range state.Weights {
			state := strings.ToLower(row.Disposition)
			if row.Outcome != "" {
				state = strings.ToLower(row.Outcome)
			}
			weights = append(weights, fmt.Sprintf("#%d %s %s receipt=%s root=%s",
				row.Attempt, row.OutputSlot, state,
				row.ReceiptDigest, row.ScratchRootID))
		}
		fields = append(fields, output.Field{K: "weights", V: weights})
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
	detached := make(chan struct{}, 1)
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case _, ok := <-interrupt:
			if !ok {
				return
			}
			// A SIGNAL NEVER CANCELS THE JOB (cl-108): a dying follower is not a person
			// asking for cancellation. The daemon owns the accepted job; this client
			// merely detaches, and only an explicit `cozy job cancel` cancels.
			fmt.Fprintf(ctx.Err,
				"\ndetached — the job keeps running; `cozy run watch %s` reattaches, `cozy job cancel %s` cancels\n",
				jobID, jobID)
			detached <- struct{}{}
			stopWatch()
		case <-done:
		}
	}()

	lines := NewProgress(ctx, false, began)
	terminal, e := c.WatchContext(watchCtx, jobID, 0, lines.On)
	lines.Done()
	if e != nil {
		return e
	}
	state, e := c.Job(jobID)
	if e != nil {
		return e
	}
	select {
	case <-detached:
		if !settled(state.Status) {
			return renderSubmittedJob(ctx, state, false)
		}
	default:
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
	fields := append(jobFields(ctx.Mode(), state, true),
		output.Field{K: "wall_ms", V: time.Since(began).Milliseconds()})
	defaults := []string{"job", "status"}
	if state.Result != nil {
		defaults = append(defaults, "result")
	}
	if state.Publication != nil {
		defaults = append(defaults, "publication")
	}
	if len(state.ModelOutputs) > 0 {
		defaults = append(defaults, "model_outputs")
	}
	// A verdict is not a --full detail. It is the answer to the question the operator is
	// asking, so it stands in the default view.
	if modelSourceVerdict(state.ModelSources) {
		defaults = append(defaults, "model_sources")
	}
	defaults = append(defaults, "wall_ms")
	rec := compactRecord(fields, defaults...)
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
		if hint := modelPublishHint(state); hint != "" {
			rec.Next = []string{hint}
		} else if state.Publication != nil {
			rec.Next = []string{"cozy run list --full"}
		}
		return emit(ctx, rec)
	}
	err := exit.Named(code, status, "job %s ended %s", state.JobID, status)
	if state.Error != "" {
		err.Message = fmt.Sprintf("job %s ended %s: %s — %s",
			state.JobID, status, state.ErrorType, state.Error)
	}
	// A canceled job is LOUD about WHO ended it (cl-108).
	if mapTerminal(status) == "canceled" && state.CanceledBy != "" {
		err.Message = fmt.Sprintf("job %s was canceled by %s", state.JobID, state.CanceledBy)
		if state.Error != "" {
			err.Message += ": " + state.Error
		}
	}
	if state.Requeues > 0 {
		err.WithRemedy("the orchestrator's retry projection spent %d of a %d-attempt budget "+
			"on neutral outcomes before settling", state.Requeues, state.RetryBudget)
	}
	if state.Triage != nil {
		err.WithNext("cozy run list --full")
	}
	if hint := modelPublishHint(state); hint != "" {
		err.WithNext(hint)
	}
	return err
}

// modelSourceVerdict answers whether any listed member carries the pod's own reason. A
// member merely still in flight is progress; a member with a code is an explanation.
func modelSourceVerdict(sources []api.ModelSourceState) bool {
	for _, source := range sources {
		if source.SafeCode != "" {
			return true
		}
	}
	return false
}

// modelSourceLines renders the unverified members, the ones that failed first: a stalled
// transfer is read by its verdicts, and a member with no verdict is read after them.
func modelSourceLines(sources []api.ModelSourceState) []string {
	if len(sources) == 0 {
		return nil
	}
	ordered := append([]api.ModelSourceState(nil), sources...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return (ordered[i].SafeCode != "") && (ordered[j].SafeCode == "")
	})
	rows := make([]string, 0, len(ordered))
	for _, source := range ordered {
		line := source.Member + " " + source.State
		if source.Length > 0 {
			line += " " + output.Bytes(source.Transferred) + "/" + output.Bytes(source.Length)
		}
		if source.SafeCode != "" {
			line += " — " + source.SafeCode
			if source.SafeDetail != "" {
				line += ": " + source.SafeDetail
			}
		}
		rows = append(rows, line)
	}
	return rows
}

func modelPublishHint(state api.JobState) string {
	if state.ModelDestination == "" || len(state.ModelOutputs) == 0 ||
		strings.HasPrefix(state.ModelDestination, "local/") {
		return ""
	}
	names := make([]string, 0, len(state.ModelOutputs))
	for name := range state.ModelOutputs {
		names = append(names, name)
	}
	sort.Strings(names)
	command := "cozy model publish " + state.ModelDestination + " --release <label>"
	for _, name := range names {
		command += " --lane " + name + "=" + state.ModelOutputs[name]
	}
	return command
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
		fields := append(jobFields(ctx.Mode(), state, true), output.Field{K: "changed", V: false})
		return emit(ctx, compactRecord(fields, "job", "status", "changed"))
	}
	if e := c.CancelJob(jobID, "cozy job cancel"); e != nil {
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
	fields := append(jobFields(ctx.Mode(), final, true), output.Field{K: "changed", V: true})
	return emit(ctx, compactRecord(fields, "job", "status", "changed"))
}

func settled(status string) bool {
	switch status {
	case "completed", "failed", "canceled":
		return true
	}
	return false
}
