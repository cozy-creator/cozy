package cli

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/transfer"
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
	if names := ctx.Inv.Values["--allow-publish"]; len(names) > 0 {
		normalized, problem := hub.NormalizePublicationRepositories(names)
		if problem != nil {
			return problem
		}
		if !rentalRequested(ctx) {
			return exit.Named(exit.Structural, "publication.machine_identity_required", "--allow-publish requires a rented transaction with its own certificate identity")
		}
		ctx.Inv.Values["--allow-publish"] = normalized
	}
	deadline, problem := runDeadline(ctx)
	if problem != nil {
		return problem
	}
	terms, destination, e := launch.ConversionTerms(target.Package+"/"+target.Function, job, ctx.Inv.Args[1:])
	if e != nil {
		return e
	}
	if destination != "" {
		if named := ctx.Inv.Value("--publish-to"); named != "" && named != destination {
			return exit.Usagef("destination %s disagrees with --publish-to=%s", destination, named)
		}
		ctx.Inv.Values["--publish-to"] = []string{destination}
	}
	// THE PAYLOAD IS TYPED AGAINST THE RECORDED SCHEMA before a job exists — the same
	// client-side check `cozy run` makes, over the job's own declared request struct.
	input, overrides, e := launch.ParsePayload(&launch.Entrypoint{
		Name: job.Name, Request: job.Request, Result: job.Result, Models: job.Models,
	}, terms, ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	if overrides.AttentionKernel != "" {
		return exit.Usagef("kernel.attention applies only to serving callables")
	}
	prepareImage := imagePreparer(ctx)
	input, assetFiles, assetTrees, e := launch.ParseTreeAssets(job, input, ctx.Inv.Values["--asset"])
	if e != nil {
		return e
	}
	input, assets, e := launch.ParseAssets(job, input, assetFiles, ctx.Inv.Values["--asset-fidelity"], prepareImage)
	if e != nil {
		return e
	}
	if e := validateInvocationPayload(ctx, target.Package, job, input); e != nil {
		return e
	}
	trees, e := parseTrees(append(append([]string(nil), ctx.Inv.Values["--input"]...), assetTrees...))
	if e != nil {
		return e
	}
	outputDirectory, e := requestedOutputDirectory(ctx)
	if e != nil {
		return e
	}

	selectedRental, e := requestedRental(ctx, target, job.Name)
	if e != nil {
		return e
	}
	sub := api.JobSubmission{Package: target.Package, Function: target.Function, Input: input, LocalAssets: assets,
		AllowPublish: ctx.Inv.Values["--allow-publish"],
		TimeoutMS:    int64(deadline / time.Millisecond),
		RetainWork:   strings.HasPrefix(target.Package, "local/"), RetryOf: ctx.Inv.Value("--retry"),
		Org: ctx.Inv.Value("--org"), Trees: trees, InstallID: target.InstallID,
		Release: target.Release, Rental: rentalRequested(ctx),
		RentNew: ctx.Inv.Bool("--rent-new"), RentalRequired: ctx.Inv.Bool("--rental-only") || ctx.Inv.Bool("--rent-new") || selectedRental != "", RequestedRental: selectedRental, OutputDirectory: outputDirectory,
		PlannedSourceBytes: ctx.ingestBytes}
	if deadline%time.Millisecond != 0 {
		sub.TimeoutMS++
	}
	source, profiles, models, e := resolveJobModelInputs(ctx, target, job, overrides.Models)
	if e != nil {
		return e
	}
	sub.Models = models
	if source != "" {
		if len(trees) > 0 {
			return exit.Usagef("foreign model inputs cannot also use local input trees")
		}
		return submitSourceTransfer(ctx, "model-upload", source, ctx.Inv.Value("--publish-to"),
			&sourceInvocation{Target: target, Job: job, Profiles: profiles}, sub)
	}
	if e := jobOutputDestination(ctx, job, &sub); e != nil {
		return e
	}
	c, e := dial(ctx)
	if e != nil {
		return e
	}
	key := requestKey(ctx.Inv.Value("--idempotency-key"))
	began := ctx.commandStarted
	handle, e := c.SubmitJob(sub, key)
	releaseSnapshotReader(target)
	if e != nil {
		return e
	}
	if !ctx.Mode().JSON {
		fmt.Fprintf(ctx.Err, "Queued run %s: %s/%s\n", runReference(handle.Number, handle.JobID), handle.Package, handle.Function)
		if rental := ctx.Inv.Value("--rental"); rental != "" {
			fmt.Fprintf(ctx.Err, "Machine: %s\n", rental)
		}
	}
	if handle.MachineExecution {
		// The daemon already owns this run. Waiting for the worker's receipt is
		// part of --await, not a second acceptance gate for a detached command.
		if !ctx.Inv.Bool("--follow") {
			state, problem := c.Job(handle.JobID)
			if problem != nil {
				return problem
			}
			return renderSubmittedJob(ctx, state, !handle.Replay)
		}
		state, detached, problem := waitMachineAcceptance(ctx, c, handle)
		if problem != nil {
			return problem
		}
		if detached {
			return renderSubmittedJob(ctx, state, !handle.Replay)
		}
		if settled(state.Status) {
			// Even an immediately completed/replayed machine job has a durable
			// terminal event. Read it for the same wall clock as later watches.
			return followJob(ctx, c, handle.JobID, began)
		}
		if state.Status == "blocked" || state.Status == "paused" {
			return renderJobTerminal(ctx, state, nil)
		}
	}
	if ctx.Inv.Bool("--follow") {
		return followJob(ctx, c, handle.JobID, began)
	}
	terminal, problem := observe(ctx, c, handle.JobID, optimisticObservation, began)
	if problem != nil {
		return problem
	}
	state, problem := c.Job(handle.JobID)
	if problem != nil {
		return problem
	}
	if terminal != nil || settled(state.Status) || state.Status == "paused" || state.Status == "blocked" {
		return renderJobTerminal(ctx, state, terminal)
	}
	return renderSubmittedJob(ctx, state, !handle.Replay)
}

func renderSubmittedJob(ctx *Context, state api.JobState, changed bool) *exit.Error {
	state.Status = publicObservedStatus(state.Status)
	reference := runReference(state.Number, state.JobID)
	fields := []output.Field{
		{K: "run", V: reference}, {K: "id", V: state.JobID},
		{K: "target", V: state.Package + "/" + state.Function},
		{K: "status", V: runStatus(state.Status)},
	}
	defaults := []string{"target", "status"}
	if machine := state.MachineExecution; machine != nil {
		fields = append(fields, output.Field{K: "machine_accepted", V: machine.Accepted}, output.Field{K: "machine", V: machine.Machine})
		defaults = append(defaults, "machine")
		if ctx.Mode().JSON {
			defaults = append(defaults, "machine_accepted")
		}
		if !machine.Accepted {
			stage := state.Stage
			if stage == "" || stage == "waiting for durable machine acceptance" {
				stage = "queued locally; machine preparation continues in the background"
			}
			fields = append(fields, output.Field{K: "stage", V: stage})
			defaults = append(defaults, "stage")
		}
	}
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
	if state.RetainWork {
		if state.Status == "paused" {
			rec.Next = append(rec.Next, "cozy run resume "+reference)
		} else if state.Status == "failed" && state.RetryAvailable {
			rec.Next = append(rec.Next, "cozy run <updated-script-or-package> --retry "+reference)
		} else if state.Status != "pausing" && !invocationSettled(state.Status) {
			rec.Next = append(rec.Next, "cozy run pause "+reference)
		}
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
		if dir == "~" || strings.HasPrefix(dir, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				return nil, exit.New(exit.NotFound, "cannot resolve input directory home: %s", err)
			}
			dir = filepath.Join(home, strings.TrimPrefix(strings.TrimPrefix(dir, "~"), "/"))
		}
		info, err := os.Stat(dir)
		if err != nil || !info.IsDir() {
			return nil, exit.New(exit.NotFound, "--input-tree %s: %s is not a directory", ref, dir).
				WithRemedy("an input tree is a MATERIALIZED directory the orchestrator grants a read of")
		}
		absolute, err := filepath.Abs(dir)
		if err != nil {
			return nil, exit.New(exit.Validation, "input tree directory is invalid")
		}
		out = append(out, ref+"="+absolute)
	}
	return out, nil
}

// ---------------------------------------------------------------------- job status

func jobFields(mode output.Mode, state api.JobState, full bool) []output.Field {
	status := state.Status
	if state.Status == "canceled" {
		status = humanCancellationStatus(state.CanceledBy)
	}
	if export := state.OutputExport; export != nil && export.State == "failed" && status == "completed" {
		status += " (export pending: " + export.ErrorCode + ")"
	}
	fields := []output.Field{
		{K: "job", V: runReference(state.Number, state.JobID)},
		{K: "id", V: state.JobID},
		{K: "package", V: state.Package},
		{K: "function", V: state.Function},
		{K: "status", V: status},
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
	if export := state.OutputExport; export != nil {
		fields = append(fields,
			output.Field{K: "output_directory", V: export.Directory},
			output.Field{K: "output_export", V: outputExportHint(mode, export)})
		if saved := exportedOutputs(api.Lifecycle{OutputExport: export, Outputs: state.Outputs}); len(saved) > 0 {
			if mode.JSON {
				fields = append(fields, output.Field{K: "saved", V: saved})
			} else {
				fields = append(fields, output.Field{K: "saved", V: outputExportHint(mode, export)})
			}
		}
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
	if len(state.NativeOutputs) > 0 {
		fields = append(fields, output.Field{K: "native_outputs", V: state.NativeOutputs})
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
	interrupt, restoreInput, _, problem := liveSignals(ctx, nil)
	if problem != nil {
		return problem
	}
	defer restoreInput()
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
			if !ctx.Mode().JSON {
				fmt.Fprintf(ctx.Err,
					"\ndetached — the job keeps running; `cozy run watch %s` reattaches, `cozy run cancel %s` cancels\n",
					jobID, jobID)
			}
			detached <- struct{}{}
			stopWatch()
		case <-done:
		}
	}()

	lines := NewProgress(ctx, ctx.Mode().JSON, began)
	var stopped *localapi.Event
	terminal, e := c.WatchContext(watchCtx, jobID, 0, func(event localapi.Event) bool {
		if event.Type == "request.blocked" {
			state, problem := c.Job(jobID)
			if problem == nil && currentManualStop(state.Status, state.StoppedEventID, event.EventID) {
				projected := publicFailureEvent(event)
				lines.On(projected)
				stopped = &projected
				return false
			}
			return true // historical stop or actively reconciling acceptance
		}
		keep := lines.On(event)
		if event.Type == "request.paused" {
			// A resumed request may replay an older pause event. Stop only when
			// its current state still matches the event being observed.
			state, problem := c.Job(jobID)
			if problem == nil && "request."+state.Status == event.Type {
				stopped = &event
				return false
			}
		}
		return keep
	})
	if terminal == nil {
		terminal = stopped
	}
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
	return renderJobTerminal(ctx, state, terminal)
}

func renderJobTerminal(ctx *Context, state api.JobState, terminal *localapi.Event) *exit.Error {
	state.Status = publicObservedStatus(state.Status)
	if machine := state.MachineExecution; machine != nil && state.Status == "completed" && !machine.Collected {
		collected, problem := awaitMachineCollection(ctx, state)
		if problem != nil {
			return problem
		}
		state = collected
		state.Status = publicObservedStatus(state.Status)
	}
	if state.Status == "completed" && state.OutputExport != nil {
		client, problem := dial(ctx)
		if problem != nil {
			return problem
		}
		life, problem := waitOutputExport(client, api.Lifecycle{
			RequestID: state.JobID, Status: state.Status, OutputExport: state.OutputExport,
		})
		if problem != nil {
			return problem
		}
		state.OutputExport = life.OutputExport
	}
	status := localapi.StreamStatus(terminal)
	if status == "" {
		status = state.Status
	}
	if status == "paused" {
		return renderSubmittedJob(ctx, state, false)
	}
	// A REQUEST THAT ENDED BEFORE ANY ATTEMPT has its reason only in the terminal EVENT:
	// there is no attempt row, so the state document has no terminal to read a cause off.
	// Without this the client printed `failed` and nothing else — which is exactly the
	// case cl-004's publication-escape arm produces, and the reason it exists.
	if state.ErrorType == "" && terminal != nil {
		state.ErrorType, _ = terminal.Payload["error_type"].(string)
		state.Error, _ = terminal.Payload["error"].(string)
	}
	fields := jobFields(ctx.Mode(), state, true)
	wallMS, wallKnown := recordedRunWall(state.CreatedAt, terminal)
	if wallKnown {
		fields = append(fields, output.Field{K: "wall_ms", V: wallMS})
	}
	defaults := []string{"job", "status"}
	if state.OutputExport != nil {
		defaults = append(defaults, "output_export")
		if saved := exportedOutputs(api.Lifecycle{OutputExport: state.OutputExport, Outputs: state.Outputs}); len(saved) > 0 {
			defaults = append(defaults, "saved")
		}
	}
	if state.Result != nil {
		defaults = append(defaults, "result")
	}
	if state.Publication != nil {
		defaults = append(defaults, "publication")
	}
	if len(state.NativeOutputs) > 0 {
		defaults = append(defaults, "native_outputs")
	}
	if len(state.ModelOutputs) > 0 {
		defaults = append(defaults, "model_outputs")
	}
	// A verdict is not a --full detail. It is the answer to the question the operator is
	// asking, so it stands in the default view.
	if modelSourceVerdict(state.ModelSources) {
		defaults = append(defaults, "model_sources")
	}
	if wallKnown {
		defaults = append(defaults, "wall_ms")
	}
	rec := compactRecord(fields, defaults...)
	if export := state.OutputExport; export != nil && export.State == "failed" {
		rec.Notes = append(rec.Notes, fmt.Sprintf("output export to %s failed (%s): %s; accepted output bytes remain in internal custody",
			export.Directory, export.ErrorCode, export.Error))
	}
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
		if hint := modelPublishHint(state); hint != "" {
			rec.Next = []string{hint}
		} else if len(state.NativeOutputs) > 0 {
			rec.Next = []string{"cozy run watch " + runReference(state.Number, state.JobID) + " --full"}
		} else if state.Publication != nil {
			rec.Next = []string{"cozy run list --full"}
		}
		return emit(ctx, rec)
	}
	humanStatus := status
	if status == "canceled" {
		humanStatus = humanCancellationStatus(state.CanceledBy)
	}
	err := exit.Named(code, status, "job %s ended %s", state.JobID, humanStatus)
	if status == "failed" && state.RetryAvailable {
		err.WithNext("cozy run <updated-script-or-package> --retry " + runReference(state.Number, state.JobID))
	}
	if state.Error != "" {
		err.Message = fmt.Sprintf("job %s ended %s: %s — %s",
			state.JobID, humanStatus, state.ErrorType, state.Error)
	}
	// A canceled job is LOUD about WHO ended it (cl-108).
	if mapTerminal(status) == "canceled" && state.CanceledBy != "" {
		err.Message = fmt.Sprintf("job %s was %s", state.JobID, humanCancellationStatus(state.CanceledBy))
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
	state, e := c.RecordedJob(jobID) // as handleRunCancel
	if e != nil {
		return e
	}
	// ALREADY TERMINAL = IDEMPOTENT 0 printing the terminal. A cancel that arrives after
	// the terminal is late, not wrong.
	if settled(state.Status) && !state.Retaining {
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

// awaitMachineCollection follows the result collection the daemon owns after a run
// completes on its machine. Each read waits on the daemon's own progress-bounded
// collection; only an unmet observation that stays unchanged for the whole stall budget
// is reported, and the run and its retained result are untouched either way.
func awaitMachineCollection(ctx *Context, state api.JobState) (api.JobState, *exit.Error) {
	client, problem := dial(ctx)
	if problem != nil {
		return state, problem
	}
	const poll = time.Second
	last, since := "", time.Now()
	for state.Status == "completed" && state.MachineExecution != nil && !state.MachineExecution.Collected {
		if observed := state.MachineExecution.ObservationError; observed != last {
			last, since = observed, time.Now()
		} else if time.Since(since) >= transfer.StallBudget {
			return state, exit.Named(exit.Unavailable, "machine_execution.result_collection_pending",
				"run %s completed on its machine, but its result has not been collected: %s", state.JobID, last).
				WithRemedy("the result stays retained on the machine; `cozy run watch %s` collects it once the machine answers", state.JobID)
		}
		time.Sleep(poll)
		if state, problem = client.Job(state.JobID); problem != nil {
			return state, problem
		}
		state.Status = publicObservedStatus(state.Status)
	}
	return state, nil
}
