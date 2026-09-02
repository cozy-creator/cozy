package cli

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/mattn/go-isatty"
	"golang.org/x/term"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
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
	return localapi.Open(ctx.Cfg, ctx.Daemon)
}

// ----------------------------------------------------------------------------- run

func handleRunExecute(ctx *Context) *exit.Error {
	if problem := validateRunPlacement(ctx); problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--stream") && !ctx.Inv.Bool("--await") {
		return exit.Usagef("--stream requires --await").
			WithRemedy("use --await for a terminal progress stream, or omit --stream for a short optimistic observation")
	}
	if ctx.Inv.Value("--timeout") != "" && !ctx.Inv.Bool("--await") {
		return exit.Usagef("--timeout requires --await").
			WithRemedy("a detached run has no client waiting to enforce a caller deadline")
	}
	target, packageInterface, problem := invocationTarget(ctx)
	if problem != nil {
		return problem
	}
	if target.Function == "" {
		return emitFunctions(ctx, target, packageInterface)
	}
	callable, problem := packageInterface.Function(target.Function)
	if problem != nil {
		return unknownFunction(target, packageInterface)
	}
	if ctx.Inv.Bool("--describe") {
		return emitDescribe(ctx, target, packageInterface, callable)
	}
	if callable.Kind != "job" {
		if len(ctx.Inv.Values["--input"]) > 0 {
			return exit.Usagef("--input-tree applies only to a job callable")
		}
		if ctx.Inv.Value("--org") != "" {
			return exit.Usagef("--org applies only to a job callable")
		}
		return handleRun(ctx, target, callable)
	}
	if ctx.Inv.Bool("--stream") || len(ctx.Inv.Values["--asset"]) > 0 ||
		ctx.Inv.Value("--out") != "" || ctx.Inv.Value("--timeout") != "" {
		return exit.Usagef("the selected callable is a job and received a serving-only flag").
			WithRemedy("jobs accept payload values, --in, --input-tree, --org, --await, and --rental")
	}
	if rentalRequested(ctx) && len(callable.Models) > 0 {
		return exit.Named(exit.Unavailable, "rental.modeled_job_unsupported",
			"remote jobs with model slots are not supported yet")
	}
	if rentalRequested(ctx) && len(ctx.Inv.Values["--input"]) > 0 {
		return exit.Named(exit.Unavailable, "rental.job_input_tree_unsupported",
			"remote jobs cannot grant a local input-tree directory")
	}
	if ctx.Inv.Bool("--await") {
		ctx.Inv.Bools["--follow"] = true
	}
	return handleJobSubmit(ctx, target, callable)
}

func validateRunPlacement(ctx *Context) *exit.Error {
	if ctx.Inv.Bool("--rental") && ctx.Inv.Bool("--rental-only") {
		return exit.Usagef("--rental and --rental-only are mutually exclusive")
	}
	managedRental := rentalRequested(ctx)
	if managedRental && ctx.Cfg.RentalsMaxHourlySpendUSDMicros <= 0 {
		return exit.Named(exit.Usage, "rental.spend_cap_required",
			"rented execution requires a positive rentals.max_hourly_spend_usd in %s", filepath.Join(ctx.Cfg.Home, "config.yaml")).
			WithRemedy("set the fleet-wide hourly ceiling before authorizing rental spend")
	}
	return nil
}

func rentalRequested(ctx *Context) bool {
	return ctx.Inv.Bool("--rental") || ctx.Inv.Bool("--rental-only")
}

func handleRun(ctx *Context, target Target, ep *launch.Entrypoint) *exit.Error {
	deadline, e := runDeadline(ctx)
	if e != nil {
		return e
	}
	key := ctx.Inv.Value("--idempotency-key")
	if key == "" {
		key = mintKey()
	}

	// THE PAYLOAD IS TYPED AGAINST THE RECORDED SCHEMA — the surface the release's own
	// runtime vouched for at install — so a typo costs a millisecond instead of a model
	// load, and `steps=2` is an int because the schema says int.
	managedRental := rentalRequested(ctx)
	if legacy := launch.LegacyFileTerm(ctx.Inv.Args[1:]); managedRental && legacy != "" {
		return exit.Named(exit.Usage, "remote_file_input_ambiguous",
			"%s embeds file bytes into a JSON string and cannot name a remote input grant", legacy).
			WithRemedy("use `--asset <field-path>=<file>`; the field path becomes the exact worker-protocol input id")
	}
	input, overrides, e := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	input, assets, e := launch.ParseAssets(ep, input, ctx.Inv.Values["--asset"])
	if e != nil {
		return e
	}
	input, e = finalizeInputPayload(ep, input, key)
	if e != nil {
		return e
	}
	models, e := resolveInvocationModels(ctx, target, ep, overrides, managedRental)
	if e != nil {
		return e
	}
	outputDirectory := ""
	if requested := ctx.Inv.Value("--out"); requested != "" {
		absolute, err := filepath.Abs(requested)
		if err != nil {
			return exit.Usagef("cannot resolve --out %q: %s", requested, err)
		}
		outputDirectory = filepath.Clean(absolute)
	}

	c, e := dial(ctx)
	if e != nil {
		return e
	}
	began := time.Now()
	handle, e := c.Submit(api.Submission{
		Package: target.Package, Function: target.Function, Input: input,
		LocalAssets: assets, InstallID: target.InstallID,
		Release: target.Release, Rental: managedRental,
		RentalRequired:  ctx.Inv.Bool("--rental-only"),
		Models:          models,
		OutputDirectory: outputDirectory,
	}, key)
	if e != nil {
		return e
	}
	submitted := time.Since(began)
	stream := ctx.Inv.Bool("--stream")
	if !stream && !ctx.Mode().JSON {
		if ctx.Mode().Full {
			fmt.Fprintf(ctx.Err, "request %s · attempt %d · %s\n",
				handle.RequestID, handle.Attempt, handle.Status)
		} else {
			fmt.Fprintf(ctx.Err, "Invoking %s/%s...\n", target.Package, target.Function)
			if len(launch.AssetPaths(ep.Result)) > 0 {
				fmt.Fprintf(ctx.Err, "Saving outputs to %s\n",
					errLink(ctx, outputDirectoryHint(ctx, target, outputDirectory)))
			}
		}
		if handle.Replay {
			fmt.Fprintln(ctx.Err, "note: this key returned the existing invocation")
		}
	}

	var terminal *localapi.Event
	stopped := ""
	if ctx.Inv.Bool("--await") {
		terminal, stopped, e = watch(ctx, c, handle.RequestID, stream, deadline, began)
	} else {
		terminal, e = observe(ctx, c, handle.RequestID, stream, optimisticObservation, began)
	}
	if e != nil {
		return e
	}
	life, e := c.Request(handle.RequestID)
	if e != nil {
		return e
	}
	if terminal == nil && !invocationSettled(life.Status) {
		return renderSubmittedRun(ctx, life, !handle.Replay)
	}
	life, e = waitOutputExport(c, life)
	if e != nil {
		return e
	}
	saved := exportedOutputs(life)
	return renderRun(ctx, life, terminal, stopped, saved, submitted, began)
}

type invocationModelSpec struct {
	Slot string
	Ref  string
	Lane string
}

// resolveInvocationModels applies the one binding ladder for both local and rented
// execution: an explicit `model.<param>=` run key, then the hub default binding. Local
// acquisition freezes the exact Manifest and length before submission; remote
// acquisition freezes the same Manifest in the signed worker download request.
func resolveInvocationModels(ctx *Context, target Target, ep *launch.Entrypoint,
	overrides map[string]string, remote bool,
) ([]orchestrator.ModelRef, *exit.Error) {
	if target.InstallID != "" {
		row, problem := exactInvocationInstall(ctx, target)
		if problem != nil {
			return nil, problem
		}
		if row.SourceKind == "local" {
			// An editable install froze its exact model selection from package.toml at
			// install; a local worker reads it from the PlacementSet and a rental is handed
			// the same rows (cl-101). There is no hub default to ask for a `local/` package.
			if len(overrides) > 0 {
				return nil, exit.Named(exit.Unavailable, "editable_model_override_unsupported",
					"editable package model overrides are not available on the published-package BYOM lane").
					WithRemedy("publish the package code, then invoke it with model.<param>=org/model@release")
			}
			return nil, nil
		}
	}
	selected, problem := invocationModelSpecs(ctx, target, ep, overrides)
	if problem != nil || len(selected) == 0 {
		return nil, problem
	}
	if remote {
		out := make([]orchestrator.ModelRef, 0, len(selected))
		for _, spec := range selected {
			row, problem := resolveRemoteModel(ctx, target.Package, spec.Slot, spec.Ref, spec.Lane)
			if problem != nil {
				return nil, problem
			}
			out = append(out, row)
		}
		return out, nil
	}
	installRow, problem := exactInvocationInstall(ctx, target)
	if problem != nil {
		return nil, problem
	}
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return nil, problem
	}
	work, problem := scratch.Temp(layout.Tmp, "invoke-models-")
	if problem != nil {
		return nil, problem
	}
	defer work.Release()
	root := work.Path
	hctx, cancel := hub.LongContext()
	defer cancel()
	out := make([]orchestrator.ModelRef, 0, len(selected))
	for index, spec := range selected {
		packagePublishStatus(ctx, "Resolving model for %s...", spec.Slot)
		model, problem := acquirePublishedModel(hctx, ctx, tool, client(ctx), spec.Ref,
			spec.Lane, target.Package, spec.Slot,
			filepath.Join(root, fmt.Sprintf("%03d", index)))
		if problem != nil {
			return nil, problem
		}
		out = append(out, orchestrator.ModelRef{Package: target.Package, Slot: spec.Slot,
			Model: model.Model, Release: model.Release, Lane: model.Lane,
			Manifest: model.Manifest, ManifestLength: model.ManifestLength})
	}
	retained, retainProblem := exactInvocationInstall(ctx, target)
	if retainProblem != nil || retained.ID != installRow.ID ||
		retained.SourceDigest != installRow.SourceDigest {
		return nil, exit.Named(exit.Conflict, "package_install_changed",
			"the selected package install disappeared or changed during model acquisition").
			WithRemedy("retry against the current installed package")
	}
	return out, nil
}

func invocationModelSpecs(ctx *Context, target Target, ep *launch.Entrypoint,
	overrides map[string]string,
) ([]invocationModelSpec, *exit.Error) {
	if len(ep.Models) == 0 {
		return nil, nil
	}
	// ParsePayload already resolved every `model.<param>=` key onto a declared slot
	// path; here each ref parses once through the one grammar, with the lane carried
	// beside the ref the way the resolvers take it.
	selected := make(map[string]invocationModelSpec, len(overrides))
	for slotPath, raw := range overrides {
		model, release, lane, manifest, problem := parseModelRef(raw)
		if problem != nil {
			return nil, problem
		}
		ref := model
		if release != "" {
			ref += "@" + release
		}
		if manifest != "" {
			ref += "#" + manifest
		}
		selected[slotPath] = invocationModelSpec{Slot: slotPath, Ref: ref, Lane: lane}
	}
	defaults := map[string]hub.PackageBindingRow{}
	if len(selected) < len(ep.Models) {
		var problem *exit.Error
		defaults, problem = invocationDefaultBindings(ctx, target)
		if problem != nil {
			return nil, problem
		}
	}
	out := make([]invocationModelSpec, 0, len(ep.Models))
	for _, slot := range ep.Models {
		if spec, ok := selected[slot.Path]; ok {
			out = append(out, spec)
			continue
		}
		binding, ok := defaults[slot.Path]
		if !ok {
			return nil, exit.Named(exit.NotFound, "package_default_model_unavailable",
				"%s has no usable configured default for model slot %s", target.Package, slot.Path).
				WithRemedy("override it explicitly: model.%s=org/model@release, or bind a default: cozy package bind %s %s org/model@release", slot.Param, target.Package, slot.Path)
		}
		out = append(out, invocationModelSpec{Slot: slot.Path, Ref: binding.Ref(), Lane: binding.Lane})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Slot < out[j].Slot })
	return out, nil
}

// invocationDefaultBindings reads the package's CURRENT default bindings from
// the hub (th-116). The rows are mutable pointers seeded from the shipped
// package.toml at release commit and owner-retargetable afterwards; no
// invocation re-reads the in-release toml, so an owner's retarget takes effect
// on the very next bare run. A `model.<param>=` run key still overrides per
// invocation.
func invocationDefaultBindings(ctx *Context, target Target) (
	map[string]hub.PackageBindingRow, *exit.Error,
) {
	ref, problem := hub.ParseRef(target.Package)
	if problem != nil {
		return nil, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	rows, problem := client(ctx).PackageBindings(hctx, ref)
	if problem != nil {
		return nil, exit.Named(problem.Code, "package_default_model_unavailable",
			"%s default bindings are not readable: %s", target.Package, problem.Message).
			WithRemedy("supply model.<param>=org/model@release to bypass the hub default")
	}
	out := make(map[string]hub.PackageBindingRow, len(rows))
	for _, row := range rows {
		out[row.Slot] = row
	}
	return out, nil
}

func exactInvocationInstall(ctx *Context, target Target) (*records.PackageInstall, *exit.Error) {
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, problem
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return nil, problem
	}
	defer store.Close()
	row, problem := store.Install(target.InstallID)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.Package != target.Package {
		return nil, exit.New(exit.NotFound, "installed package %s is no longer available", target.Package)
	}
	return row, nil
}

func resolveRemoteModel(ctx *Context, packageName, slotPath, raw, wantedLane string) (
	orchestrator.ModelRef, *exit.Error,
) {
	// A caller may narrow by Manifest spelling, but cannot introduce one: the Hub-authored
	// release card below must contain it in an exact lane before it enters request identity
	// or a signed worker download delegation. No caller bytes or local path are trusted.
	var empty orchestrator.ModelRef
	modelName, release, refLane, manifest, problem := parseModelRef(raw)
	if problem != nil {
		return empty, problem
	}
	if refLane != "" {
		if wantedLane != "" && wantedLane != refLane {
			return empty, exit.Usagef("%q asks for lane %q while lane %q was already selected",
				raw, refLane, wantedLane)
		}
		wantedLane = refLane
	}
	ref, problem := hub.ParseRef(modelName)
	if problem != nil {
		return empty, problem
	}
	if ref.Org == "local" {
		return empty, exit.Named(exit.Unavailable, "rental_local_model_sync_required",
			"%s is a private local model and cannot be granted to a rented worker by path", ref.String()).
			WithRemedy("upload it under a non-local org, or explicitly sync it through the model upload workflow")
	}
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx).ModelCard(hctx, ref)
	if problem != nil {
		return empty, problem
	}
	if card.Model.Ref() != ref.String() {
		return empty, exit.Named(exit.Conflict, "rental.model_catalog_changed",
			"Tensorhub returned model %s while resolving %s", card.Model.Ref(), ref.String())
	}
	if release == "" {
		for _, candidate := range card.Releases {
			if !candidate.Yanked && candidate.YankedAt == "" &&
				(release == "" || candidate.Release > release) {
				release = candidate.Release
			}
		}
	}
	var selected *hub.ModelReleaseSummary
	for i := range card.Releases {
		candidate := &card.Releases[i]
		if candidate.Release == release && !candidate.Yanked && candidate.YankedAt == "" {
			selected = candidate
			break
		}
	}
	if selected == nil {
		return empty, exit.New(exit.NotFound, "model %s has no available release %q", ref.String(), release)
	}
	manifestLanes := map[string][]string{}
	manifestBytes := map[string]int64{}
	for _, lane := range selected.Lanes {
		if (manifest == "" || lane.ManifestID == manifest) &&
			(wantedLane == "" || lane.Lane == wantedLane) {
			manifestLanes[lane.ManifestID] = append(manifestLanes[lane.ManifestID], lane.Lane)
			manifestBytes[lane.ManifestID] = lane.Bytes
		}
	}
	if manifest != "" && len(manifestLanes[manifest]) == 0 {
		return empty, exit.New(exit.NotFound, "model %s@%s does not contain manifest %s",
			ref.String(), release, manifest)
	}
	if manifest == "" && wantedLane != "" && len(manifestLanes) == 0 {
		return empty, exit.New(exit.NotFound, "model %s@%s has no lane %q",
			ref.String(), release, wantedLane)
	}
	if manifest == "" {
		if len(manifestLanes) != 1 {
			return empty, exit.Usagef("model %s@%s has %d manifests", ref.String(), release, len(manifestLanes)).
				WithRemedy("append #sha256:<digest> to select one exact manifest")
		}
		for digest := range manifestLanes {
			manifest = digest
		}
	}
	if _, err := canonical.Raw(manifest); err != nil {
		return empty, exit.Named(exit.Conflict, "rental.model_manifest_invalid",
			"Tensorhub returned an invalid manifest for %s@%s", ref.String(), release)
	}
	lanes := manifestLanes[manifest]
	sort.Strings(lanes)
	return orchestrator.ModelRef{Package: packageName, Slot: slotPath,
		Model: ref.String(), Release: release, Lane: lanes[0], Manifest: manifest,
		Bytes: manifestBytes[manifest]}, nil
}

func handleRunCancel(ctx *Context) *exit.Error {
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
	if before.Kind == "job" {
		return handleJobCancel(ctx)
	}
	if invocationSettled(before.Status) {
		fields := append(invocationFields(before), output.Field{K: "changed", V: false})
		return emit(ctx, compactRecord(fields, "number", "target", "status", "changed"))
	}
	if problem := client.Cancel(id, "cozy run cancel"); problem != nil {
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
	defaults := []string{"number", "target", "status", "changed"}
	if after.CanceledBy != "" {
		defaults = append(defaults, "canceled_by")
	}
	return emit(ctx, compactRecord(fields, defaults...))
}

func handleRunList(ctx *Context) *exit.Error {
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
	explicit, disabled := ctx.Inv.Bool("--watch"), ctx.Inv.Bool("--no-watch")
	if explicit && disabled {
		return exit.Usagef("--watch and --no-watch cannot be used together")
	}
	watching := explicit || ctx.Mode().TTY && !disabled
	if watching && (ctx.Mode().JSON || !ctx.Mode().TTY) {
		return exit.Usagef("--watch requires interactive terminal output").
			WithRemedy("omit --watch for one snapshot, or use --json for automation")
	}
	if watching {
		return watchRunList(ctx, client, limit)
	}
	list, problem := runList(context.Background(), client, ctx.Inv.Value("--state"), ctx.Inv.Value("--package"), limit)
	if problem != nil {
		return problem
	}
	return emit(ctx, list)
}

func runList(requestCtx context.Context, client *localapi.Client, state, packageName string, limit int) (output.List, *exit.Error) {
	pkg := strings.TrimSpace(packageName)
	rows, problem := client.Requests(requestCtx, state, pkg, limit)
	if problem != nil {
		return output.List{}, problem
	}
	list := output.List{
		Name: "invocations", Fields: []string{"number", "target", "machine", "status", "completion", "execution"},
		AllFields: []string{"number", "id", "kind", "target", "machine", "rental_id", "status", "completion", "progress_stage", "queued", "execution", "attempts", "created"},
		// The raw rental id is a machine fact: JSON always carries it, the compact
		// human table never does — the human word is the MACHINE column (cl-107).
		Machine: []string{"rental_id"},
	}
	states := map[string]int{}
	for _, life := range rows {
		kind := life.Kind
		if kind == "" {
			kind = "invocation"
		}
		// A canceled run is LOUD about its cause (cl-108): the status cell itself names
		// the recorded actor, so a list is never a quiet no-output ending.
		status := life.Status
		if life.Status == "canceled" && life.CanceledBy != "" {
			status = "canceled by " + life.CanceledBy
		}
		list.Rows = append(list.Rows, map[string]string{
			"number": strconv.FormatInt(life.Number, 10), "id": life.RequestID, "kind": kind,
			"target": life.Package + "/" + life.Function, "machine": life.Machine,
			"rental_id":  life.RentalID,
			"status":     status,
			"completion": completion(life), "progress_stage": life.ProgressStage,
			"queued":    seconds(life.QueuedMS),
			"execution": seconds(life.ExecutionMS),
			"attempts":  strconv.Itoa(life.Attempts), "created": life.CreatedAt,
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
	return list, nil
}

func completion(life api.Lifecycle) string {
	if life.Status != "in_progress" || life.Completion == nil {
		return ""
	}
	value := fmt.Sprintf("%.0f%%", *life.Completion*100)
	if life.RemainingMS != nil {
		value += " · ~" + shortDuration(time.Duration(*life.RemainingMS)*time.Millisecond)
	}
	return value
}

func watchRunList(ctx *Context, client *localapi.Client, limit int) *exit.Error {
	signalCtx, stopSignals := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	watchCtx, cancel := context.WithCancel(signalCtx)
	defer stopSignals()
	defer cancel()
	navigation := make(chan runListNavigation, 32)
	restoreInput, mouse := startRunListInput(cancel, navigation)
	controls := "\x1b[?1049h\x1b[?25l"
	if mouse {
		controls += "\x1b[?1000h\x1b[?1006h"
	}
	if _, err := io.WriteString(ctx.Out, controls); err != nil {
		restoreInput()
		return exit.As(err)
	}
	defer func() {
		if mouse {
			_, _ = io.WriteString(ctx.Out, "\x1b[?1006l\x1b[?1000l")
		}
		_, _ = io.WriteString(ctx.Out, "\x1b[?25h\x1b[?1049l")
		restoreInput()
	}()
	var viewport runListViewport
	draw := func() *exit.Error {
		list := viewport.page(terminalHeight(ctx.Out), ctx.Mode().Full)
		var frame strings.Builder
		if err := list.Emit(&frame, ctx.Mode()); err != nil {
			return exit.As(err)
		}
		body := frame.String()
		if mouse {
			// MakeRaw disables terminal newline translation. An explicit carriage return keeps
			// every table row in column one on every refresh.
			body = strings.ReplaceAll(body, "\n", "\r\n")
		}
		if _, err := fmt.Fprintf(ctx.Out, "\x1b[H\x1b[J%s", body); err != nil {
			return exit.As(err)
		}
		return nil
	}
	refresh := func() *exit.Error {
		list, problem := runList(watchCtx, client, ctx.Inv.Value("--state"), ctx.Inv.Value("--package"), limit)
		if problem != nil {
			return problem
		}
		viewport.update(list)
		return draw()
	}
	if problem := refresh(); problem != nil {
		if watchCtx.Err() != nil {
			return nil
		}
		return problem
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-watchCtx.Done():
			return nil
		case move := <-navigation:
			viewport.move(move, terminalHeight(ctx.Out), ctx.Mode().Full)
			if problem := draw(); problem != nil {
				return problem
			}
		case <-ticker.C:
			if problem := refresh(); problem != nil {
				if watchCtx.Err() != nil {
					return nil
				}
				return problem
			}
		}
	}
}

type runListNavigation uint8

const (
	runListUp runListNavigation = iota + 1
	runListDown
	runListPageUp
	runListPageDown
	runListHome
	runListEnd
)

type runListViewport struct {
	list   output.List
	top    int
	anchor string
}

func (v *runListViewport) update(list output.List) {
	if v.top > 0 && v.anchor != "" {
		for index, row := range list.Rows {
			if row["id"] == v.anchor {
				v.top = index
				break
			}
		}
	}
	v.list = list
}

func (v *runListViewport) capacity(height int, full bool) int {
	if height <= 0 {
		height = 24
	}
	rows := max(1, height-len(v.list.Aggregates)-4)
	if !full {
		rows = min(rows, 20)
	}
	return rows
}

func (v *runListViewport) clamp(pageRows int) {
	v.top = min(max(v.top, 0), max(0, len(v.list.Rows)-pageRows))
	if v.top == 0 || len(v.list.Rows) == 0 {
		v.anchor = ""
		return
	}
	v.anchor = v.list.Rows[v.top]["id"]
}

func (v *runListViewport) move(move runListNavigation, height int, full bool) {
	pageRows := v.capacity(height, full)
	switch move {
	case runListUp:
		v.top--
	case runListDown:
		v.top++
	case runListPageUp:
		v.top -= pageRows
	case runListPageDown:
		v.top += pageRows
	case runListHome:
		v.top = 0
	case runListEnd:
		v.top = len(v.list.Rows)
	}
	v.clamp(pageRows)
}

func (v *runListViewport) page(height int, full bool) output.List {
	pageRows := v.capacity(height, full)
	v.clamp(pageRows)
	end := min(v.top+pageRows, len(v.list.Rows))
	page := v.list
	page.Rows = v.list.Rows[v.top:end]
	page.Total = len(page.Rows)
	location := "no rows"
	if end > v.top {
		location = fmt.Sprintf("rows %d-%d/%d", v.top+1, end, len(v.list.Rows))
	}
	page.Trail = []string{fmt.Sprintf(
		"1s refresh · %s · wheel/↑↓/PgUp/PgDn/Home/End · q/Ctrl-C exits", location)}
	return page
}

func terminalHeight(w io.Writer) int {
	file, ok := w.(*os.File)
	if !ok {
		return 0
	}
	_, height, err := term.GetSize(int(file.Fd()))
	if err != nil {
		return 0
	}
	return height
}

func startRunListInput(cancel context.CancelFunc, navigation chan<- runListNavigation) (func(), bool) {
	fd := int(os.Stdin.Fd()) //cozy:stdin-value live-list navigation, never a prompt
	if !term.IsTerminal(fd) {
		return func() {}, false
	}
	state, err := term.MakeRaw(fd)
	if err != nil {
		return func() {}, false
	}
	go readRunListInput(bufio.NewReader(os.Stdin), cancel, navigation) //cozy:stdin-value navigation only
	return func() { _ = term.Restore(fd, state) }, true
}

func readRunListInput(in *bufio.Reader, cancel context.CancelFunc, navigation chan<- runListNavigation) {
	send := func(move runListNavigation) {
		select {
		case navigation <- move:
		default:
		}
	}
	for {
		key, err := in.ReadByte()
		if err != nil {
			return
		}
		switch key {
		case 3, 'q', 'Q':
			cancel()
			return
		case 'k':
			send(runListUp)
		case 'j':
			send(runListDown)
		case 'g':
			send(runListHome)
		case 'G':
			send(runListEnd)
		case 0x1b:
			if next, err := in.ReadByte(); err == nil && next == '[' {
				if sequence := readRunListCSI(in); sequence != "" {
					switch {
					case sequence == "A":
						send(runListUp)
					case sequence == "B":
						send(runListDown)
					case sequence == "5~":
						send(runListPageUp)
					case sequence == "6~":
						send(runListPageDown)
					case sequence == "H", sequence == "1~", sequence == "7~":
						send(runListHome)
					case sequence == "F", sequence == "4~", sequence == "8~":
						send(runListEnd)
					case strings.HasPrefix(sequence, "<64;") && strings.HasSuffix(sequence, "M"):
						for range 3 {
							send(runListUp)
						}
					case strings.HasPrefix(sequence, "<65;") && strings.HasSuffix(sequence, "M"):
						for range 3 {
							send(runListDown)
						}
					}
				}
			}
		}
	}
}

func readRunListCSI(in *bufio.Reader) string {
	var sequence strings.Builder
	for sequence.Len() < 64 {
		value, err := in.ReadByte()
		if err != nil {
			return ""
		}
		sequence.WriteByte(value)
		if value >= 0x40 && value <= 0x7e {
			return sequence.String()
		}
	}
	return ""
}

func invocationFields(life api.Lifecycle) []output.Field {
	kind := life.Kind
	if kind == "" {
		kind = "invocation"
	}
	fields := []output.Field{
		{K: "number", V: life.Number}, {K: "id", V: life.RequestID}, {K: "kind", V: kind},
		{K: "target", V: life.Package + "/" + life.Function},
		{K: "machine", V: life.Machine},
		{K: "rental_id", V: life.RentalID},
		{K: "status", V: life.Status}, {K: "attempts", V: life.Attempts},
	}
	if life.CanceledBy != "" {
		fields = append(fields, output.Field{K: "canceled_by", V: life.CanceledBy})
	}
	return fields
}

func invocationSettled(status string) bool {
	switch status {
	case "completed", "failed", "canceled":
		return true
	default:
		return false
	}
}

// optimisticObservation is long enough for a warm, weightless callable to answer in the
// foreground without turning an ordinary GPU submission into a terminal hostage. Expiry
// never cancels the request and never predicts its duration: the daemon keeps owning it.
const optimisticObservation = 3 * time.Second

// observe follows the real event stream for one bounded optimistic window. A context
// expiry merely detaches this client; it does not enter the cancellation path used by an
// explicit --await. A terminal that arrives inside the window is still rendered through
// the ordinary result/error path, including a fast refusal.
func observe(ctx *Context, c *localapi.Client, requestID string, stream bool,
	window time.Duration, began time.Time,
) (*localapi.Event, *exit.Error) {
	watchCtx, stop := context.WithTimeout(context.Background(), window)
	defer stop()
	lines := NewProgress(ctx, stream, began)
	terminal, problem := c.WatchContext(watchCtx, requestID, 0, lines.On)
	lines.Done()
	return terminal, problem
}

func renderSubmittedRun(ctx *Context, life api.Lifecycle, changed bool) *exit.Error {
	status := runStatus(life.Status)
	reference := runReference(life.Number, life.RequestID)
	fields := []output.Field{
		{K: "run", V: reference}, {K: "id", V: life.RequestID},
		{K: "target", V: life.Package + "/" + life.Function},
		{K: "machine", V: life.Machine},
		{K: "rental_id", V: life.RentalID},
		{K: "status", V: status},
	}
	defaults := []string{"target", "status"}
	if life.Machine != "" {
		defaults = append(defaults, "machine")
	}
	if life.QueuePosition != nil {
		queue := strconv.Itoa(*life.QueuePosition)
		if life.QueueDepth != nil && *life.QueueDepth >= *life.QueuePosition {
			queue += "/" + strconv.Itoa(*life.QueueDepth)
		}
		fields = append(fields, output.Field{K: "queue_position", V: queue})
		defaults = append(defaults, "queue_position")
	}
	if export := life.OutputExport; export != nil {
		fields = append(fields, output.Field{K: "output", V: outputExportHint(ctx.Mode(), export)})
		defaults = append(defaults, "output")
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

func waitOutputExport(c *localapi.Client, life api.Lifecycle) (api.Lifecycle, *exit.Error) {
	// Publication is an obligation of a SUCCESSFUL execution. A failed or canceled run
	// has no accepted output bytes to publish, so its terminal must reach the caller even
	// if an older daemon left the separately-settled export row pending. This is also the
	// client-side fence for the terminal/export race: execution failure is already an
	// absorbing answer and can never become success by waiting on that row.
	if life.Status == "failed" || life.Status == "canceled" {
		return life, nil
	}
	for life.OutputExport != nil {
		export := life.OutputExport
		switch export.State {
		case "published", "skipped":
			return life, nil
		case "failed":
			// A failed export is NOT a run failure: the run's own terminal stands, the
			// bytes are safe in internal media, and the export stays a durable obligation
			// the daemon retries. renderRun reports the completed-with-export-owed verdict.
			return life, nil
		case "pending", "exporting":
			time.Sleep(50 * time.Millisecond)
			updated, problem := c.Request(life.RequestID)
			if problem != nil {
				return life, problem
			}
			life = updated
		default:
			return life, exit.Internalf("run %s has unknown output export state %q",
				life.RequestID, export.State)
		}
	}
	return life, nil
}

func outputExportHint(mode output.Mode, export *api.OutputExportRef) string {
	if export == nil {
		return ""
	}
	if len(export.Paths) > 0 {
		paths := make([]string, 0, len(export.Paths))
		for _, path := range export.Paths {
			paths = append(paths, mode.Hyperlink(path))
		}
		return strings.Join(paths, ", ")
	}
	return mode.Hyperlink(export.Directory)
}

// outputDirectoryHint is where this run's files will land: the caller's --out, else the
// package's own store — the same directory the daemon derives, spelled by the one layout.
func outputDirectoryHint(ctx *Context, target Target, explicit string) string {
	if explicit != "" {
		return explicit
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return filepath.Join(ctx.Cfg.Home, "outputs")
	}
	return layout.PackageOutputs(target.Package)
}

// errLink hyperlinks a path for the stderr progress lane. The gate is stderr's
// own fd: the result writer's mode says nothing about where progress goes.
func errLink(ctx *Context, path string) string {
	file, ok := ctx.Err.(*os.File)
	if !ok || ctx.Mode().JSON ||
		!(isatty.IsTerminal(file.Fd()) || isatty.IsCygwinTerminal(file.Fd())) {
		return path
	}
	mode := ctx.Mode()
	mode.TTY = true
	return mode.Hyperlink(path)
}

// savedFile is one exported result file as `cozy run` reports it: the path a person
// opens, and the facts a program wants beside it.
type savedFile struct {
	Output string `json:"output"`
	Path   string `json:"path"`
	Bytes  int64  `json:"bytes"`
	Mime   string `json:"mime"`
	Digest string `json:"digest"`
}

// exportedOutputs joins the published paths to the outputs they carry. A file is named
// by its content digest, so the join is exact rather than positional.
func exportedOutputs(life api.Lifecycle) []savedFile {
	if life.OutputExport == nil || life.OutputExport.State != "published" {
		return nil
	}
	byDigest := make(map[string]api.MediaRef, len(life.Outputs))
	for _, o := range life.Outputs {
		byDigest[strings.TrimPrefix(o.Digest, "sha256:")] = o
	}
	result := make([]savedFile, 0, len(life.OutputExport.Paths))
	for _, path := range life.OutputExport.Paths {
		stem := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		row := savedFile{Path: path}
		if o, ok := byDigest[stem]; ok {
			row.Output, row.Bytes, row.Mime, row.Digest = o.OutputID, o.Length, o.MimeType, o.Digest
		}
		result = append(result, row)
	}
	return result
}

func runStatus(status string) string {
	if status == "in_progress" {
		return "running"
	}
	return status
}

// watch consumes the request's own event stream to its terminal, rendering progress as
// it goes. A SIGNAL NEVER CANCELS THE RUN (cl-108): SIGINT/SIGTERM merely DETACHES this
// client — the daemon owns the accepted run, which keeps running, exports, and stays
// watchable — because a dying watcher is not a person asking for cancellation, and the
// recurring "CLIENT_CANCELED on disconnect" silent-cancel is exactly what that
// translation produced. Only `cozy run cancel` and the caller-authored `--timeout`
// deadline cancel, and both are attributed.
// watch returns the terminal event and WHY the client stopped waiting, which is not the
// same question as what the terminal says: a canceled terminal caused by `--timeout` is
// exit 10, because a caller that set a deadline wants to know the deadline is what
// happened.
func watch(ctx *Context, c *localapi.Client, requestID string, stream bool,
	deadline time.Duration, began time.Time) (*localapi.Event, string, *exit.Error) {
	interrupt := make(chan os.Signal, 2)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)

	watchCtx, stopWatch := context.WithCancel(context.Background())
	defer stopWatch()
	stopped := make(chan string, 1)
	cancelFailed := make(chan *exit.Error, 1)
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case _, ok := <-interrupt:
			if !ok {
				return
			}
			stopped <- "detached"
			fmt.Fprintf(ctx.Err,
				"\ndetached — the run keeps running; `cozy run watch %s` reattaches, `cozy run cancel %s` cancels\n",
				requestID, requestID)
			stopWatch()
			return
		case <-deadlineC(deadline):
			// `--timeout` is a REQUEST DEADLINE the client enforces the only way a client
			// honestly can: by asking the orchestrator to cancel. It is not the
			// supervisor's watchdog deadline (that one is on the attempt, and this host
			// has no wire field for it) — walking away instead would leave the card held.
			stopped <- "deadline"
			fmt.Fprintf(ctx.Err,
				"\n--timeout %s expired; cancel requested — the attempt's own terminal still settles it\n",
				deadline)
		case <-done:
			return
		}
		cancelResult := make(chan *exit.Error, 1)
		go func() { cancelResult <- c.Cancel(requestID, fmt.Sprintf("cozy run --timeout %s", deadline)) }()
		select {
		case <-interrupt:
			fmt.Fprintln(ctx.Err, "detached — the canceled terminal still lands in `cozy run list`")
			stopWatch()
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
			fmt.Fprintln(ctx.Err, "detached — the canceled terminal still lands in `cozy run list`")
			stopWatch()
		case <-done:
		}
	}()

	lines := NewProgress(ctx, stream, began)
	terminal, e := c.WatchContext(watchCtx, requestID, 0, lines.On)
	lines.Done()
	select {
	case problem := <-cancelFailed:
		return nil, "cancel_failed", problem
	default:
	}
	reason := ""
	select {
	case reason = <-stopped:
	default:
	}
	return terminal, reason, e
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

// progress renders the live lane. Three shapes, and they are not the same surface:
// `--stream` is NDJSON of the typed envelope for a machine, a terminal gets ONE
// carriage-return-rewritten line, and a redirected human command gets sparse
// append-only lines — a stage change, each new tenth of the work, one line per five
// quiet seconds — never the full lossy tick stream. --json stays untouched. Exported —
// with HumanWaitLine and WaitPatience — so the product suite (#661: verification's one
// home) drives this exact render path.
type RunProgress struct {
	ctx         *Context
	stream      bool
	mu          sync.Mutex
	last        string
	dirty       bool
	closed      bool
	began       time.Time
	waitedSince time.Time
	waitEvent   localapi.Event
	patience    *time.Timer
	stepStage   string
	stepSeconds float64
	stepSamples int

	// The sparse lane's memory: which tenth of which stage was last appended, and when.
	sparseStage  string
	sparseDecile int
	sparseAt     time.Time

	// Injected clock and measure, so tests drive the REAL renderer deterministically.
	now   func() time.Time
	width func() int
}

func NewProgress(ctx *Context, stream bool, began time.Time) *RunProgress {
	return &RunProgress{
		ctx: ctx, stream: stream, began: began, sparseDecile: -1,
		now: time.Now, width: func() int { return terminalWidth(ctx.Err) },
	}
}

func (p *RunProgress) On(e localapi.Event) bool {
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
	p.mu.Lock()
	defer p.mu.Unlock()
	p.observeWait(e)
	// A redirected human command has no status line to rewrite: it gets the sparse
	// append lane. --full deliberately restores the complete diagnostic stream.
	if !p.ctx.Mode().Color && !p.ctx.Mode().Full {
		p.sparse(e)
		return true
	}
	p.render(p.line(e, p.ctx.Mode().Full))
	return true
}

// render rewrites the status line. Callers hold p.mu.
func (p *RunProgress) render(line string) {
	if line == "" || line == p.last {
		return
	}
	p.last, p.dirty = line, true
	// stderr, deliberately: stdout carries the RESULT, so a piped `cozy run` is not
	// polluted by the progress of producing it.
	if p.ctx.Mode().Color {
		// The one in-place line. Clamped to the CURRENT terminal width on every write:
		// a line that wraps leaves rows \r can never reach again, so on a narrow or
		// just-resized terminal the tail is dropped instead.
		fmt.Fprintf(p.ctx.Err, "\r\033[K%s", clampLine(line, p.width()))
	} else {
		fmt.Fprintln(p.ctx.Err, line)
	}
}

// observeWait keeps the wait clock. It starts on a queued/parked event — at the event's
// own recorded time, so a reattached watcher inherits the wait already served — and
// stops on any event that says the queue let go of the request. Callers hold p.mu.
func (p *RunProgress) observeWait(e localapi.Event) {
	switch strings.TrimPrefix(e.Type, "request.") {
	case "queued", "parked":
		p.waitEvent = e
		if p.waitedSince.IsZero() {
			p.waitedSince = eventTime(e)
			p.armPatience()
		}
	case "rentals", "log", "metric":
		// Still the same wait; these narrate it without ending it.
	default:
		p.waitedSince = time.Time{}
		p.disarmPatience()
	}
}

// armPatience schedules the one time-driven render: a wait that outlives WaitPatience
// re-renders with the diagnostic even when no new event arrives — the stuck case emits
// exactly one parked event and then silence. Only the human status line needs it; the
// diagnostic surfaces (--full, --stream, --json) carry the detail from the start.
// Callers hold p.mu.
func (p *RunProgress) armPatience() {
	if !p.ctx.Mode().Color || p.ctx.Mode().Full {
		return
	}
	remaining := max(WaitPatience-time.Since(p.waitedSince), 0)
	p.patience = time.AfterFunc(remaining, func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		if p.closed || p.waitedSince.IsZero() {
			return
		}
		p.render(p.waitLine(p.waitEvent))
	})
}

func (p *RunProgress) disarmPatience() {
	if p.patience != nil {
		p.patience.Stop()
		p.patience = nil
	}
}

// eventTime is the event's own recorded time, this process's clock when it carries none.
func eventTime(e localapi.Event) time.Time {
	if at, err := time.Parse(time.RFC3339Nano, e.At); err == nil {
		return at
	}
	return time.Now()
}

// sparse appends at most a few lines per stage to a redirected human stream: pass a
// non-progress status line once, and pass step telemetry only on a stage change, on
// each new tenth of the work, or after five quiet seconds.
func (p *RunProgress) sparse(e localapi.Event) {
	if strings.TrimPrefix(e.Type, "request.") != "progress" {
		p.appendOnce(progressLine(e, false))
		return
	}
	fields, ok := e.Payload["value"].(map[string]any)
	if !ok {
		p.appendOnce(humanProgress(e.Payload["value"]))
		return
	}
	facts, ok := p.observe(fields)
	if !ok {
		p.appendOnce(humanStage(map[string]any{"name": stageLabel(facts.label)}))
		return
	}
	decile := int(facts.fraction * 10)
	stale := !p.sparseAt.IsZero() && p.now().Sub(p.sparseAt) >= 5*time.Second
	if facts.label == p.sparseStage && decile == p.sparseDecile && !stale {
		return
	}
	p.sparseStage, p.sparseDecile, p.sparseAt = facts.label, decile, p.now()
	line := "  " + facts.label
	if facts.counted {
		line += fmt.Sprintf(" %d/%d", facts.current, facts.total)
	}
	line += fmt.Sprintf(" · %.0f%% · elapsed %s", facts.fraction*100, shortDuration(p.now().Sub(p.began)))
	fmt.Fprintln(p.ctx.Err, line)
}

func (p *RunProgress) appendOnce(line string) {
	if line == "" || line == p.last {
		return
	}
	p.last = line
	fmt.Fprintln(p.ctx.Err, line)
}

// stepFacts is one progress frame read through the accumulator: a display label, the
// fraction, and — when the frame counts steps — current/total plus mean seconds per step.
type stepFacts struct {
	label    string
	fraction float64
	counted  bool
	current  int64
	total    int64
	perStep  float64
}

// observe folds one progress payload into the per-stage step-time accumulator and
// returns the frame's facts; ok is false when the frame carries no usable fraction.
func (p *RunProgress) observe(fields map[string]any) (stepFacts, bool) {
	name, _ := fields["name"].(string)
	name = strings.TrimSpace(name)
	fraction, fractionOK := number(fields["fraction"])
	position, positionOK := number(fields["position"])
	stepMS, stepOK := number(fields["step_ms"])
	if name != "" && name != p.stepStage {
		p.stepStage, p.stepSeconds, p.stepSamples = name, 0, 0
	}
	if stepOK && stepMS >= 0 {
		p.stepSeconds += stepMS / 1000
		p.stepSamples++
	}
	label := stageLabel(name)
	if label == "" {
		label = "running"
	}
	facts := stepFacts{label: label, fraction: fraction}
	if !fractionOK || fraction < 0 || fraction > 1 {
		facts.label = name
		return facts, false
	}
	if positionOK && position > 0 && fraction > 0 {
		facts.counted = true
		facts.current = int64(position)
		facts.total = int64(position/fraction + 0.5)
		if facts.total < facts.current {
			facts.total = facts.current
		}
	}
	if p.stepSamples > 0 {
		facts.perStep = p.stepSeconds / float64(p.stepSamples)
	}
	return facts, true
}

func (p *RunProgress) line(e localapi.Event, full bool) string {
	if full {
		return diagnosticProgressLine(e)
	}
	switch strings.TrimPrefix(e.Type, "request.") {
	case "queued", "parked":
		return p.waitLine(e)
	}
	if strings.TrimPrefix(e.Type, "request.") != "progress" {
		return progressLine(e, false)
	}
	fields, ok := e.Payload["value"].(map[string]any)
	if !ok {
		return humanProgress(e.Payload["value"])
	}
	facts, ok := p.observe(fields)
	if !ok {
		return humanStage(map[string]any{"name": stageLabel(facts.label)})
	}
	elapsed := p.now().Sub(p.began)
	bar := progressBar(facts.fraction, 18)
	if !facts.counted {
		return fmt.Sprintf("  %s %s %.0f%% · %s", facts.label, bar, facts.fraction*100, shortDuration(elapsed))
	}
	line := fmt.Sprintf("  %s %s %d/%d · %.0f%%", facts.label, bar, facts.current, facts.total, facts.fraction*100)
	if facts.perStep > 0 {
		line += fmt.Sprintf(" · %.2fs/step · %.2f steps/s", facts.perStep, 1/facts.perStep)
		remaining := time.Duration(float64(facts.total-facts.current) * facts.perStep * float64(time.Second))
		return line + fmt.Sprintf(" · elapsed %s · ETA ~%s", shortDuration(elapsed), shortDuration(remaining))
	}
	return line + " · elapsed " + shortDuration(elapsed)
}

// clampLine bounds one rewritten status line to the terminal: a wrapped line leaves
// debris \r cannot reach.
func clampLine(line string, width int) string {
	if width <= 0 {
		return line
	}
	runes := []rune(line)
	if len(runes) < width {
		return line
	}
	return string(runes[:width-1])
}

func progressBar(fraction float64, width int) string {
	filled := int(fraction*float64(width) + 0.5)
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}
	return "[" + strings.Repeat("=", filled) + strings.Repeat(".", width-filled) + "]"
}

func stageLabel(name string) string {
	switch strings.TrimSpace(name) {
	case "tokenize", "encode", "condition", "condition_text", "condition_media":
		return "conditioning"
	case "denoise":
		return "denoising"
	case "decode", "decode_image", "decode_video", "decode_audio":
		return "decoding"
	case "encode_png", "encode_webp", "encode_outputs":
		return "saving"
	default:
		return strings.TrimSpace(name)
	}
}

func shortDuration(value time.Duration) string {
	if value < time.Second {
		return fmt.Sprintf("%.1fs", value.Seconds())
	}
	return value.Round(time.Second).String()
}

func (p *RunProgress) Done() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	p.disarmPatience()
	if p.dirty && p.ctx.Mode().Color {
		fmt.Fprintln(p.ctx.Err)
	}
}

// progressLine is the human projection of one event. Runtime's exact typed envelope stays
// available through --stream and --full; the ordinary status line shows only a named stage
// or completion percentage. Timing samples and metrics are diagnostics, not user progress.
func progressLine(e localapi.Event, full bool) string {
	if full {
		return diagnosticProgressLine(e)
	}
	kind := strings.TrimPrefix(e.Type, "request.")
	switch kind {
	case "progress":
		return humanProgress(e.Payload["value"])
	case "stage":
		return humanStage(e.Payload["value"])
	case "metric":
		return ""
	case "queued", "parked":
		return HumanWaitLine(e.Payload)
	case "rentals":
		if line, ok := e.Payload["line"].(string); ok {
			return line
		}
	case "submitted":
		return ""
	case "dispatched":
		return "  worker selected"
	case "accepted":
		return "  running"
	case "requeued":
		return "  retrying"
	case "attempt_failed":
		return "  attempt failed; retrying"
	default:
		return ""
	}
	return ""
}

func humanProgress(value any) string {
	name := "running"
	fraction, ok := number(value)
	if fields, isMap := value.(map[string]any); isMap {
		if named, exists := fields["name"].(string); exists && strings.TrimSpace(named) != "" {
			name = strings.TrimSpace(named)
		} else if named, exists := fields["stage"].(string); exists && strings.TrimSpace(named) != "" {
			name = strings.TrimSpace(named)
		}
		for _, key := range []string{"fraction", "value"} {
			if fraction, ok = number(fields[key]); ok {
				break
			}
		}
	}
	if !ok || fraction < 0 || fraction > 1 {
		return ""
	}
	return fmt.Sprintf("  %s — %.0f%%", name, fraction*100)
}

// WaitPatience is how long a wait stays a calm one-liner (cl-103). Past it, the
// dispatcher's own diagnostic joins the line: a long wait is the abnormal case, and the
// detail is how a person sees exactly what the queue is stuck on. The full diagnostic is
// always in --full and --stream/--json regardless.
const WaitPatience = 90 * time.Second

// HumanWaitLine says what the queue is DOING, never how it thinks: the event's stable
// `wait` cause becomes a calm stage line with no digests and no dispatcher vocabulary.
// The verbatim diagnostic stays in the payload's `reason` for the machine surfaces.
func HumanWaitLine(payload map[string]any) string {
	pkg, _ := payload["package"].(string)
	on, _ := payload["waiting_on"].(string)
	cause, _ := payload["wait"].(string)
	switch cause {
	case orchestrator.WaitWorkerStart:
		if pkg != "" {
			return "  starting a worker for " + pkg
		}
		return "  starting a worker"
	case orchestrator.WaitWorkerWarming:
		if on != "" {
			return "  warming the model on " + on
		}
		return "  warming the model"
	case orchestrator.WaitSlotBusy:
		if on != "" {
			return "  waiting for a free slot on " + on
		}
		return "  waiting for a free slot"
	case orchestrator.WaitQueueAhead:
		if position, ok := number(payload["position"]); ok && position > 1 {
			return fmt.Sprintf("  waiting in line — position %.0f", position)
		}
		return "  waiting in line"
	case orchestrator.WaitRental:
		return "  waiting for a rental machine"
	case orchestrator.WaitModelTransfer:
		return "  downloading the model"
	}
	return "  waiting for capacity"
}

// waitLine is HumanWaitLine plus patience: once the wait outlives WaitPatience the raw
// diagnostic earns its place on the human line too. Callers hold p.mu.
func (p *RunProgress) waitLine(e localapi.Event) string {
	line := HumanWaitLine(e.Payload)
	if p.waitedSince.IsZero() || time.Since(p.waitedSince) < WaitPatience {
		return line
	}
	reason, _ := e.Payload["reason"].(string)
	if reason == "" {
		return line
	}
	return fmt.Sprintf("%s — %s so far: %s",
		line, shortDuration(time.Since(p.waitedSince)), reason)
}

func humanStage(value any) string {
	if name, ok := value.(string); ok && strings.TrimSpace(name) != "" {
		return "  " + strings.TrimSpace(name)
	}
	if fields, ok := value.(map[string]any); ok {
		if name, ok := fields["name"].(string); ok && strings.TrimSpace(name) != "" {
			return "  " + strings.TrimSpace(name)
		}
	}
	return ""
}

func number(value any) (float64, bool) {
	switch value := value.(type) {
	case float64:
		return value, true
	case float32:
		return float64(value), true
	case int:
		return float64(value), true
	case int64:
		return float64(value), true
	case json.Number:
		parsed, err := value.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

// diagnosticProgressLine preserves the old lossless human spelling for --full. The
// machine surface remains --stream, which emits the complete JSON envelope unchanged.
func diagnosticProgressLine(e localapi.Event) string {
	kind := strings.TrimPrefix(e.Type, "request.")
	switch kind {
	case "progress", "stage", "metric":
		if v, ok := e.Payload["value"].(map[string]any); ok {
			return "  " + strings.TrimSpace(fmt.Sprintf("%s %s", kind, compactValue(v)))
		}
	case "queued", "parked":
		if reason, ok := e.Payload["reason"].(string); ok {
			return "  " + kind + " — " + reason
		}
		return "  " + kind
	case "rentals":
		if line, ok := e.Payload["line"].(string); ok {
			return line
		}
	default:
		return ""
	}
	return "  " + kind + " " + compactValue(e.Payload)
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

// opaqueType is the type an output carries when nobody declared one.
const opaqueType = "application/octet-stream"

// renderRun prints the run's answer and maps the terminal onto the SHARED matrix:
// succeeded 0 · failed 11 · canceled 12 · deadline 10, from `exit.JobTerminal`.
func renderRun(ctx *Context, life api.Lifecycle, terminal *localapi.Event, stopped string,
	saved []savedFile, submitted time.Duration, began time.Time) *exit.Error {
	status := localapi.StreamStatus(terminal)
	if status == "" {
		status = life.Status
	}
	if stopped == "deadline" && status == "canceled" {
		// The DEADLINE is why this ended, and the shared matrix has a code for it.
		status = "deadline"
	}
	// A SUCCEEDED run whose export failed is a completed run with an export still owed —
	// the daemon retries the durable obligation — and it must never wear a run-failure
	// verdict. The status says both facts; the exit code stays the run terminal's own.
	export := life.OutputExport
	exportOwed := export != nil && export.State == "failed" && mapTerminal(status) == "succeeded"
	shownStatus := life.Status
	if exportOwed {
		shownStatus = life.Status + " (export pending: " + export.ErrorCode + ")"
	}
	if life.Status == "canceled" && life.CanceledBy != "" {
		shownStatus = "canceled by " + life.CanceledBy
	}
	fields := []output.Field{
		{K: "number", V: life.Number}, {K: "id", V: life.RequestID},
		{K: "target", V: life.Package + "/" + life.Function},
		{K: "package", V: life.Package},
		{K: "function", V: life.Function},
		{K: "machine", V: life.Machine},
		{K: "rental_id", V: life.RentalID},
		{K: "status", V: shownStatus},
		{K: "attempts", V: life.Attempts},
	}
	if life.CanceledBy != "" {
		fields = append(fields, output.Field{K: "canceled_by", V: life.CanceledBy})
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
	if exportOwed {
		notes = append(notes, fmt.Sprintf(
			"output export to %s failed (%s): %s — fix the recorded destination and repeat "+
				"the same idempotency key; the daemon retries this durable export",
			export.Directory, export.ErrorCode, export.Error))
	}
	if len(saved) > 0 {
		paths := make([]string, 0, len(saved))
		opaque := false
		for _, s := range saved {
			paths = append(paths, ctx.Mode().Hyperlink(s.Path)+" ("+output.Bytes(s.Bytes)+")")
			opaque = opaque || s.Mime == opaqueType || s.Mime == ""
		}
		if ctx.Mode().JSON {
			fields = append(fields, output.Field{K: "saved", V: saved})
		} else {
			fields = append(fields, output.Field{K: "saved", V: paths})
		}
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
	// Two facts, never one sum: how long the request waited, and how long it ran.
	fields = append(fields,
		output.Field{K: "queued", V: seconds(life.QueuedMS)},
		output.Field{K: "execution", V: seconds(life.ExecutionMS)},
		output.Field{K: "submit_ms", V: submitted.Milliseconds()},
		output.Field{K: "wall_ms", V: time.Since(began).Milliseconds()})

	defaults := []string{"target", "status"}
	if life.Machine != "" {
		defaults = append(defaults, "machine")
	}
	if life.Result != nil {
		defaults = append(defaults, "result")
	}
	if len(saved) > 0 {
		defaults = append(defaults, "saved")
	}
	defaults = append(defaults, "queued", "execution")
	rec := compactRecord(fields, defaults...)
	if ctx.Mode().Human && !ctx.Mode().Full && len(saved) > 0 {
		rec.Fields = expandSavedResult(rec.Fields)
	}
	rec.Notes = notes
	code := exit.JobTerminal(mapTerminal(status))
	if code == exit.OK {
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
	// A canceled run is LOUD about WHO ended it (cl-108) — never a quiet no-output end.
	if mapTerminal(status) == "canceled" && life.CanceledBy != "" {
		e.Message = fmt.Sprintf("request %s was canceled by %s", life.RequestID, life.CanceledBy)
		if why != "" {
			e.Message += ": " + why
		}
	}
	if life.Triage != nil {
		e.WithRemedy("%s", triageRemedy(ctx, life.Triage, terminal))
	}
	return e
}

// triageRemedy names the one document that explains a failed attempt, and says so when
// it is not there. A kept bundle is a path on THIS host — a pod's bundle is fetched over
// its media plane and kept here like a local worker's (cl-101) — and the traceback's tail
// is the part a person reads first, so it is quoted rather than pointed at. A bundle that
// was not kept names the fault the daemon recorded instead of pretending one exists.
func triageRemedy(ctx *Context, ref *api.TriageRef, terminal *localapi.Event) string {
	if !ref.Kept {
		fault := eventText(terminal, "triage_fault")
		if fault == "" {
			fault = "no reason recorded"
		}
		return fmt.Sprintf("triage bundle %s was NOT kept: %s", ref.SubjectID, fault)
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return "the retained triage bundle explains it"
	}
	path := layout.TriageFile(ref.SubjectID)
	remedy := "triage bundle kept at " + ctx.Mode().Hyperlink(path)
	data, err := os.ReadFile(path)
	if err != nil {
		return remedy
	}
	var bundle struct {
		Terminal struct {
			Traceback string `json:"traceback"`
		} `json:"terminal"`
	}
	if json.Unmarshal(data, &bundle) != nil || strings.TrimSpace(bundle.Terminal.Traceback) == "" {
		return remedy
	}
	lines := strings.Split(strings.TrimRight(bundle.Terminal.Traceback, "\n"), "\n")
	if len(lines) > 6 {
		lines = lines[len(lines)-6:]
	}
	return remedy + "; the executor's traceback ends:\n  " + strings.Join(lines, "\n  ")
}

func runReference(number int64, id string) string {
	if number > 0 {
		return strconv.FormatInt(number, 10)
	}
	return id
}

// expandSavedResult avoids printing an asset handle twice: once as an internal JSON
// object and again as the useful saved path. Scalar result facts remain first-class human
// fields; --json and --full retain the exact result envelope.
func expandSavedResult(fields []output.Field) []output.Field {
	result := make([]output.Field, 0, len(fields)+4)
	reserved := make(map[string]bool, len(fields))
	for _, field := range fields {
		if field.K != "result" {
			reserved[field.K] = true
		}
	}
	for _, field := range fields {
		if field.K != "result" {
			result = append(result, field)
			continue
		}
		values, ok := field.V.(map[string]any)
		if !ok {
			result = append(result, field)
			continue
		}
		keys := make([]string, 0, len(values))
		for key, value := range values {
			lower := strings.ToLower(key)
			if reserved[key] || strings.Contains(lower, "digest") || strings.HasSuffix(lower, "_ref") ||
				strings.HasSuffix(lower, "_id") || !resultScalar(value) {
				continue
			}
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			result = append(result, output.Field{K: key, V: values[key]})
		}
	}
	return result
}

func resultScalar(value any) bool {
	switch value.(type) {
	case string, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32,
		uint64, float32, float64, json.Number:
		return true
	default:
		return false
	}
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

// finalizeInputPayload closes the one intentional client-side default that changes
// output identity: an omitted top-level integer `seed`. Its value is derived from the
// already-minted idempotency key, so a normal invocation gets fresh entropy while an
// explicit-key retry reconstructs byte-identical input instead of conflicting with its
// recorded request. An explicitly supplied seed is never changed.
func finalizeInputPayload(ep *launch.Entrypoint, input json.RawMessage,
	idempotencyKey string,
) (json.RawMessage, *exit.Error) {
	var document map[string]json.RawMessage
	if err := json.Unmarshal(input, &document); err != nil || document == nil {
		return nil, exit.Internalf("cannot finalize the invocation payload: %v", err)
	}
	for _, field := range ep.Request.Fields {
		if field.Name != "seed" || string(field.Type) != `"int"` {
			continue
		}
		if _, supplied := document[field.Name]; !supplied {
			digest := sha256.Sum256([]byte("cozy-seed\x00" + idempotencyKey))
			// 53 bits stay exact in every JSON implementation used by the worker protocol.
			seed := binary.BigEndian.Uint64(digest[:8]) & ((uint64(1) << 53) - 1)
			document[field.Name] = json.RawMessage(strconv.FormatUint(seed, 10))
		}
		break
	}
	rendered, err := json.Marshal(document)
	if err != nil {
		return nil, exit.Internalf("cannot render the finalized invocation payload: %s", err)
	}
	return rendered, nil
}

// ------------------------------------------------------------------- target parsing

// Target is one installed package and optional callable selected for invocation.
type Target struct {
	Package   string
	Function  string
	InstallID string
	Release   string
}

// parseTarget reads the user-facing package grammar. Versions are flags, not path
// segments: the common spelling stays stable while installed releases change.
func parseTarget(raw string) (Target, *exit.Error) {
	parts := strings.Split(strings.TrimSpace(raw), "/")
	usage := exit.Usagef("%q is not org/package[/function]", raw).
		WithRemedy("use org/package/function; omit /function to list the package's callables").
		WithNext("cozy package list", "cozy help run")
	if len(parts) == 4 && strings.HasPrefix(parts[2], "v") {
		return Target{}, exit.Usagef("a version does not belong in the invocation path").
			WithRemedy("use %s/%s/%s; Cozy always runs the installed release", parts[0], parts[1], parts[3]).
			WithNext("cozy help run")
	}
	if len(parts) != 2 && len(parts) != 3 {
		return Target{}, usage
	}
	if parts[0] == "" || parts[1] == "" || (len(parts) == 3 && parts[2] == "") {
		return Target{}, usage
	}
	target := Target{Package: parts[0] + "/" + parts[1]}
	if len(parts) == 3 {
		target.Function = parts[2]
	}
	return target, nil
}

func invocationTarget(ctx *Context) (Target, *launch.PackageInterface, *exit.Error) {
	target, problem := parseTarget(ctx.Inv.Args[0])
	if problem != nil {
		return Target{}, nil, problem
	}
	if rentalRequested(ctx) && strings.HasPrefix(target.Package, "local/") {
		facts, problem := activeInstallFacts(ctx, target.Package)
		if problem != nil {
			return Target{}, nil, problem
		}
		target.InstallID = facts.Install.ID
		target.Release = facts.Install.Version
		return target, facts.PackageInterface, nil
	}
	if rentalRequested(ctx) {
		ref, problem := hub.ParseRef(target.Package)
		if problem != nil {
			return Target{}, nil, problem
		}
		hctx, cancel := hub.Context()
		defer cancel()
		catalog := client(ctx)
		card, problem := catalog.PackageCard(hctx, ref)
		if problem != nil {
			return Target{}, nil, problem
		}
		release, problem := newestPackageRelease(card.Releases)
		if problem != nil {
			return Target{}, nil, problem
		}
		detail, problem := catalog.PackageRelease(hctx, ref, release)
		if problem != nil {
			return Target{}, nil, problem
		}
		packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
		if problem != nil || packageInterface.Digest != detail.Release.PackageInterfaceDigest ||
			detail.Release.PackageInterfaceLength != int64(len(detail.PackageInterface)) {
			return Target{}, nil, exit.Named(exit.Conflict, "rental.package_interface_invalid",
				"Tensorhub returned an invalid package interface")
		}
		if detail.Release.Release != release {
			return Target{}, nil, exit.Named(exit.Conflict, "rental.package_release_invalid",
				"Tensorhub returned no immutable package release identity")
		}
		target.Release = release
		return target, packageInterface, nil
	}
	facts, problem := activeInstallFacts(ctx, target.Package)
	if problem != nil && problem.Code == exit.NotFound {
		if problem = autoInstallPackage(ctx, target.Package); problem != nil {
			return Target{}, nil, problem
		}
		facts, problem = activeInstallFacts(ctx, target.Package)
	}
	if problem != nil {
		return Target{}, nil, problem
	}
	target.InstallID = facts.Install.ID
	return target, facts.PackageInterface, nil
}

func newestPackageRelease(releases []hub.ReleaseSummary, majors ...int) (string, *exit.Error) {
	wantedMajor := -1
	if len(majors) == 1 {
		wantedMajor = majors[0]
	}
	best := ""
	bestVersion := [3]int{-1, -1, -1}
	for _, row := range releases {
		if row.Yanked || row.YankedAt != "" {
			continue
		}
		parts := strings.Split(row.Release, ".")
		if len(parts) != 3 {
			continue
		}
		var version [3]int
		valid := true
		for i, part := range parts {
			value, err := strconv.Atoi(part)
			if err != nil || value < 0 || strconv.Itoa(value) != part {
				valid = false
				break
			}
			version[i] = value
		}
		if valid && (wantedMajor < 0 || version[0] == wantedMajor) &&
			(version[0] > bestVersion[0] ||
				version[0] == bestVersion[0] && version[1] > bestVersion[1] ||
				version[0] == bestVersion[0] && version[1] == bestVersion[1] &&
					version[2] > bestVersion[2]) {
			best, bestVersion = row.Release, version
		}
	}
	if best == "" {
		return "", exit.New(exit.NotFound, "package has no non-yanked numeric release")
	}
	return best, nil
}

// emitDescribe prints the callable's contract from the PackageInterface — the SAME facts
// submit validates the payload against (cl-105/cl-106), rendered by the one contract
// printer. JSON mode emits the raw request struct verbatim.
func emitDescribe(ctx *Context, target Target, packageInterface *launch.PackageInterface,
	ep *launch.Entrypoint,
) *exit.Error {
	if ctx.Mode().JSON {
		raw, ok := packageInterface.RawRequest(ep.Name)
		if !ok {
			return exit.Internalf("the package interface does not carry %s's request struct", ep.Name)
		}
		fmt.Fprintln(ctx.Out, string(raw))
		return nil
	}
	fmt.Fprint(ctx.Out, launch.DescribeContract(target.Package+"/"+ep.Name, ep,
		describeBindings(ctx, target, ep)))
	return nil
}

// describeBindings resolves each slot's CURRENT default binding — the same mutable hub
// pointers a bare run resolves (th-116) — best effort: the contract stays readable
// offline, with an unreadable default rendered as no default rather than a refusal.
func describeBindings(ctx *Context, target Target, ep *launch.Entrypoint) map[string]string {
	if len(ep.Models) == 0 || strings.HasPrefix(target.Package, "local/") {
		return nil
	}
	rows, problem := invocationDefaultBindings(ctx, target)
	if problem != nil {
		return nil
	}
	out := make(map[string]string, len(rows))
	for slot, row := range rows {
		out[slot] = row.Ref()
	}
	return out
}

func emitFunctions(ctx *Context, target Target, packageInterface *launch.PackageInterface) *exit.Error {
	list := output.List{Name: "functions", Fields: []string{"function"}, AllFields: []string{"function"}}
	for _, name := range packageInterface.Names() {
		list.Rows = append(list.Rows, map[string]string{"function": name})
		if len(list.Next) < 2 {
			list.Next = append(list.Next, "cozy run "+target.Package+"/"+name)
		}
	}
	return emit(ctx, list)
}

func unknownFunction(target Target, packageInterface *launch.PackageInterface) *exit.Error {
	names := packageInterface.Names()
	problem := exit.New(exit.NotFound, "%s registers no function %q", target.Package, target.Function)
	if len(names) == 0 {
		return problem.WithRemedy("this release registers no callable functions")
	}
	problem.WithRemedy("available functions: %s", strings.Join(names, ", "))
	for _, name := range names {
		problem.WithNext("cozy run " + target.Package + "/" + name)
	}
	return problem
}

func autoInstallPackage(ctx *Context, pkg string) *exit.Error {
	if ctx.Mode().Human {
		fmt.Fprintf(ctx.Err, "%s is not installed; installing it from Tensorhub...\n", pkg)
	}
	sub := *ctx
	sub.Out = io.Discard
	sub.Inv = &Invocation{
		Args: []string{pkg}, Bools: map[string]bool{}, Values: map[string][]string{}, Mode: ctx.Mode(),
	}
	if problem := handleRegistryInstall(&sub); problem != nil {
		return problem
	}
	if ctx.Mode().Human {
		fmt.Fprintf(ctx.Err, "Installed %s.\n", pkg)
	}
	return nil
}

// activeInstallFacts resolves the package's one active install.
func activeInstallFacts(ctx *Context, pkg string) (*launch.Facts, *exit.Error) {
	install, e := installedPackage(ctx, pkg)
	if e != nil {
		return nil, e
	}
	return launch.Read(*install, ctx.Cfg.Home, ctx.Cfg.Tool())
}

func installedPackage(ctx *Context, pkg string) (*records.PackageInstall, *exit.Error) {
	bare, major, hasMajor := splitMajor(pkg)
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return nil, e
	}
	store, e := records.Open(l.DB)
	if e != nil {
		return nil, e
	}
	defer store.Close()
	pins, e := store.Pins(bare)
	if e != nil {
		return nil, e
	}
	if len(pins) == 0 {
		return nil, exit.New(exit.NotFound, "%s is not installed on this host", pkg).
			WithRemedy("`cozy package list` lists what is").
			WithNext("cozy package search "+pkg, "cozy package list")
	}
	eligible := pins[:0]
	for _, pin := range pins {
		if !hasMajor || pin.Major == major {
			eligible = append(eligible, pin)
		}
	}
	if len(eligible) == 0 {
		return nil, exit.New(exit.NotFound, "%s is not installed on this host", pkg).
			WithRemedy("`cozy package list` lists installed majors").
			WithNext("cozy package list")
	}
	chosen := eligible[0]
	for _, pin := range eligible[1:] {
		if pin.ActivatedAt > chosen.ActivatedAt {
			chosen = pin
		}
	}
	install, e := store.Install(chosen.InstallID)
	if e != nil {
		return nil, e
	}
	if install == nil {
		return nil, exit.Internalf("%s is pinned to install %s and that row is gone", pkg, chosen.InstallID)
	}
	return install, nil
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

// seconds spells a millisecond count as the CLI's duration cell.
func seconds(ms int64) string { return fmt.Sprintf("%.1fs", float64(ms)/1000) }
