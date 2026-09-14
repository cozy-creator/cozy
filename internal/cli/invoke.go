package cli

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mattn/go-isatty"

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
	if ctx.Daemon.Addr == "" {
		state, _, problem := ensureDaemon(ctx)
		if problem != nil {
			return nil, problem
		}
		ctx.Daemon = state
	}
	return localapi.Open(ctx.Cfg, ctx.Daemon)
}

// ----------------------------------------------------------------------------- run

func handleRunExecute(ctx *Context) *exit.Error {
	if problem := validateRunPlacement(ctx); problem != nil {
		return problem
	}
	target, packageInterface, problem := invocationTarget(ctx)
	if problem != nil {
		return problem
	}
	defer func() { reclaimSnapshot(ctx, target) }()
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
	if callable.Kind == "job" && strings.HasPrefix(target.Package, "local/") && !target.Snapshot {
		target, packageInterface, problem = snapshotLocalJob(ctx, target)
		if problem != nil {
			return problem
		}
		callable, problem = packageInterface.Function(target.Function)
		if problem != nil {
			return unknownFunction(target, packageInterface)
		}
	}
	if callable.Kind == "job" && ctx.Inv.Value("--attention-kernel") != "" {
		return exit.Usagef("--attention-kernel applies only to serving callables")
	}
	if ctx.Inv.Bool("--dry-run") && ctx.Inv.Bool("--await") {
		return exit.Usagef("--dry-run and --await conflict")
	}
	if callable.Kind != "job" {
		if len(ctx.Inv.Values["--allow-publish"]) > 0 {
			return exit.Usagef("--allow-publish applies only to Runtime-owned job transactions")
		}
		if ctx.Inv.Value("--timeout") != "" && !ctx.Inv.Bool("--await") {
			return exit.Usagef("--timeout requires --await for serving callables").
				WithRemedy("a detached serving call has no client waiting to enforce a caller deadline")
		}
		if ctx.Inv.Value("--retry") != "" {
			return exit.Usagef("--retry applies only to job transactions")
		}
		if ctx.Inv.Value("--publish-to") != "" || len(ctx.Inv.Values["--source-profile"]) > 0 || ctx.Inv.Bool("--dry-run") {
			return exit.Usagef("--publish-to, --source-profile, and --dry-run apply only to job callables")
		}
		if len(ctx.Inv.Values["--input"]) > 0 {
			return exit.Usagef("--input-tree applies only to a job callable")
		}
		if ctx.Inv.Value("--org") != "" {
			return exit.Usagef("--org applies only to a job callable")
		}
		return handleRun(ctx, target, callable)
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
	if ctx.Inv.Value("--rental") != "" && (ctx.Inv.Bool("--rental") || ctx.Inv.Bool("--rental-only")) {
		return exit.Usagef("a named --rental cannot be combined with --rental-only")
	}
	managedRental := ctx.Inv.Bool("--rental") || ctx.Inv.Bool("--rental-only")
	if managedRental && ctx.Cfg.RentalsMaxHourlySpendUSDMicros <= 0 {
		return exit.Named(exit.Usage, "rental.spend_cap_required",
			"rented execution requires a positive rentals.max_hourly_spend_usd in %s", filepath.Join(ctx.Cfg.Home, "config.yaml")).
			WithRemedy("set the fleet-wide hourly ceiling before authorizing rental spend")
	}
	return nil
}

func rentalRequested(ctx *Context) bool {
	return ctx.Inv.Bool("--rental") || ctx.Inv.Bool("--rental-only") || ctx.Inv.Value("--rental") != ""
}

func handleRun(ctx *Context, target Target, ep *launch.Entrypoint) *exit.Error {
	deadline, e := runDeadline(ctx)
	if e != nil {
		return e
	}
	key := requestKey(ctx.Inv.Value("--idempotency-key"))

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
	if pin := strings.TrimSpace(ctx.Inv.Value("--attention-kernel")); pin != "" {
		if overrides.AttentionKernel != "" {
			return exit.Usagef("attention kernel was pinned more than once")
		}
		if strings.ContainsAny(pin, "= \t\r\n") {
			return exit.Usagef("--attention-kernel must be one kernel name")
		}
		overrides.AttentionKernel = pin
	}
	prepareImage := imagePreparer(ctx)
	input, assets, e := launch.ParseAssets(ep, input, ctx.Inv.Values["--asset"], ctx.Inv.Values["--asset-fidelity"], prepareImage)
	if e != nil {
		return e
	}
	input, e = finalizeInputPayload(ep, input, key)
	if e != nil {
		return e
	}
	if e := validateInvocationPayload(ctx, target.Package, ep, input); e != nil {
		return e
	}
	selectedRental, e := requestedRental(ctx, target, ep.Name)
	if e != nil {
		return e
	}
	models, e := resolveInvocationModels(ctx, target, ep, overrides.Models, managedRental)
	if e != nil {
		return e
	}
	outputDirectory, e := requestedOutputDirectory(ctx)
	if e != nil {
		return e
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
		RentalRequired:  ctx.Inv.Bool("--rental-only") || selectedRental != "",
		RequestedRental: selectedRental,
		Models:          models,
		OutputDirectory: outputDirectory,
		AttentionKernel: overrides.AttentionKernel,
	}, key)
	if e != nil {
		return e
	}
	submitted := time.Since(began)
	if !ctx.Mode().JSON {
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
		terminal, stopped, e = watch(ctx, c, handle.RequestID, deadline, began)
	} else {
		terminal, e = observe(ctx, c, handle.RequestID, optimisticObservation, began)
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

// invocationModelSpec is one slot's selection before the card is read: an explicit
// `model.<param>=` run key pins Ref (and possibly Lane); a bare run takes Ref from the
// owner binding or authored default; its ladder decides the lane. Beside a run key the binding is
// only the owner's word on where that lane fits (cl-170): evidence, never a choice.
type invocationModelSpec struct {
	Slot     string
	Ref      string
	Lane     string
	Explicit bool
	Binding  *hub.PackageBindingRow
}

// resolveInvocationModels applies the one selection order for both local and rented
// execution: an explicit `model.<param>=` run key, then a Hub owner override, then the
// selected callable's authored default ladder. Local
// acquisition freezes the exact Manifest and length before submission — the lane is the
// host GPU's rung; remote acquisition carries the whole ladder and the winning machine
// pins its rung. Editable packages use authored defaults without a Hub override.
func resolveInvocationModels(ctx *Context, target Target, ep *launch.Entrypoint,
	overrides map[string]string, remote bool,
) ([]orchestrator.ModelRef, *exit.Error) {
	selected, problem := invocationModelSpecs(ctx, target, ep, overrides)
	if problem != nil || len(selected) == 0 {
		return nil, problem
	}
	return resolveSelectedInvocationModels(ctx, target, ep, selected, remote)
}

func resolveSelectedInvocationModels(ctx *Context, target Target, ep *launch.Entrypoint,
	selected []invocationModelSpec, remote bool,
) ([]orchestrator.ModelRef, *exit.Error) {
	slots := make(map[string]launch.Slot, len(ep.Models))
	for _, slot := range ep.Models {
		slots[slot.Path] = slot
	}
	if remote {
		out := make([]orchestrator.ModelRef, 0, len(selected))
		for _, spec := range selected {
			slot, ok := slots[spec.Slot]
			if !ok {
				return nil, exit.Internalf("resolved model slot %s is absent from the package interface", spec.Slot)
			}
			var row orchestrator.ModelRef
			var problem *exit.Error
			if spec.Explicit {
				row, problem = resolveRemoteModel(ctx, target.Package, slot, spec.Ref, spec.Lane, spec.Binding)
			} else {
				row, problem = resolveRemoteLadder(ctx, target.Package, slot, spec)
			}
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
		slot, ok := slots[spec.Slot]
		if !ok {
			return nil, exit.Internalf("resolved model slot %s is absent from the package interface", spec.Slot)
		}
		lane := spec.Lane
		if !spec.Explicit {
			// GPU-specific defaults take precedence; an unlisted local GPU uses the
			// first declared lane and still goes through ordinary Runtime admission.
			rung, problem := localRung(ctx, spec.Binding.Ladder)
			if problem != nil {
				return nil, problem
			}
			lane = rung.Lane
		}
		model, problem := acquirePublishedModel(hctx, ctx, tool, client(ctx), spec.Ref,
			lane, target.Package, slot,
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
		source, provider, problem := providerModelSource(raw)
		if problem != nil {
			return nil, problem
		}
		if provider {
			selected[slotPath] = invocationModelSpec{Slot: slotPath, Ref: source, Explicit: true}
			continue
		}
		model, release, lane, manifest, problem := hub.ParseModelRef(raw)
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
		selected[slotPath] = invocationModelSpec{Slot: slotPath, Ref: ref, Lane: lane, Explicit: true}
	}
	// Editable packages have no Hub override; their selected source still declares
	// the same immutable default metadata as a published package.
	editable := strings.HasPrefix(target.Package, "local/")
	defaults := map[string]hub.PackageBindingRow{}
	if !editable {
		var problem *exit.Error
		if defaults, problem = invocationDefaultBindings(ctx, target, ep.Models); problem != nil {
			// A slot a run key covers needs no binding: beside it the read is only the
			// owner's word on the lane's fit (cl-170), and a hub that cannot give it
			// leaves the lane held to its own bytes rather than refusing the run.
			if len(selected) < len(ep.Models) {
				return nil, problem
			}
			defaults = map[string]hub.PackageBindingRow{}
		}
	} else {
		defaults = effectiveModelBindings(ep.Models, nil)
	}
	out := make([]invocationModelSpec, 0, len(ep.Models))
	for _, slot := range ep.Models {
		if spec, ok := selected[slot.Path]; ok {
			if binding, bound := defaults[slot.Path]; bound {
				spec.Binding = &binding
			}
			out = append(out, spec)
			continue
		}
		binding, ok := defaults[slot.Path]
		if !ok {
			if editable {
				return nil, exit.Named(exit.Usage, "package_model_override_required",
					"%s has no authored default for model slot %s", target.Package, slot.Path).
					WithRemedy("model.%s=org/model@release[/lane]", slot.Param)
			}
			message := fmt.Sprintf("%s has no owner binding or authored default for model slot %s", target.Package, slot.Path)
			if ep.Kind != "job" {
				message = fmt.Sprintf("%s/%s is disabled in this deployment: no default model binding for %s", target.Package, ep.Name, slot.Param)
			}
			return nil, exit.Named(exit.NotFound, "package_default_model_unavailable", "%s", message).
				WithRemedy("bind it: %s — or override this run: model.%s=org/model@release[/lane]",
					bindRemedy(target.Package, slot.Path), slot.Param)
		}
		out = append(out, invocationModelSpec{Slot: slot.Path, Ref: binding.Ref(), Binding: &binding})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Slot < out[j].Slot })
	return out, nil
}

// invocationDefaultBindings reads the package's CURRENT default bindings from the hub
// (th-116, cl-166): mutable owner-written rows, each a model release and its ladder.
// Owner rows take precedence; only absent rows use the selected callable metadata.
// A failed Hub read cannot establish absence and therefore cannot silently fall back.
func invocationDefaultBindings(ctx *Context, target Target, slots []launch.Slot) (
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
	return effectiveModelBindings(slots, rows), nil
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

func resolveRemoteModel(ctx *Context, packageName string, slot launch.Slot, raw, wantedLane string,
	binding *hub.PackageBindingRow) (orchestrator.ModelRef, *exit.Error) {
	// Exact checkpoint inputs use Hub-owned facts; named selections use the release
	// card. Both freeze a verified repository/manifest identity before preparation.
	var empty orchestrator.ModelRef
	modelName, release, refLane, manifest, problem := hub.ParseModelRef(raw)
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
		return empty, localModelOnRental(ref)
	}
	hctx, cancel := hub.Context()
	defer cancel()
	if manifest != "" && release == "" {
		if wantedLane != "" {
			return empty, exit.Usagef("a checkpoint digest without a release cannot select a lane")
		}
		resolved, problem := client(ctx).ResolveModel(hctx, ref.String()+"@"+manifest, "")
		if problem != nil {
			return empty, problem
		}
		if resolved.Model != ref.String() || resolved.ManifestID != manifest ||
			resolved.ManifestLength <= 0 || resolved.Bytes <= 0 {
			return empty, exit.Named(exit.Conflict, "rental.model_resolution_changed",
				"Tensorhub returned different or incomplete checkpoint facts for %s", raw)
		}
		if problem := requireCheckpointComponents(raw, slot, resolved.Components); problem != nil {
			return empty, problem
		}
		return orchestrator.ModelRef{Package: packageName, Slot: slot.Path,
			Model: ref.String(), Manifest: manifest, HubCheckpoint: true,
			ManifestLength: resolved.ManifestLength, Bytes: resolved.Bytes,
			ComponentBytes: resolved.ComponentBytes, ComponentUse: slot.ComponentUse}, nil
	}
	_, selected, problem := modelReleaseCard(hctx, client(ctx), ref, release)
	if problem != nil {
		return empty, problem
	}
	release = selected.Release
	if wantedLane != "" {
		if _, problem := laneOf(ref, selected, wantedLane); problem != nil {
			return empty, problem
		}
	}
	manifestLanes := map[string][]string{}
	manifestBytes := map[string]int64{}
	manifestComponents := map[string][]string{}
	manifestComponentBytes := map[string]map[string]int64{}
	for _, lane := range selected.Lanes {
		if (manifest == "" || lane.ManifestID == manifest) &&
			(wantedLane == "" || lane.Lane == wantedLane) {
			manifestLanes[lane.ManifestID] = append(manifestLanes[lane.ManifestID], lane.Lane)
			manifestBytes[lane.ManifestID] = lane.Bytes
			manifestComponents[lane.ManifestID] = lane.Components
			manifestComponentBytes[lane.ManifestID] = lane.ComponentBytes
		}
	}
	if manifest != "" && len(manifestLanes[manifest]) == 0 {
		return empty, exit.New(exit.NotFound, "model %s@%s does not contain manifest %s",
			ref.String(), release, manifest)
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
	if problem := requireCheckpointComponents(raw, slot, manifestComponents[manifest]); problem != nil {
		return empty, problem
	}
	return orchestrator.ModelRef{Package: packageName, Slot: slot.Path,
		Model: ref.String(), Release: release, Lane: lanes[0], Manifest: manifest,
		Bytes: manifestBytes[manifest], ComponentBytes: manifestComponentBytes[manifest],
		ComponentUse: slot.ComponentUse, Ladder: assertedRungs(binding, ref, selected)}, nil
}

// assertedRungs is the owner's ladder carried beside an explicit lane (cl-170): each rung
// resolved to the selected release's lane of that name, saying where the owner puts each
// lane of THIS model — the evidence a machine decision reads for the fit of the lane the
// run key chose, never a choice. The binding's release is not required (cl-174): a rung
// names a card and a lane, and on 2026-09-08 a package bound to rc.2 sized an explicit
// rc.1 lane of the same name by whole-lane bytes, refusing two H100s that were running
// that very lane. A binding for another model, or a rung naming a lane the selected
// release lacks, says nothing here.
func assertedRungs(binding *hub.PackageBindingRow, ref hub.Ref, selected *hub.ModelReleaseSummary) []records.ModelRung {
	if binding == nil || binding.Model != ref.String() {
		return nil
	}
	var rungs []records.ModelRung
	for _, rung := range binding.Ladder {
		lane, problem := laneOf(ref, selected, rung.Lane)
		if problem != nil {
			continue
		}
		rungs = append(rungs, records.ModelRung{GPU: rung.GPU, Lane: rung.Lane,
			Manifest: lane.ManifestID, Bytes: lane.Bytes, ComponentBytes: lane.ComponentBytes})
	}
	return rungs
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
		Name: "invocations", Fields: []string{"number", "target", "machine", "status", "progress", "execution"},
		AllFields: []string{"number", "id", "kind", "target", "machine", "rental_id", "status",
			"progress", "phase", "progress_stage", "stage_fraction", "overall_fraction",
			"position", "total", "queued", "execution", "attempts", "created"},
		TypedFields: []string{"number", "target", "machine", "rental_id", "requested_rental", "requested_machine", "status",
			"phase", "progress_stage", "stage_fraction", "overall_fraction", "position", "total",
			"remaining_ms", "execution_ms"},
		TypedAllFields: []string{"number", "id", "kind", "target", "machine", "rental_id", "requested_rental", "requested_machine",
			"status", "canceled_by", "phase", "phase_machine", "waiting_for", "phase_elapsed_ms",
			"phase_moved_bytes", "phase_total_bytes", "phase_rate_bytes_per_second",
			"phase_remaining_ms", "progress_stage", "stage_fraction", "overall_fraction",
			"position", "total", "remaining_ms", "queued_ms", "execution_ms", "attempts",
			"created_at"},
		TypedRows: make([]map[string]any, 0, len(rows)),
		// The raw rental id is a machine fact: JSON always carries it, the compact
		// human table never does — the human word is the MACHINE column (cl-107).
		Machine: []string{"rental_id"},
	}
	states := map[string]int{}
	for _, life := range rows {
		life.Machine = life.DisplayMachine()
		kind := life.Kind
		if kind == "" {
			kind = "invocation"
		}
		// A canceled run is LOUD about its cause (cl-108): the status cell itself names
		// the recorded actor, so a list is never a quiet no-output ending.
		status := life.Status
		if life.Status == "canceled" && life.CanceledBy != "" {
			status = humanCancellationStatus(life.CanceledBy)
		} else if life.Status == "canceled" {
			status = "cancelled"
		}
		machine := life.Machine
		if machine == "" {
			machine = "—"
		}
		list.Rows = append(list.Rows, map[string]string{
			"number": strconv.FormatInt(life.Number, 10), "id": life.RequestID, "kind": kind,
			"target": life.Package + "/" + life.Function, "machine": machine,
			"rental_id": life.RentalID,
			"status":    status, "progress": progressValue(life), "phase": life.Phase,
			"progress_stage":   life.ProgressStage,
			"stage_fraction":   fractionValue(life.StageFraction),
			"overall_fraction": fractionValue(life.OverallFraction),
			"position":         integerValue(life.Position),
			"total":            integerValue(life.Total),
			"queued":           seconds(life.QueuedMS),
			"execution":        seconds(life.ExecutionMS),
			"attempts":         strconv.Itoa(life.Attempts), "created": life.CreatedAt,
		})
		typed := map[string]any{
			"number": life.Number, "id": life.RequestID, "kind": kind,
			"target": life.Package + "/" + life.Function, "machine": life.Machine,
			"status": life.Status, "queued_ms": life.QueuedMS, "execution_ms": life.ExecutionMS,
			"attempts": life.Attempts, "created_at": life.CreatedAt,
		}
		if life.RentalID != "" {
			typed["rental_id"] = life.RentalID
		}
		if life.RequestedRental != "" {
			typed["requested_rental"] = life.RequestedRental
			typed["requested_machine"] = life.RequestedMachine
		}
		// The preparation facts are machine-readable as NUMBERS and absence, never as the
		// human cell: a reader must be able to tell "no rate was measured" from "the rate
		// was zero", and a rendered string cannot say that.
		if life.Phase != "" {
			typed["phase"] = life.Phase
			if life.WaitingFor != nil {
				typed["waiting_for"] = life.WaitingFor
			}
			if life.PhaseMachine != "" {
				typed["phase_machine"] = life.PhaseMachine
			}
			if life.PhaseElapsedMS != nil {
				typed["phase_elapsed_ms"] = *life.PhaseElapsedMS
			}
			if life.PhaseMovedBytes != nil {
				typed["phase_moved_bytes"] = *life.PhaseMovedBytes
			}
			if life.PhaseTotalBytes != nil {
				typed["phase_total_bytes"] = *life.PhaseTotalBytes
			}
			if life.PhaseRate != nil {
				typed["phase_rate_bytes_per_second"] = *life.PhaseRate
			}
			if life.PhaseRemainingMS != nil {
				typed["phase_remaining_ms"] = *life.PhaseRemainingMS
			}
		}
		if life.CanceledBy != "" {
			typed["canceled_by"] = life.CanceledBy
		}
		// Stage, position and ETA describe live execution. Terminal rows retain only
		// the API's overall fraction: completed=1, otherwise the last measured value.
		if life.Status == "in_progress" {
			if life.ProgressStage != "" {
				typed["progress_stage"] = life.ProgressStage
			}
			if life.StageFraction != nil {
				typed["stage_fraction"] = *life.StageFraction
			}
			if life.Position != nil {
				typed["position"] = *life.Position
			}
			if life.Total != nil {
				typed["total"] = *life.Total
			}
			if life.RemainingMS != nil {
				typed["remaining_ms"] = *life.RemainingMS
			}
		}
		if life.OverallFraction != nil && (life.Status == "in_progress" || life.Status == "completed" || life.Status == "failed" || life.Status == "canceled") {
			typed["overall_fraction"] = *life.OverallFraction
		}
		list.TypedRows = append(list.TypedRows, typed)
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

func fractionValue(value *float64) string {
	if value == nil {
		return ""
	}
	return strconv.FormatFloat(*value, 'f', -1, 64)
}

func integerValue(value *int64) string {
	if value == nil {
		return ""
	}
	return strconv.FormatInt(*value, 10)
}

// phaseValue describes the activity holding a queued request. Machine and elapsed
// time have their own fields; only measured transfer progress belongs beside it.
func phaseValue(life api.Lifecycle) string { return PhaseCell(life) }

// PhaseCell is exported so the product suite drives this exact renderer rather than a
// copy of it (#661: verification's one home).
func PhaseCell(life api.Lifecycle) string {
	if life.Phase == "" {
		return ""
	}
	if life.Phase == orchestrator.WaitSlotBusy || life.Phase == orchestrator.WaitQueueAhead {
		if life.WaitingFor != nil {
			return fmt.Sprintf("waiting: run %d", life.WaitingFor.Number)
		}
		if life.Phase == orchestrator.WaitQueueAhead {
			return "waiting: earlier runs"
		}
		return "waiting: free worker slot"
	}
	activity := strings.ReplaceAll(life.Phase, "_", " ")
	switch life.Phase {
	case orchestrator.PhaseAcquiring:
		activity = "acquiring rental"
	case orchestrator.PhaseReplanning:
		activity = "finding another rental"
	case orchestrator.PhaseProvisioning:
		activity = "provisioning machine"
	case orchestrator.PhaseBooting:
		activity = "starting machine"
	case orchestrator.PhaseResolving:
		activity = "preparing downloads"
	case orchestrator.PhaseDownloading:
		activity = "downloading models"
	case orchestrator.PhasePreparing:
		activity = "setting up package"
		if life.Kind == "job" {
			activity = "preparing inputs"
		}
	case orchestrator.PhaseWarming:
		activity = "loading models"
	}
	parts := []string{activity}
	// Older daemons carried completed download counters into these phases.
	// Job preparation may report real conversion progress; serving setup cannot.
	if life.Phase == orchestrator.PhaseWarming || life.Phase == orchestrator.PhasePreparing && life.Kind != "job" {
		return activity
	}
	if life.PhaseMovedBytes != nil {
		moved := output.Bytes(*life.PhaseMovedBytes)
		if life.PhaseTotalBytes != nil && *life.PhaseTotalBytes > 0 {
			moved += " / " + output.Bytes(*life.PhaseTotalBytes)
		}
		parts = append(parts, moved)
	}
	if life.PhaseRate != nil {
		parts = append(parts, output.Bytes(int64(*life.PhaseRate))+"/s")
	}
	if life.PhaseRemainingMS != nil {
		parts = append(parts, "~"+shortDuration(time.Duration(*life.PhaseRemainingMS)*time.Millisecond))
	}
	return strings.Join(parts, " · ")
}

func progressValue(life api.Lifecycle) string {
	if life.Status == "queued" {
		if life.RequestedRental != "" && (life.Phase == "" || life.Phase == orchestrator.WaitRental ||
			life.Phase == orchestrator.WaitSlotBusy || life.Phase == orchestrator.WaitQueueAhead) {
			name := either(life.RequestedMachine, life.RequestedRental)
			return "waiting for rental " + name
		}
		if cell := phaseValue(life); cell != "" {
			return cell
		}
		return "-"
	}
	if life.Status == "completed" {
		return "100%"
	}
	if life.Status != "in_progress" {
		if (life.Status == "failed" || life.Status == "canceled") && life.OverallFraction != nil {
			return fmt.Sprintf("%.0f%%", *life.OverallFraction*100)
		}
		return "-"
	}
	// ONE NUMBER AND THE STAGE. This cell carried four facts at once -- an overall
	// percent, a remaining estimate, the stage, and a SECOND percent scoped to that
	// stage -- so `35% overall (~2s) · denoise 90% stage` asked a reader to hold two
	// unrelated denominators in mind to learn one thing. The stage says what is
	// happening; the overall percent says how far along it is. A stage-scoped percent
	// answers a question nobody asked of a list, and the remaining estimate moves
	// faster than the row it sits in.
	stage := life.ProgressStage
	if life.OverallFraction == nil {
		if stage == "" {
			return "-"
		}
		return stage
	}
	overall := fmt.Sprintf("%.0f%%", *life.OverallFraction*100)
	if stage == "" {
		return overall
	}
	return stage + " " + overall
}

func watchRunList(ctx *Context, client *localapi.Client, limit int) *exit.Error {
	return watchList(ctx, "id", func(watchCtx context.Context) (output.List, *exit.Error) {
		return runList(watchCtx, client, ctx.Inv.Value("--state"), ctx.Inv.Value("--package"), limit)
	})
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
func observe(ctx *Context, c *localapi.Client, requestID string,
	window time.Duration, began time.Time,
) (*localapi.Event, *exit.Error) {
	watchCtx, stop := context.WithTimeout(context.Background(), window)
	defer stop()
	watchCtx, restoreInput, _, problem := liveWatchContext(ctx, watchCtx, nil)
	if problem != nil {
		return nil, problem
	}
	defer restoreInput()
	lines := NewProgress(ctx, false, began)
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
func watch(ctx *Context, c *localapi.Client, requestID string,
	deadline time.Duration, began time.Time) (*localapi.Event, string, *exit.Error) {
	interrupt, restoreInput, _, problem := liveSignals(ctx, nil)
	if problem != nil {
		return nil, "", problem
	}
	defer restoreInput()

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
			if !ctx.Mode().JSON {
				fmt.Fprintf(ctx.Err,
					"\ndetached — the run keeps running; `cozy run watch %s` reattaches, `cozy run cancel %s` cancels\n",
					requestID, requestID)
			}
			stopWatch()
			return
		case <-deadlineC(deadline):
			// `--timeout` is a REQUEST DEADLINE the client enforces the only way a client
			// honestly can: by asking the orchestrator to cancel. It is not the
			// supervisor's watchdog deadline (that one is on the attempt, and this host
			// has no wire field for it) — walking away instead would leave the card held.
			stopped <- "deadline"
			if !ctx.Mode().JSON {
				fmt.Fprintf(ctx.Err,
					"\n--timeout %s expired; cancel requested — the attempt's own terminal still settles it\n",
					deadline)
			}
		case <-done:
			return
		}
		cancelResult := make(chan *exit.Error, 1)
		go func() { cancelResult <- c.Cancel(requestID, fmt.Sprintf("cozy run --timeout %s", deadline)) }()
		select {
		case <-interrupt:
			if !ctx.Mode().JSON {
				fmt.Fprintln(ctx.Err, "detached — the cancelled terminal still lands in `cozy run list`")
			}
			stopWatch()
		case problem := <-cancelResult:
			if problem != nil {
				if !ctx.Mode().JSON {
					fmt.Fprintf(ctx.Err, "cancel: %s\n", problem.Message)
				}
				cancelFailed <- problem
				stopWatch()
				return
			}
		case <-done:
			return
		}
		select {
		case <-interrupt:
			if !ctx.Mode().JSON {
				fmt.Fprintln(ctx.Err, "detached — the cancelled terminal still lands in `cozy run list`")
			}
			stopWatch()
		case <-done:
		}
	}()

	lines := NewProgress(ctx, ctx.Mode().JSON, began)
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
// Awaited `--json` writes NDJSON of the typed envelope on stderr, a terminal keeps
// finished stages above a refreshed active block, and a redirected human command gets sparse
// append-only lines — a stage change, each new tenth of the work, one line per five
// quiet seconds — never the full lossy tick stream. JSON results stay on stdout. Exported —
// with HumanWaitLine — so the product suite (#661: verification's one
// home) drives this exact render path.
type RunProgress struct {
	ctx             *Context
	rawJSON         bool
	mu              sync.Mutex
	last            string
	rentalLine      string
	placementLine   string
	closed          bool
	began           time.Time
	stepStage       string
	stepSeconds     float64
	stepSamples     int
	stepPosition    float64
	progressAttempt uint64
	terminal        liveProgress
	overallSeen     bool
	overallFraction float64
	overallDelta    float64
	overallSeconds  float64

	// The sparse lane's memory: which tenth of which stage was last appended, and when.
	sparseStage   string
	sparseDecile  int
	sparseAt      time.Time
	sparseStarted time.Time

	// Injected clock and measure, so tests drive the REAL renderer deterministically.
	now   func() time.Time
	width func() int
}

func NewProgress(ctx *Context, rawJSON bool, began time.Time) *RunProgress {
	return &RunProgress{
		ctx: ctx, rawJSON: rawJSON, began: began, sparseDecile: -1,
		now: time.Now, width: func() int { return terminalWidth(ctx.Err) },
	}
}

func (p *RunProgress) On(e localapi.Event) bool {
	if p.rawJSON {
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
	kind := strings.TrimPrefix(e.Type, "request.")
	if !p.ctx.Mode().Full && (kind == "rentals" || kind == "placement") {
		p.rentalNotice(e)
		if kind != "placement" || !waitingPlacement(e.Payload) {
			return true
		}
		// The placement record remains unchanged on the wire. Its attaching verdict
		// is an ordinary rental wait on the human progress surface.
		e.Type = "request.parked"
		e.Payload = map[string]any{"wait": orchestrator.WaitRental}
	}
	if strings.TrimPrefix(e.Type, "request.") == "progress" && p.progressAttempt != e.Attempt {
		if p.progressAttempt != 0 && p.ctx.Mode().Color && !p.ctx.Mode().Full {
			p.finishLive("retrying", eventTime(e))
		}
		p.progressAttempt = e.Attempt
		p.stepStage, p.stepSeconds, p.stepSamples = "", 0, 0
		p.overallSeen, p.overallFraction, p.overallDelta, p.overallSeconds = false, 0, 0, 0
	}
	// A redirected human command has no status line to rewrite: it gets the sparse
	// append lane. --full deliberately restores the complete diagnostic stream.
	if !p.ctx.Mode().Color && !p.ctx.Mode().Full {
		p.sparse(e)
		return true
	}
	if p.ctx.Mode().Color && !p.ctx.Mode().Full {
		p.interactive(e)
	} else {
		p.render(diagnosticProgressLine(e))
	}
	return true
}

// render appends diagnostic text; only the ordinary interactive lane owns a live block.
func (p *RunProgress) render(line string) {
	if line == "" || line == p.last {
		return
	}
	p.last = line
	fmt.Fprintln(p.ctx.Err, line)
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
	kind := strings.TrimPrefix(e.Type, "request.")
	if kind == "phase" {
		p.sparsePhase(e)
		return
	}
	if kind == "queued" || kind == "parked" {
		line := HumanWaitLine(e.Payload)
		if line != p.sparseStage {
			p.sparseStarted = eventTime(e)
		} else if p.now().Sub(p.sparseAt) < 5*time.Second {
			return
		}
		p.sparseStage, p.sparseAt = line, p.now()
		p.appendOnce(line + " · elapsed " + shortDuration(max(p.now().Sub(p.sparseStarted), 0)))
		return
	}
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
	fraction := facts.stageFraction
	if facts.hasOverall {
		fraction = facts.overallFraction
	}
	decile := int(fraction * 10)
	stale := !p.sparseAt.IsZero() && p.now().Sub(p.sparseAt) >= 5*time.Second
	if facts.label == p.sparseStage && decile == p.sparseDecile && !stale {
		return
	}
	p.sparseStage, p.sparseDecile, p.sparseAt = facts.label, decile, p.now()
	line := "  " + facts.label
	if facts.counted {
		line += fmt.Sprintf(" %d/%d", facts.current, facts.total)
	}
	if facts.hasStageFraction {
		line += fmt.Sprintf(" · %.0f%% stage", facts.stageFraction*100)
	}
	line += facts.timing()
	if facts.hasOverall {
		line += fmt.Sprintf(" · %.0f%% overall", facts.overallFraction*100)
	}
	line += " · elapsed " + shortDuration(p.now().Sub(p.began))
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
	label            string
	stageFraction    float64
	hasStageFraction bool
	overallFraction  float64
	hasOverall       bool
	counted          bool
	current          int64
	total            int64
	perStep          float64
	overallRemaining time.Duration
	hasOverallETA    bool
}

// observe folds one progress payload into the per-stage step-time accumulator and
// returns the frame's facts; ok is false when the frame carries no usable fraction.
func (p *RunProgress) observe(fields map[string]any) (stepFacts, bool) {
	name, _ := fields["stage"].(string)
	name = strings.TrimSpace(name)
	stageFraction, stageFractionOK := number(fields["stage_fraction"])
	overallFraction, overallOK := number(fields["overall_fraction"])
	position, positionOK := number(fields["position"])
	total, totalOK := number(fields["total"])
	stepMS, stepOK := number(fields["step_ms"])
	previousPosition := p.stepPosition
	if name != "" && name != p.stepStage {
		p.stepStage, p.stepSeconds, p.stepSamples, p.stepPosition = name, 0, 0, -1
		previousPosition = -1
	}
	if stepOK && stepMS >= 0 {
		if stepMS > 0 && (!positionOK || position > p.stepPosition) {
			p.stepSeconds += stepMS / 1000
			p.stepSamples++
			p.stepPosition = position
		}
	}
	label := stageLabel(name)
	if label == "" {
		label = "running"
	}
	facts := stepFacts{label: label}
	if positionOK != totalOK || positionOK &&
		(position < 0 || total <= 0 || position > total || math.Trunc(position) != position || math.Trunc(total) != total) {
		return facts, false
	}
	if positionOK {
		facts.counted = true
		facts.current = int64(position)
		facts.total = int64(total)
		if !stageFractionOK {
			stageFraction, stageFractionOK = position/total, true
		}
	}
	if stageFractionOK && stageFraction >= 0 && stageFraction <= 1 {
		facts.stageFraction, facts.hasStageFraction = stageFraction, true
	}
	if overallOK && overallFraction >= 0 && overallFraction <= 1 {
		if p.overallSeen && overallFraction < p.overallFraction {
			return facts, false
		}
		if p.overallSeen && overallFraction > p.overallFraction && stepOK && stepMS > 0 {
			elapsed := stepMS / 1000
			if positionOK {
				// A coalesced frame carries the last step's interval, not the
				// elapsed time for every step since the previous coordinate.
				advanced := position - math.Max(0, previousPosition)
				elapsed *= math.Max(0, advanced)
			}
			p.overallDelta += overallFraction - p.overallFraction
			p.overallSeconds += elapsed
		}
		p.overallSeen, p.overallFraction = true, overallFraction
	}
	if p.overallSeen {
		facts.overallFraction, facts.hasOverall = p.overallFraction, true
		if p.overallDelta > 0 && p.overallSeconds > 0 {
			facts.overallRemaining = time.Duration(
				(1 - p.overallFraction) * p.overallSeconds / p.overallDelta * float64(time.Second))
			facts.hasOverallETA = true
		}
	}
	if p.stepSamples > 0 {
		facts.perStep = p.stepSeconds / float64(p.stepSamples)
	}
	return facts, facts.hasStageFraction || facts.hasOverall
}

// clampLine bounds one rewritten status line to the terminal: a wrapped line leaves
// debris \r cannot reach.
func clampLine(line string, width int) string {
	var out strings.Builder
	cells := 0
	for _, r := range line {
		if r < 32 || r == 127 {
			continue
		}
		n := runeCells(r)
		if width > 0 && cells+n >= width {
			break
		}
		out.WriteRune(r)
		cells += n
	}
	return out.String()
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
	if p.terminal.timer != nil {
		p.terminal.timer.Stop()
	}
	if p.ctx.Mode().Color && !p.ctx.Mode().Full {
		p.finishLive("", p.now())
		return
	}
}

// progressLine is the human projection of one event. Runtime's exact typed envelope stays
// available through --json and --full; the ordinary status line shows only Runtime's named
// stage and explicit progress coordinates. Timing samples and metrics are diagnostics.
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
	case "phase":
		return HumanPhaseLine(e.Payload["value"])
	case "metric":
		return ""
	case "queued", "parked":
		return HumanWaitLine(e.Payload)
	case "rentals", "placement":
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
		return attemptFailedLine(e.Payload)
	default:
		return ""
	}
	return ""
}

// attemptFailedLine names WHY the attempt failed, not merely that it did. Every one of
// these causes is deterministic, so an operator who is shown only "retrying" watches a
// budget burn on a fact that was on the wire the whole time (cl-206).
func attemptFailedLine(payload map[string]any) string {
	kind, _ := payload["error_type"].(string)
	detail, _ := payload["error"].(string)
	kind, detail = strings.TrimSpace(kind), strings.TrimSpace(detail)
	if len(detail) > 240 {
		detail = strings.ToValidUTF8(detail[:240], "") + "…"
	}
	switch {
	case kind != "" && detail != "":
		return "  attempt failed (" + kind + "): " + detail + "; retrying"
	case kind != "":
		return "  attempt failed (" + kind + "); retrying"
	case detail != "":
		return "  attempt failed: " + detail + "; retrying"
	}
	return "  attempt failed; retrying"
}

func humanProgress(value any) string {
	fields, ok := value.(map[string]any)
	if !ok {
		return ""
	}
	stage, _ := fields["stage"].(string)
	stage = strings.TrimSpace(stage)
	stageFraction, hasStage := number(fields["stage_fraction"])
	if !hasStage {
		position, hasPosition := number(fields["position"])
		total, hasTotal := number(fields["total"])
		if hasPosition && hasTotal && position >= 0 && total > 0 && position <= total {
			stageFraction, hasStage = position/total, true
		}
	}
	overall, hasOverall := number(fields["overall_fraction"])
	if hasOverall && overall >= 0 && overall <= 1 {
		line := fmt.Sprintf("  overall — %.0f%%", overall*100)
		if stage != "" && hasStage && stageFraction >= 0 && stageFraction <= 1 {
			line += fmt.Sprintf(" · %s %.0f%% stage", stage, stageFraction*100)
		}
		return line
	}
	if stage != "" && hasStage && stageFraction >= 0 && stageFraction <= 1 {
		return fmt.Sprintf("  %s — %.0f%% stage", stage, stageFraction*100)
	}
	if stage != "" {
		return "  " + stage
	}
	return ""
}

// HumanWaitLine says what the queue is DOING, never how it thinks: the event's stable
// `wait` cause becomes a calm stage line with no digests and no dispatcher vocabulary.
// The verbatim diagnostic stays in the payload's `reason` for the machine surfaces.
func HumanWaitLine(payload map[string]any) string {
	pkg, _ := payload["package"].(string)
	on, _ := payload["waiting_on"].(string)
	cause, _ := payload["wait"].(string)
	if cause == orchestrator.WaitSlotBusy || cause == orchestrator.WaitQueueAhead {
		if waiting, ok := payload["waiting_for"].(map[string]any); ok {
			if run, ok := number(waiting["number"]); ok && run > 0 {
				line := fmt.Sprintf("  waiting for run %.0f", run)
				if on != "" {
					line += " on " + on
				}
				return line
			}
		}
	}
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

// HumanPhaseLine renders one preparation phase for the attached run (cl-121). It is the
// same information `run list` shows in its PROGRESS cell, said as a sentence: what is
// happening, on which machine, and — where the producer measured them — how much has moved
// and how fast. Absent numbers are absent, never rendered as zero.
func HumanPhaseLine(value any) string {
	fields, ok := value.(map[string]any)
	if !ok {
		return ""
	}
	name, _ := fields["phase"].(string)
	if strings.TrimSpace(name) == "" {
		return ""
	}
	if name == orchestrator.WaitSlotBusy || name == orchestrator.WaitQueueAhead {
		return HumanWaitLine(map[string]any{"wait": name, "waiting_on": fields["machine"],
			"waiting_for": fields["waiting_for"]})
	}
	line := "  " + strings.ReplaceAll(name, "_", " ")
	if machine, _ := fields["machine"].(string); machine != "" {
		line += " on " + machine
	}
	moved, hasMoved := number(fields["moved_bytes"])
	if hasMoved {
		line += " — " + output.Bytes(int64(moved))
		if total, ok := number(fields["total_bytes"]); ok && total > 0 {
			line += " of " + output.Bytes(int64(total))
		}
	}
	if rate, ok := number(fields["rate_bytes_per_second"]); ok && rate > 0 {
		line += " · " + output.Bytes(int64(rate)) + "/s"
	}
	if remaining, ok := number(fields["remaining_ms"]); ok && remaining > 0 {
		line += " · ETA ~" + shortDuration(time.Duration(remaining)*time.Millisecond)
	} else if !hasMoved {
		if elapsed, ok := number(fields["elapsed_ms"]); ok && elapsed > 0 {
			line += " — " + shortDuration(time.Duration(elapsed)*time.Millisecond)
		}
	}
	return line
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
// machine surface is awaited --json, which emits the complete JSON envelope unchanged.
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
	case "rentals", "placement":
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
		shownStatus = humanCancellationStatus(life.CanceledBy)
	} else if life.Status == "canceled" {
		shownStatus = "cancelled"
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
	humanStatus := status
	if status == "canceled" {
		humanStatus = humanCancellationStatus(life.CanceledBy)
	}
	e := exit.Named(code, status, "request %s ended %s", life.RequestID, humanStatus)
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
			life.RequestID, humanStatus, errType, why)
	}
	// A canceled run is LOUD about WHO ended it (cl-108) — never a quiet no-output end.
	if mapTerminal(status) == "canceled" && life.CanceledBy != "" {
		e.Message = fmt.Sprintf("request %s was %s", life.RequestID, humanCancellationStatus(life.CanceledBy))
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
	remedy := "triage bundle " + ref.SubjectID + " kept; read it with `GET " + ref.URL + "`"
	c, problem := dial(ctx)
	if problem != nil {
		return remedy
	}
	data, problem := c.Triage(ref.AttemptKey)
	if problem != nil {
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

// requestKey preserves an explicit key or mints this invocation's key. One key names one request forever, so
// a fresh invocation gets a fresh one and a caller that wants retry safety across process
// restarts passes its own.
func requestKey(supplied string) string {
	if supplied != "" {
		return supplied
	}
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
// The same validation and argument help apply to serving functions and jobs,
// before either path resolves or acquires a model.
func validateInvocationPayload(ctx *Context, pkg string, ep *launch.Entrypoint, input json.RawMessage) *exit.Error {
	problem := launch.ValidatePayload(pkg, ep, input)
	if problem != nil && !ctx.Mode().JSON {
		problem.Message += "\n\n" + launch.DescribeArguments(ep)
	}
	return problem
}

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
	Snapshot  bool
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
	if isScriptTarget(ctx.Inv.Args[0]) {
		return scriptTarget(ctx)
	}
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
		installed, problem := installedPackage(ctx, target.Package)
		if problem != nil && problem.Code != exit.NotFound {
			return Target{}, nil, problem
		}
		release := ""
		if installed != nil {
			release = installed.Version
		} else {
			card, problem := catalog.PackageCard(hctx, ref)
			if problem != nil {
				return Target{}, nil, problem
			}
			release, problem = newestPackageRelease(card.Releases)
			if problem != nil {
				return Target{}, nil, problem
			}
		}
		detail, problem := catalog.PackageRelease(hctx, ref, release)
		if problem != nil {
			return Target{}, nil, problem
		}
		packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
		if problem != nil {
			return Target{}, nil, exit.Named(exit.Conflict, "rental.package_interface_invalid",
				"Tensorhub returned an invalid package interface: %s", problem.Message)
		}
		if packageInterface.Digest != detail.Release.PackageInterfaceDigest ||
			detail.Release.PackageInterfaceLength != int64(len(detail.PackageInterface)) {
			return Target{}, nil, exit.Named(exit.Conflict, "rental.package_interface_invalid",
				"Tensorhub package interface does not match its committed digest or length")
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
	rows, problem := invocationDefaultBindings(ctx, target, ep.Models)
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
	slots := declaredModelSlots(packageInterface.Entrypoints)
	defaults := effectiveModelBindings(slots, nil)
	var bindingProblem *exit.Error
	if len(slots) > 0 && !strings.HasPrefix(target.Package, "local/") {
		defaults, bindingProblem = invocationDefaultBindings(ctx, target, slots)
	}
	list := output.List{Name: "functions", Fields: []string{"function", "availability"}, AllFields: []string{"function", "availability"}}
	for _, name := range packageInterface.Names() {
		callable, _ := packageInterface.Function(name)
		availability := modelDefaultAvailability(callable, defaults)
		if bindingProblem != nil && callable.Kind != "job" && len(callable.Models) > 0 {
			availability = "unknown: defaults unavailable"
		}
		list.Rows = append(list.Rows, map[string]string{"function": name, "availability": availability})
		if availability == "available" && len(list.Next) < 2 {
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
