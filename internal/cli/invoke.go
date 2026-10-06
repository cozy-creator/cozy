package cli

import (
	"bytes"
	"cmp"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/mattn/go-isatty"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
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
	if ctx.foregroundClient != nil {
		return ctx.foregroundClient, nil
	}
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

// foregroundTarget takes a named --rental the running daemon cannot carry (it predates
// cozy.machine.v1) as this process's own run: the rental's machine as an explicit endpoint,
// signed with the rental's own key. close ends the controller; nil: the daemon carries it.
func foregroundTarget(ctx *Context) (func(), *exit.Error) {
	name := ctx.Inv.Value("--rental")
	if name == "" || ctx.Inv.Value("--machine-endpoint-file") != "" {
		return nil, nil
	}
	ep, problem := foregroundRental(ctx, name)
	if problem != nil || ep == nil {
		return nil, problem
	}
	adoptRentalHub(ctx, name)
	close, problem := endpointController(ctx, ep)
	if problem != nil {
		return nil, problem
	}
	delete(ctx.Inv.Values, "--rental")
	return close, nil
}

func handleRunExecute(ctx *Context) *exit.Error {
	close, problem := foregroundTarget(ctx)
	if problem != nil {
		return problem
	}
	if close != nil {
		defer close()
		if problem := validateRunPlacement(ctx); problem != nil {
			return problem
		}
		target, packageInterface, problem := invocationTarget(ctx)
		if problem != nil {
			return problem
		}
		return runTarget(ctx, target, packageInterface)
	}
	if path := ctx.Inv.Value("--machine-endpoint-file"); path != "" {
		if rentalRequested(ctx) {
			return exit.Usagef("--machine-endpoint-file cannot select or buy a rental")
		}
		close, problem := startEndpointController(ctx, path)
		if problem != nil {
			return problem
		}
		defer close()
	}
	if problem := validateRunPlacement(ctx); problem != nil {
		return problem
	}
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	target, packageInterface, problem := invocationTarget(ctx)
	if problem != nil {
		return problem
	}
	return runTarget(ctx, target, packageInterface)
}

// runTarget runs one resolved target; ctx.Inv.Args is the target followed by its terms.
func runTarget(ctx *Context, target Target, packageInterface *launch.PackageInterface) *exit.Error {
	defer func() { reclaimSnapshot(ctx, target) }()
	if target.Function == "" {
		return emitFunctions(ctx, target, packageInterface)
	}
	callable, problem := packageInterface.Function(target.Function)
	if problem != nil {
		return unknownFunction(target, packageInterface)
	}
	if problem := callable.RequirePublic(); problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--describe") {
		return emitDescribe(ctx, target, packageInterface, callable)
	}
	if level := ctx.Inv.Value("--warm"); level != "" {
		return handleWarm(ctx, target, callable, level)
	}
	if callable.Kind == "job" && strings.HasPrefix(target.Package, "local/") && !target.Snapshot {
		resolved := target.lease
		target, packageInterface, problem = snapshotLocalJob(ctx, target)
		resolved.Release() // the job runs its own snapshot install
		if problem != nil {
			return problem
		}
		callable, problem = packageInterface.Function(target.Function)
		if problem != nil {
			return unknownFunction(target, packageInterface)
		}
		if problem := callable.RequirePublic(); problem != nil {
			return problem
		}
	}
	if callable.Kind != "job" {
		if len(ctx.Inv.Values["--allow-upload"]) > 0 {
			return exit.Usagef("--allow-upload applies only to Runtime-owned job transactions")
		}
		if ctx.Inv.Value("--timeout") != "" && !ctx.Inv.Bool("--await") {
			return exit.Usagef("--timeout requires --await for serving callables").
				WithRemedy("a detached serving call has no client waiting to enforce a caller deadline")
		}
		if ctx.Inv.Value("--retry") != "" {
			return exit.Usagef("--retry applies only to job transactions")
		}
		if ctx.Inv.Value("--upload-to") != "" {
			return exit.Usagef("--upload-to applies only to job callables")
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
	if ctx.Inv.Bool("--rent-new") && (ctx.Inv.Bool("--rental") || ctx.Inv.Value("--rental") != "" || ctx.Inv.Value("--retry") != "") {
		return exit.Usagef("--rent-new cannot be combined with a named --rental or retained --retry")
	}
	if ctx.Inv.Bool("--rental") && ctx.Inv.Bool("--rental-only") {
		return exit.Usagef("--rental and --rental-only are mutually exclusive")
	}
	if ctx.Inv.Value("--rental") != "" && (ctx.Inv.Bool("--rental") || ctx.Inv.Bool("--rental-only")) {
		return exit.Usagef("a named --rental cannot be combined with --rental-only")
	}
	return nil
}

// remoteRun is a run on another computer: a rental, or the endpoint a foreground run took
// from its --rental. Its preparation never needs this computer's machine.
func remoteRun(ctx *Context) bool {
	return ctx.endpoint != nil || rentalRequested(ctx)
}

func rentalRequested(ctx *Context) bool {
	return ctx.Inv.Bool("--rent-new") || ctx.Inv.Bool("--rental") || ctx.Inv.Bool("--rental-only") || ctx.Inv.Value("--rental") != ""
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
	if legacy := launch.LegacyFileTerm(ctx.Inv.Args[1:]); legacy != "" {
		return exit.Named(exit.Usage, "remote_file_input_ambiguous",
			"%s embeds file bytes into a JSON string and cannot name a remote input grant", legacy).
			WithRemedy("use `--asset <field-path>=<file>`; the field path becomes the exact worker-protocol input id")
	}
	input, overrides, e := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	if pin := ctx.Inv.Value("--attention-kernel"); pin != "" {
		if overrides.AttentionKernel != "" && overrides.AttentionKernel != pin {
			return exit.Usagef("attention kernel was pinned as both %s and %s", overrides.AttentionKernel, pin)
		}
		if problem := launch.ValidateAttentionOverride(pin); problem != nil {
			return problem
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
	input, ignored, e := validateInvocationPayload(ctx, target.Package, ep, input)
	if e != nil {
		return e
	}
	if ctx.endpoint != nil {
		normalized, err := canonical.NormalizeApplication(input)
		if err != nil {
			return exit.New(exit.Validation, "explicit machine payload cannot be canonicalized: %s", err)
		}
		input = normalized
	}
	selectedRental, e := requestedRental(ctx)
	if e != nil {
		return e
	}
	if ctx.endpoint != nil {
		packagePublishStatus(ctx, "Execution target: machine %s at %s", ctx.endpoint.WorkerID, ctx.endpoint.Address)
	} else if !managedRental {
		packagePublishStatus(ctx, "Execution target: local machine")
	} else if selectedRental != "" {
		packagePublishStatus(ctx, "Execution target: rental %s", ctx.Inv.Value("--rental"))
	} else {
		packagePublishStatus(ctx, "Finding a rental machine...")
	}
	models, chosen, e := modelChoices(ctx, target, ep, overrides.Models)
	if e != nil {
		return e
	}
	if sourced(models) && (!chosen || managedRental && selectedRental == "") {
		return exit.Usagef("a provider-source model runs on a named machine").
			WithRemedy("add --rental=<name>; the machine resolves, narrows and converts the source itself")
	}
	if !chosen || managedRental && selectedRental == "" {
		// Choosing a machine to rent reads the ladders; so do editable code and provider
		// sources.
		children := capturedModelChoices(models)
		if models, e = resolveInvocationModels(ctx, target, ep, overrides.Models); e != nil {
			return e
		}
		models = append(models, children...)
	}
	if models, e = applyModelAdapters(ctx, target, ep, models, overrides.Overlays); e != nil {
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
	began := ctx.commandStarted
	submitting := time.Now()
	handle, e := c.Submit(api.Submission{
		MachineEndpoint: ctx.endpoint,
		Package:         target.Package, Function: target.Function, Input: input,
		LocalAssets: assets, InstallID: target.InstallID,
		Release: target.Release, Rental: managedRental,
		RentalRequired:  ctx.Inv.Bool("--rental-only") || ctx.Inv.Bool("--rent-new") || selectedRental != "",
		RentNew:         ctx.Inv.Bool("--rent-new"),
		RequestedRental: selectedRental,
		Models:          models,
		OutputDirectory: outputDirectory,
		AttentionKernel: overrides.AttentionKernel,
		Ignored:         ignored,
	}, key)
	releaseSnapshotReader(target)
	if e != nil {
		return e
	}
	submitted := time.Since(submitting)
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
		notice := &reattachNotice{ctx: ctx}
		c = c.Following(notice.say)
		terminal, stopped, e = watch(ctx, c, c.Cancel, handle.RequestID, deadline, began, notice)
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
	return renderRun(ctx, life, terminal, stopped, saved, submitted)
}

// modelChoices are a published call's explicit `model.<param>=` selections, parsed and
// nothing more: the machine that runs the call resolves them and every other slot for its
// own devices. A provider source is only pinned to its commit (one metadata read, none when
// the caller named the commit); the machine narrows and converts it. It answers false for a
// call the machine cannot take that way.
func modelChoices(ctx *Context, target Target, ep *launch.Entrypoint, overrides map[string]string) ([]orchestrator.ModelRef, bool, *exit.Error) {
	profiles, problem := parseSourceProfileFlags(ctx)
	if problem != nil {
		return nil, false, problem
	}
	var out []orchestrator.ModelRef
	slots := append([]launch.Slot(nil), ep.Models...)
	for selector := range overrides {
		if slices.ContainsFunc(slots, func(slot launch.Slot) bool { return slot.Path == selector }) {
			continue
		}
		if _, path, ok := launch.CapturedModelSlot(selector); ok {
			_, parameter, _ := strings.Cut(path, ".models.")
			slots = append(slots, launch.Slot{Path: selector, Param: parameter})
		}
	}
	sort.Slice(slots, func(i, j int) bool { return slots[i].Path < slots[j].Path })
	for _, slot := range slots {
		raw, chosen := overrides[slot.Path]
		parameter := slot.Path[strings.LastIndex(slot.Path, ".")+1:]
		profileKey := parameter
		if choice := modelChoiceSlot(target, slot.Path); choice.Callable != "" {
			profileKey = slot.Path
		}
		profile, profiled := profiles[profileKey]
		delete(profiles, profileKey)
		if !chosen {
			if profiled {
				return nil, false, exit.Usagef("--source-profile %s=%s names no provider-source model", parameter, profile)
			}
			continue
		}
		if strings.Contains(raw, "://") {
			source, problem := pinnedProviderSource(ctx, raw)
			if problem != nil {
				return nil, false, problem
			}
			row := modelChoiceSlot(target, slot.Path)
			row.Source = source
			if profiled {
				row.Profiles = []string{profile}
			}
			out = append(out, row)
			continue
		}
		if profiled {
			return nil, false, exit.Usagef("--source-profile %s=%s names no provider-source model", parameter, profile)
		}
		model, release, lane, manifest, problem := hub.ParseModelRef(raw)
		if problem != nil {
			return nil, false, problem
		}
		row := modelChoiceSlot(target, slot.Path)
		row.Model, row.CatalogRepository, row.Release, row.Lane, row.Manifest = model, model, release, lane, manifest
		out = append(out, row)
	}
	for parameter := range profiles {
		return nil, false, exit.Usagef("--source-profile %s names no model parameter of %s", parameter, target.Function)
	}
	// Unpublished code is a root naming its installation, whose open slots the machine
	// resolves under the owner the root names. Code that calls other unpublished code goes by
	// capture instead, and resolves its own slots, unless every slot names one provider source.
	if (strings.HasPrefix(target.Package, "local/") || target.Snapshot) && !oneSource(ep, out) {
		calls, problem := callsUnpublished(ctx, target.InstallID)
		if problem != nil || calls {
			return capturedModelChoices(out), false, problem
		}
	}
	return out, true, nil
}

// callsUnpublished answers whether an installation's callables reach another unpublished
// installation, which only a capture carries to a machine.
func callsUnpublished(ctx *Context, install string) (bool, *exit.Error) {
	_, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return false, problem
	}
	defer store.Close()
	rows, problem := store.ChildBindings(install)
	if problem != nil {
		return false, problem
	}
	return slices.ContainsFunc(rows, func(row records.ChildBinding) bool { return row.ChildInstallID != install }), nil
}

func sourced(models []orchestrator.ModelRef) bool {
	return slices.ContainsFunc(models, func(model orchestrator.ModelRef) bool { return model.Source != "" })
}

// oneSource answers whether every Model slot of ep names the same provider source.
func oneSource(ep *launch.Entrypoint, models []orchestrator.ModelRef) bool {
	if len(models) == 0 || len(models) != len(ep.Models) {
		return false
	}
	for _, model := range models {
		if model.Source == "" || model.Source != models[0].Source {
			return false
		}
	}
	return true
}

// pinnedProviderSource spells a Hugging Face or Civitai source at an immutable revision.
func pinnedProviderSource(ctx *Context, raw string) (string, *exit.Error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", exit.Internalf("cannot resolve the current directory: %s", err)
	}
	source, problem := modelsource.Parse(raw, cwd)
	if problem != nil {
		return "", problem
	}
	if source.Kind != modelsource.HuggingFace && source.Kind != modelsource.Civitai {
		return "", exit.Usagef("model input %q is not a Tensorhub, Hugging Face, or Civitai reference", raw)
	}
	token := ctx.Cfg.HuggingFaceToken
	if source.Kind == modelsource.Civitai {
		token = ctx.Cfg.CivitaiToken
	}
	resolver, problem := modelsource.NewResolver(source.Kind, token)
	if problem != nil {
		return "", problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	if source, problem = resolver.Pin(hctx, source); problem != nil {
		return "", problem
	}
	return source.Canonical, nil
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

// resolveInvocationModels applies the one selection order for every machine: an explicit
// `model.<param>=` run key, then a Hub owner override, then the selected callable's
// authored default ladder. The whole ladder travels; the machine that runs the call pins
// its rung from the devices it measured. Editable packages use authored defaults without a
// Hub override.
func resolveInvocationModels(ctx *Context, target Target, ep *launch.Entrypoint,
	overrides map[string]string,
) ([]orchestrator.ModelRef, *exit.Error) {
	selected, problem := invocationModelSpecs(ctx, target, ep, overrides)
	if problem != nil || len(selected) == 0 {
		return nil, problem
	}
	return resolveSelectedInvocationModels(ctx, target, ep, selected)
}

func resolveSelectedInvocationModels(ctx *Context, target Target, ep *launch.Entrypoint,
	selected []invocationModelSpec,
) ([]orchestrator.ModelRef, *exit.Error) {
	slots := make(map[string]launch.Slot, len(ep.Models))
	for _, slot := range ep.Models {
		slots[slot.Path] = slot
	}
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
		defaults = effectiveModelBindings(ep.Models, nil, packageOwner(target.Package, ep.Models, commandNamespace(ctx)))
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
			if editable && slot.RelativeDefault() {
				_, problem := commandNamespace(ctx)()
				message := "the caller on this Tensorhub is unknown"
				if problem != nil {
					message = problem.Message
				}
				return nil, exit.Named(exit.Unavailable, "package_model_default_owner_unknown",
					"%s default %s for model slot %s names your account's model, but %s",
					target.Package, slot.DefaultBinding.Ref(), slot.Path, message).
					WithRemedy("sign in on %s, or override this run: model.%s=org/model@release[/lane]", ctx.Cfg.HubURL, slot.Param)
			}
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

// invocationDefaultBindings reads the package's CURRENT default bindings for these slots from
// the hub (th-116, cl-166): mutable owner-written rows, each a model release and its ladder.
// Owner rows take precedence; only absent rows use the selected callable metadata. A Hub that
// cannot answer (one without owner bindings, or out of reach) leaves the release's declared
// defaults, with a warning.
func invocationDefaultBindings(ctx *Context, target Target, slots []launch.Slot) (
	map[string]hub.PackageBindingRow, *exit.Error,
) {
	ref, problem := hub.ParseRef(target.Package)
	if problem != nil {
		return nil, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	paths := make([]string, len(slots))
	for i, slot := range slots {
		paths[i] = slot.Path
	}
	rows, problem := client(ctx).PackageBindings(hctx, ref, paths...)
	if problem != nil && problem.ErrName() != "hub.package_bindings_invalid" {
		fmt.Fprintf(ctx.Err, "warning: %s owner bindings are not readable (%s); using the release's declared defaults\n", target.Package, problem.Message)
		rows, problem = nil, nil
	}
	if problem != nil {
		remedy := "choose the model per run: model.<param>=org/model@release"
		if problem.Remedy != "" {
			remedy = problem.Remedy + " — or " + remedy
		}
		return nil, exit.Named(problem.Code, "package_default_model_unavailable",
			"%s default bindings are unusable: %s", target.Package, problem.Message).
			WithRemedy("%s", remedy)
	}
	return effectiveModelBindings(slots, rows, ref.Org), nil
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
		return empty, localModelOnMachine(ref)
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
			Model: ref.String(), CatalogRepository: ref.String(), Manifest: manifest, HubCheckpoint: true,
			ManifestLength: resolved.ManifestLength, Bytes: resolved.Bytes,
			ComponentBytes: resolved.ComponentBytes, ComponentUse: slot.ComponentUse}, nil
	}
	_, selected, problem := modelReleaseCardForLane(hctx, client(ctx), ref, release, wantedLane)
	if problem != nil {
		return empty, problem
	}
	warnYanked(ctx, ref, selected)
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
		if len(manifestLanes) != 1 && binding != nil && binding.Model == ref.String() {
			// Several lanes and no lane named: the owner's ladder for this model picks
			// one per machine, exactly as it does for an unselected slot.
			if rungs, problem := ladderRungs(ctx, ref, selected, slot, raw, binding.Ladder); problem == nil && len(rungs) > 0 {
				return orchestrator.ModelRef{Package: packageName, Slot: slot.Path, Model: ref.String(),
					CatalogRepository: ref.String(), Release: release, ComponentUse: slot.ComponentUse, Ladder: rungs}, nil
			}
		}
		if len(manifestLanes) != 1 {
			return empty, exit.Usagef("model %s@%s has %d manifests", ref.String(), release, len(manifestLanes)).
				WithRemedy("append /<lane> or #sha256:<digest> to select one exact manifest")
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
		Model: ref.String(), CatalogRepository: ref.String(), Release: release, Lane: lanes[0], Manifest: manifest,
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
		rungs = append(rungs, records.ModelRung{GPU: rung.GPU, GPUs: rung.GPUs, Lane: rung.Lane,
			Manifest: lane.ManifestID, Bytes: lane.Bytes, ComponentBytes: lane.ComponentBytes})
	}
	return rungs
}

func handleRunCancel(ctx *Context) *exit.Error {
	if ctx.Inv.Bool("--abandon") {
		return handleRunAbandon(ctx)
	}
	id := ctx.Inv.Args[0]
	close, problem := endpointForRecordedRun(ctx, id)
	if problem != nil {
		return problem
	}
	if close != nil {
		defer close()
	}
	if strings.HasPrefix(id, "job-") {
		return cancelJob(ctx)
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	// The recorded run decides the route; the daemon reads the machine's own state right
	// before it sends the cancel, so this read does not dial the machine first.
	before, problem := client.RecordedRequest(id)
	if problem != nil {
		return problem
	}
	if before.Kind == "job" {
		return cancelJob(ctx)
	}
	// A settled run holding nothing is final. A completed machine run holds nothing a cancel
	// would release unless its machine keeps the result: this host's own collection, in
	// flight or done, releases the rest.
	if invocationSettled(before.Status) && (!before.Retaining || collectedOrCollecting(before)) {
		fields := append(invocationFields(before), output.Field{K: "changed", V: false})
		return emit(ctx, compactRecord(fields, "number", "target", "status", "changed"))
	}
	if problem := client.Cancel(id, "cozy run cancel"); problem != nil {
		return problem
	}
	if canceling, problem := client.RecordedRequest(id); problem != nil {
		return problem
	} else if problem := startForCancel(ctx, canceling.Number, id, canceling.Status, canceling.MachineExecution); problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--await") {
		if _, problem := client.Watch(id, 0, func(localapi.Event) bool { return true }); problem != nil {
			return problem
		}
	}
	after, problem := client.RecordedRequest(id)
	if problem != nil {
		return problem
	}
	// Changed is what the cancel did: the run ended canceled, or the work it held was
	// released. A run that finished first keeps its own terminal.
	changed := after.Status != before.Status || (before.Retaining && !after.Retaining)
	fields := append(invocationFields(after), output.Field{K: "changed", V: changed})
	defaults := []string{"number", "target", "status", "changed"}
	if after.CanceledBy != "" {
		defaults = append(defaults, "canceled_by")
	}
	return emit(ctx, compactRecord(fields, defaults...))
}

func handleRunAbandon(ctx *Context) *exit.Error {
	if ctx.Inv.Bool("--await") {
		return exit.Usagef("--abandon is local only and cannot be combined with --await; it does not confirm remote cancellation")
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	result, problem := client.Abandon(ctx.Inv.Args[0], "cozy run cancel --abandon")
	if problem != nil {
		return problem
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "id", V: result.ID}, {K: "number", V: result.Number}, {K: "status", V: result.Status}, {K: "changed", V: result.Changed},
		{K: "abandoned_locally", V: true}, {K: "remote_stop_confirmed", V: false}, {K: "rental_released", V: false},
	}, Notes: []string{"Local abandonment does not confirm remote work stopped or end any rental."}})
}

func handleRunList(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	every := everyHub(ctx)
	if every {
		client = client.AllHubs()
	}
	limit := 0
	if raw := ctx.Inv.Value("--limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 0 {
			return exit.Usagef("--limit %q must be zero (all history) or a positive number", raw)
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
	if ctx.Inv.Value("--limit") == "" {
		limit = 50
	}
	list, older, problem := runList(context.Background(), client, ctx.Inv.Value("--state"), ctx.Inv.Value("--package"), limit)
	if problem != nil {
		return problem
	}
	nameHubs(ctx.Cfg, &list)
	switch {
	case older != nil:
		list.Total, list.More = len(list.Rows)+*older, "Use --limit 0 to show all."
	case len(list.Rows) == limit:
		// A daemon that does not count what it left out.
		list.Trail = append(list.Trail, fmt.Sprintf("Showing the newest %d runs; older ones may not be shown. Use --limit 0 to show all.", limit))
	}
	return emit(ctx, list)
}

// nameHubs names each run's hub where it has a name, for --full. A run goes straight to its
// machine, which `run list` shows; the hub that rented the machine is not a column.
func nameHubs(cfg config.Config, list *output.List) {
	for _, row := range list.Rows {
		row["hub"] = cfg.HubLabel(row["hub"])
	}
}

// runList reads up to limit runs (0: all), newest first, and how many matching runs it left
// out: nil when the daemon does not count them.
func runList(requestCtx context.Context, client *localapi.Client, state, packageName string, limit int) (output.List, *int, *exit.Error) {
	pkg := strings.TrimSpace(packageName)
	var rows []api.Lifecycle
	var before int64
	none := 0
	older := &none
	for {
		pageSize := 500
		if limit > 0 {
			pageSize = min(pageSize, limit-len(rows))
		}
		history, problem := client.RequestPage(requestCtx, state, pkg, pageSize, before)
		if problem != nil {
			return output.List{}, nil, problem
		}
		page := history.Requests
		if len(page) == 0 {
			break
		}
		next := page[len(page)-1].Number
		if next < 1 || (before > 0 && next >= before) {
			return output.List{}, nil, exit.New(exit.Conflict, "run history pagination did not advance; restart the Cozy daemon to load the current API")
		}
		rows = append(rows, page...)
		if len(page) < pageSize {
			break
		}
		if limit > 0 && len(rows) >= limit {
			older = history.Older
			break
		}
		before = next
	}
	return runListRows(rows), older, nil
}

func runListRows(rows []api.Lifecycle) output.List {
	list := output.List{
		Uncapped: true,
		Name:     "invocations", Fields: []string{"number", "target", "machine", "status", "progress", "execution", "reason"},
		AllFields: []string{"number", "id", "kind", "target", "machine", "rental_id", "status",
			"progress", "phase", "progress_stage", "stage_fraction", "overall_fraction",
			"position", "total", "queued", "execution", "attempt_wall", "attempts", "created", "reason", "hub"},
		TypedFields: []string{"number", "target", "machine", "rental_id", "requested_rental", "requested_machine", "status",
			"phase", "progress_stage", "stage_fraction", "overall_fraction", "position", "total",
			"remaining_ms", "execution_ms", "execution_known", "error_type", "error_code", "error", "retaining", "retry_available"},
		TypedAllFields: []string{"number", "id", "kind", "target", "machine", "rental_id", "requested_rental", "requested_machine",
			"status", "canceled_by", "phase", "phase_machine", "waiting_for", "rental_boot", "wait_reason", "phase_elapsed_ms",
			"phase_moved_bytes", "phase_total_bytes", "phase_rate_bytes_per_second",
			"phase_remaining_ms", "phase_sample_age_ms", "progress_stage", "stage_fraction", "overall_fraction",
			"position", "total", "progress_unit", "progress_rate", "remaining_ms", "queued_ms", "execution_ms", "execution_known", "attempt_wall_ms", "attempts",
			"created_at", "error_type", "error_code", "error", "triage", "retaining", "retry_available", "hub"},
		TypedRows: make([]map[string]any, 0, len(rows)),
		// The raw rental id is a machine fact: JSON always carries it, the compact
		// human table never does — the human word is the MACHINE column (cl-107).
		Machine: []string{"rental_id"},
	}
	states := map[string]int{}
	for _, life := range rows {
		life.Status = publicObservedStatus(life.Status)
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
			"execution":        executionValue(life.ExecutionMS, life.ExecutionKnown, life.MachineExecution != nil),
			"attempt_wall":     seconds(life.AttemptWallMS),
			"attempts":         strconv.Itoa(life.Attempts), "created": life.CreatedAt,
			"reason": reasonCell(life),
		})
		typed := map[string]any{
			"number": life.Number, "id": life.RequestID, "kind": kind,
			"target": life.Package + "/" + life.Function, "machine": life.Machine,
			"status": life.Status, "queued_ms": life.QueuedMS, "execution_ms": life.ExecutionMS,
			"attempts": life.Attempts, "created_at": life.CreatedAt,
		}
		if life.MachineExecution != nil {
			typed["execution_known"], typed["attempt_wall_ms"] = life.ExecutionKnown, life.AttemptWallMS
			if !life.ExecutionKnown {
				typed["execution_ms"] = nil
			}
		}
		if life.Hub != "" {
			list.Rows[len(list.Rows)-1]["hub"], typed["hub"] = life.Hub, life.Hub
		}
		if life.Retaining {
			typed["retaining"] = true
		}
		if life.RetryAvailable {
			typed["retry_available"] = true
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
			if life.PhaseSampleAgeMS != nil {
				typed["phase_sample_age_ms"] = *life.PhaseSampleAgeMS
			}
			if life.PhaseRate != nil {
				typed["phase_rate_bytes_per_second"] = *life.PhaseRate
			}
			if life.PhaseRemainingMS != nil {
				typed["phase_remaining_ms"] = *life.PhaseRemainingMS
			}
		}
		if life.RentalBoot != nil {
			typed["rental_boot"] = life.RentalBoot
		}
		if life.WaitReason != "" {
			typed["wait_reason"] = life.WaitReason
		}
		if life.CanceledBy != "" {
			typed["canceled_by"] = life.CanceledBy
		}
		for key, value := range map[string]string{"error_type": life.ErrorType,
			"error_code": life.ErrorCode, "error": life.Error} {
			if value != "" {
				typed[key] = value
			}
		}
		if life.Triage != nil {
			typed["triage"] = life.Triage
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
			if life.ProgressUnit != "" {
				typed["progress_unit"] = life.ProgressUnit
			}
			if life.ProgressRate != nil {
				typed["progress_rate"] = *life.ProgressRate
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
	return list
}

// reasonCell is why a run is where it is, cut to one line: for a queued run what it waits
// on (its rental's boot, else its newest park), for a run that ended without its result the
// recorded cause. `cozy run show <number>` shows it whole.
func reasonCell(life api.Lifecycle) string {
	reason := ""
	switch {
	case life.Status == "queued" && life.RentalBoot != nil:
		reason = describeBoot(bootMachine(life), life.RentalBoot, time.Now()).facts()
	case life.Status == "queued":
		reason = life.WaitReason
	case life.Status != "completed" && life.Status != "in_progress":
		// The list already identifies the failed run; keep its specific cause.
		reason = strings.TrimPrefix(life.Error, "Runtime refused machine execution: ")
	}
	reason = strings.Join(strings.Fields(reason), " ")
	if runes := []rune(reason); len(runes) > failureReasonCell {
		reason = string(runes[:failureReasonCell-1]) + "…"
	}
	return reason
}

// bootMachine is the machine a queued run's rental boot is for.
func bootMachine(life api.Lifecycle) string {
	return either(life.PhaseMachine, either(life.Machine, life.RentalID))
}

const failureReasonCell = 72

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
	if life.RentalBoot != nil {
		w := describeBoot(bootMachine(life), life.RentalBoot, time.Now())
		return joinParts(w.subject(bootMachine(life)), w.elapsed)
	}
	if life.Phase == orchestrator.PhaseGPUWait {
		detail := life.PhaseDetail
		if life.WaitingFor != nil {
			detail += fmt.Sprintf(", behind run %d", life.WaitingFor.Number)
		}
		return "waiting for GPU (" + detail + ")"
	}
	if life.Phase == orchestrator.PhaseOwnerReconciliation {
		return "awaiting owner reconciliation: machine authorization expired (publication " + life.PhaseDetail + ")"
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
	activity := preparationLabel(life.Phase)
	activity = strings.ReplaceAll(activity, "_", " ")
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
	if life.PhaseSampleAgeMS != nil && time.Duration(*life.PhaseSampleAgeMS)*time.Millisecond > orchestrator.PreparationRateMaxAge {
		parts = append(parts, "last update "+shortDuration(time.Duration(*life.PhaseSampleAgeMS)*time.Millisecond)+" ago")
		return strings.Join(parts, " · ")
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
	if life.Phase == orchestrator.PhaseGPUWait || life.Phase == orchestrator.PhaseOwnerReconciliation {
		return phaseValue(life)
	}
	if life.Status == "queued" {
		if life.RentalBoot != nil {
			return phaseValue(life)
		}
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
	stage := stageLabel(life.ProgressStage)
	if life.ProgressUnit == "bytes" && life.Position != nil && life.Total != nil {
		// A byte stage has no whole-run fraction; its moved/total and rate ARE the number.
		parts := []string{output.Bytes(*life.Position) + " / " + output.Bytes(*life.Total)}
		if stage != "" {
			parts = append([]string{stage}, parts...)
		}
		if life.ProgressRate != nil {
			parts = append(parts, output.Bytes(int64(*life.ProgressRate))+"/s")
		}
		stage = strings.Join(parts, " · ")
	}
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
	cfg := ctx.Cfg
	history := runHistory{client: client, state: ctx.Inv.Value("--state"), packageName: ctx.Inv.Value("--package"), limit: limit, more: true, hubs: &cfg}
	return watchListPages(ctx, "id", history.refresh, history.next)
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

// custodyOwed is whether a finished machine run's result has yet to settle where it stays.
func collectedOrCollecting(life api.Lifecycle) bool {
	view := life.MachineExecution
	return view != nil && life.Status == "completed" && view.Retained == "" && view.CollectionRefused == ""
}

func custodyOwed(view *api.MachineExecutionView, status string, products int) bool {
	// A failed or canceled run's result is what it published: nothing to wait for without.
	ended := status == "completed" || products > 0 && (status == "failed" || status == "canceled")
	return view != nil && view.Accepted && ended && !view.Collected && view.Retained == "" && view.CollectionRefused == ""
}

// awaitResultCustody follows a finished machine run's events past its terminal until its
// result's custody settles: collected, refused until its owner acts, kept on the machine,
// or lost with it. The daemon's observer collects; a client only listens.
func awaitResultCustody(c *localapi.Client, id string) *exit.Error {
	terminal, problem := c.Watch(id, 0, func(localapi.Event) bool { return true })
	if problem != nil || terminal == nil {
		return problem
	}
	_, problem = c.Watch(id, terminal.SequenceNumber, func(event localapi.Event) bool { return !records.CustodyEvent(event.Type) })
	return problem
}

func waitOutputExport(c *localapi.Client, life api.Lifecycle) (api.Lifecycle, *exit.Error) {
	// A machine run's result is the fold of its products whatever its terminal, so a failed
	// or canceled one exports what it made too. Otherwise publication is an obligation of a
	// SUCCESSFUL execution: failure is an absorbing answer, never success by waiting.
	if life.MachineExecution == nil && (life.Status == "failed" || life.Status == "canceled") {
		return life, nil
	}
	if custodyOwed(life.MachineExecution, life.Status, len(life.Output)) {
		if problem := awaitResultCustody(c, life.RequestID); problem != nil {
			return life, problem
		}
		updated, problem := c.Request(life.RequestID)
		if problem != nil {
			return life, problem
		}
		life = updated
	}
	if view := life.MachineExecution; view != nil && !view.Collected {
		return life, nil // nothing to export until collected; renderRun says why
	}
	for life.OutputExport != nil {
		export := life.OutputExport
		switch export.State {
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
			return life, nil
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

// exportedOutputs joins the published paths to the outputs they carry: each output names its
// stable file, so the join is exact rather than positional.
func exportedOutputs(life api.Lifecycle) []savedFile {
	if life.OutputExport == nil || life.OutputExport.State != "published" {
		return nil
	}
	byPath := make(map[string]savedFile, len(life.Outputs)+len(life.Output))
	for _, o := range life.Outputs {
		byPath[o.Path] = savedFile{Output: o.OutputID, Path: o.Path, Bytes: o.Length, Mime: o.MimeType, Digest: o.Digest}
	}
	// Native products are retained output facts even when nothing was published
	// to a Hub. Join their committed file paths, without inventing a receipt.
	for _, o := range life.Output {
		if o.Status == "completed" && o.Path != "" {
			byPath[o.Path] = savedFile{Output: o.Name, Path: o.Path, Bytes: o.Length, Mime: o.MediaType, Digest: o.Sha256}
		}
	}
	result := make([]savedFile, 0, len(life.OutputExport.Paths))
	for _, path := range life.OutputExport.Paths {
		row := savedFile{Path: path}
		if o, ok := byPath[path]; ok && path != "" {
			row = o
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
// invocationEventObserver is the observation-only view used by awaited runs.
// Keep cancellation out of this dependency: a terminal disconnect is a local
// observer event, never a request to stop durable work.
type invocationEventObserver interface {
	Request(string) (api.Lifecycle, *exit.Error)
	WatchContext(context.Context, string, int64, func(localapi.Event) bool) (*localapi.Event, *exit.Error)
}

func watch(ctx *Context, c invocationEventObserver, cancel func(string, string) *exit.Error, requestID string,
	deadline time.Duration, began time.Time, notice *reattachNotice) (*localapi.Event, string, *exit.Error) {
	interrupt, restoreInput, _, problem := liveSignals(ctx, nil)
	if problem != nil {
		return nil, "", problem
	}
	defer restoreInput()

	watchCtx, stopWatch := context.WithCancel(context.Background())
	defer stopWatch()
	lines := NewProgress(ctx, ctx.Mode().JSON, began)
	if life, e := c.Request(requestID); e == nil && lines.liveMode() {
		lines.describeRun(life)
	}
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
				lines.Detach(fmt.Sprintf(
					"\ndetached — the run keeps running; `cozy run watch %s` reattaches, `cozy run cancel %s` cancels\n",
					requestID, requestID))
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
				lines.Say(fmt.Sprintf(
					"\n--timeout %s expired; cancel requested — the attempt's own terminal still settles it\n",
					deadline))
			}
		case <-done:
			return
		}
		cancelResult := make(chan *exit.Error, 1)
		go func() { cancelResult <- cancel(requestID, fmt.Sprintf("%s %s", deadlineActor, deadline)) }()
		select {
		case <-interrupt:
			if !ctx.Mode().JSON {
				lines.Detach("detached — the cancelled terminal still lands in `cozy run list`\n")
			}
			stopWatch()
		case problem := <-cancelResult:
			if problem != nil {
				if !ctx.Mode().JSON {
					lines.Say(fmt.Sprintf("cancel: %s\n", problem.Message))
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
				lines.Detach("detached — the cancelled terminal still lands in `cozy run list`\n")
			}
			stopWatch()
		case <-done:
		}
	}()

	notice.lines = lines
	var manualStop *localapi.Event
	terminal, e := c.WatchContext(watchCtx, requestID, 0, func(event localapi.Event) bool {
		if event.Type == "request.blocked" {
			state, problem := c.Request(requestID)
			if problem == nil && currentManualStop(state.Status, state.StoppedEventID, event.SequenceNumber) {
				projected := publicFailureEvent(event)
				lines.On(projected)
				manualStop = &projected
				return false
			}
			return true
		}
		return lines.On(event)
	})
	if terminal == nil {
		terminal = manualStop
	}
	lines.Done()
	notice.lines = nil
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
// Awaited `--json` writes NDJSON of the typed envelope on stderr, a terminal redraws the
// run in place (liveView), and a redirected human command gets sparse
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
	view            liveView
	overallSeen     bool
	overallFraction float64
	overallDelta    float64
	overallSeconds  float64
	overallScoped   bool

	// The sparse lane's memory: which tenth of which stage was last appended, and when.
	sparseStage   string
	sparseDecile  int
	sparseAt      time.Time
	sparseStarted time.Time

	// items are the run's output items as their output_item.added events named them.
	items map[string]records.OutputItem
	// rented is a run placed on a rental, whose video a browser can play (`cozy run play`).
	rented bool

	// Injected clock and measure, so tests drive the REAL renderer deterministically.
	now   func() time.Time
	width func() int
}

// outputLines are a run's output_item events as lines (progressive-outputs.md §3): an item's
// file once, when it is added; then one line per revision with its size, media time and what
// it added. A list element never changes, so its one revision is one line.
func (p *RunProgress) outputLines(e localapi.Event) []string {
	raw, _ := json.Marshal(e.Payload)
	switch e.Type {
	case records.OutputItemAdded:
		var added struct {
			Item records.OutputItem `json:"item"`
		}
		if json.Unmarshal(raw, &added) != nil {
			return nil
		}
		if p.items == nil {
			p.items = map[string]records.OutputItem{}
		}
		p.items[added.Item.ID] = added.Item
		var lines []string
		if added.Item.Path != "" {
			lines = append(lines, "  "+itemName(added.Item)+"  "+p.ctx.Mode().Hyperlink(added.Item.Path))
		}
		if p.rented && added.Item.Type == "video" {
			play := "cozy run play " + e.RequestID
			if added.Item.Name != "video" {
				play += " --output " + added.Item.Name
			}
			lines = append(lines, "  "+itemName(added.Item)+"  play in a browser: "+play)
		}
		return lines
	case records.OutputItemDelta:
		var delta records.OutputDelta
		if json.Unmarshal(raw, &delta) != nil {
			return nil
		}
		item := p.items[delta.ItemID]
		if item.Index > 0 {
			return []string{"  " + itemName(item) + "  " + strings.TrimSpace(output.Bytes(delta.Length)+"  "+delta.Label)}
		}
		line := fmt.Sprintf("  %s r%d  %s", itemName(item), delta.Rev, output.Bytes(delta.Length))
		if delta.DurationUs > 0 {
			line += "  " + mediaTime(delta.DurationUs)
		}
		if delta.AppendedFrom != nil {
			line += "  +" + output.Bytes(delta.Length-*delta.AppendedFrom)
		}
		if delta.Label != "" {
			line += "  " + delta.Label
		}
		return []string{line}
	}
	return nil
}

// itemName is an item as the CLI names it: its output, and a list element's 1-based index.
func itemName(item records.OutputItem) string {
	if item.Index > 0 {
		return fmt.Sprintf("%s %d", item.Name, item.Index)
	}
	return item.Name
}

// mediaTime is a media duration as m:ss.
func mediaTime(us uint64) string {
	seconds := us / 1_000_000
	return fmt.Sprintf("%d:%02d", seconds/60, seconds%60)
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
	e = machineProgressEvent(e)
	p.mu.Lock()
	defer p.mu.Unlock()
	kind := strings.TrimPrefix(e.Type, "request.")
	if kind == "warning" {
		if !p.ctx.warned(e.Payload) {
			p.notice(e, warningLine(e.Payload))
		}
		return true
	}
	if strings.HasPrefix(e.Type, "output_item.") {
		for _, line := range p.outputLines(e) {
			p.notice(e, line)
		}
		return true
	}
	p.rented = p.rented || kind == "rentals" || kind == "placement"
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
		if p.progressAttempt != 0 && p.liveMode() {
			p.view.retireAll(eventTime(e), true)
		}
		p.progressAttempt = e.Attempt
		p.stepStage, p.stepSeconds, p.stepSamples = "", 0, 0
		p.overallSeen, p.overallFraction, p.overallDelta, p.overallSeconds = false, 0, 0, 0
		p.overallScoped = false
	}
	// A redirected human command has no status line to rewrite: it gets the sparse
	// append lane. --full deliberately restores the complete diagnostic stream.
	if !p.ctx.Mode().Live && !p.ctx.Mode().Full {
		p.sparse(e)
		return true
	}
	if p.liveMode() {
		p.onLive(e)
	} else {
		p.render(diagnosticProgressLine(e))
	}
	return true
}

// machineProgressEvent reads a Runtime-owned execution's imported progress sample in the
// live lane's shape, so a machine run's stages render exactly as a local run's do.
func machineProgressEvent(e localapi.Event) localapi.Event {
	value, ok := e.Payload["payload"].(map[string]any)
	if e.Type != "machine.progress" || e.Payload["type"] != "progress" || !ok {
		return e
	}
	e.Type, e.Payload = "request.progress", map[string]any{"value": value}
	return e
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
	line += facts.countLabel()
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
	bytes            bool
	rate             float64
	hasRate          bool
	perStep          float64
	overallRemaining time.Duration
	hasOverallETA    bool
	scoped           bool
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
	facts := stepFacts{label: label, scoped: strings.Contains(name, " / ")}
	p.overallScoped = p.overallScoped || facts.scoped
	if positionOK != totalOK || positionOK &&
		(position < 0 || total <= 0 || position > total || math.Trunc(position) != position || math.Trunc(total) != total) {
		return facts, false
	}
	if positionOK {
		facts.counted = true
		facts.current = int64(position)
		facts.total = int64(total)
		facts.bytes = fields["unit"] == "bytes"
		facts.rate, facts.hasRate = number(fields["rate"])
		facts.hasRate = facts.hasRate && facts.rate >= 0
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
		// A child range is work allocation, not a prediction that later children
		// take the same time. Keep its measured step ETA scoped to that child.
		if !p.overallScoped && p.overallDelta > 0 && p.overallSeconds > 0 {
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
	parts := strings.Split(strings.TrimSpace(name), " / ")
	if len(parts) > 1 {
		video := strings.HasPrefix(parts[0], "Shot ")
		for i, part := range parts {
			if video && part == "denoise" {
				parts[i] = "Generating video"
			} else {
				parts[i] = stageLabel(part)
			}
		}
		return strings.Join(parts, " · ")
	}
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
		return preparationLabel(strings.TrimSpace(name))
	}
}

// These preparation operations inspect or fetch model inputs; none proves that
// weights are loaded onto a GPU. Unknown author labels remain intact.
func preparationLabel(name string) string {
	switch name {
	case "select_model_defaults":
		return "Selecting model"
	case "materialize_model_defaults":
		return "Preparing model files"
	case "prepare_model_binding", "prep_model_binding":
		return "Checking model compatibility"
	case "checkpoint_input_admission":
		return "Checking model inputs"
	default:
		return name
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
	if p.liveMode() {
		p.settle("", p.now())
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
	case "run.created":
		return ""
	case "dispatched":
		return "  worker selected"
	case "run.in_progress":
		return "  Waiting for a progress update"
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
func warningLine(payload map[string]any) string {
	if message, _ := payload["message"].(string); strings.TrimSpace(message) != "" {
		return "  warning: " + strings.TrimSpace(message)
	}
	return ""
}

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
	stage = stageLabel(stage)
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
		if on != "" {
			return "  waiting for " + on
		}
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
	if boot := bootOf(fields); boot != nil {
		machine, _ := fields["machine"].(string)
		return "  " + describeBoot(machine, boot, time.Now()).line(machine)
	}
	line := "  " + strings.ReplaceAll(preparationLabel(name), "_", " ")
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
	if age, ok := number(fields["sample_age_ms"]); ok && time.Duration(age)*time.Millisecond > orchestrator.PreparationRateMaxAge {
		return line + " · last update " + shortDuration(time.Duration(age)*time.Millisecond) + " ago"
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
		return "  " + stageLabel(name)
	}
	if fields, ok := value.(map[string]any); ok {
		if name, ok := fields["name"].(string); ok && strings.TrimSpace(name) != "" {
			return "  " + stageLabel(name)
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
	saved []savedFile, submitted time.Duration) *exit.Error {
	life.Status = publicObservedStatus(life.Status)
	status := localapi.StreamStatus(terminal)
	if status == "" {
		status = life.Status
	}
	if (stopped == "deadline" || deadlineCancel(life.CanceledBy)) && status == "canceled" {
		// The DEADLINE is why this ended, and the shared matrix has a code for it.
		status = "deadline"
	}
	if mapTerminal(status) == "succeeded" {
		if problem := undelivered(life); problem != nil {
			return problem
		}
	}
	shownStatus := life.Status
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
		output.Field{K: "execution", V: executionValue(life.ExecutionMS, life.ExecutionKnown, life.MachineExecution != nil)},
		output.Field{K: "submit_ms", V: submitted.Milliseconds()})
	if wallMS, known := recordedRunWall(life.CreatedAt, terminal); known {
		fields = append(fields, output.Field{K: "wall_ms", V: wallMS})
	}

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
	// ONE SOURCE: the daemon's lifecycle, which `cozy run list` reads too. A failure before
	// any attempt is filled there from its settled event, so list and watch cannot disagree.
	errType, errCode, why := life.ErrorType, life.ErrorCode, life.Error
	e.Cause = errCode
	e.Details = failureDetails(ctx, life, errType, errCode, why, terminal)
	defer func() {
		// What it made before it ended is its result: keep it in view.
		if len(saved) > 0 {
			kept := make([]string, 0, len(saved))
			for _, file := range saved {
				kept = append(kept, file.Path)
			}
			e.Details["saved"] = saved
			e.Message += "; kept " + strings.Join(kept, ", ")
		}
	}()
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

// undelivered is why a completed run's outputs are not all in its folder, each at its final
// revision with its sha256 verified; nil once they are. A caller that awaited the run learns
// which outputs are missing and where their bytes still are, never a bare success.
func undelivered(life api.Lifecycle) *exit.Error {
	export := life.OutputExport
	if export == nil {
		return nil
	}
	if export.State == "published" {
		return nil
	}
	var missing, names []string
	for _, item := range life.Output {
		if export.State != "failed" || item.Status == "undelivered" {
			missing, names = append(missing, item.ID), append(names, item.Name)
		}
	}
	named := "its outputs are"
	if names = slices.Compact(names); len(names) > 0 {
		named = strings.Join(names, ", ") + map[bool]string{true: " is", false: " are"}[len(missing) == 1]
	}
	cause, where := strings.Trim(export.ErrorCode+": "+export.Error, ": "), fmt.Sprintf("this computer's record of run %d", life.Number)
	if view := life.MachineExecution; view != nil && !view.Collected {
		cause = strings.Trim(view.CollectionRefused+": "+cmp.Or(view.ObservationError, view.Retained), ": ")
		where = "machine " + view.Machine
	}
	e := exit.Named(exit.Unavailable, "run.outputs_undelivered", "run %d completed, but %s not in %s: %s — its bytes stay on %s; fix the cause and `cozy run watch %d` delivers them",
		life.Number, named, export.Directory, cmp.Or(cause, "not collected yet"), where, life.Number)
	e.Details = map[string]any{"number": life.Number, "request_id": life.RequestID, "missing": missing,
		"directory": export.Directory, "cause": cause, "bytes_on": where}
	return e
}

// failureDetails is the whole recorded failure for machine readers: the run, its typed
// cause, and the triage bundle itself when one was kept.
func failureDetails(ctx *Context, life api.Lifecycle, errType, errCode, why string,
	terminal *localapi.Event,
) map[string]any {
	details := map[string]any{"number": life.Number, "request_id": life.RequestID, "status": life.Status}
	for key, value := range map[string]string{"error_type": errType, "error_code": errCode,
		"error": why, "canceled_by": life.CanceledBy, "machine": life.Machine} {
		if value != "" {
			details[key] = value
		}
	}
	if life.Triage == nil {
		return details
	}
	triage := map[string]any{"subject_id": life.Triage.SubjectID, "attempt_key": life.Triage.AttemptKey,
		"kept": life.Triage.Kept, "length": life.Triage.Length}
	details["triage"] = triage
	if !life.Triage.Kept {
		if fault := eventText(terminal, "triage_fault"); fault != "" {
			triage["fault"] = fault
		}
		return details
	}
	if !ctx.Mode().JSON {
		return details // the human remedy quotes the bundle's traceback instead
	}
	c, problem := dial(ctx)
	if problem != nil {
		triage["unreadable"] = problem.Message
		return details
	}
	data, problem := c.Triage(life.Triage.AttemptKey)
	if problem != nil {
		triage["unreadable"] = problem.Message
		return details
	}
	var bundle any
	if err := json.Unmarshal(data, &bundle); err != nil {
		triage["unreadable"] = "triage bundle is not JSON: " + err.Error()
		return details
	}
	triage["bundle"] = bundle
	return details
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

// validateInvocationPayload applies the same validation and argument help to serving
// functions and jobs, before either path resolves or acquires a model. It answers the
// payload without its undeclared fields, and warns once naming them.
func validateInvocationPayload(ctx *Context, pkg string, ep *launch.Entrypoint, input json.RawMessage) (json.RawMessage, []string, *exit.Error) {
	cleaned, ignored, problem := launch.ValidatePayload(pkg, ep, input)
	if len(ignored) > 0 {
		ctx.warn(launch.IgnoredWarning(pkg+"/"+ep.Name, ignored))
	}
	if problem != nil && !ctx.Mode().JSON {
		problem.Message += "\n\n" + launch.DescribeArguments(ep)
	}
	return cleaned, ignored, problem
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
	Package        string
	Function       string
	InstallID      string
	Release        string
	Snapshot       bool
	releaseCapture func()
	// lease holds the resolved install until the daemon has recorded the submission.
	lease *install.Lease
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
	// Org and package names are never case-sensitive; the callable name is Python's.
	target := Target{Package: strings.ToLower(parts[0] + "/" + parts[1])}
	if len(parts) == 3 {
		target.Function = parts[2]
	}
	return target, nil
}

func invocationTarget(ctx *Context) (Target, *launch.PackageInterface, *exit.Error) {
	if isScriptTarget(ctx.Inv.Args[0]) {
		if ambiguousScriptTarget(ctx.Inv.Args[0]) {
			fmt.Fprintf(ctx.Err, "running the local file %s; spell a package as org/name/function\n", ctx.Inv.Args[0])
		}
		return scriptTarget(ctx)
	}
	target, problem := parseTarget(ctx.Inv.Args[0])
	if problem != nil {
		return Target{}, nil, problem
	}
	// An installed package is validated against its installed interface with no hub read;
	// the machine that runs it prepares that release itself.
	facts, lease, problem := leasedInstallFacts(ctx, target.Package)
	if problem == nil {
		target.lease = lease
		target.Release = facts.Install.Version
		if strings.HasPrefix(target.Package, "local/") {
			target.InstallID = facts.Install.ID
		} else {
			// The run's results are read against the release's interface with no Hub call.
			keepReleaseInterface(home.Paths(ctx.Cfg.Home).Root, target.Package, target.Release, facts.PackageInterface.Raw,
				strings.Split(facts.Install.Closure, "\n"))
		}
		adoptInstallHub(ctx, facts.Install)
		return target, facts.PackageInterface, nil
	}
	if problem.Code != exit.NotFound || strings.HasPrefix(target.Package, "local/") {
		return Target{}, nil, problem
	}
	// A release this client has not installed: the one it read last, else the one the machine
	// that runs it names at its own Hub. Only a run with no machine yet reads the Hub, once.
	root := home.Paths(ctx.Cfg.Home).Root
	if release, surface := keptNewestRelease(root, ctx.Cfg.HubURL, target.Package); release != "" {
		target.Release = release
		return target, surface, nil
	}
	if release, surface, problem := describeOnMachine(ctx, target.Package); problem != nil {
		return Target{}, nil, problem
	} else if release != "" {
		keepNewestRelease(root, ctx.Cfg.HubURL, target.Package, release)
		target.Release = release
		return target, surface, nil
	}
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
	if problem != nil {
		return Target{}, nil, exit.Named(exit.Conflict, "rental.package_interface_invalid",
			"Tensorhub returned an invalid package interface: %s", problem.Message)
	}
	if detail.Release.Release != release {
		return Target{}, nil, exit.Named(exit.Conflict, "rental.package_release_invalid",
			"Tensorhub returned no immutable package release identity")
	}
	requirements, problem := detail.Requirements()
	if problem != nil {
		return Target{}, nil, problem
	}
	target.Release = release
	// The run's results, and its rental's machine class, read this immutable release with
	// no further Hub call.
	keepReleaseInterface(root, target.Package, release, detail.PackageInterface, requirements)
	keepNewestRelease(root, ctx.Cfg.HubURL, target.Package, release)
	return target, packageInterface, nil
}

// describeOnMachine is pkg's newest release and its interface as the machine the run goes to
// installs it at its own Hub (describe/1), so the client reads no Hub. "" when the run has no
// machine yet, or the machine or daemon cannot describe.
func describeOnMachine(ctx *Context, pkg string) (string, *launch.PackageInterface, *exit.Error) {
	machine, known, problem := knownMachine(ctx)
	if problem != nil || !known {
		return "", nil, problem
	}
	var described api.ReleaseDescription
	if ctx.endpoint != nil {
		install, problem := foregroundPrewarm(ctx, ctx.endpoint, machine, records.RentalInstallSelection{Package: pkg})
		if problem != nil || json.Unmarshal(install.Result, &described) != nil {
			return "", nil, describeFallback(problem)
		}
	} else {
		c, problem := dial(ctx)
		if problem != nil {
			return "", nil, problem
		}
		if described, problem = c.DescribeRelease(machine, pkg); problem != nil {
			return "", nil, describeFallback(problem)
		}
	}
	if described.Release == "" || len(described.Interface) == 0 {
		return "", nil, nil
	}
	raw, err := canonical.NormalizeJCS(described.Interface)
	if err != nil {
		return "", nil, exit.Named(exit.Conflict, "machine.package_interface_invalid", "the machine described an invalid package interface")
	}
	surface, problem := launch.DecodePackageInterface(raw)
	if problem != nil {
		return "", nil, exit.Named(exit.Conflict, "machine.package_interface_invalid",
			"the machine described an invalid package interface: %s", problem.Message)
	}
	keepReleaseInterface(home.Paths(ctx.Cfg.Home).Root, pkg, described.Release, raw, nil)
	return described.Release, surface, nil
}

// describeFallback keeps a describe refusal that ends the run, and drops one that only says
// the machine or daemon cannot describe now (an older one, no route, or an unreachable
// machine, whose run is still recorded and waits for it): the Hub names it.
func describeFallback(problem *exit.Error) *exit.Error {
	if problem == nil || problem.Code == exit.Unavailable || problem.Code == exit.NotFound ||
		problem.ErrName() == "untyped_answer" || problem.ErrName() == "hub.untyped_refusal" {
		return nil
	}
	return problem
}

// knownMachine is the machine a run names without renting one: this computer's, or a named
// rental. It answers false when the run asks the fleet to choose or buy one.
func knownMachine(ctx *Context) (string, bool, *exit.Error) {
	if ctx.endpoint != nil {
		return ctx.endpoint.Name(), true, nil
	}
	if ctx.Inv.Value("--rental") != "" {
		selected, problem := requestedRental(ctx)
		return selected, problem == nil, problem
	}
	return machines.Local, !rentalRequested(ctx), nil
}

// newestPackageRelease is the newest non-yanked release by PEP 440 order, preferring a
// final release and falling back to prereleases when the package has published only those.
func newestPackageRelease(releases []hub.ReleaseSummary) (string, *exit.Error) {
	var best, bestPre string
	var bestVersion, bestPreVersion pep440.Version
	for _, row := range releases {
		if row.Yanked || row.YankedAt != "" {
			continue
		}
		version, err := pep440.Parse(row.Release)
		if err != nil {
			continue
		}
		if version.IsPreRelease() {
			if bestPre == "" || version.GreaterThan(bestPreVersion) {
				bestPre, bestPreVersion = row.Release, version
			}
		} else if best == "" || version.GreaterThan(bestVersion) {
			best, bestVersion = row.Release, version
		}
	}
	if best == "" {
		best = bestPre
	}
	if best == "" {
		return "", exit.New(exit.NotFound, "package has no non-yanked release")
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
		// A job's accelerator declaration rides beside its request: true, false, or null.
		var doc map[string]json.RawMessage
		if ep.Kind == "job" && json.Unmarshal(raw, &doc) == nil {
			doc["accelerator"], _ = json.Marshal(ep.Accelerator)
			var out bytes.Buffer
			encoder := json.NewEncoder(&out)
			encoder.SetEscapeHTML(false)
			if encoder.Encode(doc) == nil {
				raw = bytes.TrimSpace(out.Bytes())
			}
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
	var public []launch.Entrypoint
	for _, entrypoint := range packageInterface.Entrypoints {
		if !entrypoint.Internal {
			public = append(public, entrypoint)
		}
	}
	slots := declaredModelSlots(public)
	defaults := effectiveModelBindings(slots, nil, packageOwner(target.Package, slots, commandNamespace(ctx)))
	var bindingProblem *exit.Error
	if len(slots) > 0 && !strings.HasPrefix(target.Package, "local/") {
		defaults, bindingProblem = invocationDefaultBindings(ctx, target, slots)
	}
	list := output.List{Name: "functions", Fields: []string{"function", "availability"}, AllFields: []string{"function", "availability"}}
	for _, name := range packageInterface.PublicNames() {
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
	if unavailable := packageInterface.Unavailable[target.Function]; unavailable != nil {
		return unavailable
	}
	names := packageInterface.PublicNames()
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

// leasedInstallFacts resolves the active install of pkg and leases it for this submission,
// so no superseding install can reclaim it before the daemon records the request. An
// install reclaimed before the lease was taken is resolved again.
func leasedInstallFacts(ctx *Context, pkg string) (*launch.Facts, *install.Lease, *exit.Error) {
	layout, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return nil, nil, problem
	}
	defer store.Close()
	for {
		active, problem := installedPackage(store, pkg)
		if problem != nil {
			return nil, nil, problem
		}
		lease, problem := install.LeaseInstall(layout, store, active.ID)
		if problem != nil && problem.Code == exit.NotFound {
			continue
		}
		if problem != nil {
			return nil, nil, problem
		}
		facts, problem := launch.Read(*active, ctx.Cfg.Home, ctx.Cfg.Tool())
		if problem != nil {
			lease.Release()
			return nil, nil, problem
		}
		return facts, lease, nil
	}
}

func installedPackage(store *records.Store, pkg string) (*records.PackageInstall, *exit.Error) {
	bare, major, hasMajor := splitMajor(pkg)
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

func executionValue(ms int64, known, machine bool) string {
	if machine && !known {
		return "—"
	}
	return seconds(ms)
}
