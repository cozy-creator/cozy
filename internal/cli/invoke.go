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
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
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
	target, descriptor, problem := invocationTarget(ctx)
	if problem != nil {
		return problem
	}
	if target.Function == "" {
		return emitFunctions(ctx, target, descriptor)
	}
	callable, problem := descriptor.Function(target.Function)
	if problem != nil {
		return unknownFunction(target, descriptor)
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
	if len(ctx.Inv.Values["--model"]) > 0 {
		return exit.Usagef("--model applies to serving callables; remote modeled jobs are not supported yet")
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
	input, e := launch.ParsePayload(ep, ctx.Inv.Args[1:], ctx.Inv.Value("--in"))
	if e != nil {
		return e
	}
	input, assets, e := launch.ParseAssets(ep, input, ctx.Inv.Values["--asset"])
	if e != nil {
		return e
	}
	input, outputHash, e := finalizeInputPayload(ep, input, key)
	if e != nil {
		return e
	}
	models := []orchestrator.ModelRef(nil)
	if managedRental {
		models, e = resolveInvocationModels(ctx, target, ep, ctx.Inv.Values["--model"], true)
		if e != nil {
			return e
		}
	} else {
		models, e = resolveInvocationModels(ctx, target, ep, ctx.Inv.Values["--model"], false)
		if e != nil {
			return e
		}
	}
	outputDirectory := ""
	outputIntentHash := ""
	if requested := ctx.Inv.Value("--out"); requested != "" {
		absolute, err := filepath.Abs(requested)
		if err != nil {
			return exit.Usagef("cannot resolve --out %q: %s", requested, err)
		}
		outputDirectory = filepath.Clean(absolute)
		outputIntentHash = outputHash
	}

	c, e := dial(ctx)
	if e != nil {
		return e
	}
	began := time.Now()
	handle, e := c.Submit(api.Submission{
		Package: target.Package, Function: target.Function, Input: input,
		LocalAssets: assets, InstallID: target.InstallID,
		Release: target.Release, ReleaseDigest: target.ReleaseDigest, Rental: managedRental,
		RentalRequired:  ctx.Inv.Bool("--rental-only"),
		Models:          models,
		OutputDirectory: outputDirectory, OutputPayloadHash: outputIntentHash,
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
			if dir := ctx.Inv.Value("--out"); dir != "" {
				fmt.Fprintf(ctx.Err, "Saving outputs as %s\n", outputPathHint(ep, dir, outputHash))
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
// execution: explicit --model, then package.toml default. Local acquisition freezes the
// exact Manifest and length before submission; remote acquisition freezes the same
// Manifest in the signed worker download request.
func resolveInvocationModels(ctx *Context, target Target, ep *launch.Entrypoint,
	raw []string, remote bool,
) ([]orchestrator.ModelRef, *exit.Error) {
	if !remote && target.InstallID != "" {
		row, problem := exactInvocationInstall(ctx, target)
		if problem != nil {
			return nil, problem
		}
		if row.SourceKind == "local" {
			if len(raw) > 0 {
				return nil, exit.Named(exit.Unavailable, "editable_model_override_unsupported",
					"editable package model overrides are not available on the published-package BYOM lane").
					WithRemedy("publish the package code, then invoke it with --model [slot=]org/model@release")
			}
			return nil, nil
		}
	}
	selected, problem := invocationModelSpecs(ctx, target, ep, raw)
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
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return nil, exit.Internalf("cannot create model selection scratch: %s", err)
	}
	root, err := os.MkdirTemp(layout.Transfer, "invoke-models-")
	if err != nil {
		return nil, exit.Internalf("cannot create model selection scratch: %s", err)
	}
	defer os.RemoveAll(root)
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
	raw []string,
) ([]invocationModelSpec, *exit.Error) {
	if len(ep.Models) == 0 {
		if len(raw) > 0 {
			return nil, exit.Usagef("%s declares no model slots", ep.Name)
		}
		return nil, nil
	}
	overrides := make(map[string]string, len(raw))
	for _, value := range raw {
		left, right, qualified := strings.Cut(strings.TrimSpace(value), "=")
		if !qualified {
			if len(ep.Models) != 1 || left == "" {
				return nil, exit.Usagef("--model %q must name one of %d model slots", value, len(ep.Models)).
					WithRemedy("use --model <slot>=org/model@release")
			}
			right = left
			left = ep.Models[0].Path
		}
		left, right = strings.TrimSpace(left), strings.TrimSpace(right)
		if left == "" || right == "" {
			return nil, exit.Usagef("%q is not [slot=]org/model@release", value)
		}
		slot, problem := modelSlot(ep, left)
		if problem != nil {
			return nil, problem
		}
		if _, exists := overrides[slot.Path]; exists {
			return nil, exit.Usagef("model slot %s was bound more than once", slot.Path)
		}
		overrides[slot.Path] = right
	}
	defaults := map[string]publishedDefaultBinding{}
	if len(overrides) < len(ep.Models) {
		var problem *exit.Error
		defaults, problem = invocationDefaultBindings(ctx, target)
		if problem != nil {
			return nil, problem
		}
	}
	out := make([]invocationModelSpec, 0, len(ep.Models))
	for _, slot := range ep.Models {
		if ref, ok := overrides[slot.Path]; ok {
			out = append(out, invocationModelSpec{Slot: slot.Path, Ref: ref})
			continue
		}
		binding, ok := defaults[slot.Path]
		if !ok {
			return nil, exit.Named(exit.NotFound, "package_default_model_unavailable",
				"%s has no usable configured default for model slot %s", target.Package, slot.Path).
				WithRemedy("override it explicitly: --model %s=org/model@release", slot.Param)
		}
		out = append(out, invocationModelSpec{Slot: slot.Path, Ref: binding.Ref, Lane: binding.Lane})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Slot < out[j].Slot })
	return out, nil
}

func invocationDefaultBindings(ctx *Context, target Target) (
	map[string]publishedDefaultBinding, *exit.Error,
) {
	var packageConfig, descriptor install.ExactDocument
	runtimeBin := ""
	if target.InstallID != "" {
		row, problem := exactInvocationInstall(ctx, target)
		if problem != nil {
			return nil, problem
		}
		var problem2 *exit.Error
		packageConfig, problem2 = exactInstalledDocument(filepath.Join(row.ProjectDir, "package.toml"))
		if problem2 != nil {
			return nil, problem2
		}
		descriptor, problem2 = exactInstalledDocument(launch.DescriptorPath(row.Dir))
		if problem2 != nil {
			return nil, problem2
		}
		runtimeBin = row.Runtime
	} else {
		ref, problem := hub.ParseRef(target.Package)
		if problem != nil {
			return nil, problem
		}
		hctx, cancel := hub.Context()
		defer cancel()
		plan, problem := client(ctx).PackageDownloads(hctx, ref, target.Release)
		if problem != nil {
			return nil, problem
		}
		packageConfig, problem = exactPackageInstallDocument("package.toml", plan.PackageConfig)
		if problem != nil {
			return nil, problem
		}
		descriptor, problem = exactPackageInstallDocument("package descriptor", plan.PackageDescriptor)
		if problem != nil {
			return nil, problem
		}
		runtimeBin, problem = launch.HostRuntime()
		if problem != nil {
			return nil, problem
		}
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return nil, problem
	}
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return nil, exit.Internalf("cannot create binding metadata scratch: %s", err)
	}
	root, err := os.MkdirTemp(layout.Transfer, "binding-defaults-")
	if err != nil {
		return nil, exit.Internalf("cannot create binding metadata scratch: %s", err)
	}
	defer os.RemoveAll(root)
	rows, problem := publishedDefaultBindings(context.Background(), ctx.Cfg, root,
		runtimeBin, packageConfig, descriptor)
	if problem != nil {
		return nil, exit.Named(problem.Code, "package_default_model_invalid",
			"%s configured default model is not usable: %s", target.Package, problem.Message).
			WithRemedy("supply --model <slot>=org/model@release to bypass the configured default")
	}
	out := make(map[string]publishedDefaultBinding, len(rows))
	for _, row := range rows {
		out[row.ModelBindingPath] = row
	}
	return out, nil
}

func exactInstalledDocument(path string) (install.ExactDocument, *exit.Error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return install.ExactDocument{}, exit.New(exit.NotFound, "cannot read installed metadata %s: %s", path, err)
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	return install.ExactDocument{Bytes: raw, Digest: digest, Length: int64(len(raw))}, nil
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

func modelSlot(ep *launch.Entrypoint, asked string) (*launch.Slot, *exit.Error) {
	var match *launch.Slot
	for i := range ep.Models {
		slot := &ep.Models[i]
		if asked != slot.Path && asked != slot.Param {
			continue
		}
		if match != nil {
			return nil, exit.Usagef("model slot %q is ambiguous; use its full descriptor path", asked)
		}
		match = slot
	}
	if match == nil {
		return nil, exit.Usagef("%q is not a model slot for %s", asked, ep.Name)
	}
	return match, nil
}

func resolveRemoteModel(ctx *Context, packageName, slotPath, raw, wantedLane string) (
	orchestrator.ModelRef, *exit.Error,
) {
	// A caller may narrow by Manifest spelling, but cannot introduce one: the Hub-authored
	// release card below must contain it in an exact lane before it enters request identity
	// or a signed worker download delegation. No caller bytes or local path are trusted.
	var empty orchestrator.ModelRef
	if strings.Count(raw, "#") > 1 {
		return empty, exit.Usagef("%q carries more than one manifest", raw)
	}
	modelRelease, manifest, hasManifest := strings.Cut(raw, "#")
	if hasManifest && manifest == "" {
		return empty, exit.Usagef("%q carries an empty manifest", raw)
	}
	if manifest != "" {
		if _, err := canonical.Raw(manifest); err != nil {
			return empty, exit.Usagef("%q is not a sha256 model manifest", manifest)
		}
	}
	if strings.Count(modelRelease, "@") > 1 {
		return empty, exit.Usagef("%q carries more than one release", modelRelease)
	}
	modelName, release, hasRelease := strings.Cut(modelRelease, "@")
	if hasRelease && release == "" {
		return empty, exit.Usagef("%q carries an empty release", raw)
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
	for _, lane := range selected.Lanes {
		if (manifest == "" || lane.ManifestID == manifest) &&
			(wantedLane == "" || lane.Lane == wantedLane) {
			manifestLanes[lane.ManifestID] = append(manifestLanes[lane.ManifestID], lane.Lane)
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
		Model: ref.String(), Release: release, Lane: lanes[0], Manifest: manifest}, nil
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
	lines := newProgress(ctx, stream, began)
	terminal, problem := c.WatchContext(watchCtx, requestID, 0, lines.on)
	lines.done()
	return terminal, problem
}

func renderSubmittedRun(ctx *Context, life api.Lifecycle, changed bool) *exit.Error {
	status := runStatus(life.Status)
	fields := []output.Field{
		{K: "run", V: life.RequestID},
		{K: "target", V: life.Package + "/" + life.Function},
		{K: "status", V: status},
	}
	defaults := []string{"target", "status"}
	if life.QueuePosition != nil {
		queue := strconv.Itoa(*life.QueuePosition)
		if life.QueueDepth != nil && *life.QueueDepth >= *life.QueuePosition {
			queue += "/" + strconv.Itoa(*life.QueueDepth)
		}
		fields = append(fields, output.Field{K: "queue_position", V: queue})
		defaults = append(defaults, "queue_position")
	}
	if export := life.OutputExport; export != nil {
		fields = append(fields, output.Field{K: "output", V: outputExportHint(export)})
		defaults = append(defaults, "output")
	}
	fields = append(fields, output.Field{K: "changed", V: changed})
	defaults = append(defaults, "run")
	rec := compactRecord(fields, defaults...)
	rec.Next = []string{
		"cozy run watch " + life.RequestID,
		"cozy run cancel " + life.RequestID,
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
			return life, exit.Named(exit.Unavailable, "output_export_failed",
				"run %s completed, but output export failed: %s — %s",
				life.RequestID, export.ErrorCode, export.Error).
				WithRemedy("fix the recorded destination and repeat the same idempotency key; the daemon retries this durable export")
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

func outputExportHint(export *api.OutputExportRef) string {
	if export == nil {
		return ""
	}
	if len(export.Paths) == 1 {
		return export.Paths[0]
	}
	if len(export.Paths) > 1 {
		return strings.Join(export.Paths, ", ")
	}
	return filepath.Join(export.Directory, export.PayloadHash+"*")
}

func exportedOutputs(life api.Lifecycle) []map[string]string {
	if life.OutputExport == nil || life.OutputExport.State != "published" {
		return nil
	}
	paths := life.OutputExport.Paths
	result := make([]map[string]string, 0, len(paths))
	for index, path := range paths {
		row := map[string]string{"path": path}
		if index < len(life.Outputs) {
			row["output"] = life.Outputs[index].OutputID
			row["bytes"] = output.Bytes(life.Outputs[index].Length)
			row["mime"] = life.Outputs[index].MimeType
			row["digest"] = life.Outputs[index].Digest
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
// it goes. SIGINT does not kill this process: it CANCELS the request through the
// orchestrator and keeps watching, because the attempt's own journaled terminal is what
// settles it and a client that walked away would leave the card held.
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
	forced := make(chan struct{}, 1)
	cancelFailed := make(chan *exit.Error, 1)
	done := make(chan struct{})
	defer close(done)
	go func() {
		reason, message := "", ""
		select {
		case _, ok := <-interrupt:
			if !ok {
				return
			}
			reason, message = "canceled", "cancel requested"
		case <-deadlineC(deadline):
			// `--timeout` is a REQUEST DEADLINE the client enforces the only way a client
			// honestly can: by asking the orchestrator to cancel. It is not the
			// supervisor's watchdog deadline (that one is on the attempt, and this host
			// has no wire field for it) — walking away instead would leave the card held.
			reason, message = "deadline", fmt.Sprintf("--timeout %s expired", deadline)
		case <-done:
			return
		}
		stopped <- reason
		fmt.Fprintf(ctx.Err, "\n%s — the attempt's own terminal still settles it\n", message)
		cancelResult := make(chan *exit.Error, 1)
		go func() { cancelResult <- c.Cancel(requestID) }()
		select {
		case <-interrupt:
			fmt.Fprintln(ctx.Err, "second interrupt — stopped waiting; the request remains recorded")
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
			fmt.Fprintln(ctx.Err, "second interrupt — stopped waiting; the request remains recorded")
			forced <- struct{}{}
			stopWatch()
		case <-done:
		}
	}()

	lines := newProgress(ctx, stream, began)
	terminal, e := c.WatchContext(watchCtx, requestID, 0, lines.on)
	lines.done()
	select {
	case problem := <-cancelFailed:
		return nil, "cancel_failed", problem
	default:
	}
	select {
	case <-forced:
		return nil, "interrupted", exit.New(exit.Canceled,
			"stopped waiting for %s; it remains visible in `cozy run list`", requestID).
			WithNext("cozy run list")
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

// progress renders the live lane. Two shapes, and they are not the same surface:
// `--stream` is NDJSON of the typed envelope for a machine, and the default is one
// rewritten line for a person.
type runProgress struct {
	ctx         *Context
	stream      bool
	last        string
	dirty       bool
	began       time.Time
	stepStage   string
	stepSeconds float64
	stepSamples int
}

func newProgress(ctx *Context, stream bool, began time.Time) *runProgress {
	return &runProgress{ctx: ctx, stream: stream, began: began}
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
	// A redirected human command has no status line to rewrite. Its final result is the
	// useful record; spraying every lossy tick into logs is neither progress nor a stable
	// interface. --full deliberately restores the diagnostic stream.
	if !p.ctx.Mode().Color && !p.ctx.Mode().Full {
		return true
	}
	line := p.line(e, p.ctx.Mode().Full)
	if line == "" || line == p.last {
		return true
	}
	p.last, p.dirty = line, true
	// stderr, deliberately: stdout carries the RESULT, so a piped `cozy run` is not
	// polluted by the progress of producing it.
	if p.ctx.Mode().Color {
		fmt.Fprintf(p.ctx.Err, "\r\033[K%s", line)
	} else {
		fmt.Fprintln(p.ctx.Err, line)
	}
	return true
}

func (p *runProgress) line(e localapi.Event, full bool) string {
	if full {
		return diagnosticProgressLine(e)
	}
	if strings.TrimPrefix(e.Type, "request.") != "progress" {
		return progressLine(e, false)
	}
	fields, ok := e.Payload["value"].(map[string]any)
	if !ok {
		return humanProgress(e.Payload["value"])
	}
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
	if !fractionOK || fraction < 0 || fraction > 1 {
		return humanStage(map[string]any{"name": stageLabel(name)})
	}
	elapsed := time.Since(p.began)
	label := stageLabel(name)
	if label == "" {
		label = "running"
	}
	bar := progressBar(fraction, 18)
	if !positionOK || position <= 0 || fraction <= 0 {
		return fmt.Sprintf("  %s %s %.0f%% · %s", label, bar, fraction*100, shortDuration(elapsed))
	}
	total := int64(position/fraction + 0.5)
	current := int64(position)
	if total < current {
		total = current
	}
	line := fmt.Sprintf("  %s %s %d/%d", label, bar, current, total)
	if p.stepSamples > 0 {
		seconds := p.stepSeconds / float64(p.stepSamples)
		if seconds > 0 {
			line += fmt.Sprintf(" · %.2fs/step · %.2f steps/s", seconds, 1/seconds)
			remaining := time.Duration(float64(total-current) * seconds * float64(time.Second))
			line += fmt.Sprintf(" · elapsed %s · ETA ~%s", shortDuration(elapsed), shortDuration(remaining))
			return line
		}
	}
	return line + " · elapsed " + shortDuration(elapsed)
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

func (p *runProgress) done() {
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
	case "queued":
		return "  preparing a local worker"
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
	case "queued":
		if reason, ok := e.Payload["reason"].(string); ok {
			return "  queued — " + reason
		}
		return "  queued"
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

func outputPathHint(ep *launch.Entrypoint, dir, payloadHash string) string {
	paths := launch.AssetPaths(ep.Result)
	if len(paths) != 1 {
		return filepath.Join(dir, payloadHash+"*")
	}
	spec, ok := launch.ResultAssetSpec(ep, paths[0])
	if !ok || len(spec.MediaTypes) != 1 {
		return filepath.Join(dir, payloadHash+"*")
	}
	extension := resultfiles.Extension(spec.MediaTypes[0])
	if extension == "" {
		return filepath.Join(dir, payloadHash+"*")
	}
	return filepath.Join(dir, payloadHash+extension)
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
	elapsed := time.Since(began)
	fields = append(fields,
		output.Field{K: "elapsed", V: fmt.Sprintf("%.1fs", elapsed.Seconds())},
		output.Field{K: "submit_ms", V: submitted.Milliseconds()},
		output.Field{K: "wall_ms", V: elapsed.Milliseconds()})

	defaults := []string{"target", "status"}
	if life.Result != nil {
		defaults = append(defaults, "result")
	}
	if len(saved) > 0 {
		defaults = append(defaults, "saved")
	}
	defaults = append(defaults, "elapsed")
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
	if life.Triage != nil {
		e.WithRemedy("the retained triage bundle explains it").
			WithNext("cozy run list --full")
	}
	return e
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
//
// The returned hash is SHA-256 of the exact normalized JSON bytes submitted to the daemon.
// It therefore exists before execution and names the output independently of result bytes.
func finalizeInputPayload(ep *launch.Entrypoint, input json.RawMessage,
	idempotencyKey string,
) (json.RawMessage, string, *exit.Error) {
	var document map[string]json.RawMessage
	if err := json.Unmarshal(input, &document); err != nil || document == nil {
		return nil, "", exit.Internalf("cannot finalize the invocation payload: %v", err)
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
		return nil, "", exit.Internalf("cannot render the finalized invocation payload: %s", err)
	}
	digest := sha256.Sum256(rendered)
	return rendered, hex.EncodeToString(digest[:]), nil
}

// ------------------------------------------------------------------- target parsing

// Target is one installed package and optional callable selected for invocation.
type Target struct {
	Package       string
	Function      string
	InstallID     string
	Release       string
	ReleaseDigest string
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

func invocationTarget(ctx *Context) (Target, *launch.PackageDescriptor, *exit.Error) {
	target, problem := parseTarget(ctx.Inv.Args[0])
	if problem != nil {
		return Target{}, nil, problem
	}
	if rentalRequested(ctx) && strings.HasPrefix(target.Package, "local/") {
		facts, problem := generationFacts(ctx, target.Package)
		if problem != nil {
			return Target{}, nil, problem
		}
		target.InstallID = facts.Install.ID
		target.Release = facts.Install.Version
		target.ReleaseDigest = facts.Install.SourceDigest
		return target, facts.PackageDescriptor, nil
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
		descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
		if problem != nil || descriptor.Digest != detail.Release.PackageDescriptorDigest ||
			detail.Release.PackageDescriptorLength != int64(len(detail.PackageDescriptor)) {
			return Target{}, nil, exit.Named(exit.Conflict, "rental.package_descriptor_invalid",
				"Tensorhub returned an invalid package descriptor")
		}
		if _, err := canonical.Raw(detail.Release.ReleaseDigest); err != nil ||
			detail.Release.Release != release {
			return Target{}, nil, exit.Named(exit.Conflict, "rental.package_release_invalid",
				"Tensorhub returned no immutable package release identity")
		}
		target.Release, target.ReleaseDigest = release, detail.Release.ReleaseDigest
		return target, descriptor, nil
	}
	facts, problem := generationFacts(ctx, target.Package)
	if problem != nil && problem.Code == exit.NotFound {
		if problem = autoInstallPackage(ctx, target.Package); problem != nil {
			return Target{}, nil, problem
		}
		facts, problem = generationFacts(ctx, target.Package)
	}
	if problem != nil {
		return Target{}, nil, problem
	}
	target.InstallID = facts.Install.ID
	return target, facts.PackageDescriptor, nil
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

func emitFunctions(ctx *Context, target Target, descriptor *launch.PackageDescriptor) *exit.Error {
	list := output.List{Name: "functions", Fields: []string{"function"}, AllFields: []string{"function"}}
	for _, name := range descriptor.Names() {
		list.Rows = append(list.Rows, map[string]string{"function": name})
		if len(list.Next) < 2 {
			list.Next = append(list.Next, "cozy run "+target.Package+"/"+name)
		}
	}
	return emit(ctx, list)
}

func unknownFunction(target Target, descriptor *launch.PackageDescriptor) *exit.Error {
	names := descriptor.Names()
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

// generationFacts resolves the package's one active install.
func generationFacts(ctx *Context, pkg string) (*launch.Facts, *exit.Error) {
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
