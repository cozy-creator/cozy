package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	packageref "github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

type publishSource struct {
	Canonical string
	Selection string
	License   string
	Lane      string
	Files     int
	Bytes     int64
	Exact     []modeltransfer.SourceFile
	Access    []sourceCapability
	// The provider resolution this source came from, ALREADY NARROWED to the reviewed
	// carriers, and the resolver that answered it. Carried so a header-first conversion
	// preflight (tfs-076) can range-read those carriers' headers without re-asking the
	// provider everything it has just answered. Nil for a local file, a `local/` alias,
	// or a Tensorhub checkpoint — none of which has a provider header to read.
	Resolver   *modelsource.Resolver
	Resolution modelsource.Plan
}

type sourceCapability struct {
	Member, ObjectID, Provider, URL string
	Length                          int64
	ExpiresAtUnix                   uint64
}

type producerPlan struct {
	Name             string
	InstallID        string
	Release          string
	SourceDigest     string
	PackageInterface *launch.PackageInterface
	Job              *launch.Entrypoint
	Pin              modeltransfer.JobPin
	SourceProfiles   map[string]string
	Outputs          []modeltransfer.OutputPin
	NeedsAccelerator bool
}

func sourceProfileNames(profiles map[string]string) []string {
	names := make([]string, 0, len(profiles))
	for _, name := range profiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return slices.Compact(names)
}

func handleModelUpload(ctx *Context) *exit.Error {
	return handleModelTransfer(ctx, "model-upload")
}

func handleModelDownload(ctx *Context) *exit.Error {
	return handleModelTransfer(ctx, "model-download")
}

func handleModelTransfer(ctx *Context, kind string) *exit.Error {
	sourceArg, destinationArg := ctx.Inv.Args[0], strings.TrimSpace(ctx.Inv.Args[1])
	destination := destinationArg
	if kind == "model-upload" {
		ref, problem := hub.ParseRef(destinationArg)
		if problem != nil {
			return problem
		}
		if ref.Org == "local" {
			return exit.Usagef("local/ is reserved for private aliases and cannot be a Tensorhub destination").
				WithRemedy("upload under your Tensorhub account, for example alice/%s", ref.Name)
		}
		destination = ref.String()
	} else {
		if !strings.HasPrefix(destinationArg, "local/") {
			return exit.Usagef("model download destination %q is not local/name", destinationArg)
		}
		if problem := modelsource.LocalName(strings.TrimPrefix(destinationArg, "local/")); problem != nil {
			return problem
		}
		if ctx.Inv.Bool("--rental-only") {
			return exit.Named(exit.Unavailable, "model_download.rented_return_unavailable",
				"rented model download cannot yet return an output to local TensorFS").
				WithRemedy("run without --rental or --rental-only; tracked remote-return support is not landed")
		}
	}
	if ctx.Inv.Bool("--dry-run") && ctx.Inv.Bool("--await") {
		return exit.Usagef("--dry-run and --await conflict: a dry-run creates no durable run")
	}
	preflightLocalOnly := localOnlyModelSource(sourceArg)
	if !(preflightLocalOnly && ctx.Inv.Bool("--rental") && !ctx.Inv.Bool("--rental-only")) {
		if problem := validateRunPlacement(ctx); problem != nil {
			return problem
		}
	}
	producerName := strings.TrimSpace(ctx.Inv.Value("--producer"))
	if producerName != "" {
		if _, problem := parseProductionCallable(producerName); problem != nil {
			return problem
		}
	}
	suppliedProfiles, problem := parseSourceProfileFlags(ctx, producerName)
	if problem != nil {
		return problem
	}
	instructionSource, problem := canonicalProductionSource(ctx, sourceArg)
	if problem != nil {
		return problem
	}
	placement := ""
	if ctx.Inv.Bool("--rental") {
		placement = "rental"
	} else if ctx.Inv.Bool("--rental-only") {
		placement = "rental-only"
	}
	instruction := modeltransfer.Instruction{Kind: kind, Destination: destination,
		Source: instructionSource, InputLane: strings.TrimSpace(ctx.Inv.Value("--lane")),
		Producer: producerName, Placement: placement, SourceProfiles: suppliedProfiles}
	localOnly := localOnlyModelSource(instructionSource)
	if localOnly && ctx.Inv.Bool("--rental-only") {
		return exit.Usagef("a local model source cannot run under --rental-only").
			WithRemedy("omit --rental-only or use an addressable provider/Tensorhub source")
	}
	effectiveRental := rentalRequested(ctx) && !localOnly
	if kind == "model-download" && effectiveRental {
		return exit.Named(exit.Unavailable, "model_download.rented_return_unavailable",
			"rented model download cannot yet return an output to local TensorFS").
			WithRemedy("run locally until the negotiated weights-read return plane is active")
	}
	if producerName == "" && effectiveRental {
		return exit.Named(exit.Unavailable, "model_transfer.rented_pass_through_unavailable",
			"rented pass-through cannot choose a TensorFS source profile without a producer job").
			WithRemedy("omit the rental flag for local pass-through, or select a typed --producer")
	}
	if localOnly {
		ctx.Inv.Bools["--rental"] = false
		ctx.Inv.Bools["--rental-only"] = false
	}
	producer, problem := resolveProducerPlan(ctx, producerName, suppliedProfiles)
	if problem != nil {
		return problem
	}
	var sourceProfiles []string
	if producer != nil {
		sourceProfiles = sourceProfileNames(producer.SourceProfiles)
	}
	source, problem := resolvePublishSource(ctx, sourceArg, sourceProfiles)
	if problem != nil {
		return problem
	}
	if effectiveRental && catalogModelSpelling(source.Canonical) {
		return exit.Named(exit.Unavailable, "model_transfer.rented_catalog_source_unavailable",
			"rented model transfer cannot yet bind a Tensorhub checkpoint through the worker download set").
			WithRemedy("run locally or use the original pinned provider source until the tracked catalog binding lands")
	}
	plan := modeltransfer.Plan{
		Instruction: instruction,
		Destination: destination,
		Source:      source.Canonical, SourceSelection: source.Selection,
		SourceLicense: source.License, InputLane: source.Lane,
		SourceFiles: source.Exact,
	}
	if producer != nil {
		plan.Producer, plan.ProducerInstallID = producer.Name, producer.InstallID
		plan.ProducerRelease = producer.Release
		plan.ProducerSourceDigest = producer.SourceDigest
		plan.PackageInterfaceDigest = producer.PackageInterface.Digest
		pin := producer.Pin
		plan.Job, plan.SourceProfiles, plan.Outputs = &pin, producer.SourceProfiles, producer.Outputs
	} else {
		plan.Outputs = []modeltransfer.OutputPin{{Name: "model"}}
	}
	if kind == "model-download" && len(plan.Outputs) != 1 {
		return exit.Named(exit.Unavailable, "model_download.multiple_outputs_unavailable",
			"model download producer %s emits %d outputs; one local alias currently retains exactly one",
			producerName, len(plan.Outputs)).WithRemedy(
			"use a one-output producer or model upload until tracked multi-output local aliases land")
	}
	// THE GUARD (tfs-076). A conversion plan is a function of the source HEADERS, so it is
	// decidable HERE — owner-side, before a rental is requested, before a byte of payload
	// moves, and before this process submits anything. Runs 290, 294 and 309 each moved
	// 210.3 GB and then refused on facts that were in the first few kilobytes of each
	// member; a `--dry-run` in front of each of them reported `status: planned`, because
	// it validated the dispatch plan and never the conversion plan.
	//
	// Placed on the two paths that pay for the omission: a DRY RUN, whose whole job is to
	// answer this question, and a RENTED transfer, which is the one that spends money to
	// find out. A local transfer already plans from headers itself, in
	// prepareLocalTransferSources, and paying for a second header read here would be the
	// same work twice.
	conversion := conversionPreflight{Undecided: "not run on this path"}
	if ctx.Inv.Bool("--dry-run") || effectiveRental {
		pctx, cancel := hub.LongContext()
		decided, problem := preflightConversionPlan(pctx, ctx, source, plan.SourceProfiles)
		cancel()
		if problem != nil {
			return problem
		}
		conversion = decided
	}
	id := plan.ID()
	if ctx.Inv.Bool("--dry-run") {
		if kind == "model-upload" {
			ref, _ := hub.ParseRef(destination)
			if _, problem := ownedPublication(ctx, ref); problem != nil {
				return problem
			}
		}
		fields := []output.Field{
			{K: "id", V: id}, {K: "kind", V: kind},
			{K: "model", V: destination},
			{K: "source", V: source.Canonical}, {K: "source_selection", V: source.Selection},
			{K: "source_files", V: source.Files}, {K: "source_bytes", V: output.Bytes(source.Bytes)},
			{K: "status", V: "planned"}, {K: "changed", V: false},
		}
		if producer != nil {
			fields = append(fields,
				output.Field{K: "producer", V: producer.Name + "@" + producer.Release},
				output.Field{K: "source_profiles", V: plan.ProfileNames()},
				output.Field{K: "steps", V: 1},
				output.Field{K: "outputs", V: plan.OutputNames()},
				output.Field{K: "needs_accelerator", V: producer.NeedsAccelerator},
			)
			// What the dispatch plan cannot say and this can: the reviewed headers admit a
			// conversion, and these are the journal keys the pod will open.
			if conversion.decided() {
				fields = append(fields,
					output.Field{K: "conversion", V: "planned"},
					output.Field{K: "conversion_sessions", V: conversion.Sessions()})
			} else {
				fields = append(fields,
					output.Field{K: "conversion", V: "undecided"},
					output.Field{K: "conversion_undecided", V: conversion.Undecided})
			}
		}
		defaults := []string{"id", "kind", "model", "source"}
		if producer != nil {
			defaults = append(defaults, "producer", "outputs", "conversion")
		}
		defaults = append(defaults, "status", "changed")
		return emit(ctx, compactRecord(fields, defaults...))
	}
	if kind == "model-upload" {
		ref, _ := hub.ParseRef(destination)
		if _, problem := ownedPublication(ctx, ref); problem != nil {
			return problem
		}
	}
	intent := modelTransferIntent(plan)
	for i := range intent.SourceFiles {
		intent.SourceFiles[i].Header = conversion.Headers[intent.SourceFiles[i].Member]
	}
	intent.LocalOnly = localOnly
	daemonState, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = daemonState
	local, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	submission := api.JobSubmission{Input: []byte("{}"), ModelTransfer: &intent,
		Rental: effectiveRental, RentalRequired: placement == "rental-only"}
	if producer != nil {
		submission.Package, submission.Function = producer.Pin.Package, producer.Pin.Function
		submission.InstallID, submission.Release = producer.Pin.InstallID, producer.Pin.Release
		if kind == "model-upload" {
			submission.Org = strings.Split(destination, "/")[0]
		}
	}
	planDigest, err := plan.Digest()
	if err != nil {
		return exit.Internalf("cannot digest model transfer submission: %s", err)
	}
	handle, problem := local.SubmitJob(submission,
		"model-transfer-"+strings.TrimPrefix(planDigest, "sha256:"))
	if problem != nil {
		return problem
	}
	state, problem := local.Job(handle.JobID)
	if problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--await") {
		return watchJob(ctx, local, state)
	}
	return renderSubmittedJob(ctx, state, !handle.Replay)
}

func localOnlyModelSource(source string) bool {
	if strings.HasPrefix(source, "local/") {
		return true
	}
	cwd, err := os.Getwd()
	if err != nil {
		return false
	}
	parsed, problem := modelsource.Parse(source, cwd)
	return problem == nil && parsed.Kind == modelsource.LocalFile
}

func modelTransferIntent(plan modeltransfer.Plan) records.ModelTransferIntent {
	files := make([]records.ModelTransferSourceFile, 0, len(plan.SourceFiles))
	for _, file := range plan.SourceFiles {
		files = append(files, records.ModelTransferSourceFile{Member: file.Member,
			SHA256: file.SHA256, Length: file.Length})
	}
	outputs := make([]records.ModelTransferOutput, 0, len(plan.Outputs))
	for _, output := range plan.Outputs {
		outputs = append(outputs, records.ModelTransferOutput{Name: output.Name})
	}
	return records.ModelTransferIntent{Kind: plan.Instruction.Kind, Destination: plan.Destination,
		Source: plan.Source, SourceSelection: plan.SourceSelection, SourceLicense: plan.SourceLicense,
		SourceFiles: files, InputLane: plan.InputLane, SourceProfiles: plan.SourceProfiles,
		Outputs: outputs}
}

func canonicalProductionSource(ctx *Context, raw string) (string, *exit.Error) {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "local/") {
		if problem := modelsource.LocalName(strings.TrimPrefix(raw, "local/")); problem != nil {
			return "", problem
		}
		return raw, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", exit.Internalf("cannot resolve the current directory: %s", err)
	}
	if parsed, problem := modelsource.Parse(raw, cwd); problem == nil {
		return parsed.Canonical, nil
	}
	if !catalogModelSpelling(raw) {
		_, problem := modelsource.Parse(raw, cwd)
		return "", problem
	}
	name, release, pinned := strings.Cut(raw, "@")
	if !pinned || strings.TrimSpace(release) == "" {
		return "", exit.Usagef("model transfer source %q is not pinned to one Tensorhub release", raw).
			WithRemedy("use org/model@release; mutable source latest cannot enter operation identity")
	}
	ref, problem := hub.ParseRef(name)
	if problem != nil {
		return "", problem
	}
	return ref.String() + "@" + strings.TrimSpace(release), nil
}

func resolvePublishSource(ctx *Context, raw string, sourceProfiles []string) (publishSource, *exit.Error) {
	raw = strings.TrimSpace(raw)
	lane := strings.TrimSpace(ctx.Inv.Value("--lane"))
	if strings.HasPrefix(raw, "local/") {
		name := strings.TrimPrefix(raw, "local/")
		if problem := modelsource.LocalName(name); problem != nil {
			return publishSource{}, problem
		}
		if ctx.Inv.Bool("--rental") {
			return publishSource{}, exit.Usagef("a local/%s alias cannot be read by a rented worker", name).
				WithRemedy("use its addressable Tensorhub release or original pinned foreign source")
		}
		if lane != "" {
			return publishSource{}, exit.Usagef("--lane selects only a Tensorhub model release")
		}
		tool, _, problem := localTensorFS(ctx)
		if problem != nil {
			return publishSource{}, problem
		}
		alias, problem := tool.ResolveLocal(name)
		if problem != nil {
			return publishSource{}, problem
		}
		return publishSource{Canonical: "local/" + name, Selection: alias.ManifestDigest}, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return publishSource{}, exit.Internalf("cannot resolve the current directory: %s", err)
	}
	parsed, parseProblem := modelsource.Parse(raw, cwd)
	if parseProblem == nil {
		if lane != "" {
			return publishSource{}, exit.Usagef("--lane selects only a Tensorhub model release")
		}
		if parsed.Kind == modelsource.LocalFile {
			if ctx.Inv.Bool("--rental") {
				return publishSource{}, exit.Usagef("a local model file cannot be read by a rented worker").
					WithRemedy("import it locally or use an addressable pinned foreign source")
			}
			file, err := os.Open(parsed.Path)
			if err != nil {
				return publishSource{}, exit.New(exit.NotFound, "cannot open local model source: %s", err)
			}
			defer file.Close()
			hash := sha256.New()
			length, err := io.Copy(hash, file)
			if err != nil || length != parsed.Bytes {
				return publishSource{}, exit.Named(exit.Conflict, "model_source.local_changed",
					"local model source changed while measuring its exact identity")
			}
			digest := hex.EncodeToString(hash.Sum(nil))
			return publishSource{Canonical: parsed.Canonical, Selection: "sha256:" + digest,
				Files: 1, Bytes: parsed.Bytes, Exact: []modeltransfer.SourceFile{{
					Member: filepath.Base(parsed.Path), SHA256: digest, Length: parsed.Bytes}}}, nil
		}
		var token = ctx.Cfg.HuggingFaceToken
		if parsed.Kind == modelsource.Civitai {
			token = ctx.Cfg.CivitaiToken
		}
		resolver, problem := modelsource.NewResolver(parsed.Kind, token)
		if problem != nil {
			return publishSource{}, problem
		}
		hctx, cancel := hub.LongContext()
		defer cancel()
		resolved, problem := resolver.Resolve(hctx, parsed)
		if problem != nil {
			return publishSource{}, problem
		}
		// A single carrier is already exact (for example Civitai's primary
		// checkpoint). Multi-carrier provider repositories must be narrowed by
		// TensorFS's reviewed profiles before any body is persisted or granted.
		if len(resolved.Files) > 1 && len(sourceProfiles) > 0 {
			tool, _, problem := localTensorFS(ctx)
			if problem != nil {
				return publishSource{}, problem
			}
			members, problem := tool.SourceProfileMembers(ctx.Cfg.TensorFSRegistry, sourceProfiles)
			if problem != nil {
				return publishSource{}, problem
			}
			resolved, problem = resolved.Select(members)
			if problem != nil {
				return publishSource{}, problem
			}
		}
		exact := make([]modeltransfer.SourceFile, 0, len(resolved.Files))
		access := make([]sourceCapability, 0, len(resolved.Files))
		provider := string(parsed.Kind)
		for _, file := range resolved.Files {
			exact = append(exact, modeltransfer.SourceFile{
				Member: file.Member, SHA256: file.SHA256, Length: file.Length,
			})
			access = append(access, sourceCapability{
				Member: file.Member, ObjectID: "sha256:" + file.SHA256,
				Length: file.Length, Provider: provider, URL: file.URL,
			})
		}
		return publishSource{Canonical: resolved.Canonical,
			Selection: "sha256:" + resolved.SelectionSHA256,
			License:   resolved.License, Files: len(resolved.Files), Bytes: resolved.Bytes,
			Exact: exact, Access: access,
			Resolver: resolver, Resolution: resolved}, nil
	}
	if !catalogModelSpelling(raw) {
		return publishSource{}, parseProblem
	}
	hctx, cancel := hub.LongContext()
	defer cancel()
	resolved, problem := client(ctx).ResolveModel(hctx, raw, lane)
	if problem != nil {
		return publishSource{}, problem
	}
	if resolved.Release == "" || resolved.Lane == "" || resolved.ManifestID == "" {
		return publishSource{}, exit.Named(exit.Structural, "model_source_resolution_incomplete",
			"Tensorhub did not resolve one exact checkpoint from the model release lane")
	}
	return publishSource{Canonical: resolved.Model + "@" + resolved.Release,
		Selection: resolved.ManifestID, Lane: resolved.Lane,
		Files: resolved.Objects, Bytes: resolved.Bytes}, nil
}

func catalogModelSpelling(value string) bool {
	if strings.Contains(value, "://") || filepath.IsAbs(value) || strings.HasPrefix(value, ".") {
		return false
	}
	name, _, _ := strings.Cut(value, "@")
	parts := strings.Split(name, "/")
	return len(parts) == 2 && parts[0] != "" && parts[1] != ""
}

// parseSourceProfileFlags reads repeatable --source-profile slot=profile pairs.
// They bind a producer's model inputs to reviewed TensorFS source profiles when
// the job declares none; pass-through has no slots and takes no profiles.
func parseSourceProfileFlags(ctx *Context, producerName string) (map[string]string, *exit.Error) {
	values := ctx.Inv.Values["--source-profile"]
	if len(values) == 0 {
		return nil, nil
	}
	if producerName == "" {
		return nil, exit.Usagef("--source-profile binds a producer job's model inputs; pass-through takes no source profiles").
			WithRemedy("select the producer job whose inputs these profiles narrow, e.g. --producer org/package@vN/function")
	}
	supplied := make(map[string]string, len(values))
	for _, value := range values {
		slot, profile, ok := strings.Cut(value, "=")
		slot, profile = strings.TrimSpace(slot), strings.TrimSpace(profile)
		if !ok || slot == "" || profile == "" {
			return nil, exit.Usagef("--source-profile %q is not slot=profile", value)
		}
		if _, duplicate := supplied[slot]; duplicate {
			return nil, exit.Usagef("--source-profile names slot %s twice", slot)
		}
		supplied[slot] = profile
	}
	return supplied, nil
}

func resolveProducerPlan(ctx *Context, raw string, supplied map[string]string) (*producerPlan, *exit.Error) {
	if raw == "" {
		return nil, nil
	}
	target, problem := parseProductionCallable(raw)
	if problem != nil {
		return nil, problem
	}
	selected, problem := resolveProductionPackage(ctx, target.Selector, rentalRequested(ctx))
	if problem != nil {
		return nil, problem
	}
	packageInterface := selected.PackageInterface
	job, problem := packageInterface.Function(target.Function)
	if problem != nil || job.Kind != "job" {
		return nil, exit.Named(exit.Validation, "model_producer.not_job",
			"--producer %s does not name one ordinary job callable", raw)
	}
	if problem := modeltransfer.ValidateProducer(raw, job, supplied); problem != nil {
		return nil, problem
	}
	plan := &producerPlan{Name: target.Selector + "/" + target.Function,
		InstallID: selected.InstallID, Release: selected.Release,
		SourceDigest:     selected.SourceDigest,
		PackageInterface: packageInterface, Job: job,
		NeedsAccelerator: launch.AcceleratorRequired(selected.Requirements),
		SourceProfiles:   map[string]string{}}
	plan.Pin = modeltransfer.JobPin{Callable: plan.Name, Package: target.Package,
		Function: target.Function, InstallID: selected.InstallID, Release: selected.Release,
		SourceDigest: selected.SourceDigest, DescriptorID: job.DescriptorID}
	for _, slot := range job.Models {
		plan.SourceProfiles[slot.Param] = supplied[slot.Param]
	}
	for _, output := range job.WeightsOutputs {
		plan.Outputs = append(plan.Outputs, modeltransfer.OutputPin{Name: output.OutputID})
	}
	return plan, nil
}

type productionPackage struct {
	InstallID, Release, SourceDigest string
	PackageInterface                 *launch.PackageInterface
	Requirements                     []string
}

func resolveProductionPackage(ctx *Context, packageName string, remote bool) (productionPackage, *exit.Error) {
	selector, problem := packageref.ParseRef(packageName)
	if problem != nil {
		return productionPackage{}, problem
	}
	key := selector.String()
	if !remote {
		install, problem := installedPackage(ctx, key)
		if problem != nil {
			return productionPackage{}, problem.WithRemedy(
				"install the exact producer package before running it locally")
		}
		facts, problem := launch.Read(*install, ctx.Cfg.Home, ctx.Cfg.Tool())
		if problem != nil {
			return productionPackage{}, problem
		}
		selected := productionPackage{InstallID: install.ID, Release: install.Version,
			SourceDigest: install.SourceDigest, PackageInterface: facts.PackageInterface,
			Requirements: strings.Split(install.Closure, "\n")}
		return selected, nil
	}

	ref, problem := hub.ParseRef(selector.Package)
	if problem != nil {
		return productionPackage{}, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx).PackageCard(hctx, ref)
	if problem != nil {
		return productionPackage{}, problem
	}
	if card.Package.Ref() != selector.Package {
		return productionPackage{}, exit.Named(exit.Conflict,
			"model_transfer.package_catalog_changed",
			"Tensorhub returned package %s while resolving %s", card.Package.Ref(), key)
	}
	var release string
	if selector.HasMajor {
		release, problem = newestPackageRelease(card.Releases, selector.Major)
	} else {
		release, problem = newestPackageRelease(card.Releases)
	}
	if problem != nil {
		detail := ""
		if selector.HasMajor {
			detail = " in v" + strconv.Itoa(selector.Major)
		}
		return productionPackage{}, exit.Named(exit.NotFound,
			"model_transfer.package_release_absent",
			"Tensorhub package %s has no active immutable release%s", selector.Package, detail)
	}
	detail, problem := client(ctx).PackageRelease(hctx, ref, release)
	if problem != nil {
		return productionPackage{}, problem
	}
	if detail.Release.Release != release || detail.Release.Yanked || detail.Release.YankedAt != "" ||
		detail.Release.PackageInterfaceLength != int64(len(detail.PackageInterface)) {
		return productionPackage{}, exit.Named(exit.Conflict, "model_transfer.package_changed",
			"Tensorhub package %s@%s returned inconsistent immutable release metadata",
			selector.Package, release)
	}
	requirements, problem := detail.Requirements()
	if problem != nil {
		return productionPackage{}, problem
	}
	packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return productionPackage{}, problem
	}
	if packageInterface.Digest != detail.Release.PackageInterfaceDigest {
		return productionPackage{}, exit.Named(exit.Conflict, "model_transfer.package_interface_changed",
			"Tensorhub package interface does not match its release fact")
	}
	selected := productionPackage{Release: release,
		PackageInterface: packageInterface, Requirements: requirements}
	return selected, nil
}

type productionCallable struct {
	Package, Selector, Function string
}

func parseProductionCallable(value string) (productionCallable, *exit.Error) {
	parts := strings.Split(value, "/")
	if len(parts) != 3 || parts[2] == "" {
		return productionCallable{}, exit.Usagef("%q is not org/package@vN/function", value)
	}
	selector, problem := packageref.ParseRef(parts[0] + "/" + parts[1])
	if problem != nil {
		return productionCallable{}, problem
	}
	return productionCallable{Package: selector.Package, Selector: selector.String(),
		Function: parts[2]}, nil
}
