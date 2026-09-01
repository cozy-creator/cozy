package cli

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	packageref "github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/tfs"
)

var modelReleasePattern = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

type publishSource struct {
	Canonical string
	Selection string
	License   string
	Lane      string
	Files     int
	Bytes     int64
	Exact     []modelproduction.SourceFile
	Access    []api.ModelProductionSourceCapability
}

type producerPlan struct {
	Name          string
	InstallID     string
	Release       string
	ReleaseDigest string
	Descriptor    *launch.PackageDescriptor
	Production    *launch.ModelProduction
	Jobs          []modelproduction.JobPin
	GPUCount      int64
	Requires      []string
	Needs         modelproduction.ResourceNeeds
}

func handleModelPublish(ctx *Context) *exit.Error {
	destination, problem := hub.ParseRef(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	if destination.Org == "local" {
		return exit.Usagef("local/ is reserved for private aliases and cannot be a Tensorhub destination").
			WithRemedy("publish under your Tensorhub account, for example alice/%s", destination.Name)
	}
	release := strings.TrimSpace(ctx.Inv.Value("--release"))
	if !modelReleasePattern.MatchString(release) {
		return exit.Usagef("--release %q is not an immutable N.M.P semantic version", release)
	}
	if ctx.Inv.Bool("--dry-run") && ctx.Inv.Bool("--detach") {
		return exit.Usagef("--dry-run and --detach conflict: a dry-run creates no durable operation")
	}
	if ctx.Inv.Bool("--rental") {
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
	if producerName == "" && !ctx.Inv.Bool("--rental") {
		_, manifestProblem := tfs.ManifestID(ctx.Inv.Args[1])
		if strings.HasPrefix(ctx.Inv.Args[1], "local/") || manifestProblem == nil {
			if ctx.Inv.Bool("--detach") {
				return exit.Usagef("--detach requires a durable model production")
			}
			return handleDirectModelPublish(ctx)
		}
	}
	instructionSource, problem := canonicalProductionSource(ctx, ctx.Inv.Args[1])
	if problem != nil {
		return problem
	}
	instruction := modelproduction.Instruction{Destination: destination.String(), Release: release,
		Source: instructionSource, InputLane: strings.TrimSpace(ctx.Inv.Value("--lane")),
		Producer: producerName, Rental: ctx.Inv.Bool("--rental")}
	if !ctx.Inv.Bool("--dry-run") && ctx.Inv.Bool("--rental") && producerName != "" {
		daemonState, _, problem := ensureDaemon(ctx)
		if problem != nil {
			return problem
		}
		ctx.Daemon = daemonState
		local, problem := dial(ctx)
		if problem != nil {
			return problem
		}
		productionState, problem := local.SubmitModelProduction(instruction)
		if problem != nil {
			return problem
		}
		if ctx.Inv.Bool("--detach") {
			return emitModelProductionState(ctx, productionState, true)
		}
		return followModelProduction(ctx, local, productionState)
	}

	source, problem := resolvePublishSource(ctx, ctx.Inv.Args[1])
	if problem != nil {
		return problem
	}
	producer, problem := resolveProducerPlan(ctx, producerName)
	if problem != nil {
		return problem
	}
	plan := modelproduction.Plan{
		Instruction: instruction,
		Destination: destination.String(), Release: release,
		Source: source.Canonical, SourceSelection: source.Selection,
		SourceLicense: source.License, InputLane: source.Lane,
		SourceFiles: source.Exact,
	}
	if producer != nil {
		plan.Producer, plan.ProducerInstallID = producer.Name, producer.InstallID
		plan.ProducerRelease = producer.Release
		plan.ProducerDigest, plan.DescriptorDigest = producer.ReleaseDigest, producer.Descriptor.Digest
		plan.Production, plan.Jobs = producer.Production, producer.Jobs
		plan.Resources = producer.Needs
	}
	id := plan.ID()
	if ctx.Inv.Bool("--dry-run") {
		if _, problem := ownedPublication(ctx, destination); problem != nil {
			return problem
		}
		fields := []output.Field{
			{K: "id", V: id}, {K: "kind", V: "model-publication"},
			{K: "model", V: destination.String()}, {K: "release", V: release},
			{K: "source", V: source.Canonical}, {K: "source_selection", V: source.Selection},
			{K: "source_files", V: source.Files}, {K: "source_bytes", V: output.Bytes(source.Bytes)},
			{K: "status", V: "planned"}, {K: "changed", V: false},
		}
		if producer != nil {
			fields = append(fields,
				output.Field{K: "producer", V: producer.Name + "@" + producer.Release},
				output.Field{K: "production", V: producer.Production.Name},
				output.Field{K: "source_profiles", V: plan.SourceProfiles()},
				output.Field{K: "steps", V: len(producer.Jobs)},
				output.Field{K: "lanes", V: plan.Lanes()},
				output.Field{K: "gpu_count", V: producer.GPUCount},
				output.Field{K: "requires", V: producer.Requires},
			)
		}
		defaults := []string{"id", "kind", "model", "release", "source"}
		if producer != nil {
			defaults = append(defaults, "producer", "production", "lanes")
		}
		defaults = append(defaults, "status", "changed")
		return emit(ctx, compactRecord(fields, defaults...))
	}
	return exit.Named(exit.Unavailable, "model_publication_execution_unavailable",
		"model publication %s is planned, but this build cannot yet submit its durable source publication", id)
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
		return "", exit.Usagef("model production source %q is not pinned to one Tensorhub release", raw).
			WithRemedy("use org/model@release; mutable source latest cannot enter operation identity")
	}
	ref, problem := hub.ParseRef(name)
	if problem != nil {
		return "", problem
	}
	return ref.String() + "@" + strings.TrimSpace(release), nil
}

func resolvePublishSource(ctx *Context, raw string) (publishSource, *exit.Error) {
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
			return publishSource{Canonical: parsed.Canonical, Files: 1, Bytes: parsed.Bytes}, nil
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
		exact := make([]modelproduction.SourceFile, 0, len(resolved.Files))
		access := make([]api.ModelProductionSourceCapability, 0, len(resolved.Files))
		provider := string(parsed.Kind)
		for _, file := range resolved.Files {
			exact = append(exact, modelproduction.SourceFile{
				Member: file.Member, SHA256: file.SHA256, Length: file.Length,
			})
			access = append(access, api.ModelProductionSourceCapability{
				Member: file.Member, ObjectID: "sha256:" + file.SHA256,
				Length: file.Length, Provider: provider, URL: file.URL,
			})
		}
		return publishSource{Canonical: resolved.Canonical,
			Selection: "sha256:" + resolved.SelectionSHA256,
			License:   resolved.License, Files: len(resolved.Files), Bytes: resolved.Bytes,
			Exact: exact, Access: access}, nil
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
			"Tensorhub did not resolve one immutable model release lane")
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

func resolveProducerPlan(ctx *Context, raw string) (*producerPlan, *exit.Error) {
	if raw == "" {
		return nil, nil
	}
	parts := strings.Split(raw, "/")
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return nil, exit.Usagef("--producer %q is not org/package[@vN]/production", raw)
	}
	packageName := parts[0] + "/" + parts[1]
	producerSelector, problem := packageref.ParseRef(packageName)
	if problem != nil {
		return nil, problem
	}
	remote := ctx.Inv.Bool("--rental")
	packages := map[string]productionPackage{}
	selected, problem := resolveProductionPackage(ctx, producerSelector.String(), remote, packages)
	if problem != nil {
		return nil, problem
	}
	if major, majorProblem := packageref.MajorOf(selected.Release); majorProblem == nil {
		// Self-steps select the producer's major but must reuse the already-frozen
		// producer release, even if the catalog advances during this one plan.
		packages[producerSelector.Package+"@v"+strconv.Itoa(major)] = selected
	}
	descriptor := selected.Descriptor
	production, problem := descriptor.Production(parts[2])
	if problem != nil {
		return nil, problem
	}
	ordered, problem := production.OrderedSteps()
	if problem != nil {
		return nil, problem
	}
	plan := &producerPlan{Name: raw, InstallID: selected.InstallID, Release: selected.Release,
		ReleaseDigest: selected.ReleaseDigest,
		Descriptor:    descriptor, Production: production}
	requires := map[string]bool{}
	for _, step := range ordered {
		target, parseProblem := parseProductionCallable(step.Callable)
		if parseProblem != nil {
			return nil, exit.Named(exit.Validation, "model_production_callable_invalid",
				"production step %s does not name one job callable", step.Name)
		}
		stepPackage, packageProblem := resolveProductionPackage(ctx, target.Selector, remote,
			packages)
		if packageProblem != nil {
			return nil, packageProblem
		}
		stepDescriptor := stepPackage.Descriptor
		job, jobProblem := stepDescriptor.Function(target.Function)
		if jobProblem != nil || job.Kind != "job" {
			return nil, exit.Named(exit.Validation, "model_production_callable_not_job",
				"production step %s callable %s is not one job", step.Name, step.Callable)
		}
		if validation := validateProductionInvocation(step, job); validation != nil {
			return nil, validation
		}
		gpu := max(step.Resources.GPUCount, job.Resources.GPUCount)
		plan.GPUCount = max(plan.GPUCount, gpu)
		for _, value := range []string{step.Resources.Requires, job.Resources.Requires} {
			for _, token := range strings.Split(value, ",") {
				if token = strings.TrimSpace(token); token != "" {
					requires[token] = true
				}
			}
		}
		plan.Jobs = append(plan.Jobs, modelproduction.JobPin{
			Step: step.Name, Callable: step.Callable, Release: stepPackage.Release,
			InstallID: stepPackage.InstallID, ReleaseDigest: stepPackage.ReleaseDigest,
			DescriptorID: job.DescriptorID,
		})
	}
	for token := range requires {
		plan.Requires = append(plan.Requires, token)
	}
	sort.Strings(plan.Requires)
	plan.Needs, problem = productionResourceNeeds(plan.GPUCount, plan.Requires)
	if problem != nil {
		return nil, problem
	}
	return plan, nil
}

type productionPackage struct {
	InstallID, Release, ReleaseDigest string
	Descriptor                        *launch.PackageDescriptor
}

func resolveProductionPackage(ctx *Context, packageName string, remote bool,
	cache map[string]productionPackage,
) (productionPackage, *exit.Error) {
	selector, problem := packageref.ParseRef(packageName)
	if problem != nil {
		return productionPackage{}, problem
	}
	key := selector.String()
	if selected, ok := cache[key]; ok {
		return selected, nil
	}
	if !remote {
		install, problem := installedPackage(ctx, key)
		if problem != nil {
			return productionPackage{}, problem.WithRemedy(
				"install every exact production package before running it locally")
		}
		facts, problem := launch.Read(*install, ctx.Cfg.Home, ctx.Cfg.Tool())
		if problem != nil {
			return productionPackage{}, problem
		}
		selected := productionPackage{InstallID: install.ID, Release: install.Version,
			ReleaseDigest: install.SourceDigest, Descriptor: facts.PackageDescriptor}
		cache[key] = selected
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
			"model_production.package_catalog_changed",
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
			"model_production.package_release_absent",
			"Tensorhub package %s has no active immutable release%s", selector.Package, detail)
	}
	detail, problem := client(ctx).PackageRelease(hctx, ref, release)
	if problem != nil {
		return productionPackage{}, problem
	}
	if detail.Release.Release != release || detail.Release.Yanked || detail.Release.YankedAt != "" ||
		detail.Release.PackageDescriptorLength != int64(len(detail.PackageDescriptor)) {
		return productionPackage{}, exit.Named(exit.Conflict, "model_production.package_changed",
			"Tensorhub package %s@%s returned inconsistent immutable release metadata",
			selector.Package, release)
	}
	if _, err := canonical.Raw(detail.Release.ReleaseDigest); err != nil {
		return productionPackage{}, exit.Named(exit.Conflict,
			"model_production.package_release_digest_invalid",
			"Tensorhub package %s@%s has no exact immutable release digest",
			selector.Package, release)
	}
	descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
	if problem != nil {
		return productionPackage{}, problem
	}
	if descriptor.Digest != detail.Release.PackageDescriptorDigest {
		return productionPackage{}, exit.Named(exit.Conflict, "model_production.descriptor_changed",
			"Tensorhub package descriptor does not match its release fact")
	}
	selected := productionPackage{Release: release, ReleaseDigest: detail.Release.ReleaseDigest,
		Descriptor: descriptor}
	cache[key] = selected
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

var productionResourcePattern = regexp.MustCompile(`^(sm|vram|ram)([1-9][0-9]*)(\+|g)?$`)

func productionResourceNeeds(gpuCount int64, requires []string) (modelproduction.ResourceNeeds, *exit.Error) {
	needs := modelproduction.ResourceNeeds{GPUCount: gpuCount}
	for _, value := range requires {
		match := productionResourcePattern.FindStringSubmatch(strings.ToLower(value))
		if len(match) != 4 || match[1] == "sm" && match[3] != "+" ||
			match[1] != "sm" && match[3] != "g" {
			return needs, exit.Named(exit.Validation, "model_production_resource_unknown",
				"production resource requirement %q is not smN+, vramNg, or ramNg", value)
		}
		amount, err := strconv.ParseInt(match[2], 10, 64)
		if err != nil {
			return needs, exit.Named(exit.Validation, "model_production_resource_invalid",
				"production resource requirement %q is outside the supported range", value)
		}
		switch match[1] {
		case "sm":
			needs.MinSM = max(needs.MinSM, amount)
		case "vram":
			needs.VRAMGB = max(needs.VRAMGB, amount)
		case "ram":
			needs.RAMGB = max(needs.RAMGB, amount)
		}
	}
	if needs.GPUCount != 1 || needs.MinSM == 0 || needs.VRAMGB == 0 || needs.RAMGB == 0 {
		return needs, exit.Named(exit.Validation, "model_production_resources_incomplete",
			"rented model production requires exactly one GPU plus explicit smN+, vramNg, and ramNg floors")
	}
	return needs, nil
}

func validateProductionInvocation(step launch.ModelProductionStep, job *launch.Entrypoint) *exit.Error {
	models := map[string]bool{}
	for _, slot := range job.Models {
		models[slot.Param] = true
	}
	if !sameNames(models, keys(step.Models)) {
		return exit.Named(exit.Validation, "model_production_model_inputs_mismatch",
			"production step %s model inputs do not match job %s", step.Name, job.Name)
	}
	for _, field := range job.Request.Fields {
		if field.Wire == "required" {
			return exit.Named(exit.Validation, "model_production_argument_missing",
				"production step %s job %s requires argument %s", step.Name, job.Name, field.Name)
		}
	}
	outputs := map[string]bool{}
	for _, output := range job.ArtifactOutputs {
		outputs[output.OutputID] = true
	}
	wanted := make(map[string]bool, len(step.Outputs))
	for _, output := range step.Outputs {
		wanted[output] = true
	}
	if !sameNames(outputs, wanted) {
		return exit.Named(exit.Validation, "model_production_outputs_mismatch",
			"production step %s outputs do not match job %s ArtifactSink slots", step.Name, job.Name)
	}
	return nil
}

func keys(values map[string]string) map[string]bool {
	out := make(map[string]bool, len(values))
	for key := range values {
		out[key] = true
	}
	return out
}

func sameNames(first, second map[string]bool) bool {
	if len(first) != len(second) {
		return false
	}
	for name := range first {
		if !second[name] {
			return false
		}
	}
	return true
}
