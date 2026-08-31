package cli

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
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
			WithRemedy("publish under your Tensorhub organization, for example alice/%s", destination.Name)
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
	if producerName == "" && !ctx.Inv.Bool("--rental") {
		_, manifestProblem := tfs.ManifestID(ctx.Inv.Args[1])
		if strings.HasPrefix(ctx.Inv.Args[1], "local/") || manifestProblem == nil {
			if ctx.Inv.Bool("--detach") {
				return exit.Usagef("--detach requires a durable model production")
			}
			return handleDirectModelPublish(ctx)
		}
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
				output.Field{K: "nodes", V: len(producer.Jobs)},
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
	if ctx.Inv.Bool("--rental") && len(source.Exact) > 0 {
		return exit.Named(exit.Unavailable, "model_source_host_exchange_unavailable",
			"model publication %s is planned, but pod-supervisor cannot yet prepare its selected foreign source files", id).
			WithRemedy("the minor-9 PrepareModelSources host exchange must land; Creator will not download model bodies or hand a broad provider credential to the worker")
	}
	if producer != nil {
		return exit.Named(exit.Unavailable, "model_job_bindings_unavailable",
			"model publication %s is planned, but job InvocationSpec cannot yet bind exact prepared Model sources", id).
			WithRemedy("the minor-9 job model-binding contract must land; an input-tree path is not an ArtifactSink Model capability")
	}
	return exit.Named(exit.Unavailable, "model_publication_execution_unavailable",
		"model publication %s is planned, but this build cannot yet submit its durable source publication", id)
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
		for _, file := range resolved.Files {
			exact = append(exact, modelproduction.SourceFile{
				Member: file.Member, SHA256: file.SHA256, Length: file.Length,
			})
		}
		return publishSource{Canonical: resolved.Canonical,
			Selection: "sha256:" + resolved.SelectionSHA256,
			License:   resolved.License, Files: len(resolved.Files), Bytes: resolved.Bytes,
			Exact: exact}, nil
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
		return nil, exit.Usagef("--producer %q is not org/package/production", raw)
	}
	packageName := parts[0] + "/" + parts[1]
	install, problem := installedPackage(ctx, packageName)
	if problem != nil {
		return nil, problem.WithRemedy("install the exact producer package before planning its production")
	}
	var descriptor *launch.PackageDescriptor
	if ctx.Inv.Bool("--rental") {
		descriptor, problem = publishedDescriptorForInstall(client(ctx), install)
	} else {
		var facts *launch.Facts
		facts, problem = launch.Read(*install, ctx.Cfg.Home, ctx.Cfg.Tool())
		if facts != nil {
			descriptor = facts.PackageDescriptor
		}
	}
	if problem != nil {
		return nil, problem
	}
	production, problem := descriptor.Production(parts[2])
	if problem != nil {
		return nil, problem
	}
	ordered, problem := production.OrderedNodes()
	if problem != nil {
		return nil, problem
	}
	plan := &producerPlan{Name: raw, InstallID: install.ID, Release: install.Version,
		ReleaseDigest: install.SourceDigest,
		Descriptor:    descriptor, Production: production}
	requires := map[string]bool{}
	for _, node := range ordered {
		target, parseProblem := parseTarget(node.Callable)
		if parseProblem != nil || target.Function == "" {
			return nil, exit.Named(exit.Validation, "model_production_callable_invalid",
				"production node %s does not name one job callable", node.Name)
		}
		nodeInstall, installProblem := installedPackage(ctx, target.Package)
		if installProblem != nil {
			return nil, installProblem.WithRemedy("install every exact production job package before planning")
		}
		var nodeDescriptor *launch.PackageDescriptor
		if ctx.Inv.Bool("--rental") {
			nodeDescriptor, installProblem = publishedDescriptorForInstall(client(ctx), nodeInstall)
		} else {
			var facts *launch.Facts
			facts, installProblem = launch.Read(*nodeInstall, ctx.Cfg.Home, ctx.Cfg.Tool())
			if facts != nil {
				nodeDescriptor = facts.PackageDescriptor
			}
		}
		if installProblem != nil {
			return nil, installProblem
		}
		job, jobProblem := nodeDescriptor.Function(target.Function)
		if jobProblem != nil || job.Kind != "job" {
			return nil, exit.Named(exit.Validation, "model_production_callable_not_job",
				"production node %s callable %s is not one installed job", node.Name, node.Callable)
		}
		if validation := validateProductionInvocation(node, job); validation != nil {
			return nil, validation
		}
		gpu := max(node.Resources.GPUCount, job.Resources.GPUCount)
		plan.GPUCount = max(plan.GPUCount, gpu)
		for _, value := range []string{node.Resources.Requires, job.Resources.Requires} {
			for _, token := range strings.Split(value, ",") {
				if token = strings.TrimSpace(token); token != "" {
					requires[token] = true
				}
			}
		}
		facts, factsProblem := launch.Read(*nodeInstall, ctx.Cfg.Home, ctx.Cfg.Tool())
		if factsProblem != nil {
			return nil, factsProblem
		}
		jobFacts, factsProblem := facts.Job(target.Function)
		if factsProblem != nil {
			return nil, factsProblem
		}
		plan.Jobs = append(plan.Jobs, modelproduction.JobPin{
			Node: node.Name, Callable: node.Callable, Release: nodeInstall.Version,
			InstallID: nodeInstall.ID, ReleaseDigest: nodeInstall.SourceDigest,
			DescriptorID: jobFacts.DescriptorID,
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

func publishedDescriptorForInstall(c *hub.Client,
	install *records.PackageInstall,
) (*launch.PackageDescriptor, *exit.Error) {
	if install.SourceKind != "tensorhub" || !install.Verified || install.SourceDigest == "" {
		return nil, exit.Named(exit.Validation, "model_production.package_unpublished",
			"production package %s is not one exact Tensorhub release", install.Package)
	}
	ref, problem := hub.ParseRef(install.Package)
	if problem != nil {
		return nil, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	detail, problem := c.PackageRelease(hctx, ref, install.Version)
	if problem != nil {
		return nil, problem
	}
	if detail.Release.ReleaseDigest != install.SourceDigest ||
		detail.Release.PackageDescriptorLength != int64(len(detail.PackageDescriptor)) {
		return nil, exit.Named(exit.Conflict, "model_production.package_changed",
			"Tensorhub package %s@%s no longer matches the installed exact release",
			install.Package, install.Version)
	}
	descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
	if problem != nil {
		return nil, problem
	}
	if descriptor.Digest != detail.Release.PackageDescriptorDigest {
		return nil, exit.Named(exit.Conflict, "model_production.descriptor_changed",
			"Tensorhub package descriptor does not match its release fact")
	}
	return descriptor, nil
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

func validateProductionInvocation(node launch.ModelProductionNode, job *launch.Entrypoint) *exit.Error {
	models := map[string]bool{}
	for _, slot := range job.Models {
		models[slot.Param] = true
	}
	if !sameNames(models, keys(node.Models)) {
		return exit.Named(exit.Validation, "model_production_model_inputs_mismatch",
			"production node %s model inputs do not match job %s", node.Name, job.Name)
	}
	for _, field := range job.Request.Fields {
		if field.Wire == "required" {
			return exit.Named(exit.Validation, "model_production_argument_missing",
				"production node %s job %s requires argument %s", node.Name, job.Name, field.Name)
		}
	}
	outputs := map[string]bool{}
	for _, output := range job.ArtifactOutputs {
		outputs[output.OutputID] = true
	}
	wanted := make(map[string]bool, len(node.Outputs))
	for _, output := range node.Outputs {
		wanted[output] = true
	}
	if !sameNames(outputs, wanted) {
		return exit.Named(exit.Validation, "model_production_outputs_mismatch",
			"production node %s outputs do not match job %s ArtifactSink slots", node.Name, job.Name)
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
