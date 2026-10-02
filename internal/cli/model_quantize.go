package cli

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// quantizedLanes are the lanes `cozy model quantize` can ask a package for, one flag each.
var quantizedLanes = []string{"fp8", "mxfp8"}

// handleModelQuantize runs the quantizer of the package that serves a model. The package
// owns the recipe (which components and layers); a quantizer is its job that takes the model
// and declares one weights output named for the lane. It is the ordinary conversion run,
// `cozy run <package>/<job> <checkpoint> <org/model>`, on the machine the run names.
func handleModelQuantize(ctx *Context) *exit.Error {
	lane := ""
	for _, name := range quantizedLanes {
		if !ctx.Inv.Bool("--" + name) {
			continue
		}
		if lane != "" {
			return exit.Usagef("one run writes one lane: choose --%s or --%s", lane, name)
		}
		lane = name
	}
	if lane == "" {
		return exit.Usagef("name the lane to write: --%s", strings.Join(quantizedLanes, " or --"))
	}
	if problem := validateRunPlacement(ctx); problem != nil {
		return problem
	}
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	ref, checkpoint, problem := quantizeSource(ctx, ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	destination := ref.String()
	if len(ctx.Inv.Args) > 1 && strings.TrimSpace(ctx.Inv.Args[1]) != "" {
		destination = strings.TrimSpace(ctx.Inv.Args[1])
	}
	pkg := ctx.Inv.Value("--package")
	if pkg == "" {
		if pkg, problem = servingQuantizer(ctx, lane, ref, checkpoint); problem != nil {
			return problem
		}
	}
	ctx.Inv.Args = []string{pkg}
	target, surface, problem := invocationTarget(ctx)
	if problem != nil {
		return problem
	}
	job := quantizer(surface, lane)
	if job == nil {
		reclaimSnapshot(ctx, target)
		return noQuantizer(lane, "%s %s has no %s quantizer", target.Package, target.Release, lane)
	}
	source := ref.String() + "#" + checkpoint
	if problem := holdCheckpoint(ctx, source); problem != nil {
		reclaimSnapshot(ctx, target)
		return problem
	}
	target.Function = job.Name
	ctx.Inv.Args = []string{target.Package + "/" + job.Name, source, destination}
	return runTarget(ctx, target, surface)
}

// holdCheckpoint has the machine a run names hold the checkpoint first: the request `cozy
// model download <checkpoint> [--rental]` makes, which moves no byte the machine already has.
// Runtime before 0.18.102 refuses a run naming a checkpoint whose bytes it holds without its
// repository, as the pod that uploaded it does. A run that buys its machine fetches it itself.
func holdCheckpoint(ctx *Context, source string) *exit.Error {
	machine, known, problem := knownMachine(ctx)
	if problem != nil || !known {
		return problem
	}
	model, problem := resolveRemoteModel(ctx, "", launch.Slot{}, source, "", nil)
	if problem != nil {
		return problem
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	install, problem := client.PrepareRentalPackage(machine, records.RentalInstallSelection{Models: []records.ModelRef{model}})
	if problem != nil {
		return problem
	}
	return watchRentalInstall(ctx, client, either(ctx.Inv.Value("--rental"), machine), install)
}

// quantizer is the package's job that writes `lane`: a model input and one weights output
// named for the lane.
func quantizer(surface *launch.PackageInterface, lane string) *launch.Entrypoint {
	for i := range surface.Jobs {
		job := &surface.Jobs[i]
		if !job.Internal && launch.ConversionSlot(job) != nil && len(job.WeightsOutputs) == 1 &&
			job.WeightsOutputs[0].OutputID == lane {
			return job
		}
	}
	return nil
}

func noQuantizer(lane, format string, args ...any) *exit.Error {
	return exit.Named(exit.NotFound, "model.quantizer_not_found", format, args...).
		WithRemedy("a package quantizes its models with a job that takes the model and declares one weights output named %s, "+
			"such as `%s(ctx, *, source: <Model>, tel) -> ModelArtifact` registered with `weights=(WeightsOutput(%q, ...),)`; "+
			"name the package with --package org/name", lane, lane, lane)
}

// quantizeSource is the one checkpoint a spelling names.
func quantizeSource(ctx *Context, raw string) (hub.Ref, string, *exit.Error) {
	model, release, lane, manifest, problem := hub.ParseModelRef(raw)
	if problem != nil {
		return hub.Ref{}, "", problem
	}
	ref, problem := hub.ParseRef(model)
	if problem != nil {
		return hub.Ref{}, "", problem
	}
	if ref.Org == "local" {
		return ref, "", exit.Usagef("%s is a local alias; quantize a Tensorhub model", raw).
			WithRemedy("upload it first: cozy model upload %s <org/model>", raw)
	}
	if manifest != "" && release == "" {
		return ref, manifest, nil
	}
	hctx, cancel := hub.Context()
	defer cancel()
	_, selected, problem := modelReleaseCardForLane(hctx, client(ctx), ref, release, lane)
	if problem != nil {
		if release == "" {
			problem.WithRemedy("name one checkpoint: %s#sha256:<checkpoint>; `cozy model upload` prints it", ref.String())
		}
		return ref, "", problem
	}
	var found []hub.ModelLaneSummary
	var names []string
	for _, row := range selected.Lanes {
		names = append(names, row.Lane)
		if (lane == "" || row.Lane == lane) && (manifest == "" || row.ManifestID == manifest) {
			found = append(found, row)
		}
	}
	if len(found) != 1 {
		sort.Strings(names)
		return ref, "", exit.Named(exit.NotFound, "model.lane_not_found",
			"%s names no single lane of %s@%s (lanes: %s)", raw, ref.String(), selected.Release,
			orNone(strings.Join(names, ", "))).
			WithRemedy("name the lane to quantize: %s@%s/<lane>", ref.String(), selected.Release)
	}
	return ref, found[0].ManifestID, nil
}

// servingQuantizer is the one installed package, published on this hub, whose `lane`
// quantizer takes this checkpoint. A package states which components its model class reads;
// a checkpoint that supplies them all is one it serves.
func servingQuantizer(ctx *Context, lane string, ref hub.Ref, checkpoint string) (string, *exit.Error) {
	hctx, cancel := hub.Context()
	defer cancel()
	resolved, problem := client(ctx).ResolveModel(hctx, ref.String()+"@"+checkpoint, "")
	if problem != nil {
		return "", problem
	}
	components := resolved.Components
	_, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return "", problem
	}
	defer store.Close()
	installed, problem := store.Installed()
	if problem != nil {
		return "", problem
	}
	var found []string
	for _, install := range installed {
		// A run of an install uses the hub it came from (adoptInstallHub); the model is this one's.
		if install.SourceKind != "tensorhub" || install.Hub != "" && install.Hub != strings.TrimRight(ctx.Cfg.HubURL, "/") ||
			len(found) > 0 && found[len(found)-1] == install.Package {
			continue
		}
		surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir))
		if problem != nil {
			continue
		}
		if job := quantizer(surface, lane); job != nil && len(launch.MissingComponents(job.Models[0], components)) == 0 {
			found = append(found, install.Package)
		}
	}
	switch len(found) {
	case 1:
		return found[0], nil
	case 0:
		return "", noQuantizer(lane, "no installed package quantizes %s (components: %s) to %s",
			ref.String(), orNone(strings.Join(components, ", ")), lane)
	}
	return "", exit.Usagef("%s quantize %s to %s", strings.Join(found, " and "), ref.String(), lane).
		WithRemedy("choose one: --package %s", found[0])
}
