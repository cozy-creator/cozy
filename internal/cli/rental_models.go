package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/units"
)

// manualRentalWorkload is what a manual rental declares so the Hub sizes its disk: Hub
// models by identity, provider sources not yet on the Hub by their planned ingest bytes,
// and an explicit --disk-gb. Resuming a paid operation replays its exact declaration.
func manualRentalWorkload(ctx *Context, existing *records.RentalOperation) (hub.DeclaredWorkload, *exit.Error) {
	var models, sources []string
	for _, spec := range ctx.Inv.Values["--model"] {
		if strings.Contains(spec, "://") {
			sources = append(sources, spec)
		} else {
			models = append(models, spec)
		}
	}
	if len(ctx.Inv.Values["--source-profile"]) > 0 && len(sources) != 1 {
		return hub.DeclaredWorkload{}, exit.Usagef("--source-profile selects the profiles of one --model provider source")
	}
	disk, problem := manualRentalDisk(ctx, existing)
	if problem != nil {
		return hub.DeclaredWorkload{}, problem
	}
	if existing != nil {
		request, problem := hub.ParseRentalRequestBytes(existing.RequestBody)
		if problem != nil {
			return hub.DeclaredWorkload{}, problem
		}
		serving, problem := manualRentalModels(ctx, models, existing, request.ServingModels)
		return hub.DeclaredWorkload{SourceBytes: request.PlannedSourceBytes, ServingModels: serving,
			ContainerDiskGB: disk}, problem
	}
	workload := hub.DeclaredWorkload{ContainerDiskGB: disk}
	cwd, err := os.Getwd()
	if err != nil {
		return workload, exit.Internalf("cannot resolve working directory: %s", err)
	}
	for _, spec := range sources {
		parsed, problem := modelsource.Parse(spec, cwd)
		if problem != nil {
			return workload, problem
		}
		if parsed.Kind != modelsource.HuggingFace && parsed.Kind != modelsource.Civitai {
			return workload, exit.Usagef("--model %q is not a Hugging Face or Civitai source", spec)
		}
		plan, problem := planNativeIngest(ctx, cwd, parsed, ctx.Inv.Values["--source-profile"])
		if problem != nil {
			return workload, problem
		}
		workload.SourceBytes += plan.source.Bytes
		fmt.Fprintf(ctx.Err, "Sizing rental for %s: %d files, %s (%s)\n", plan.source.Canonical,
			plan.source.Files, units.Bytes(plan.source.Bytes), strings.Join(plan.profiles, ", "))
	}
	serving, problem := manualRentalModels(ctx, models, nil, nil)
	workload.ServingModels = serving
	return workload, problem
}

// manualRentalModels declares identity only. The same Hub resolver used by model
// downloads chooses the exact checkpoint; the existing rental encoder canonicalizes
// the set and Hub alone measures its deduplicated closure and disk headroom.
func manualRentalModels(ctx *Context, specs []string, existing *records.RentalOperation, pinned []hub.ServingModel) ([]hub.ServingModel, *exit.Error) {
	if existing != nil && len(specs) == 0 {
		return pinned, nil // resuming the paid operation reuses its exact body
	}
	selected := make([]hub.ServingModel, 0, len(specs))
	seen := map[hub.ServingModel]bool{}
	for _, spec := range specs {
		model, release, lane, manifest, problem := hub.ParseModelRef(spec)
		if problem != nil {
			return nil, problem
		}
		if manifest == "" && (release == "" || lane == "") {
			return nil, exit.Usagef("--model %q must pin org/model@release/lane or org/model#sha256:<digest>", spec)
		}
		if release == "" && lane != "" {
			return nil, exit.Usagef("--model %q selects a lane without a release", spec)
		}
		ref, problem := hub.ParseRef(model)
		if problem != nil {
			return nil, problem
		}
		if ref.Org == "local" {
			return nil, exit.Usagef("--model %q is local; upload it before sizing a remote rental", spec)
		}
		var exact hub.ServingModel
		if existing != nil {
			for _, held := range pinned {
				if held.Model == model && held.Release == release && held.Lane == lane &&
					(manifest == "" || held.Manifest == manifest) {
					exact = held
					break
				}
			}
			if exact.Model == "" {
				return nil, rentalModelsChanged(existing.Key)
			}
		} else {
			hctx, cancel := hub.Context()
			resolveRef := model + "@" + release
			if release == "" {
				resolveRef = model + "@" + manifest
			}
			resolved, problem := client(ctx).ResolveModel(hctx, resolveRef, lane)
			cancel()
			if problem != nil {
				return nil, problem
			}
			if _, err := canonical.Raw(resolved.ManifestID); err != nil || resolved.Model != model ||
				resolved.Release != release || resolved.Lane != lane || manifest != "" && manifest != resolved.ManifestID {
				return nil, exit.Named(exit.Conflict, "rental.model_resolution_changed",
					"Tensorhub returned a different or incomplete checkpoint for --model %q", spec)
			}
			exact = hub.ServingModel{Model: model, Release: release, Lane: lane, Manifest: resolved.ManifestID}
		}
		if !seen[exact] {
			seen[exact] = true
			selected = append(selected, exact)
			if exact.Release == "" {
				fmt.Fprintf(ctx.Err, "Sizing rental for %s#%s\n", exact.Model, exact.Manifest)
			} else {
				fmt.Fprintf(ctx.Err, "Sizing rental for %s@%s/%s#%s\n", exact.Model, exact.Release, exact.Lane, exact.Manifest)
			}
		}
	}
	if existing != nil && len(selected) != len(pinned) {
		return nil, rentalModelsChanged(existing.Key)
	}
	return selected, nil
}

func rentalModelsChanged(operation string) *exit.Error {
	return exit.Named(exit.Conflict, "rental.idempotency_conflict",
		"rental operation %s already declares a different serving model set", operation).
		WithRemedy("resume without --model to reuse the pinned set, or use a new operation key")
}
