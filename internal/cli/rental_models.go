package cli

import (
	"fmt"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// manualRentalModels declares identity only. The same Hub resolver used by model
// downloads chooses the exact checkpoint; the existing rental encoder canonicalizes
// the set and Hub alone measures its deduplicated closure and disk headroom.
func manualRentalModels(ctx *Context, existing *records.RentalOperation) ([]hub.ServingModel, *exit.Error) {
	specs := ctx.Inv.Values["--model"]
	var pinned []hub.ServingModel
	if existing != nil {
		request, problem := hub.ParseRentalRequestBytes(existing.RequestBody)
		if problem != nil {
			return nil, problem
		}
		pinned = request.ServingModels
		if len(specs) == 0 {
			return pinned, nil // resuming the paid operation reuses its exact body
		}
	}
	selected := make([]hub.ServingModel, 0, len(specs))
	seen := map[hub.ServingModel]bool{}
	for _, spec := range specs {
		model, release, lane, manifest, problem := parseModelRef(spec)
		if problem != nil {
			return nil, problem
		}
		if release == "" || lane == "" {
			return nil, exit.Usagef("--model %q must pin org/model@release/lane", spec)
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
			resolved, problem := client(ctx).ResolveModel(hctx, model+"@"+release, lane)
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
			fmt.Fprintf(ctx.Err, "Sizing rental for %s@%s/%s#%s\n", exact.Model, exact.Release, exact.Lane, exact.Manifest)
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
