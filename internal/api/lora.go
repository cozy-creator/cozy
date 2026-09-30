package api

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// validateModelAdapters validates selection syntax; the executing Runtime owns
// compatibility, immutable resolution and the operation's capability check.
func validateModelAdapters(models []orchestrator.ModelRef) *exit.Error {
	for _, model := range models {
		for i := range model.Adapters {
			adapter := &model.Adapters[i]
			if adapter.Scale == "" {
				adapter.Scale = "1"
			}
			canonical, problem := launch.CanonicalLoRAScale(adapter.Scale)
			if problem != nil {
				return problem
			}
			adapter.Scale = canonical
			if adapter.SourceComponent == "" {
				adapter.SourceComponent = "adapter"
			}
			if strings.ContainsAny(adapter.Component+adapter.SourceComponent, "=,: \t\n\r") {
				return exit.Usagef("adapter component names must be single names")
			}
			if adapter.Source != "" {
				if adapter.Model != "" || adapter.Release != "" || adapter.Lane != "" || adapter.Manifest != "" {
					return exit.Usagef("an adapter names either a provider source or a catalog checkpoint")
				}
			} else if adapter.Model == "" || len(adapter.Profiles) > 0 {
				return exit.Usagef("a catalog adapter needs a model reference; profiles apply only to provider sources")
			}
		}
	}
	return nil
}

// Model selections have one identity shape for serving roots and job captures.
// Observational sizing is excluded; adapter order, sources and scales are retained.
func modelSelectionIdentity(model orchestrator.ModelRef) canonical.Value {
	row := map[string]canonical.Value{
		"package": model.Package, "slot": model.Slot, "model": model.Model,
		"release": model.Release, "lane": model.Lane, "manifest": model.Manifest,
		"manifest_length": model.ManifestLength,
	}
	addSourceIdentity(row, model.Source, model.Profiles)
	if len(model.Adapters) > 0 {
		adapters := make([]canonical.Value, 0, len(model.Adapters))
		for _, adapter := range model.Adapters {
			entry := map[string]canonical.Value{
				"component": adapter.Component, "model": adapter.Model, "release": adapter.Release,
				"lane": adapter.Lane, "manifest": adapter.Manifest,
				"source_component": adapter.SourceComponent, "scale": adapter.Scale,
			}
			addSourceIdentity(entry, adapter.Source, adapter.Profiles)
			adapters = append(adapters, entry)
		}
		row["adapters"] = adapters
	}
	return row
}

func addSourceIdentity(row map[string]canonical.Value, source string, profiles []string) {
	if source == "" {
		return
	}
	values := make([]canonical.Value, 0, len(profiles))
	for _, profile := range profiles {
		values = append(values, profile)
	}
	row["source"], row["profiles"] = source, values
}
