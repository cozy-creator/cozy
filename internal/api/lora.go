package api

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

func validateModelAdapters(models []orchestrator.ModelRef) *exit.Error {
	for _, model := range models {
		if len(model.Adapters) > 0 && !model.Downloadable() && model.BindingPath == "" {
			return exit.Named(exit.Structural, "native_model_adapters_unsupported", "this base has no downloadable or native model binding; upload the base checkpoint first")
		}
		for _, adapter := range model.Adapters {
			if adapter.Component == "" || adapter.SourceComponent == "" || len(adapter.Component) > 128 || len(adapter.SourceComponent) > 128 || strings.ContainsAny(adapter.Component+adapter.SourceComponent, "/\\\u0000\r\n\t ") {
				return exit.Usagef("model adapters require bounded explicit target and source components")
			}
			if adapter.Bytes < 0 || adapter.ManifestLength < 0 {
				return exit.Usagef("adapter byte observations cannot be negative")
			}
			if _, problem := hub.ParseRef(adapter.Model); problem != nil {
				return problem
			}
			if _, err := canonical.Raw(adapter.Manifest); err != nil {
				return exit.Usagef("model adapters require exact manifest digests")
			}
			scale, problem := launch.CanonicalLoRAScale(adapter.Scale)
			if problem != nil {
				return problem
			}
			if scale != adapter.Scale {
				return exit.Usagef("model adapter scale must use canonical spelling %s", scale)
			}
		}
	}
	return nil
}
