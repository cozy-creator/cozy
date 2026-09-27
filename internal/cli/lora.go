package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func resolveInvocationLoRAs(ctx *Context, target Target, models []orchestrator.ModelRef,
	selected []launch.LoRAOverride) *exit.Error {
	for _, request := range selected {
		var base *orchestrator.ModelRef
		for i := range models {
			if models[i].Slot == request.Slot {
				base = &models[i]
				break
			}
		}
		if base == nil {
			return exit.Usagef("LoRA target slot %s is not bound", request.Slot)
		}
		// Runtime verifies exact target geometry. Reject a component already disproved
		// by a pinned source card before any adapter payload can be transferred.
		if len(base.ComponentBytes) > 0 && base.ComponentBytes[request.Component] == 0 {
			return exit.Usagef("base model has no component %q", request.Component)
		}
		slot := launch.Slot{Path: request.Slot, ComponentUse: map[string][]string{"apply": {"adapter"}}}
		source, problem := resolveRemoteModel(ctx, target.Package, slot, request.Ref, "", nil)
		if problem != nil {
			return problem
		}
		base.Adapters = append(base.Adapters, records.ModelAdapterRef{
			Component: request.Component, Model: source.Model, Release: source.Release, Lane: source.Lane,
			Manifest: source.Manifest, ManifestLength: source.ManifestLength, SourceComponent: "adapter",
			Scale: request.Scale, Bytes: source.Bytes,
		})
	}
	return nil
}
