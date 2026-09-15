package cli

import (
	"fmt"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

func resolveInvocationLoRAs(ctx *Context, target Target, models []orchestrator.ModelRef,
	selected []launch.LoRAOverride, remote bool) *exit.Error {
	for index, request := range selected {
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
		var source orchestrator.ModelRef
		var problem *exit.Error
		if remote {
			source, problem = resolveRemoteModel(ctx, target.Package, slot, request.Ref, "", nil)
		} else {
			source, problem = acquireLocalLoRA(ctx, target, slot, request.Ref, index)
		}
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

func acquireLocalLoRA(ctx *Context, target Target, slot launch.Slot, ref string, index int) (orchestrator.ModelRef, *exit.Error) {
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	work, problem := scratch.Temp(layout.Tmp, "invoke-lora-")
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	defer work.Release()
	hctx, cancel := hub.LongContext()
	defer cancel()
	model, problem := acquirePublishedModel(hctx, ctx, tool, client(ctx), ref, "", target.Package, slot, filepath.Join(work.Path, fmt.Sprintf("%03d", index)))
	if problem != nil {
		return orchestrator.ModelRef{}, problem
	}
	return orchestrator.ModelRef{Model: model.Model, Release: model.Release, Lane: model.Lane, Manifest: model.Manifest, ManifestLength: model.ManifestLength}, nil
}
