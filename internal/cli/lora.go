package cli

import (
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// modelChoiceSlot keeps a child's callable identity separate from the root's
// parameter names. The machine validates it against the captured graph.
func modelChoiceSlot(target Target, selector string) orchestrator.ModelRef {
	pkg, path, qualified := launch.CapturedModelSlot(selector)
	if !qualified {
		path = selector
	}
	if pkg == "" {
		pkg = target.Package
	}
	out := orchestrator.ModelRef{Choice: true, Package: pkg, Slot: path, BindingPath: path}
	entrypoint, _, qualified := strings.Cut(path, ".models.")
	if qualified && (pkg != target.Package || entrypoint != target.Function) {
		out.Callable = pkg + "/" + entrypoint
	}
	return out
}

// applyModelAdapters preserves list order and leaves selection/compatibility to
// the same machine that resolves the base checkpoint. No handler payload changes.
func applyModelAdapters(ctx *Context, target Target, ep *launch.Entrypoint,
	models []orchestrator.ModelRef, overlays map[string][]launch.ModelOverlay,
) ([]orchestrator.ModelRef, *exit.Error) {
	flags, problem := launch.ParseLoRAs(ep, ctx.Inv.Values["--lora"])
	if problem != nil {
		return nil, problem
	}
	selected := make(map[string][]launch.ModelOverlay, len(overlays))
	for slot, stack := range overlays {
		if len(stack) == 0 {
			continue
		}
		choice := modelChoiceSlot(target, slot)
		key := choice.Package + "/" + choice.BindingSlot()
		if _, duplicate := selected[key]; duplicate {
			return nil, exit.Usagef("model slot %s has more than one overlay-list spelling", key)
		}
		selected[key] = append([]launch.ModelOverlay(nil), stack...)
	}
	for _, flag := range flags {
		choice := modelChoiceSlot(target, flag.Slot)
		key := choice.Package + "/" + choice.BindingSlot()
		selected[key] = append(selected[key], launch.ModelOverlay{
			Ref: flag.Ref, Weight: flag.Scale, Component: flag.Component,
		})
	}
	paths := make([]string, 0, len(selected))
	for path := range selected {
		paths = append(paths, path)
	}
	slices.Sort(paths)
	for _, path := range paths {
		choice := modelChoiceSlot(target, path)
		index := slices.IndexFunc(models, func(model orchestrator.ModelRef) bool {
			return model.Package == choice.Package && model.BindingSlot() == choice.BindingSlot()
		})
		if index < 0 {
			models = append(models, choice)
			index = len(models) - 1
		}
		for _, overlay := range selected[path] {
			if strings.ContainsAny(overlay.Component+overlay.SourceComponent, "=,: \t\n\r") {
				return nil, exit.Usagef("invalid adapter component %q", overlay.Component)
			}
			adapter := records.ModelAdapterRef{Component: overlay.Component,
				SourceComponent: overlay.SourceComponent, Scale: overlay.Weight, Profiles: overlay.Profiles}
			if adapter.SourceComponent == "" {
				adapter.SourceComponent = "adapter"
			}
			if strings.Contains(overlay.Ref, "://") {
				adapter.Source, problem = pinnedProviderSource(ctx, overlay.Ref)
			} else {
				adapter.Model, adapter.Release, adapter.Lane, adapter.Manifest, problem = hub.ParseModelRef(overlay.Ref)
			}
			if problem != nil {
				return nil, problem
			}
			models[index].Adapters = append(models[index].Adapters, adapter)
		}
	}
	return models, nil
}

func capturedModelChoices(models []orchestrator.ModelRef) []orchestrator.ModelRef {
	var out []orchestrator.ModelRef
	for _, model := range models {
		if model.Choice && model.Callable != "" {
			out = append(out, model)
		}
	}
	return out
}
