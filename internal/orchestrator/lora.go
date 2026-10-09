package orchestrator

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// ModelChoices is shared by release roots, frozen captures and private serving
// preparation. Explicit child targets keep the complete package/slot identity.
func ModelChoices(request records.Request, models []ModelRef) ([]*v1.ModelChoice, *exit.Error) {
	var out []*v1.ModelChoice
	for _, model := range models {
		path := model.BindingSlot()
		entrypoint, parameter, attributed := strings.Cut(path, ".models.")
		if !attributed {
			parameter = model.Slot
		}
		child := model.Package != "" && model.Package != request.Package || attributed && entrypoint != request.Entrypoint
		if child {
			parameter = path
			if model.Package != "" && model.Package != request.Package {
				parameter = model.Package + "/" + path
			}
		}
		choice := &v1.ModelChoice{Parameter: parameter, Repository: model.Model,
			Release: model.Release, Lane: model.Lane, Source: model.Source,
			Profiles: model.Profiles, Adapters: downloadAdapters(model.Adapters)}
		// Names for the machine (th-245): it resolves the release and lane at the Hub. A hash
		// travels only for a checkpoint by hash: the owner's at the Hub, or one the machine holds.
		switch {
		case model.Source != "":
		case model.Release == "" && model.Manifest != "":
			digest, err := canonical.Raw(model.Manifest)
			if err != nil {
				return nil, exit.Named(exit.Validation, "model.manifest_invalid", "%s names no exact manifest", path)
			}
			choice.Manifest, _ = canonical.Spell(digest)
			choice.ManifestLength = uint64(max(model.ManifestLength, 0))
		case model.Model == "":
			return nil, exit.Named(exit.Internal, "model.choice_unresolved", "%s reached its machine with no model", path)
		case model.Lane == "":
			for _, rung := range model.Ladder {
				choice.Rungs = append(choice.Rungs, &v1.ModelRung{Gpu: rung.GPU, Gpus: uint32(max(rung.GPUs, 0)), Lane: rung.Lane})
			}
		}
		out = append(out, choice)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Parameter < out[j].Parameter })
	for i := 1; i < len(out); i++ {
		if out[i-1].Parameter == out[i].Parameter {
			return nil, exit.Usagef("model slot %s was selected more than once", out[i].Parameter)
		}
	}
	return out, nil
}

func downloadAdapters(adapters []records.ModelAdapterRef) []*v1.Adapter {
	out := make([]*v1.Adapter, 0, len(adapters))
	for _, a := range adapters {
		adapter := &v1.Adapter{Component: a.Component, Model: a.Model, Release: a.Release, Lane: a.Lane, Scale: a.Scale, Source: a.Source, Profiles: a.Profiles}
		if a.Release == "" {
			adapter.Manifest = a.Manifest
		}
		out = append(out, adapter)
	}
	return out
}

func placementAdapters(slot canonical.Doc, models map[string]canonical.Doc) []records.ModelAdapterRef {
	out := make([]records.ModelAdapterRef, 0, len(slot.List("adapters")))
	for _, a := range slot.List("adapters") {
		model := models[a.Str("model_id")]
		out = append(out, records.ModelAdapterRef{
			Component: a.Str("component"), Model: model.Str("repo"), Release: model.Str("version"), Lane: model.Str("lane"),
			Manifest: model.Sub("manifest").Str("digest"), ManifestLength: model.Sub("manifest").Int("length"),
			SourceComponent: a.Str("source_component"), Scale: a.Str("scale"),
		})
	}
	return out
}
