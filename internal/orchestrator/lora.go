package orchestrator

import (
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ModelChoices is shared by release roots, frozen captures and private serving
// preparation. Explicit child targets keep the complete package/slot identity.
func ModelChoices(request records.Request, models []ModelRef) ([]*pb.ModelChoice, *exit.Error) {
	var out []*pb.ModelChoice
	for _, model := range models {
		path := model.BindingSlot()
		entrypoint, parameter, attributed := strings.Cut(path, ".models.")
		if !attributed {
			parameter = model.Slot
		}
		child := model.Package != "" && model.Package != request.Package || attributed && entrypoint != request.Entrypoint
		if child {
			if !model.Choice && len(model.Adapters) == 0 {
				continue // advisory child sizing is not an explicit override
			}
			parameter = path
			if model.Package != "" && model.Package != request.Package {
				parameter = model.Package + "/" + path
			}
		}
		choice := &pb.ModelChoice{Parameter: parameter, Repository: model.Model,
			Release: model.Release, Lane: model.Lane, Source: model.Source,
			Profiles: model.Profiles, Adapters: downloadAdapters(model.Adapters)}
		if model.Manifest != "" {
			digest, err := canonical.Raw(model.Manifest)
			if err != nil {
				return nil, exit.Named(exit.Validation, "model.manifest_invalid", "%s names no exact manifest", path)
			}
			choice.Manifest = &pb.Ref{Digest: digest, Length: uint64(max(model.ManifestLength, 0))}
		}
		out = append(out, choice)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Parameter < out[j].Parameter })
	return out, nil
}

func downloadAdapters(adapters []records.ModelAdapterRef) []*pb.DownloadAdapterRef {
	out := make([]*pb.DownloadAdapterRef, 0, len(adapters))
	for _, a := range adapters {
		out = append(out, &pb.DownloadAdapterRef{Component: a.Component, Model: a.Model, Release: a.Release, Lane: a.Lane, Manifest: a.Manifest, SourceComponent: a.SourceComponent, Scale: a.Scale, Source: a.Source, Profiles: a.Profiles})
	}
	return out
}

func hasModelAdapters(models []ModelRef) bool {
	for _, model := range models {
		if len(model.Adapters) > 0 {
			return true
		}
	}
	return false
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
