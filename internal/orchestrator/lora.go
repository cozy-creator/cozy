package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func downloadAdapters(adapters []records.ModelAdapterRef) []*pb.DownloadAdapterRef {
	out := make([]*pb.DownloadAdapterRef, 0, len(adapters))
	for _, a := range adapters {
		out = append(out, &pb.DownloadAdapterRef{Component: a.Component, Model: a.Model, Release: a.Release, Lane: a.Lane, Manifest: a.Manifest, SourceComponent: a.SourceComponent, Scale: a.Scale})
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
