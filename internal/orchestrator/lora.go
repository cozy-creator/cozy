package orchestrator

import (
	"context"
	"google.golang.org/grpc"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
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

type adapterProtocolPeer interface {
	ProtocolInfo(context.Context, *pb.ProtocolInfoRequest, ...grpc.CallOption) (*pb.ProtocolInfoResult, error)
}

func requireAdapterDownloadPeer(s *session, data []byte) *exit.Error {
	if len(data) == 0 {
		return nil
	}
	doc, err := canonical.Read(data, &pb.DownloadDelegation{})
	if err != nil {
		return exit.Named(exit.Structural, "model_adapter_selection_invalid", "download selection is not canonical: %s", err)
	}
	needed := false
	for _, model := range doc.List("models") {
		if len(model.List("adapters")) > 0 {
			needed = true
			break
		}
	}
	return requireAdapterPeer(s, needed)
}

func requireAdapterPeer(s *session, needed bool) *exit.Error {
	if !needed {
		return nil
	}
	if s == nil {
		return exit.Unavailablef("LoRA preparation awaits its worker")
	}
	var peer adapterProtocolPeer
	if s.host != nil {
		peer = s.host
	} else if s.preparation != nil {
		peer = s.preparation
	}
	if peer == nil {
		return exit.Unavailablef("LoRA preparation awaits its preparation peer")
	}
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	defer cancel()
	info, err := peer.ProtocolInfo(ctx, &pb.ProtocolInfoRequest{})
	if err != nil {
		return exit.Unavailablef("LoRA worker capability probe is unavailable")
	}
	if info == nil || info.WireMinor < pb.ModelAdapterWireMinor {
		return exit.Named(exit.Structural, "model_adapters_protocol_unsupported", "model adapters require Host and Runtime protocol minor %d or newer", pb.ModelAdapterWireMinor)
	}
	return nil
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
