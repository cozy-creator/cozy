package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestDownloadModelRefsCarriesExactCheckpointAndAdapters(t *testing.T) {
	models := orchestrator.DownloadModelRefs([]records.ModelRef{{
		Package: "alice/video", Slot: "generate.models.model", BindingPath: "generate.models.model",
		Model: "alice/h3", Release: "1.0.0", Lane: "fp8", Manifest: "sha256:" + repeatHex("ab", 32),
		Adapters: []records.ModelAdapterRef{{Component: "transformer", Model: "alice/style", Release: "1.0.0", Lane: "default", Manifest: "sha256:" + repeatHex("cd", 32), SourceComponent: "adapter", Scale: "0.5"}},
	}})
	if len(models) != 1 {
		t.Fatalf("got %d model refs, want one", len(models))
	}
	got := models[0]
	if got.Package != "alice/video" || got.Slot != "generate.models.model" || got.Model != "alice/h3" ||
		got.Release != "1.0.0" || got.Lane != "fp8" || got.Manifest != "sha256:"+repeatHex("ab", 32) {
		t.Fatalf("projection lost model identity: %+v", got)
	}
	if len(got.Adapters) != 1 || got.Adapters[0].Manifest != "sha256:"+repeatHex("cd", 32) || got.Adapters[0].Scale != "0.5" {
		t.Fatalf("projection lost adapter identity: %+v", got.Adapters)
	}
}

func repeatHex(pair string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += pair
	}
	return out
}
