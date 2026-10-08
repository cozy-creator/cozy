package producttest

import (
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/protobuf/proto"
)

// These are the actual protobuf choices shared by public roots, captured serving
// and jobs. An advisory sizing ladder must never become a lane-less override.
func TestModelChoiceWirePreservesWorkerDefaultsAndExplicitSelections(t *testing.T) {
	request := records.Request{Package: "local/profile", Entrypoint: "render"}
	base := records.ModelRef{Package: request.Package, Slot: "render.models.base", Model: "owner/model", Release: "1.0.0",
		Ladder: []records.ModelRung{{GPU: "5090", GPUs: 1, Lane: "fp8"}, {GPU: "H100", GPUs: 2, Lane: "bf16"}}}
	for _, mode := range []string{"default", "adapter-only", "explicit-lane", "explicit-unversioned", "pinned-manifest", "provider"} {
		t.Run(mode, func(t *testing.T) {
			model := base
			switch mode {
			case "adapter-only":
				model.Adapters = []records.ModelAdapterRef{{Component: "dit", Model: "owner/adapter", Lane: "lora", Scale: "0.75"}}
			case "explicit-lane":
				model.Choice, model.Lane = true, "bf16"
			case "explicit-unversioned":
				model.Choice, model.Release = true, ""
			case "pinned-manifest":
				model.Manifest, model.ManifestLength = "sha256:"+strings.Repeat("a", 64), 123
			case "provider":
				model.Choice, model.Source, model.Profiles = true, "hf://owner/checkpoint@commit", []string{"video"}
			}
			before := model
			choices, problem := orchestrator.ModelChoices(request, []records.ModelRef{model})
			fatal(t, problem)
			if !reflect.DeepEqual(before, model) {
				t.Fatal("wire conversion changed retained placement facts")
			}
			wire, err := proto.Marshal(&v1.RunRequest{Id: "same-root", Spec: &v1.RunSpec{Source: &v1.RunSpec_Release{Release: &v1.Release{Package: "proof/root"}}, Models: choices}})
			must(t, err)
			var decoded v1.RunRequest
			must(t, proto.Unmarshal(wire, &decoded))
			if decoded.Spec.GetRelease().Release != "" {
				t.Fatal("wire conversion chose a package version")
			}
			if mode == "default" {
				if len(decoded.Spec.Models) != 0 {
					t.Fatalf("advisory root suppressed authored lane selection: %v", decoded.Spec.Models)
				}
				return
			}
			if len(decoded.Spec.Models) != 1 {
				t.Fatalf("lost explicit selection: %v", decoded.Spec.Models)
			}
			got := decoded.Spec.Models[0]
			if got.Parameter != "base" {
				t.Fatalf("wrong slot: %s", got.Parameter)
			}
			if mode == "adapter-only" {
				if got.Repository != "" || got.Release != "" || got.Lane != "" || got.Manifest != "" || len(got.Adapters) != 1 || got.Adapters[0].Model != "owner/adapter" || got.Adapters[0].Scale != "0.75" {
					t.Fatalf("adapter changed default base selection: %v", got)
				}
			} else if got.Repository != model.Model || got.Release != model.Release || got.Lane != model.Lane || got.Manifest != model.Manifest || got.Source != model.Source || !reflect.DeepEqual(got.Profiles, model.Profiles) {
				t.Fatalf("explicit choice changed: %v vs %+v", got, model)
			}
		})
	}
}
