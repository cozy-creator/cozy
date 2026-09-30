package producttest

import (
	"bytes"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPrivateLoRAChoicesCarryAnExactCallableWithoutBroadcasting(t *testing.T) {
	request := records.Request{Package: "local/probe", Entrypoint: "generate", Models: []records.ModelRef{
		{Package: "local/probe", Slot: "generate.models.model", Adapters: []records.ModelAdapterRef{{Model: "proof/style", Scale: "0.5"}}},
		{Package: "local/probe", Slot: "another.models.model", Choice: true, Adapters: []records.ModelAdapterRef{{Model: "proof/other", Scale: "-0.25"}}},
	}}
	choices, problem := orchestrator.PrivateModelChoices(request)
	fatal(t, problem)
	if len(choices) != 1 || choices[0].Parameter != "generate.models.model" || len(choices[0].Adapters) != 1 || choices[0].Adapters[0].Model != "proof/style" {
		t.Fatal("private preparation selected a bare slot or another captured callable")
	}
	// Release roots supply their entrypoint separately and retain the established
	// bare root parameter. Qualifying a private RPC must not change that identity.
	root, problem := orchestrator.ModelChoices(request, request.OwnModels())
	fatal(t, problem)
	if len(root) != 1 || root[0].Parameter != "model" {
		t.Fatal("private preparation changed the release-root selection identity")
	}
}

// A private preparation resolves open catalog choices once. Its concrete stack
// must satisfy the selection, while the root still names the original base and
// a composed view may legitimately supply an individual component.
func TestPrivateLoRAPreparationReadbackChecksSelectionAndKeepsBaseCustody(t *testing.T) {
	for _, arm := range []string{"resolved", "missing", "reordered", "scale", "component", "pin", "unexpected"} {
		t.Run(arm, func(t *testing.T) {
			base := bytes.Repeat([]byte{1}, 32)
			baseID, err := canonical.Spell(base)
			must(t, err)
			request := records.Request{Package: "local/probe", Entrypoint: "generate", Models: []records.ModelRef{{
				Package: "local/probe", Slot: "generate.models.model", Model: "proof/base", Manifest: baseID,
				Adapters: []records.ModelAdapterRef{
					{Model: "proof/style-a", Release: "1.0.0", Scale: "0"},
					{Component: "dit", Model: "proof/style-b", Scale: "-0.25", SourceComponent: "adapter"},
				},
			}}}
			stack := []*pb.ModelAdapter{
				{Component: "dit", ModelId: "a", SourceComponent: "adapter", Scale: "0"},
				{Component: "dit", ModelId: "b", SourceComponent: "adapter", Scale: "-0.25"},
			}
			switch arm {
			case "missing":
				stack = stack[:1]
			case "reordered":
				stack[0], stack[1] = stack[1], stack[0]
			case "scale":
				stack[0].Scale = "1"
			case "component":
				stack[1].Component = "unselected"
			case "pin":
				request.Models[0].Adapters[0].Manifest = childDigest("e")
			case "unexpected":
				request.Models[0].Adapters = nil
			}
			placement := &pb.Placement{InstallationId: "install-probe", PlacementId: "placement-probe",
				Models: []*pb.Model{
					{Id: "base", Repo: "proof/base", Manifest: &pb.Ref{Digest: base, Length: 100}},
					{Id: "a", Repo: "proof/style-a", Version: "1.0.0", Lane: "bf16", Manifest: &pb.Ref{Digest: bytes.Repeat([]byte{2}, 32), Length: 100}},
					{Id: "b", Repo: "proof/style-b", Version: "2.0.0", Lane: "fp16", Manifest: &pb.Ref{Digest: bytes.Repeat([]byte{3}, 32), Length: 100}},
					{Id: "composed", Repo: "proof/base", Manifest: &pb.Ref{Digest: bytes.Repeat([]byte{4}, 32), Length: 100}},
				},
				Entrypoints: []*pb.Entrypoint{{Name: "generate", EntrypointBindingDigest: bytes.Repeat([]byte{5}, 32), Slots: []*pb.Slot{{
					Slot: "model", ReferenceModelId: "base", Components: []*pb.Component{{Component: "dit", ModelId: "composed"}}, Adapters: stack,
				}}}},
			}
			raw, _, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{placement}})
			must(t, err)
			_, _, problem := orchestrator.ServingPlacement(raw, "install-probe", request)
			if arm == "resolved" {
				fatal(t, problem)
				if request.Models[0].Manifest != baseID || request.Models[0].Adapters[0].Manifest != "" || request.Models[0].Adapters[0].Component != "" {
					t.Fatal("readback mutated the original base or caller's adapter choice")
				}
			} else if problem == nil || problem.ErrName() != "model_adapters_preparation_mismatch" {
				t.Fatalf("%s preparation was accepted: %v", arm, problem)
			}
		})
	}
}
