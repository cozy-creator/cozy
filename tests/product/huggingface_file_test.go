package producttest

import (
	"context"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/secret"
)

func TestHuggingFaceFileURLsPreserveSelectionAcrossCanonicalRoundTrip(t *testing.T) {
	commit := strings.Repeat("a", 40)
	for _, raw := range []string{
		"https://huggingface.co/owner/model/blob/main/adapters/my%20lora.safetensors?download=true",
		"https://huggingface.co/owner/model/resolve/" + commit + "/adapters/my%20lora.safetensors",
		"hf://owner/model@" + commit + "/adapters/my%20lora.safetensors",
	} {
		source, problem := modelsource.Parse(raw, t.TempDir())
		if problem != nil {
			t.Fatal(problem)
		}
		again, problem := modelsource.Parse(source.Canonical, t.TempDir())
		if problem != nil || again != source {
			t.Fatalf("lost selection: %#v -> %#v: %v", source, again, problem)
		}
		if source.Member != "adapters/my lora.safetensors" {
			t.Fatal(source)
		}
	}
	for _, suffix := range []string{"tool.exe", "../bad.safetensors", "%2e%2e/bad.safetensors", "foo%5cbar.safetensors", "foo%00.safetensors", "", "model.safetensors/"} {
		_, problem := modelsource.Parse("https://huggingface.co/owner/model/blob/main/"+suffix, t.TempDir())
		if problem == nil {
			t.Fatalf("accepted unsafe/non-carrier %q", suffix)
		}
	}
}

// Public-origin proof through the production provider client.
func TestHuggingFaceExplicitFilesResolveThroughPublicProvider(t *testing.T) {
	fullRun(t, "resolves pinned files on live HuggingFace")
	resolver, problem := modelsource.NewResolver(modelsource.HuggingFace, secret.Value{})
	fatal(t, problem)
	for _, row := range []struct {
		repo, commit, file string
		size               int64
	}{
		{"Jojocodex/minimax-h3-spatial-physics-lora", "476b24df28b9b7cc5481b750469698f5cc4b0558", "wushu_spatial_physics_clean_3000_pruned.safetensors", 155109672},
		{"Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora", "9598ef0b02f4590e202e956b4e6328dc71ae2bf5", "wushu_action_v7_fl2va_aitoolkit_adaln_full-int8convrot_bf16te_2000step.safetensors", 596451088},
	} {
		source, problem := modelsource.Parse("https://huggingface.co/"+row.repo+"/blob/"+row.commit+"/"+row.file, "")
		fatal(t, problem)
		plan, problem := resolver.Resolve(context.Background(), source)
		fatal(t, problem)
		if len(plan.Files) != 1 || plan.Files[0].Member != row.file || !plan.Files[0].Carrier || plan.Bytes != row.size {
			t.Fatalf("selected sibling files: %+v", plan)
		}
		if plan.Canonical != "hf://"+row.repo+"@"+row.commit+"/"+row.file {
			t.Fatal(plan.Canonical)
		}
	}
}

// A safetensors member stored outside LFS carries no provider SHA-256. Its identity is
// measured from the bytes served at the pinned commit instead of refusing the source.
func TestHuggingFaceNonLFSMemberIsIdentifiedFromItsBytes(t *testing.T) {
	fullRun(t, "reads a pinned member on live HuggingFace")
	resolver, problem := modelsource.NewResolver(modelsource.HuggingFace, secret.Value{})
	fatal(t, problem)
	source, problem := modelsource.Parse("hf://hf-internal-testing/tiny-random-bert@f171d7baecaf37b5da5a3616d8833b9969753535/model.safetensors", "")
	fatal(t, problem)
	plan, problem := resolver.Resolve(context.Background(), source)
	fatal(t, problem)
	if len(plan.Files) != 1 || plan.Files[0].SHA256 != "965f02b6a7e5520fc12f710e4e3b6132f697f1c8f648819553c5ade86752d2de" ||
		plan.Files[0].Length != 520212 {
		t.Fatalf("non-LFS member was not identified from its bytes: %+v", plan.Files)
	}
	staged, problem := resolver.Stage(context.Background(), plan, t.TempDir(), false, nil)
	fatal(t, problem)
	if len(staged) != 1 {
		t.Fatalf("non-LFS member was not staged: %+v", staged)
	}
}
