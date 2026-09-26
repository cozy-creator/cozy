package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

func overlayEntrypoint() *launch.Entrypoint {
	return &launch.Entrypoint{
		Name:   "generate",
		Models: []launch.Slot{{Path: "generate.models.base_model", Param: "base_model"}},
	}
}

func TestParsePayloadCanonicalWeightedOverlayList(t *testing.T) {
	payload, keys, problem := launch.ParsePayload(overlayEntrypoint(), []string{
		`model.base_model=org/base@1.0`,
		`model.base_model.lora:=[{"ref":"org/style-a@1","weight":0.5},{"ref":"org/style-b@2","weight":-0.25}]`,
	}, "")
	if problem != nil {
		t.Fatalf("ParsePayload refused: %v", problem)
	}
	if string(payload) != "{}" {
		t.Fatalf("payload = %s, want empty request payload", payload)
	}
	if got := keys.Models["generate.models.base_model"]; got != "org/base@1.0" {
		t.Fatalf("base ref = %q", got)
	}
	got := keys.Overlays["generate.models.base_model"]
	if len(got) != 2 || got[0].Ref != "org/style-a@1" || got[0].Weight != "0.5" || got[1].Weight != "-0.25" {
		t.Fatalf("overlays = %#v", got)
	}
}

func TestParsePayloadInputModelEnvelopePreservesOrder(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "request.json")
	if err := os.WriteFile(path, []byte(`{"prompt":"hello","models":{"base_model":{"ref":"org/base@1","lora":[{"ref":"org/a@1","weight":1},{"ref":"org/b@1","weight":"0.75"}]}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	payload, keys, problem := launch.ParsePayload(overlayEntrypoint(), nil, path)
	if problem != nil {
		t.Fatalf("ParsePayload refused: %v", problem)
	}
	if string(payload) != `{"prompt":"hello"}` {
		t.Fatalf("payload = %s", payload)
	}
	got := keys.Overlays["generate.models.base_model"]
	if len(got) != 2 || got[0].Ref != "org/a@1" || got[0].Weight != "1" || got[1].Weight != "0.75" {
		t.Fatalf("ordered overlays = %#v", got)
	}
}

func TestParsePayloadInlineOverlayShorthandRepeatsInOrder(t *testing.T) {
	_, keys, problem := launch.ParsePayload(overlayEntrypoint(), []string{
		`base_model.lora=org/a@1,weight=0.5`,
		`model.base_model.lora=org/b@1,weight=-0.25`,
	}, "")
	if problem != nil {
		t.Fatalf("ParsePayload refused: %v", problem)
	}
	got := keys.Overlays["generate.models.base_model"]
	if len(got) != 2 || got[0].Ref != "org/a@1" || got[1].Ref != "org/b@1" {
		t.Fatalf("shorthand order = %#v", got)
	}
}

func TestParsePayloadRejectsInvalidOverlayWeight(t *testing.T) {
	_, _, problem := launch.ParsePayload(overlayEntrypoint(), []string{
		`model.base_model.lora=org/style@1,weight=nan`,
	}, "")
	if problem == nil || problem.ErrName() != "usage" {
		t.Fatalf("problem = %#v, want usage refusal", problem)
	}
}

func TestParsePayloadRequiresCanonicalOverlayWeight(t *testing.T) {
	_, _, problem := launch.ParsePayload(overlayEntrypoint(), []string{
		`model.base_model.lora:=[{"ref":"org/style@1"}]`,
	}, "")
	if problem == nil || problem.ErrName() != "usage" {
		t.Fatalf("problem = %#v, want missing-weight usage refusal", problem)
	}
}

func TestParsePayloadRejectsUnknownOverlaySlot(t *testing.T) {
	_, _, problem := launch.ParsePayload(overlayEntrypoint(), []string{
		`model.other.lora=org/style@1,weight=0.5`,
	}, "")
	if problem == nil || problem.ErrName() != "model_slot_unknown" {
		t.Fatalf("problem = %#v, want model_slot_unknown", problem)
	}
}

func TestParsePayloadStartsEachRequestWithPristineOverlaySet(t *testing.T) {
	_, first, problem := launch.ParsePayload(overlayEntrypoint(), []string{
		`model.base_model.lora=org/style@1,weight=0.5`,
	}, "")
	if problem != nil {
		t.Fatalf("first parse refused: %v", problem)
	}
	if len(first.Overlays["generate.models.base_model"]) != 1 {
		t.Fatalf("first overlays = %#v", first.Overlays)
	}
	_, second, problem := launch.ParsePayload(overlayEntrypoint(), nil, "")
	if problem != nil {
		t.Fatalf("second parse refused: %v", problem)
	}
	if len(second.Overlays) != 0 {
		t.Fatalf("overlay state leaked between requests: %#v", second.Overlays)
	}
}
