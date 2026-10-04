package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

// TestLoRAModeRidesEveryRequest is the owner's LoRA ruling (2026-10-04) at the CLI: a
// request's `lora_mode` reaches the Runtime whatever the callable declares, and only
// "bake" or "per_step" pass; a callable declaring its own field of that name keeps it.
func TestLoRAModeRidesEveryRequest(t *testing.T) {
	raw := []byte(`{"application":"proof:app","entrypoints":[{"name":"run","request":{"fields":[{"name":"prompt","type":"str"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	ep, problem := iface.Function("run")
	fatal(t, problem)
	payload := []byte(`{"prompt":"a","lora_mode":"per_step","style":"noir"}`)
	cleaned, ignored, problem := launch.ValidatePayload("proof", ep, payload)
	fatal(t, problem)
	if len(ignored) != 1 || ignored[0] != "style" || string(cleaned) != `{"lora_mode":"per_step","prompt":"a"}` {
		t.Fatalf("lora_mode did not ride the request: %v %s", ignored, cleaned)
	}
	if payloadProblem("proof", ep, []byte(`{"prompt":"a","lora_mode":"fast"}`)) == nil {
		t.Fatal(`lora_mode "fast" passed`)
	}
}
