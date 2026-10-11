package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestCountedPurchasePreferencePrecedesMeasuredFallback(t *testing.T) {
	models := []records.ModelRef{{Callable: "proof/h3/run", SupportedGPUs: []int{1, 2, 4}, Model: "proof/h3", Ladder: []records.ModelRung{
		{GPU: "H100", GPUs: 2, Lane: "fp8-pruned", Manifest: "two"},
		{GPU: "H100", GPUs: 4, Lane: "fp8-pruned", Manifest: "four"},
	}}}
	skus := []hub.RentalSKU{
		{Name: "h100", AcceleratorModel: "NVIDIA H100", AcceleratorCount: 2, VRAMGB: 80},
		{Name: "h100", AcceleratorModel: "NVIDIA H100", AcceleratorCount: 4, VRAMGB: 80},
	}
	for _, tier := range []string{"fast", "cheap", "balanced"} {
		candidates := rental.Purchases(skus, models, true, true, rental.Constraints{})
		candidates[1].Measured = true
		candidates[1].TimeS = 1
		candidates[1].CostUSDMicros = 1
		if chosen := rental.Place(tier, candidates); chosen != 0 {
			t.Fatalf("%s measured4-GPU fallback displaced preferred2: %d %+v", tier, chosen, candidates)
		}
		candidates[0].Verdict = orchestrator.VerdictNoStock
		if chosen := rental.Place(tier, candidates); chosen != 1 {
			t.Fatalf("%s stock refusal did not advance to4 GPUs: %d", tier, chosen)
		}
		candidates[0].Verdict = ""
		rental.Conclude(candidates, 0)
		if candidates[1].Verdict != "later_preference" {
			t.Fatalf("fallback explanation invented a measurement: %+v", candidates[1])
		}
	}
}
