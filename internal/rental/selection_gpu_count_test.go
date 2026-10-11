package rental

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func gpuMarket() []hub.RentalSKU {
	var skus []hub.RentalSKU
	for _, count := range []int{1, 2, 4} {
		skus = append(skus, hub.RentalSKU{Name: "h100", AcceleratorModel: "NVIDIA H100",
			AcceleratorCount: count, VRAMGB: 80})
	}
	return skus
}

func TestExactRunGPUCountControlsPurchaseIndependentlyOfWeights(t *testing.T) {
	for _, model := range []records.ModelRef{
		{Model: "a/weights", Manifest: "first", GPUs: 1},
		{Model: "b/weights", Manifest: "second", GPUs: 4},
		{Model: "c/weights", Ladder: []records.ModelRung{{GPU: "H100", GPUs: 1, Manifest: "third"}}},
	} {
		rows := Purchases(gpuMarket(), []records.ModelRef{model}, true, false,
			Constraints{GPUs: 2, Degrees: []int{2, 4}})
		for _, row := range rows {
			if (row.Verdict == "") != (row.GPUs >= 2) {
				t.Fatalf("weights %s, machine %d GPUs: %q", model.Model, row.GPUs, row.Verdict)
			}
		}
	}
	rows := Purchases(gpuMarket(), nil, true, false, Constraints{GPUs: 2})
	if rows[1].Verdict == "" {
		t.Fatal("an unpartitionable package was admitted at exactly two GPUs")
	}
}

func TestExistingRentalCanLendExactSubset(t *testing.T) {
	for _, count := range []int{1, 2, 4, 6} {
		candidate := orchestrator.PlacementCandidate{GPUs: count}
		row := records.Rental{AcceleratorModel: "NVIDIA H100", AcceleratorCount: count, State: "ready"}
		Standing(&candidate, []records.ModelRef{{Manifest: "weights", GPUs: 4}}, row,
			80, true, true, false, Constraints{GPUs: 2, Degrees: []int{2, 4}}, Disk{})
		if count == 1 && candidate.Verdict == "" {
			t.Fatal("one GPU cannot supply a two-GPU subset")
		}
		if count >= 2 && candidate.Verdict != "" && candidate.Verdict != orchestrator.VerdictAttaching {
			t.Fatalf("existing %d-GPU machine refused the two-GPU subset: %s", count, candidate.Verdict)
		}
		if candidate.GPUs != count {
			t.Fatal("selection changed the rental's physical capacity")
		}
		if count >= 2 && candidate.RunGPUs != 2 {
			t.Fatalf("execution estimate used physical capacity: %+v", candidate)
		}
	}
}

func TestCompositionPurchaseUsesCodeCapabilityNotWeightRungCount(t *testing.T) {
	models := []records.ModelRef{
		{Callable: "p/image", Manifest: "image", GPUs: 8, SupportedGPUs: []int{1}},
		{Callable: "p/film", Manifest: "base", GPUs: 1, SupportedGPUs: []int{1, 2, 4}},
		{Callable: "p/film", Manifest: "adapter", GPUs: 1, SupportedGPUs: []int{1, 2, 4}},
	}
	rows := Purchases(gpuMarket(), models, true, true, Constraints{})
	for _, row := range rows {
		if row.Verdict != "" {
			t.Fatalf("code-supported %d-GPU composition refused: %s", row.GPUs, row.Verdict)
		}
	}
	models[2].SupportedGPUs = []int{1, 2}
	if got := PurchaseWidthUnusable(4, models, true, Constraints{}); got == "" {
		t.Fatal("weight metadata bypassed a package slot's missing four-GPU support")
	}
	models[1].SupportedGPUs, models[2].SupportedGPUs = nil, nil
	if got := PurchaseWidthUnusable(2, models, true, Constraints{}); got == "" {
		t.Fatal("unknown package capability was inferred from weight metadata")
	}
	// An explicit CPU orchestrator reserves the group; unsupported actual GPU children
	// refuse on the worker, while unused captured functions do not reject the request.
	if got := PurchaseWidthUnusable(2, models, true, Constraints{GPUs: 2}); got != "" {
		t.Fatalf("CPU job cannot reserve an explicit group: %s", got)
	}
}

func TestAutomaticRunUsesLargestCodeSupportedDegreeOnExistingMachine(t *testing.T) {
	candidate := orchestrator.PlacementCandidate{GPUs: 6}
	row := records.Rental{AcceleratorModel: "NVIDIA H100", AcceleratorCount: 6, State: "ready"}
	Standing(&candidate, []records.ModelRef{{Manifest: "weights", GPUs: 2}}, row,
		80, true, true, false, Constraints{Degrees: []int{2, 4, 8}}, Disk{})
	if candidate.RunGPUs != 4 || candidate.GPUs != 6 {
		t.Fatalf("want four supported GPUs on six-card machine, got %+v", candidate)
	}
}
