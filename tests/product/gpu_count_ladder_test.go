package producttest

import (
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"testing"
)

func TestCountedLaddersBuyMaximumChildGroupAndReuseExactReplica(t *testing.T) {
	models := []records.ModelRef{
		{Model: "proof/qwen", Ladder: []records.ModelRung{{GPU: "H100", GPUs: 1, Lane: "original", Manifest: "q"}}},
		{Model: "proof/h3", Ladder: []records.ModelRung{{GPU: "H100", GPUs: 2, Lane: "fp8", Manifest: "h"}, {GPU: "H100", GPUs: 4, Lane: "fp8", Manifest: "h"}}},
	}
	skus := []hub.RentalSKU{}
	for _, count := range []int{1, 2, 3, 4, 8} {
		skus = append(skus, hub.RentalSKU{Name: "h100", AcceleratorModel: "NVIDIA H100", AcceleratorCount: count, VRAMGB: 80})
	}
	candidates := rental.Purchases(skus, models, true, true, rental.Constraints{})
	for i, c := range candidates {
		expected := skus[i].AcceleratorCount == 2 || skus[i].AcceleratorCount == 4
		if (c.Verdict == "") != expected {
			t.Fatalf("width %d verdict %q", c.GPUs, c.Verdict)
		}
		if expected && c.Models[1].GPUs != c.GPUs {
			t.Fatalf("bought count %d with H3 group %d", c.GPUs, c.Models[1].GPUs)
		}
	}
	pinned, _, ok := rental.Pin(models, "NVIDIA H100", 4)
	if !ok || pinned[0].GPUs != 1 || pinned[1].GPUs != 2 {
		t.Fatalf("reuse did not preserve exact authored groups: %#v", pinned)
	}
	if _, _, ok := pinned[1].RungFor("NVIDIA H200", 4); ok {
		t.Fatal("counted H100 selection silently broadened on replacement")
	}
	legacy := records.ModelRef{Ladder: []records.ModelRung{{GPU: "H100", Lane: "fp8", Manifest: "h"}}}
	if _, _, ok := legacy.RungFor("NVIDIA H100", 4); !ok {
		t.Fatal("older uncounted ladder lost four-card compatibility")
	}
}

func TestWorkflowMemoryIsMaximumCallableAndSumWithinCallable(t *testing.T) {
	models := []records.ModelRef{
		{Callable: "p/h3/fl2va", ComponentBytes: map[string]int64{"dit": 60 << 30}},
		{Callable: "p/h3/fl2va", ComponentBytes: map[string]int64{"lora": 4 << 30}},
		{Callable: "p/h3/ref2va", ComponentBytes: map[string]int64{"dit": 60 << 30}},
		{Callable: "p/qwen/image", ComponentBytes: map[string]int64{"transformer": 30 << 30}},
	}
	got := records.Resident(models, "H100", true)
	if got.Bytes != 64<<30 {
		t.Fatalf("CPU workflow child peak %d; want 64 GiB, not 154 GiB", got.Bytes)
	}
}
