package producttest

import (
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"testing"
)

func TestCountedLaddersBuyMaximumChildGroupAndReuseWidestGroup(t *testing.T) {
	models := []records.ModelRef{
		{Callable: "proof/qwen/image", Model: "proof/qwen", Ladder: []records.ModelRung{{GPU: "H100", GPUs: 1, Lane: "original", Manifest: "q"}}},
		{Callable: "proof/h3/shot", Model: "proof/h3", Ladder: []records.ModelRung{{GPU: "H100", GPUs: 2, Lane: "fp8", Manifest: "h"}, {GPU: "H100", GPUs: 4, Lane: "fp8", Manifest: "h"}}},
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
	// Reuse takes each callable's widest authored group the machine holds: ladder order
	// among widths is what to buy, not how much of a paid machine to use.
	for count, want := range map[int]int{8: 4, 4: 4, 3: 2, 2: 2} {
		pinned, _, ok := rental.Pin(models, "NVIDIA H100", count)
		if !ok || pinned[0].GPUs != 1 || pinned[1].GPUs != want {
			t.Fatalf("reuse of %d cards: %#v", count, pinned)
		}
	}
	if _, _, ok := rental.Pin(models, "NVIDIA H100", 1); ok {
		t.Fatal("a one-card machine fit an H3 authored only at 2 and 4")
	}
	pinned, _, _ := rental.Pin(models, "NVIDIA H100", 4)
	if _, _, ok := rental.Pin(pinned, "NVIDIA H200", 4); ok {
		t.Fatal("counted H100 selection silently broadened on replacement")
	}
	legacy := records.ModelRef{Ladder: []records.ModelRung{{GPU: "H100", Lane: "fp8", Manifest: "h"}}}
	if pinned, _, ok := rental.Pin([]records.ModelRef{legacy}, "NVIDIA H100", 4); !ok || pinned[0].GPUs != 0 {
		t.Fatal("older uncounted ladder lost four-card compatibility")
	}
}

// One callable's slots are one group: H3 Turbo's base and LoRA take the widest width both
// ladders author, an uncounted rung serves any width, a pinned count is exact, and ladder
// order still ranks lanes of one width.
func TestReuseWidthIsJointAcrossOneCallablesSlots(t *testing.T) {
	rung := func(gpu string, gpus int, lane string) records.ModelRung {
		return records.ModelRung{GPU: gpu, GPUs: gpus, Lane: lane, Manifest: lane}
	}
	base := records.ModelRef{Slot: "base", Ladder: []records.ModelRung{rung("H100", 2, "fp8"), rung("H100", 4, "fp8"), rung("*", 0, "bf16")}}
	lora := func(rungs ...records.ModelRung) records.ModelRef {
		return records.ModelRef{Slot: "lora", Ladder: rungs}
	}
	const h100 = "NVIDIA H100 80GB HBM3"
	for _, arm := range []struct {
		name, accelerator string
		other             records.ModelRef
		count             int
		gpus              [2]int
		lanes             [2]string
	}{
		{"both author 4", h100, lora(rung("H100", 2, "pdd8"), rung("H100", 4, "pdd8")), 4, [2]int{4, 4}, [2]string{"fp8", "pdd8"}},
		{"the LoRA stops at 2", h100, lora(rung("H100", 2, "pdd8")), 4, [2]int{2, 2}, [2]string{"fp8", "pdd8"}},
		{"an uncounted LoRA serves any width", h100, lora(rung("*", 0, "pdd8")), 8, [2]int{4, 0}, [2]string{"fp8", "pdd8"}},
		{"a pinned count is honored", h100, records.ModelRef{Slot: "lora", GPUs: 2, Lane: "pdd8", Manifest: "pdd8"}, 4, [2]int{2, 2}, [2]string{"fp8", "pdd8"}},
		{"ladder order ranks one width's lanes", h100, lora(rung("H100", 4, "a"), rung("H100", 4, "b")), 4, [2]int{4, 4}, [2]string{"fp8", "a"}},
		{"an unlisted card takes the catch-all", "NVIDIA A100", lora(rung("*", 0, "pdd8")), 4, [2]int{0, 0}, [2]string{"bf16", "pdd8"}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			pinned, _, ok := rental.Pin([]records.ModelRef{base, arm.other}, arm.accelerator, arm.count)
			if !ok {
				t.Fatal("no fit")
			}
			for i := range pinned {
				if pinned[i].GPUs != arm.gpus[i] || pinned[i].Lane != arm.lanes[i] {
					t.Fatalf("slot %d took %d×%s; want %d×%s", i, pinned[i].GPUs, pinned[i].Lane, arm.gpus[i], arm.lanes[i])
				}
			}
		})
	}
	if _, _, ok := rental.Pin([]records.ModelRef{{Slot: "base", Ladder: base.Ladder[:2]}, lora(rung("H100", 1, "pdd8"))}, h100, 4); ok {
		t.Fatal("slots of one callable with no common width fit")
	}
}

func TestWorkflowMemoryIsMaximumCallableAndSumWithinCallable(t *testing.T) {
	models := []records.ModelRef{
		{Callable: "p/h3/fl2va", ComponentBytes: map[string]int64{"dit": 60 << 30}},
		{Callable: "p/h3/fl2va", ComponentBytes: map[string]int64{"lora": 4 << 30}},
		{Callable: "p/h3/ref2va", ComponentBytes: map[string]int64{"dit": 60 << 30}},
		{Callable: "p/qwen/image", ComponentBytes: map[string]int64{"transformer": 30 << 30}},
	}
	models = append(models, records.ModelRef{ComponentBytes: map[string]int64{"derive_only_input": 200 << 30}})
	got := records.Resident(models, "H100", true)
	if got.Bytes != 64<<30 {
		t.Fatalf("CPU workflow child peak %d; want 64 GiB, not 154 GiB", got.Bytes)
	}
}
