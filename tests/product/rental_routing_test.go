package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestPublishedJobRentalClass(t *testing.T) {
	skus := offeredSKUs()
	tests := []struct {
		name         string
		requirements []string
		wantSKU      string
	}{
		{
			name: "SDXL three-lane CPU producer",
			requirements: []string{
				"cozy-jobs==0.0.14", "cozy-runtime<1.0.0,>=0.0.34",
			},
			wantSKU: "cpu",
		},
		{
			name: "MiniMax H3 CUDA producer",
			requirements: []string{
				"cozy-jobs==0.0.13", "cozy-runtime<1.0.0,>=0.0.33", "msgspec>=0.19",
				"numpy>=1.26", "torch<3,>=2.13",
			},
			wantSKU: "rtx-4090",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sku, _, mismatch, ok := choose(skus,
				launch.AcceleratorRequired(test.requirements),
				rental.Constraints{Requirements: test.requirements})
			if !ok || sku.Name != test.wantSKU {
				t.Fatalf("selected %+v, found=%v mismatch=%q; want %s", sku, ok, mismatch, test.wantSKU)
			}
		})
	}
}

func TestPackageReleaseRequirementsBindExactInterface(t *testing.T) {
	raw := []byte(`{"format":"cozy.package.interface/1"}`)
	sum := sha256.Sum256(raw)
	detail := hub.PackageReleaseDetail{PackageInterface: raw,
		ExecutionRequirements: []string{"cozy-runtime<1.0.0,>=0.0.34", "torch<3,>=2.13"}}
	detail.Release.PackageInterfaceDigest = "sha256:" + hex.EncodeToString(sum[:])
	detail.Release.PackageInterfaceLength = int64(len(raw))
	requirements, problem := detail.Requirements()
	if problem != nil || len(requirements) != 2 || requirements[1] != "torch<3,>=2.13" {
		t.Fatalf("exact release requirements were refused: requirements=%v problem=%v", requirements, problem)
	}

	detail.PackageInterface = bytes.Replace(raw, []byte("interface"), []byte("interfaces"), 1)
	if _, problem := detail.Requirements(); problem == nil || problem.ErrName() != "hub.package_interface_identity_mismatch" {
		t.Fatalf("changed package interface bytes were admitted: %v", problem)
	}
}

// offeredSKUs is the catalog shape Tensorhub actually publishes: every product carries
// the label of the base image it boots. `torch2.13.0-cpu-cp312-linux-x86` is a CPU box
// running CPU Torch; `python3.12-cpu-linux-x86` is the torch-free one.
func offeredSKUs() []hub.RentalSKU {
	return []hub.RentalSKU{
		{Name: "rtx-4090", AcceleratorModel: "RTX 4090", AcceleratorCount: 1, PriceUSDMicrosPerHour: 740_000,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"},
		{Name: "cpu-torch", AcceleratorModel: "CPU", AcceleratorCount: 1, PriceUSDMicrosPerHour: 90_000,
			BaseWorkerProfile: "torch2.13.0-cpu-cp312-linux-x86"},
		{Name: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, PriceUSDMicrosPerHour: 70_000,
			BaseWorkerProfile: "python3.12-cpu-linux-x86"},
	}
}

// Python/platform mismatches remain pre-spend refusals. A package may choose
// a different Torch version or install it into a base that has no cached copy.
func TestReleaseRefusedBeforeRentalSpend(t *testing.T) {
	tests := []struct {
		name           string
		requirements   []string
		requiresPython string
		skus           []hub.RentalSKU
		wantSKU        string
		wantMismatch   string
	}{
		{
			name:         "torch floor above the fleet",
			requirements: []string{"cozy-runtime<1.0.0,>=0.0.34", "torch<3,>=2.14"},
			skus:         offeredSKUs(),
			wantSKU:      "rtx-4090",
		},
		{
			name:           "interpreter above the fleet",
			requirements:   []string{"cozy-runtime<1.0.0,>=0.0.34"},
			requiresPython: ">=3.13",
			skus:           offeredSKUs(),
			wantMismatch:   "this base is cp312",
		},
		{
			name:         "torch-free accelerator base",
			requirements: []string{"torch<3,>=2.13"},
			skus: []hub.RentalSKU{{Name: "rtx-4090", AcceleratorModel: "RTX 4090", AcceleratorCount: 1,
				PriceUSDMicrosPerHour: 740_000, BaseWorkerProfile: "python3.12-cpu-linux-x86"}},
			wantSKU: "rtx-4090",
		},
		{
			name:         "the fleet satisfies the release",
			requirements: []string{"cozy-runtime<1.0.0,>=0.0.34", "torch<3,>=2.13"},
			skus:         offeredSKUs(),
			wantSKU:      "rtx-4090",
		},
		{
			name:         "an unreadable label decides nothing",
			requirements: []string{"torch<3,>=2.14"},
			skus: []hub.RentalSKU{{Name: "rtx-4090", AcceleratorModel: "RTX 4090", AcceleratorCount: 1,
				PriceUSDMicrosPerHour: 740_000, BaseWorkerProfile: "some-future-spelling"}},
			wantSKU: "rtx-4090",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sku, _, mismatch, ok := choose(test.skus,
				launch.AcceleratorRequired(test.requirements),
				rental.Constraints{Requirements: test.requirements, RequiresPython: test.requiresPython})
			if test.wantSKU != "" {
				if !ok || sku.Name != test.wantSKU {
					t.Fatalf("selected %+v found=%v mismatch=%q; want %s",
						sku, ok, mismatch, test.wantSKU)
				}
				return
			}
			if ok {
				t.Fatalf("rented %s before spend; want a refusal naming %q", sku.Name, test.wantMismatch)
			}
			if !strings.Contains(mismatch, test.wantMismatch) {
				t.Fatalf("refused with %q; want it to name %q", mismatch, test.wantMismatch)
			}
		})
	}
}

// th-126 — the ranking proof: the storage adder is the same for every SKU of a
// class (one image spec per class), so the cheapest-by-total choice is the
// cheapest-by-GPU-rate choice the routing always made. A class whose products
// carried different disks would rank by what the renter actually pays.
func TestSameSpecStorageAdderLeavesRankingUnchanged(t *testing.T) {
	skus := offeredSKUs()
	for i := range skus {
		if skus[i].AcceleratorModel != "CPU" {
			skus[i].StorageUSDMicrosPerHour = 213_504
		}
	}
	skus = append(skus, hub.RentalSKU{Name: "l4", AcceleratorModel: "NVIDIA L4", AcceleratorCount: 1,
		PriceUSDMicrosPerHour: 490_000, StorageUSDMicrosPerHour: 213_504,
		BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"})
	sku, _, mismatch, ok := choose(skus, true, rental.Constraints{})
	if !ok || sku.Name != "l4" {
		t.Fatalf("selected %+v ok=%v mismatch=%q; the same-spec adder must not reorder the ladder", sku, ok, mismatch)
	}
	// Different adders DO reorder by true cost: a cheap card on a bloated spec
	// loses to a dearer card whose pod bills less in total.
	skus = append(skus, hub.RentalSKU{Name: "l4-bloated", AcceleratorModel: "NVIDIA L4", AcceleratorCount: 1,
		PriceUSDMicrosPerHour: 480_000, StorageUSDMicrosPerHour: 300_000,
		BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"})
	sku, _, _, ok = choose(skus, true, rental.Constraints{})
	if !ok || sku.Name != "l4" {
		t.Fatalf("selected %+v; want l4 by total (703504 < 780000)", sku)
	}
}

// rentalReleaseFacts supplies immutable facts for lifecycle fixtures that never
// reach package preparation. The interface identity is still validated normally.
func rentalReleaseFacts() hub.PackageReleaseDetail {
	raw := []byte(`{"format":"cozy.package.interface/1"}`)
	sum := sha256.Sum256(raw)
	detail := hub.PackageReleaseDetail{PackageInterface: raw, ExecutionRequirements: []string{}, RequiresPython: ">=3.12"}
	detail.Release.Release = "1"
	detail.Release.PackageInterfaceDigest = "sha256:" + hex.EncodeToString(sum[:])
	detail.Release.PackageInterfaceLength = int64(len(raw))
	return detail
}
