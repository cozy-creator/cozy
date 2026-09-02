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
			sku, mismatch, ok := rental.CheapestCompatibleSKU(skus,
				launch.AcceleratorRequired(test.requirements),
				rental.Constraints{Requirements: test.requirements})
			if !ok || sku.Name != test.wantSKU {
				t.Fatalf("selected %+v, found=%v mismatch=%q; want %s", sku, ok, mismatch, test.wantSKU)
			}
		})
	}
}

func TestPackageReleaseRequirementsBindExactBytes(t *testing.T) {
	raw := []byte(`{"format":"cozy.package.manifest/1","requirements":["cozy-runtime<1.0.0,>=0.0.34","torch<3,>=2.13"]}`)
	sum := sha256.Sum256(raw)
	// Go's outer JSON response encoder HTML-escapes `<` even though the stored
	// PackageRelease writer does not. RawMessage therefore sees a different token
	// spelling for the same document content.
	transported := bytes.ReplaceAll(raw, []byte("<"), []byte(`\u003c`))
	detail := hub.PackageReleaseDetail{Document: transported}
	detail.Release.ReleaseDigest = "sha256:" + hex.EncodeToString(sum[:])
	requirements, problem := detail.Requirements()
	if problem != nil || len(requirements) != 2 || requirements[1] != "torch<3,>=2.13" {
		t.Fatalf("exact release requirements were refused: requirements=%v problem=%v", requirements, problem)
	}

	detail.Document = bytes.Replace(raw, []byte("torch<3"), []byte("torch<4"), 1)
	if _, problem := detail.Requirements(); problem == nil || problem.ErrName() != "hub.package_release_digest_mismatch" {
		t.Fatalf("changed release bytes were admitted: %v", problem)
	}
}

// offeredSKUs is the catalog shape Tensorhub actually publishes: every product carries
// the label of the base image it boots. `torch2.13.0-cpu-cp312-linux-x86` is a CPU box
// running CPU Torch; `python3.12-cpu-linux-x86` is the torch-free one.
func offeredSKUs() []hub.RentalSKU {
	return []hub.RentalSKU{
		{Name: "rtx-4090", AcceleratorModel: "RTX 4090", PriceUSDMicrosPerHour: 740_000,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"},
		{Name: "cpu-torch", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 90_000,
			BaseWorkerProfile: "torch2.13.0-cpu-cp312-linux-x86"},
		{Name: "cpu", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 70_000,
			BaseWorkerProfile: "python3.12-cpu-linux-x86"},
	}
}

// TestReleaseRefusedBeforeRentalSpend is the PRE-SPEND filter: a release whose own stored
// requirements the offered base profiles already contradict never reaches a paid ask.
//
// The two arms that bite in the real catalog are the version arms — a fleet-wide Torch
// release and a fleet-wide Python ABI are exactly what the profile label names, and
// neither is visible to the accelerator-class decision that used to be the whole filter.
// The torch-free arm is defense in depth: the hub couples a torch-free profile to a
// CPU-backend target, so it cannot appear on an accelerator product today.
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
			wantMismatch: "this base carries torch 2.13.0",
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
			skus: []hub.RentalSKU{{Name: "rtx-4090", AcceleratorModel: "RTX 4090",
				PriceUSDMicrosPerHour: 740_000, BaseWorkerProfile: "python3.12-cpu-linux-x86"}},
			wantMismatch: "this base carries no Torch",
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
			skus: []hub.RentalSKU{{Name: "rtx-4090", AcceleratorModel: "RTX 4090",
				PriceUSDMicrosPerHour: 740_000, BaseWorkerProfile: "some-future-spelling"}},
			wantSKU: "rtx-4090",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sku, mismatch, ok := rental.CheapestCompatibleSKU(test.skus,
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
