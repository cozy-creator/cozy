package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestPublishedJobRentalClass(t *testing.T) {
	skus := []hub.RentalSKU{
		{Name: "rtx-4090", AcceleratorModel: "RTX 4090", PriceUSDMicrosPerHour: 740_000},
		{Name: "cpu-torch", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 90_000},
		{Name: "cpu", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 70_000},
	}
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
			sku, ok := rental.CheapestCompatibleSKU(skus, launch.AcceleratorRequired(test.requirements))
			if !ok || sku.Name != test.wantSKU {
				t.Fatalf("selected %+v, found=%v; want %s", sku, ok, test.wantSKU)
			}
		})
	}
}

func TestPackageReleaseRequirementsBindExactBytes(t *testing.T) {
	raw := []byte(`{"format":"cozy.package.release/1","requirements":["cozy-runtime<1.0.0,>=0.0.34","torch<3,>=2.13"]}`)
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
