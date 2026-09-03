package rental

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
)

// CheapestCompatibleSKU applies Creator's complete automatic machine-class
// decision to the live catalog. Price and stable product name are the only
// ordering facts after CPU versus accelerator compatibility is established.
//
// `constraints` is the published release's own Requirements/RequiresPython, and it
// narrows the choice ADVISORILY: a product whose base profile the release already
// contradicts is not worth an hour's rent, because the pod would refuse it typed on
// arrival. The SKU still chooses the machine. When every accelerator-compatible product
// is excluded this way, `mismatch` names why, so the caller refuses before the paid ask
// instead of reporting an empty catalog.
type Constraints struct {
	Requirements   []string
	RequiresPython string
}

func CheapestCompatibleSKU(skus []hub.RentalSKU, needsAccelerator bool,
	constraints Constraints) (sku hub.RentalSKU, mismatch string, found bool) {
	compatible := make([]hub.RentalSKU, 0, len(skus))
	for _, candidate := range skus {
		if (candidate.AcceleratorModel != "CPU") != needsAccelerator {
			continue
		}
		compatible = append(compatible, candidate)
	}
	// Cheapest is what the renter PAYS: the GPU rate plus the SKU's storage
	// adder (th-126). Every GPU product shares one image spec, so the adder is
	// a constant and the ordering equals the GPU-price ordering there; a class
	// whose products carried different disks would still rank by true cost.
	total := func(sku hub.RentalSKU) int64 {
		return sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
	}
	sort.Slice(compatible, func(i, j int) bool {
		if total(compatible[i]) != total(compatible[j]) {
			return total(compatible[i]) < total(compatible[j])
		}
		return compatible[i].Name < compatible[j].Name
	})
	excluded := ""
	for _, candidate := range compatible {
		profile, readable := launch.ParseBaseProfile(candidate.BaseWorkerProfile)
		if !readable {
			return candidate, "", true
		}
		reason := launch.BaseMismatch(profile, constraints.Requirements, constraints.RequiresPython)
		if reason == "" {
			return candidate, "", true
		}
		if excluded == "" {
			excluded = candidate.Name + ": " + reason
		}
	}
	return hub.RentalSKU{}, excluded, false
}
