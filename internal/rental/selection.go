package rental

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/hub"
)

// CheapestCompatibleSKU applies Creator's complete automatic machine-class
// decision to the live catalog. Price and stable product name are the only
// ordering facts after CPU versus accelerator compatibility is established.
func CheapestCompatibleSKU(skus []hub.RentalSKU, needsAccelerator bool) (hub.RentalSKU, bool) {
	compatible := make([]hub.RentalSKU, 0, len(skus))
	for _, sku := range skus {
		if (sku.AcceleratorModel != "CPU") == needsAccelerator {
			compatible = append(compatible, sku)
		}
	}
	sort.Slice(compatible, func(i, j int) bool {
		if compatible[i].PriceUSDMicrosPerHour != compatible[j].PriceUSDMicrosPerHour {
			return compatible[i].PriceUSDMicrosPerHour < compatible[j].PriceUSDMicrosPerHour
		}
		return compatible[i].Name < compatible[j].Name
	})
	if len(compatible) == 0 {
		return hub.RentalSKU{}, false
	}
	return compatible[0], true
}
