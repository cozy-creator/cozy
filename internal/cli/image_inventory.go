package cli

import (
	"context"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
)

// RentalProvided is ONE machine's registered `tensorhub.image_inventory/1` document as a
// normalized distribution-name set: the answer to "what does this pod already have?".
//
// It is a property of the placed image, so it cannot be known before a machine is chosen —
// which is the whole reason sealing an editable package's carrier set belongs after
// placement rather than at submission (cl-212). Before th-205 there was no machine-scoped
// door to ask, only `prepare-facts`, which needs a published org/name@release an editable
// install does not have, so Creator pruned against a static ABI-protection roster instead
// and carried a package's entire closure.
//
// A hub that cannot answer is an error, never an empty set: an empty set would seal the
// whole closure, which is merely wasteful, but a WRONG non-empty one would seal away a
// wheel the image does not have, and the pod would fail to import at run time.
func (r *Resolver) RentalProvided(ctx context.Context, rentalID string) (map[string]bool, *exit.Error) {
	if rentalID == "" {
		return nil, nil
	}
	raw, problem := r.catalog.RentalImageInventory(ctx, rentalID)
	if problem != nil {
		return nil, problem
	}
	var document struct {
		Format        string `json:"format"`
		Distributions []struct {
			Name string `json:"name"`
		} `json:"distributions"`
	}
	if err := json.Unmarshal(raw, &document); err != nil || document.Format != inventoryFormat {
		return nil, exit.Named(exit.Structural, "rental.image_inventory_invalid",
			"the placed image's inventory is not a %s document", inventoryFormat)
	}
	if len(document.Distributions) == 0 {
		return nil, exit.Named(exit.Structural, "rental.image_inventory_empty",
			"the placed image's inventory names no distributions")
	}
	out := make(map[string]bool, len(document.Distributions))
	for _, row := range document.Distributions {
		if name := localpackage.NormalizeDistribution(row.Name); name != "" {
			out[name] = true
		}
	}
	return out, nil
}

const inventoryFormat = "tensorhub.image_inventory/1"
