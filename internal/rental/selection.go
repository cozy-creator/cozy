package rental

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
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

// CheapestCompatibleSKU is Choose's answer without the record, for callers that only
// need the winner.
func CheapestCompatibleSKU(skus []hub.RentalSKU, needsAccelerator bool,
	constraints Constraints) (sku hub.RentalSKU, mismatch string, found bool) {
	sku, decision, found := Choose(skus, needsAccelerator, constraints)
	return sku, decision.Mismatch, found
}

// Choose picks the machine AND states the case for it (cl-132). The pick is identical to
// what CheapestCompatibleSKU always returned; the second return is the evidence that
// makes the pick auditable — see orchestrator.SKUDecision for why it has to exist.
func Choose(skus []hub.RentalSKU, needsAccelerator bool,
	constraints Constraints) (hub.RentalSKU, orchestrator.SKUDecision, bool) {
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
	decision := orchestrator.SKUDecision{
		Offered: make([]orchestrator.SKUCandidate, 0, len(compatible)),
	}
	var winner hub.RentalSKU
	found := false
	for _, candidate := range compatible {
		row := orchestrator.SKUCandidate{
			Name: candidate.Name, TotalUSDMicrosPerHour: total(candidate)}
		if found {
			// Everything after the winner is simply dearer (or an equal-priced
			// later name): the sort already ordered it, and re-deciding it here
			// would answer a question the choice never asked.
			row.Verdict = orchestrator.VerdictDearer
			decision.Offered = append(decision.Offered, row)
			continue
		}
		profile, readable := launch.ParseBaseProfile(candidate.BaseWorkerProfile)
		reason := ""
		if readable {
			reason = launch.BaseMismatch(profile, constraints.Requirements,
				constraints.RequiresPython)
		}
		if reason == "" {
			winner, found = candidate, true
			decision.Chosen = candidate.Name
		} else {
			row.Verdict = orchestrator.VerdictBaseMismatch + ": " + reason
			if decision.Mismatch == "" {
				decision.Mismatch = candidate.Name + ": " + reason
			}
		}
		decision.Offered = append(decision.Offered, row)
	}
	if found {
		// A refusal reports the exclusion that caused it; a successful choice has
		// nothing to explain away.
		decision.Mismatch = ""
	}
	return winner, decision, found
}

// AcquisitionReason is the rental's PROVENANCE: which command caused this pod to be
// bought (cl-132).
//
// The managed path used to record "" here, so a pod bought by auto-placement and one
// bought by an explicit `cozy rental new` were indistinguishable in the operations table
// and absent from `cozy rental list` entirely. On 2026-09-04 that cost an hour and two
// wrongly-filed issues: two pods bought minutes apart by an ingest job and by a serving
// run were both credited to an explicit `cozy rental new rtx-a4000` which had in fact
// been REFUSED for want of inventory, and the fleet was read as having substituted a
// dearer card for the one that was asked for. Nothing had substituted anything. A rental
// now says who bought it, so that reading is available without a database.
func AcquisitionReason(req records.Request) string {
	kind := "cozy run"
	if req.IsJob() {
		kind = "cozy job"
	}
	reason := kind + " " + req.ID
	if req.Package != "" {
		reason += " (" + req.Package + ")"
	}
	return reason
}
