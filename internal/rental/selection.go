package rental

import (
	"fmt"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// Constraints is the published release's own Requirements/RequiresPython. It narrows the
// catalog ADVISORILY: a product whose base profile the release already contradicts is not
// worth an hour's rent, because the pod would refuse it typed on arrival.
type Constraints struct {
	Requirements   []string
	RequiresPython string
}

// Step is one buyable machine in walk order, with the request's selection pinned to the
// rung that fits it.
type Step struct {
	SKU    hub.RentalSKU
	Rung   int // 1-based ladder rung
	Models []records.ModelRef
	Index  int // this product's row in the decision's Offered
}

// Plan is Creator's complete automatic machine-class decision for a buy (cl-166): the
// catalog walked rung by rung in the owner's ladder order, cheapest first within a rung,
// keeping only the products the request FITS, and the record that makes every exclusion
// readable. The ladder is a fit map — which lane belongs on which GPU class — so a product
// no rung names is not a candidate, and a product whose rung lane outweighs its device is
// not one either.
//
// The resident need is the pinned lanes' manifest bytes summed over slots: the model card
// publishes one byte total per lane and no per-component sizes, so the whole lane is what
// this reads, and it is the conservative fact — a lane's largest component is never larger
// than the lane. VRAMGB is read as GiB, the unit GPU memory is actually built in (an
// "80 GB" H100 carries 81920 MiB). A CPU-class request skips the device check: there is
// no device, and the hub sizes the pod's RAM from the SKU.
func Plan(skus []hub.RentalSKU, models []records.ModelRef, needsAccelerator bool,
	constraints Constraints) ([]Step, orchestrator.SKUDecision) {
	total := func(sku hub.RentalSKU) int64 {
		return sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
	}
	decision := orchestrator.SKUDecision{Ladder: ladderText(models)}
	type row struct {
		step    Step
		verdict string
	}
	rows := make([]row, 0, len(skus))
	for _, sku := range skus {
		if (sku.AcceleratorModel != "CPU") != needsAccelerator {
			continue
		}
		r := row{step: Step{SKU: sku}}
		r.step.Rung, r.step.Models, r.verdict = fit(sku, models, needsAccelerator)
		if r.verdict == "" {
			if profile, readable := launch.ParseBaseProfile(sku.BaseWorkerProfile); readable {
				if reason := launch.BaseMismatch(profile, constraints.Requirements,
					constraints.RequiresPython); reason != "" {
					r.verdict = orchestrator.VerdictBaseMismatch + ": " + reason
					if decision.Mismatch == "" {
						decision.Mismatch = sku.Name + ": " + reason
					}
				}
			}
		}
		rows = append(rows, r)
	}
	// Walk order: the owner's rung first, price within it, name for a stable tie; the
	// products no rung names come last, so the record still shows them.
	sort.SliceStable(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if (a.step.Rung == 0) != (b.step.Rung == 0) {
			return a.step.Rung != 0
		}
		if a.step.Rung != b.step.Rung {
			return a.step.Rung < b.step.Rung
		}
		if total(a.step.SKU) != total(b.step.SKU) {
			return total(a.step.SKU) < total(b.step.SKU)
		}
		return a.step.SKU.Name < b.step.SKU.Name
	})
	var steps []Step
	for i, r := range rows {
		candidate := orchestrator.SKUCandidate{Name: r.step.SKU.Name,
			TotalUSDMicrosPerHour: total(r.step.SKU), Rung: r.step.Rung, Verdict: r.verdict}
		if r.step.Rung != 0 {
			candidate.Lane = records.Lanes(r.step.Models)
		}
		decision.Offered = append(decision.Offered, candidate)
		if r.verdict == "" {
			r.step.Index = i
			steps = append(steps, r.step)
		}
	}
	if len(steps) > 0 {
		decision.Mismatch = ""
	}
	return steps, decision
}

// fit pins the request's selection to the product's rung and says whether it holds.
func fit(sku hub.RentalSKU, models []records.ModelRef, needsAccelerator bool) (int, []records.ModelRef, string) {
	rung := 1
	pinned := make([]records.ModelRef, 0, len(models))
	for i, model := range models {
		fitted, index, ok := model.RungFor(sku.AcceleratorModel)
		if !ok {
			return 0, nil, orchestrator.VerdictGPUMismatch
		}
		if i == 0 {
			rung = index + 1
		}
		pinned = append(pinned, model.Pin(fitted))
	}
	if needsAccelerator {
		if need, have := records.ResidentBytes(pinned), sku.VRAMGB<<30; need > have {
			return rung, pinned, fmt.Sprintf("%s: needs %.1f GiB, %s has %d GB",
				orchestrator.VerdictVRAMShort, float64(need)/(1<<30), sku.Name, sku.VRAMGB)
		}
	}
	return rung, pinned, ""
}

// ladderText renders the fit map the walk follows: the lone slot's ladder, or one
// `slot: …` line per slot when the request binds several.
func ladderText(models []records.ModelRef) []string {
	var out []string
	for _, model := range models {
		rungs := model.Lane
		if !model.Pinned() {
			parts := make([]string, 0, len(model.Ladder))
			for _, rung := range model.Ladder {
				parts = append(parts, rung.String())
			}
			rungs = strings.Join(parts, " > ")
		}
		if len(models) == 1 {
			return []string{rungs}
		}
		out = append(out, model.Slot+": "+rungs)
	}
	return out
}

// Conclude writes the walk's outcome onto the record: the winner, `no_inventory` on every
// step the hub refused before it, and `dearer` / `later_rung` on every fitting product the
// walk never reached. A record with a chosen product and a verdict-less row is the defect
// SKUDecision.UnexplainedPick exists to catch.
func Conclude(decision *orchestrator.SKUDecision, steps []Step, refused []int, winner int) {
	for _, index := range refused {
		decision.Offered[steps[index].Index].Verdict = orchestrator.VerdictNoInventory
	}
	if winner < 0 || winner >= len(steps) {
		return
	}
	chosen := steps[winner]
	decision.Chosen = chosen.SKU.Name
	for i := winner + 1; i < len(steps); i++ {
		verdict := orchestrator.VerdictDearer
		if steps[i].Rung != chosen.Rung {
			verdict = fmt.Sprintf("%s: rung %d", orchestrator.VerdictLaterRung, steps[i].Rung)
		}
		decision.Offered[steps[i].Index].Verdict = verdict
	}
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
