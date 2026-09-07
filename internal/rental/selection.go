package rental

import (
	"fmt"
	"math"
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

// Purchases is every product of the request's class as a placement candidate (cl-165),
// pinned to the rung its accelerator fits and carrying the verdict that keeps it out —
// no rung, a device the lane's resident components outweigh (cl-168), a base image the
// release contradicts — or none, when it may be bought. The rate is what the renter pays:
// price plus storage. A CPU-class request skips the device check: there is no device.
func Purchases(skus []hub.RentalSKU, models []records.ModelRef, needsAccelerator bool,
	constraints Constraints) []orchestrator.PlacementCandidate {
	var out []orchestrator.PlacementCandidate
	for _, sku := range skus {
		if (sku.AcceleratorModel != "CPU") != needsAccelerator {
			continue
		}
		c := orchestrator.PlacementCandidate{SKU: sku.Name,
			RateUSDMicrosPerHour: sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour}
		var ok bool
		if c.Models, c.Rung, ok = Pin(models, sku.AcceleratorModel); !ok {
			c.Verdict = orchestrator.VerdictNoRung
		} else {
			c.Lane = records.Lanes(c.Models)
			if needsAccelerator {
				need := records.Resident(c.Models)
				c.Fit, c.Verdict = FitNote(need, sku.VRAMGB), Fit(need, sku.VRAMGB, sku.Name)
			}
			if c.Verdict == "" {
				c.Verdict = baseMismatch(sku, constraints)
			}
		}
		out = append(out, c)
	}
	return out
}

// Pin binds the selection to the rung fitting `accelerator`: the pinned refs, the first
// slot's 1-based rung, and whether every slot fits. An empty selection fits anywhere.
func Pin(models []records.ModelRef, accelerator string) ([]records.ModelRef, int, bool) {
	rung := 1
	pinned := make([]records.ModelRef, 0, len(models))
	for i, model := range models {
		fitted, index, ok := model.RungFor(accelerator)
		if !ok {
			return nil, 0, false
		}
		if i == 0 {
			rung = index + 1
		}
		pinned = append(pinned, model.Pin(fitted))
	}
	return pinned, rung, true
}

// Fit is the one VRAM sanity floor a buy, a reuse and an explicit override are held to
// (cl-168): a device is short only when the entrypoint's resident components measurably
// outweigh it. It is never a second opinion on the owner's rung — a selection the card
// publishes no component bytes for fits by that assertion. VRAMGB is read as GiB, the
// unit GPU memory is built in (an "80 GB" H100 carries 81920 MiB).
func Fit(need records.Residency, vramGB int64, name string) string {
	if need.Fit != records.FitComponents || need.Bytes <= vramGB<<30 {
		return ""
	}
	return fmt.Sprintf("%s%s: needs %.1f GiB resident (%s), %s has %d GB", orchestrator.VerdictExcluded,
		orchestrator.ExcludedVRAMShort, float64(need.Bytes)/(1<<30), need.Need, name, vramGB)
}

// FitNote renders how the device was sized, so the record says whether a figure was
// compared or the owner's rung stood alone.
func FitNote(need records.Residency, vramGB int64) string {
	if need.Fit == records.FitComponents {
		return fmt.Sprintf("%s %.1f GiB of %d GB", need.Fit, float64(need.Bytes)/(1<<30), vramGB)
	}
	return need.Fit
}

func baseMismatch(sku hub.RentalSKU, constraints Constraints) string {
	profile, readable := launch.ParseBaseProfile(sku.BaseWorkerProfile)
	if !readable {
		return ""
	}
	reason := launch.BaseMismatch(profile, constraints.Requirements, constraints.RequiresPython)
	if reason == "" {
		return ""
	}
	return orchestrator.VerdictExcluded + orchestrator.ExcludedBaseMismatch + ": " + reason
}

// Ladder renders the fit map the candidates were pinned by: the lone slot's ladder, or
// one `slot: …` line per slot when the request binds several.
func Ladder(models []records.ModelRef) []string {
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

// Measure reads each open candidate's expected time and cost from the row measured for
// its (lane, sku) under the first slot's model release, and returns the rows used. An
// attached rental runs after the attempts ahead of it and bills this request only the
// run; a purchase pays its prepare time and bills all of it (placement-economics.md).
func Measure(candidates []orchestrator.PlacementCandidate, rows []hub.ModelThroughput,
	models []records.ModelRef) []hub.ModelThroughput {
	used := []hub.ModelThroughput{}
	if len(models) == 0 {
		return used
	}
	for i := range candidates {
		c := &candidates[i]
		if c.Verdict != "" {
			continue
		}
		for _, row := range rows {
			if row.Release != models[0].Release || row.Lane != c.Models[0].Lane || row.SKU != c.SKU {
				continue
			}
			seconds, billed := row.MedianS+row.PrepareS, row.MedianS+row.PrepareS
			if c.Attached() {
				seconds, billed = row.MedianS*float64(1+c.Ahead), row.MedianS
			}
			c.Measured, c.TimeS = true, seconds
			c.CostUSDMicros = int64(math.Round(float64(c.RateUSDMicrosPerHour) * billed / 3600))
			if !contains(used, row) {
				used = append(used, row)
			}
			break
		}
	}
	return used
}

func contains(rows []hub.ModelThroughput, row hub.ModelThroughput) bool {
	for _, have := range rows {
		if have.Lane == row.Lane && have.SKU == row.SKU {
			return true
		}
	}
	return false
}

// Place is the tier's choice among the open candidates and returns its index, or -1
// when none is open: `fast` the least time, `cheap` the least cost, `balanced` the least
// (time / best time) × (cost / best cost) — a candidate worse on both axes cannot win
// any tier. Ties break by rung, attached over purchase, rate, name. Unmeasured
// candidates count only when nothing is measured, attached first and then by rung and
// rate — the ladder's own order. Every measured candidate's score is written.
func Place(tier string, candidates []orchestrator.PlacementCandidate) int {
	bestTime, bestCost := math.Inf(1), math.Inf(1)
	for _, c := range candidates {
		if c.Verdict == "" && c.Measured {
			bestTime, bestCost = min(bestTime, c.TimeS), min(bestCost, float64(c.CostUSDMicros))
		}
	}
	measured := !math.IsInf(bestTime, 1)
	winner := -1
	for i := range candidates {
		c := &candidates[i]
		if c.Verdict != "" || c.Measured != measured {
			continue
		}
		if measured {
			c.Score = c.TimeS / bestTime * float64(c.CostUSDMicros) / bestCost
		}
		if winner < 0 || prefers(tier, *c, candidates[winner]) {
			winner = i
		}
	}
	return winner
}

func prefers(tier string, a, b orchestrator.PlacementCandidate) bool {
	if a.Measured {
		if x, y := key(tier, a), key(tier, b); x != y {
			return x < y
		}
	} else if a.Attached() != b.Attached() {
		return a.Attached()
	}
	if a.Rung != b.Rung {
		return a.Rung < b.Rung
	}
	if a.Attached() != b.Attached() {
		return a.Attached()
	}
	if a.RateUSDMicrosPerHour != b.RateUSDMicrosPerHour {
		return a.RateUSDMicrosPerHour < b.RateUSDMicrosPerHour
	}
	return a.Name() < b.Name()
}

func key(tier string, c orchestrator.PlacementCandidate) float64 {
	switch tier {
	case "fast":
		return c.TimeS
	case "cheap":
		return float64(c.CostUSDMicros)
	}
	return c.Score
}

// Conclude writes the choice onto the record: `chosen` on the winner and, on every
// candidate still open, why not it — slower or dearer than the winner, or unmeasured.
// A record with a winner and a verdict-less row is the defect
// PlacementDecision.Unexplained exists to catch.
func Conclude(candidates []orchestrator.PlacementCandidate, winner int) {
	chosen := &candidates[winner]
	chosen.Verdict = orchestrator.VerdictChosen
	for i := range candidates {
		c := &candidates[i]
		if c.Verdict != "" {
			continue
		}
		switch {
		case !c.Measured:
			c.Verdict = orchestrator.VerdictUnmeasured
		case c.TimeS > chosen.TimeS:
			c.Verdict = orchestrator.VerdictSlower
		default:
			c.Verdict = orchestrator.VerdictDearer
		}
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
