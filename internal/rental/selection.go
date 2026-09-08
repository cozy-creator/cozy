package rental

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// Constraints is the published release's own Requirements/RequiresPython and the degrees
// its package declares it can shard at. It narrows the catalog ADVISORILY: a product whose
// base profile the release already contradicts, or whose WIDTH the package cannot shard
// across, is not worth an hour's rent, because the pod would refuse it typed on arrival.
type Constraints struct {
	Requirements   []string
	RequiresPython string
	// Degrees is the intersection of every model slot's `sequence_parallel.degrees` in the
	// package interface — the author's statement of which group degrees the whole
	// construction can be built at. Empty means the package declares none, which is most
	// packages and is why a wide product is excluded rather than chosen by default.
	Degrees []int
}

// Purchases is every product of the request's class as a placement candidate (cl-165),
// pinned to the rung its accelerator fits and carrying the verdict that keeps it out —
// no rung, a device the selection outweighs (cl-168, cl-170), a base image the release
// contradicts — or none, when it may be bought. The rate is what the renter pays: price
// plus storage. A CPU-class request skips the device check: there is no device, and a
// `job` request skips the memory figure: its models are never resident (cl-180).
func Purchases(skus []hub.RentalSKU, models []records.ModelRef, needsAccelerator, job bool,
	constraints Constraints) []orchestrator.PlacementCandidate {
	var out []orchestrator.PlacementCandidate
	for _, sku := range skus {
		if (sku.AcceleratorModel != "CPU") != needsAccelerator {
			continue
		}
		c := orchestrator.PlacementCandidate{SKU: sku.Name,
			RateUSDMicrosPerHour: sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour}
		Size(&c, models, sku.AcceleratorModel, sku.VRAMGB, needsAccelerator, job)
		if c.Verdict == "" {
			c.Verdict = widthUnusable(sku, job, constraints)
		}
		if c.Verdict == "" {
			c.Verdict = baseMismatch(sku, constraints)
		}
		out = append(out, c)
	}
	return out
}

// Size pins the selection onto one machine and holds its device to what the pinned lanes
// need (cl-168, cl-170): Models, Rung and Lane, then Fit — the rule that sized it — and
// the verdict that keeps it out, no_rung or vram_short. `device` is false for a CPU-class
// request, or a machine the catalog no longer sizes: nothing is compared. `job` says the
// selection is a job's inputs, which are derive-only and never resident (cl-180); the
// rung still selects the lane, and nothing is compared against memory.
func Size(c *orchestrator.PlacementCandidate, models []records.ModelRef, accelerator string,
	vramGB int64, device, job bool) {
	var ok bool
	if c.Models, c.Rung, ok = Pin(models, accelerator); !ok {
		c.Verdict = orchestrator.VerdictNoRung
		return
	}
	c.Lane = records.Lanes(c.Models)
	if !device {
		return
	}
	need := records.Resident(c.Models, accelerator, job)
	c.Fit, c.Verdict = FitNote(need, vramGB), Fit(need, vramGB, c.SKU)
}

// Pin binds the selection to the rung fitting `accelerator`: the pinned refs, the first
// slot's 1-based rung — 0 when that slot pinned its own lane, which is not a rung
// (cl-170) — and whether every slot fits. An empty selection fits anywhere.
func Pin(models []records.ModelRef, accelerator string) ([]records.ModelRef, int, bool) {
	rung := 0
	pinned := make([]records.ModelRef, 0, len(models))
	for i, model := range models {
		fitted, index, ok := model.RungFor(accelerator)
		if !ok {
			return nil, 0, false
		}
		if i == 0 && !model.Pinned() {
			rung = index + 1
		}
		pinned = append(pinned, model.Pin(fitted))
	}
	return pinned, rung, true
}

// Fit is the one VRAM sanity floor a buy, a reuse and an explicit override are held to
// (cl-168): a device is short only when what the selection measurably needs outweighs it.
// A rung-asserted slot needs no figure — the owner's word stands. VRAMGB is read as GiB,
// the unit GPU memory is built in (an "80 GB" H100 carries 81920 MiB).
//
// ONE CARD, AT EVERY WIDTH, DELIBERATELY (cl-179). VRAMGB is the per-card figure for a
// wide product as much as a narrow one, and that is the correct test: nothing here is
// tensor- or pipeline-parallel, so under a sequence-parallel group every rank holds the
// WHOLE weights and a four-card pod fits exactly what one of its cards fits. Reading a
// width as capacity — summing it, or dividing the need by it — would buy a pod that
// cannot hold the model and only discover it after the hour was billed.
func Fit(need records.Residency, vramGB int64, name string) string {
	if need.Bytes <= vramGB<<30 {
		return ""
	}
	return fmt.Sprintf("%s%s: needs %.1f GiB resident (%s), %s has %d GB", orchestrator.VerdictExcluded,
		orchestrator.ExcludedVRAMShort, float64(need.Bytes)/(1<<30), need.Need, name, vramGB)
}

// FitNote renders how the device was sized: the rule, and the figure compared when there
// was one.
func FitNote(need records.Residency, vramGB int64) string {
	if need.Bytes == 0 {
		return need.Fit
	}
	return fmt.Sprintf("%s %.1f GiB of %d GB", need.Fit, float64(need.Bytes)/(1<<30), vramGB)
}

// widthUnusable keeps a product WIDER than one card out of the ladder unless this request
// can actually use every card it would pay for (cl-179).
//
// A wide machine is not more capacity: every rank of a sequence-parallel group holds the
// FULL weights, so width buys latency and never fit. The only thing that uses the extra
// cards is a group placement of exactly that degree, which needs two things this side
// knows before spending: the package's author must have declared the degree, and the
// request must be a serving one — a job is a single bounded attempt and shards nothing,
// so a wide pod bought for one idles every card but the first for the whole hour.
//
// The check is CHEAP INSURANCE, not the fence. The worker refuses
// `device_group_unsupported` on arrival either way; the difference is whether that refusal
// costs an hour's rent. Constraints are advisory — a hub that will not answer yields none
// — so this narrows the ladder and never widens it: with no declared degrees, only
// one-card products remain, which is exactly the behaviour before wide products existed.
func widthUnusable(sku hub.RentalSKU, job bool, constraints Constraints) string {
	if sku.AcceleratorCount < 2 {
		return ""
	}
	if job {
		return orchestrator.VerdictExcluded + orchestrator.ExcludedWidthUndeclared +
			fmt.Sprintf(": %d cards, and a job shards none of them", sku.AcceleratorCount)
	}
	for _, degree := range constraints.Degrees {
		if degree == sku.AcceleratorCount {
			return ""
		}
	}
	return orchestrator.VerdictExcluded + orchestrator.ExcludedWidthUndeclared +
		fmt.Sprintf(": %d cards, and the package declares %s", sku.AcceleratorCount,
			declaredDegrees(constraints.Degrees))
}

func declaredDegrees(degrees []int) string {
	if len(degrees) == 0 {
		return "no sequence-parallel degree"
	}
	parts := make([]string, 0, len(degrees))
	for _, degree := range degrees {
		parts = append(parts, strconv.Itoa(degree))
	}
	return "degrees " + strings.Join(parts, ", ")
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

// Ladder renders the owner's fit map the candidates were sized by: the lone slot's
// ladder, or one `slot: …` line per slot when the request binds several; nothing for a
// slot bound to none.
func Ladder(models []records.ModelRef) []string {
	var out []string
	for _, model := range models {
		if len(model.Ladder) == 0 {
			continue
		}
		parts := make([]string, 0, len(model.Ladder))
		for _, rung := range model.Ladder {
			parts = append(parts, rung.String())
		}
		rungs := strings.Join(parts, " > ")
		if len(models) == 1 {
			return []string{rungs}
		}
		out = append(out, model.Slot+": "+rungs)
	}
	return out
}

// Override renders the lane(s) the request pinned before any machine existed — an explicit
// `model.<param>=org/model@release/lane` — the lone slot's lane, or `slot=lane` pairs.
func Override(models []records.ModelRef) string {
	var parts []string
	for _, model := range models {
		if !model.Pinned() {
			continue
		}
		if len(models) == 1 {
			return model.Lane
		}
		parts = append(parts, model.Slot+"="+model.Lane)
	}
	return strings.Join(parts, ",")
}

// Attaching is the index of a fitting rental the fleet holds whose worker has not
// attached yet (cl-170), or -1. It cannot take the request now, and the request waits
// for it rather than buying around a machine already paid for (the cl-132 shape).
func Attaching(candidates []orchestrator.PlacementCandidate) int {
	for i, c := range candidates {
		if c.Verdict == orchestrator.VerdictAttaching {
			return i
		}
	}
	return -1
}

// Wait closes the record on a decision to wait for `attaching`: every candidate still
// open is passed over for it.
func Wait(candidates []orchestrator.PlacementCandidate, attaching int) {
	for i := range candidates {
		if candidates[i].Verdict == "" {
			candidates[i].Verdict = orchestrator.VerdictExcluded + orchestrator.ExcludedAttaching +
				": " + candidates[attaching].Name()
		}
	}
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
// any tier. Ties break by rung, attached over purchase, attempts ahead, rate, name.
// Unmeasured candidates count only when nothing is measured, attached first and then by
// rung, the fewest attempts ahead and rate — the ladder's own order, an idle machine
// before a busy one (cl-174). Every measured candidate's score is written.
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
	if a.Ahead != b.Ahead {
		return a.Ahead < b.Ahead
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
