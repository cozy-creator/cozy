package rental

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
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
	// GPUs requests an exact execution group independently of the selected weights.
	// Wider machines may lend a subset; this is not the rental's physical size.
	GPUs                  uint32
	Requirements          []string
	RequiresPython        string
	PythonVersion         string
	SupportedPythonMinors []string
	// Degrees is the intersection of every model slot's `sequence_parallel.degrees` in the
	// package interface — the author's statement of which group degrees the whole
	// construction can be built at. Empty means the package declares none, which is most
	// packages and is why a wide product is excluded rather than chosen by default.
	Degrees []int
	// Working is the release entrypoint's measured working memory by models digest
	// (proto-061 B). Nil or a missing digest is unmeasured.
	Working records.WorkingPeaks
}

// Purchases is every product of the request's class as a placement candidate (cl-165),
// sized by the request's lane bytes and carrying the verdict that keeps it out — a device
// the selection outweighs (cl-168), a base image the release contradicts — or none, when
// it may be bought. The rate is what the renter pays: price
// plus storage. A CPU-class request skips the device check: there is no device. `composes`
// says the request constructs no model of its own, so its callees decide the width.
func Purchases(skus []hub.RentalSKU, models []records.ModelRef, needsAccelerator, composes bool,
	constraints Constraints) []orchestrator.PlacementCandidate {
	var out []orchestrator.PlacementCandidate
	for _, sku := range skus {
		if (sku.AcceleratorModel != "CPU") != needsAccelerator {
			continue
		}
		c := orchestrator.PlacementCandidate{SKU: sku.Name, GPUs: sku.AcceleratorCount,
			RateUSDMicrosPerHour: sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour}
		count, verdict := runWidth(sku.AcceleratorCount, models, composes, constraints)
		c.Verdict = verdict
		// Do not buy several GPUs for code that can use only one. A parallel
		// package may use a supported subset of a wider offer.
		if c.Verdict == "" && constraints.GPUs == 0 && count == 1 && sku.AcceleratorCount > 1 {
			c.Verdict = WidthUnusable(sku.AcceleratorCount, composes, constraints)
		}
		if c.Verdict == "" {
			size(&c, models, sku.AcceleratorModel, sku.VRAMGB, needsAccelerator, constraints.Working, count)
		}
		if c.Verdict == "" {
			c.Verdict = baseMismatch(sku, constraints)
		}
		out = append(out, c)
	}
	if !composes && constraints.GPUs == 0 {
		fewest(out)
	}
	return out
}

// fewest keeps the fewest cards of each product that hold the run (xs-042): every card of a
// sequence-parallel group holds the whole weights, so a wider buy holds no more. A wider
// product stays only where the narrower ones cannot hold it (a working peak measured at a
// wider degree is smaller), at a degree the package declares.
func fewest(candidates []orchestrator.PlacementCandidate) {
	narrowest := map[string]int{}
	for _, c := range candidates {
		if c.Verdict == "" && (narrowest[c.SKU] == 0 || c.GPUs < narrowest[c.SKU]) {
			narrowest[c.SKU] = c.GPUs
		}
	}
	for i := range candidates {
		if c := &candidates[i]; c.Verdict == "" && c.GPUs > narrowest[c.SKU] {
			c.Verdict = fmt.Sprintf("%s%s: %s holds the run", orchestrator.VerdictExcluded,
				orchestrator.ExcludedWidthUnneeded, orchestrator.MachineLabel(c.SKU, narrowest[c.SKU]))
		}
	}
}

// PurchaseWidthUnusable uses package-code capability only. A CPU composition may
// reserve an explicit subset; each actual GPU child validates that exact count on the
// worker. Without an explicit count, a wide rental needs a child that can use it.
func PurchaseWidthUnusable(width int, models []records.ModelRef, composes bool, constraints Constraints) string {
	if composes && constraints.GPUs > 0 {
		return ""
	}
	if composes {
		groups := map[string][]int{}
		for _, model := range models {
			if model.Callable == "" {
				continue
			}
			degrees := model.SupportedGPUs
			if len(degrees) == 0 {
				degrees = []int{1}
			}
			if common, seen := groups[model.Callable]; seen {
				var overlap []int
				for _, degree := range common {
					for _, candidate := range degrees {
						if degree == candidate {
							overlap = append(overlap, degree)
						}
					}
				}
				degrees = overlap
			}
			groups[model.Callable] = degrees
		}
		for _, degrees := range groups {
			for _, degree := range degrees {
				if degree == width {
					return ""
				}
			}
		}
	}
	return WidthUnusable(width, composes, constraints)
}

// runWidth is shared by purchases and existing machines; only package code and
// an explicit request select the degree. Weight selectors do not participate.
func runWidth(available int, models []records.ModelRef, composes bool, constraints Constraints) (int, string) {
	if reason := GPUCountUnusable(available, constraints.GPUs); reason != "" {
		return 0, reason
	}
	if constraints.GPUs > 0 {
		count := int(constraints.GPUs)
		return count, PurchaseWidthUnusable(count, models, composes, constraints)
	}
	count := available
	for count > 1 && PurchaseWidthUnusable(count, models, composes, constraints) != "" {
		count--
	}
	return count, ""
}

// GPUCountUnusable rejects a machine that cannot supply the exact requested subset.
func GPUCountUnusable(available int, requested uint32) string {
	if requested > 0 && (available < 0 || uint64(available) < uint64(requested)) {
		return orchestrator.VerdictExcluded + "gpu_count_unavailable: " +
			fmt.Sprintf("requested %d GPUs; this machine has %d", requested, available)
	}
	return ""
}

// size holds every machine to the same VRAM check: the components the selection stages, or
// its whole lanes' bytes (xs-042).
func size(c *orchestrator.PlacementCandidate, models []records.ModelRef, accelerator string,
	vramGB int64, device bool, working records.WorkingPeaks, count int) {
	c.RunGPUs = count
	if accelerator == "CPU" {
		c.RunGPUs = 0
	}
	c.Models, c.Lane = models, records.Lanes(models)
	if !device {
		return
	}
	need := WithWorking(records.Resident(models), working.For(models, c.SKU, count))
	c.Fit, c.Verdict = FitNote(need, vramGB), Fit(need, vramGB, c.SKU)
}

// WithWorking applies exact total evidence only to component-staged selections.
// Other selections retain weights-plus-working.
func WithWorking(need records.Residency, peak records.WorkingPeak) records.Residency {
	need.Weights = need.Bytes
	// A total allocator peak already includes the weights resident in its scope.
	// Adding the largest unrelated scope's weights would invent co-residency.
	if need.Fit == records.FitComponents {
		if peak.TotalRuns > 0 && peak.TotalBytes > 0 {
			need.ObservedTotal, need.ObservedTotalRuns = peak.TotalBytes, peak.TotalRuns
			need.Bytes = max(need.Bytes, peak.TotalBytes)
		}
		// Legacy working peaks have no co-resident scope. Preserve them as raw
		// evidence, but do not manufacture a larger staged execution from them.
		return need
	}
	if peak.Runs == 0 {
		return need
	}
	need.Working, need.WorkingRuns = peak.Bytes, peak.Runs
	if peak.Bytes > math.MaxInt64-need.Bytes {
		need.Bytes = math.MaxInt64
	} else {
		need.Bytes += peak.Bytes
	}
	return need
}

// Fit is the one VRAM sanity floor a buy, a reuse and an explicit override are held to
// (cl-168): a device is short only when what the selection measurably needs outweighs it.
// VRAMGB is read as GiB, the unit GPU memory is built in (an "80 GB" H100 carries 81920 MiB).
//
// ONE CARD, AT EVERY WIDTH, DELIBERATELY (cl-179). VRAMGB is the per-card figure for a
// wide product as much as a narrow one, and that is the correct test: nothing here is
// tensor- or pipeline-parallel, so under a sequence-parallel group every GPU holds the
// WHOLE weights and a four-card pod fits exactly what one of its cards fits. Reading a
// width as capacity — summing it, or dividing the need by it — would buy a pod that
// cannot hold the model and only discover it after the hour was billed.
func Fit(need records.Residency, vramGB int64, name string) string {
	if need.Bytes <= vramGB<<30 {
		return ""
	}
	why := need.Need
	if need.ObservedTotalRuns > 0 {
		why = strings.TrimPrefix(why+"; ", "; ") + fmt.Sprintf("measured total %.1f GiB", gib(need.ObservedTotal))
	} else if need.WorkingRuns > 0 {
		why = strings.TrimPrefix(why+"; ", "; ") + fmt.Sprintf("measured working %.1f GiB", gib(need.Working))
	}
	return fmt.Sprintf("%s%s: needs %.1f GiB resident (%s), %s has %d GB", orchestrator.VerdictExcluded,
		orchestrator.ExcludedVRAMShort, gib(need.Bytes), why, name, vramGB)
}

// FitNote renders how the device was sized: the rule, the figures compared when there
// were any, and whether working memory was measured.
func FitNote(need records.Residency, vramGB int64) string {
	if need.Bytes == 0 {
		return need.Fit
	}
	if need.ObservedTotalRuns > 0 {
		return fmt.Sprintf("max(%s %.1f GiB, measured total %.1f GiB from %d exact runs) = %.1f GiB of %d GB",
			need.Fit, gib(need.Weights), gib(need.ObservedTotal), need.ObservedTotalRuns, gib(need.Bytes), vramGB)
	}
	if need.Fit == records.FitComponents && need.ObservedTotalRuns == 0 {
		return fmt.Sprintf("%s %.1f GiB (total memory unmeasured for this exact workload/SKU) of %d GB", need.Fit, gib(need.Weights), vramGB)
	}
	if need.WorkingRuns == 0 {
		return fmt.Sprintf("%s %.1f GiB (working memory unmeasured) of %d GB", need.Fit, gib(need.Weights), vramGB)
	}
	weights := ""
	if need.Fit != "" {
		weights = fmt.Sprintf("%s %.1f GiB + ", need.Fit, gib(need.Weights))
	}
	return fmt.Sprintf("%smeasured working %.1f GiB (%d runs) of %d GB",
		weights, gib(need.Working), need.WorkingRuns, vramGB)
}

func gib(bytes int64) float64 { return float64(bytes) / (1 << 30) }

// WidthUnusable keeps a BUY of a machine wider than one card out of the decision unless
// this request can use every card it would be billed for (cl-179). A reuse is not held to
// it: the worker runs a paid wide pod at the best declared degree that fits.
//
// A wide machine is not more capacity: every GPU of a sequence-parallel group holds the
// FULL weights, so width buys latency and never fit. The only thing that uses the extra
// cards is a group placement of exactly that degree, which needs two things this side
// knows before spending: the package's author must have declared the degree for the
// request's own models; a request that composes callees shards nothing itself, so a wide
// pod given one idles every card but the first unless a callee can use them.
//
// Constraints are advisory — a hub that will not answer yields none — so this narrows the
// decision and never widens it: with no declared degrees only one-card machines remain,
// which is exactly the behaviour before wide products existed.
func WidthUnusable(width int, composes bool, constraints Constraints) string {
	if width < 2 {
		return ""
	}
	if composes {
		return orchestrator.VerdictExcluded + orchestrator.ExcludedWidthUndeclared +
			fmt.Sprintf(": %d cards, and no callee shards over them", width)
	}
	for _, degree := range constraints.Degrees {
		if degree == width {
			return ""
		}
	}
	return orchestrator.VerdictExcluded + orchestrator.ExcludedWidthUndeclared +
		fmt.Sprintf(": %d cards, and the package declares %s", width,
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
	if constraints.SupportedPythonMinors != nil {
		supported := false
		for _, minor := range constraints.SupportedPythonMinors {
			supported = supported || profile.PythonABI == "cp"+strings.ReplaceAll(minor, ".", "")
		}
		if !supported {
			return orchestrator.VerdictExcluded + orchestrator.ExcludedBaseMismatch + ": image Python is outside the supported window"
		}
	}
	reason := launch.BaseMismatch(profile, constraints.Requirements, constraints.RequiresPython)
	if reason == "" {
		return ""
	}
	return orchestrator.VerdictExcluded + orchestrator.ExcludedBaseMismatch + ": " + reason
}

// Override renders the lane(s) the request pinned before any machine existed — an explicit
// `model.<param>=org/model@release/lane` — the lone slot's lane, or `slot=lane` pairs.
func Override(models []records.ModelRef) string {
	var parts []string
	for _, model := range models {
		if !model.Choice || model.Lane == "" {
			continue
		}
		if len(models) == 1 {
			return model.Lane
		}
		parts = append(parts, model.BindingSlot()+"="+model.Lane)
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

// Standing pins one rental the fleet already holds as a placement candidate and answers
// whether the caller must still ask its LIVE questions — what the pod is holding right
// now, and how much work is ahead of a new request. Everything decidable from the rental
// ROW alone is settled here, in the ONE order that keeps a transient state out of a
// permanent verdict (cl-185):
//
//  1. CLASS — a CPU pod cannot serve an accelerator request, ever.
//  2. GEOMETRY — what the sized lanes need outweighs its memory. That is a fact about the
//     machine and the selection, so it holds whatever the pod is doing this second.
//  3. LIFECYCLE — and only now. A hub state this build knows to be FINISHED excludes the
//     machine, naming the state. Every other state, including the whole pre-ready
//     lifecycle, is a machine on its way: VerdictAttaching, waited for (cl-170), never
//     bought around and never failed on.
//
// Asking liveness first is what cost run 412: `karam`, bought eleven minutes earlier and
// `ready` four minutes later, was excluded `not_ready` beside a genuinely dead pod, and
// the request settled FAILED on a fleet that was simply still booting.
func Standing(c *orchestrator.PlacementCandidate, models []records.ModelRef,
	row records.Rental, vramGB int64, needsAccelerator, offered, composes bool,
	constraints Constraints, disk Disk,
) bool {
	if needsAccelerator && row.AcceleratorModel == "CPU" {
		c.Verdict = orchestrator.VerdictExcluded + orchestrator.ExcludedWrongClass
		return false
	}
	// The disk the Hub bought is a fact about the machine, like its class. Too small is
	// excluded; not reported is chosen only after the rentals known to fit.
	need, retained := disk.NeedGB(), gigabytes(disk.RetainedBytes)
	if need > 0 && disk.HaveGB > 0 && disk.HaveGB-retained < need {
		held := ""
		if retained > 0 {
			held = fmt.Sprintf(" with %d GB retained", retained)
		}
		c.Verdict = fmt.Sprintf("%s%s: %d GB disk%s, the ingest needs %d GB", orchestrator.VerdictExcluded,
			orchestrator.ExcludedDiskShort, disk.HaveGB, held, need)
		return false
	}
	c.DiskUnknown = need > 0 && disk.HaveGB == 0
	// A machine the user already has up is held to the same memory floor as a buy; the
	// catalog's figure for its product is the fact (a product gone from the catalog this
	// minute decides nothing). Its GPU count is not held to the package's degrees: the
	// pod is already paid for, and the run takes the widest declared group that fits it.
	// Cards beyond that group idle; nothing refuses.
	count, verdict := runWidth(c.GPUs, models, composes, constraints)
	if c.Verdict = verdict; c.Verdict != "" {
		return false
	}
	size(c, models, row.AcceleratorModel, vramGB, needsAccelerator && offered, constraints.Working, count)
	switch {
	case c.Verdict != "":
		return false
	case records.RentalTerminalState(row.State):
		c.Verdict = orchestrator.VerdictExcluded + orchestrator.ExcludedNotReady + ": " + row.State
		return false
	case row.State == hub.RentalDegraded && row.Failure.Code == hub.RentalProviderStopped:
		// The Hub never replaces a machine its provider stopped; new work buys around it.
		c.Verdict = orchestrator.VerdictExcluded + orchestrator.ExcludedNotReady + ": stopped by provider"
		return false
	case !records.RentalReadyState(row.State) || row.Address == "" || row.CertPath == "" && row.CertificatePEM == "":
		c.Verdict = orchestrator.VerdictAttaching
		return false
	}
	return true
}

// Disk is an existing rental's container disk as the Hub reported it (0: not reported),
// the models its finished ingests retain there, and the source bytes the request ingests.
type Disk struct {
	HaveGB        int
	RetainedBytes int64
	SourceBytes   int64
}

// ingestFixedGB is the disk an ingest pod spends beside its workload: the image, the OS
// writable layer, caches and installs (the Hub's own fixed allowance).
const ingestFixedGB = 21

// NeedGB is the disk an ingest of SourceBytes needs, or 0 when it ingests nothing. The
// fetched source and the converted output built from it (at least as large) share one
// Store on the container disk.
func (d Disk) NeedGB() int {
	if d.SourceBytes <= 0 {
		return 0
	}
	return 2*gigabytes(d.SourceBytes) + ingestFixedGB
}

func gigabytes(bytes int64) int {
	if bytes <= 0 {
		return 0
	}
	return int(1 + (bytes-1)/1_000_000_000)
}

// Refusal names why nothing could be placed. The FIRST question is not which reason to
// print, it is whether the answer can change on its own (cl-185).
//
// A stock-out, and a catalog that offered no product of this class the second it was
// asked, are WEATHER: the same request placed a minute later succeeds. They exit
// Unavailable, which Orchestrator.deferUnavailable PARKS and re-asks rather than
// settling — the same treatment a fitting machine still coming up already gets, which
// never reaches here at all. Only a refusal nothing but an operator can lift is
// terminal, and exit.Capacity means what it says: a quantified shortfall against the
// physical floor. Run 412 died on `rental.no_fitting_sku` while both machines it named
// were mid-acquisition and the catalog was momentarily empty; nothing about that
// sentence was true.
func Refusal(req records.Request, d orchestrator.PlacementDecision, needsAccelerator bool,
	capped *exit.Error) *exit.Error {
	var noStock, verdicts []string
	mismatch, offered := "", 0
	for _, c := range d.Candidates {
		verdicts = append(verdicts, c.Name()+" "+c.Verdict)
		if !c.Attached() {
			offered++
		}
		if c.Verdict == orchestrator.VerdictNoStock {
			noStock = append(noStock, c.SKU)
		}
		if reason, ok := strings.CutPrefix(c.Verdict,
			orchestrator.VerdictExcluded+orchestrator.ExcludedBaseMismatch+": "); ok && mismatch == "" {
			mismatch = c.SKU + ": " + reason
		}
	}
	class := "accelerator"
	if !needsAccelerator {
		class = "CPU"
	}
	switch {
	case len(noStock) > 0:
		return exit.Named(exit.Unavailable, "rental.no_inventory",
			"every rental SKU fitting %s is out of stock right now (%s) — NOTHING was rented",
			req.Package, strings.Join(noStock, ", ")).
			WithRemedy("this is a stock-out: a run that meets it twice fails; run it again once stock returns")
	case capped != nil:
		// The one refusal that names an OPERATOR action rather than weather. Parking on
		// it would hide a misconfigured cap behind a queue that never drains.
		return capped
	case mismatch != "":
		// Refused BEFORE the paid ask, in the pod's own vocabulary. Publication stays
		// base-independent: the release is published and simply unqualified here.
		return exit.Named(exit.Unavailable, "rental.package_base_incompatible",
			"no rentable machine can run %s@%s — %s", req.Package, req.Release, mismatch).
			WithRemedy("publish a release whose requirements one of Tensorhub's offered base images satisfies")
	case offered == 0:
		// Nothing on offer to fit. Saying "no SKU fits" over an empty market is the
		// conflation this branch exists to refuse: it sends the reader to the model for
		// a fact about Tensorhub's inventory.
		return exit.Named(exit.Unavailable, "rental.catalog_empty",
			"Tensorhub offered no %s product when %s was placed; %s",
			class, req.Package, heldOrNothing(verdicts)).
			WithRemedy("this is an empty catalog: a run that meets it twice fails; run it again once Tensorhub offers one")
	}
	return exit.Named(exit.Capacity, "rental.no_fitting_sku",
		"no rental SKU on offer fits %s: %s", req.Package, strings.Join(verdicts, "; ")).
		WithRemedy("run at fewer bits (--bits=4), or choose a smaller model: model.<param>=org/model@release[/lane]")
}

// heldOrNothing renders what the fleet DID hold beside an empty catalog, so the reader
// can tell "we own nothing" from "we own two machines and none of them could take it".
func heldOrNothing(verdicts []string) string {
	if len(verdicts) == 0 {
		return "and this fleet holds no machine of its own"
	}
	return "and the machines it holds are " + strings.Join(verdicts, "; ")
}

// IdleAttached is the index of an open candidate the fleet ALREADY HOLDS with nothing
// ahead of it, or -1 — the tier's own order among those, so the choice stays the
// decision's and not an accident of row order.
//
// It is asked instead of reading the tier's overall winner when some machine is still
// coming up (cl-185). A tier may score a purchase above an attached pod; buying, or
// waiting, past a machine this fleet has already paid for and left idle is what cl-132
// and cl-174 forbid, and reading the winner alone let it happen whenever the arithmetic
// came out that way.
func IdleAttached(tier string, candidates []orchestrator.PlacementCandidate) int {
	best := -1
	for i := range candidates {
		c := candidates[i]
		if c.Verdict != "" || !c.Attached() || c.Ahead > 0 {
			continue
		}
		switch {
		case best < 0:
		case c.Measured != candidates[best].Measured:
			if !c.Measured {
				continue
			}
		case !prefers(tier, c, candidates[best]):
			continue
		}
		best = i
	}
	return best
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
// its (lane, sku, group width) under the first slot's model release, and returns the rows
// used. The width is the group the selection runs on, not the rental's card count. An
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
			if row.Release != models[0].Release || row.Lane != c.Models[0].Lane || row.SKU != c.SKU ||
				c.RunGPUs == 0 || max(row.AcceleratorCount, 1) != c.RunGPUs {
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
		if have.Lane == row.Lane && have.SKU == row.SKU && have.AcceleratorCount == row.AcceleratorCount {
			return true
		}
	}
	return false
}

// Place is the tier's choice among the open candidates and returns its index, or -1
// when none is open: `fast` the least time, `cheap` the least cost, `balanced` the least
// (time / best time) × (cost / best cost) — a candidate worse on both axes cannot win
// any tier. Ties break by attachment, queue, rate, name. Unmeasured candidates count only
// when nothing is measured, attached first and then by the fewest attempts ahead and rate,
// an idle machine before a busy one (cl-174). Every measured candidate's score is written.
func Place(tier string, candidates []orchestrator.PlacementCandidate) int {
	eligible := func(candidate orchestrator.PlacementCandidate) bool { return candidate.Verdict == "" }
	bestTime, bestCost := math.Inf(1), math.Inf(1)
	for _, c := range candidates {
		if eligible(c) && c.Measured {
			bestTime, bestCost = min(bestTime, c.TimeS), min(bestCost, float64(c.CostUSDMicros))
		}
	}
	measured := !math.IsInf(bestTime, 1)
	winner := -1
	for i := range candidates {
		c := &candidates[i]
		if !eligible(*c) || c.Measured != measured {
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
	if a.Attached() && b.Attached() && a.DiskUnknown != b.DiskUnknown {
		return !a.DiskUnknown
	}
	if a.Measured {
		if x, y := key(tier, a), key(tier, b); x != y {
			return x < y
		}
	} else if a.Attached() != b.Attached() {
		return a.Attached()
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
	reason := "cozy run " + req.ID
	if req.Package != "" {
		reason += " (" + req.Package + ")"
	}
	return reason
}
