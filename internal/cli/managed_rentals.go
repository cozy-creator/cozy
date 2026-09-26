package cli

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// managedRentals serializes fleet admission, acknowledged keepalive and release.
// Fixed idle expiry counts real work on each machine, never retained bytes or
// an open controller connection. Durable records survive daemon restarts.
type managedRentals struct {
	mu       sync.Mutex
	ctx      *Context
	layout   home.Layout
	store    *records.Store
	owner    *orchestrator.Orchestrator
	installs *rental.InstallQueue
	// retryAt holds a rental whose release the hub refused or did not answer; said holds
	// the last line printed about each rental, so the sweep speaks once per change.
	retryAt map[string]time.Time
	said    map[string]string
	closed  bool
	// unrecorded is the OTHER HALF OF THE FLEET (cl-199): the rentals the hub says
	// this account owns that this host holds no live record of. It is refreshed by
	// every reconcile from `GET /v1/rentals` (th-199) and is the only place a pod
	// this host never recorded can be counted, named, or ended.
	//
	// listingProblem is why the last reconcile could not ask. It is kept rather than
	// raised because a hub that cannot be asked is not a hub that says the account
	// owns nothing, and the difference has to reach the reader: a board that silently
	// falls back to local records is exactly the board that showed an empty fleet
	// while six H100s billed.
	unrecorded []hub.Rental
	// live is every rental the last listing says this account holds, recorded or
	// not: the fleet count and burn are the hub's, never a local sum.
	live           []hub.Rental
	listed         bool
	listingProblem *exit.Error
}

// Failed provider release is retried without changing the idle deadline.
const idleReleaseRetry = 3 * time.Minute

// totals is the reconciled fleet count and hourly burn, for a caller that spells them itself.
func (m *managedRentals) totals() (int, int64, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		return 0, 0, problem
	}
	return m.totalsLocked()
}

func (m *managedRentals) status() (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		return "", problem
	}
	return m.lineLocked()
}

// machineKey names one product at one GPU count.
type machineKey struct {
	name string
	gpus int
}

func (m *managedRentals) admit(skuName string, gpus int) (string, hub.RentalSKU, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		return "", hub.RentalSKU{}, problem
	}
	line, problem := m.lineLocked()
	if problem != nil {
		return "", hub.RentalSKU{}, problem
	}
	skus, problem := m.catalogLocked()
	if problem != nil {
		return "", hub.RentalSKU{}, problem
	}
	if sku, found := hub.FindRentalSKU(skus, skuName, gpus); found {
		return line, sku, nil
	}
	// An explicit ask is HONOURED OR REFUSED, never widened to a neighbouring card or
	// count — and the refusal has to say so out loud (cl-132), AND say which absence it
	// hit (th-150).
	return "", hub.RentalSKU{}, m.refuseSKULocked(skuName, gpus, skus)
}

// refuseSKULocked distinguishes unknown products, provider stock-outs, and
// temporary boot-failure exclusions. Reading status never acquires a rental.
func (m *managedRentals) refuseSKULocked(skuName string, gpus int, skus []hub.RentalSKU) *exit.Error {
	hctx, cancel := hub.Context()
	status, problem := client(m.ctx).RentalSKUStatus(hctx, skuName, gpus)
	cancel()
	if problem != nil {
		// The lookup is an EXPLANATION, never the refusal itself: a hub that fails
		// the read must still get a refusal about the SKU rather than an error
		// about the lookup.
		return SKURefusal(skuName, gpus, skus, nil, time.Now())
	}
	return SKURefusal(skuName, gpus, skus, &status, time.Now())
}

// SKURefusal composes the refusal for a (name, GPU count) the live catalog does not
// carry. `status` is the hub's answer about that ask, or nil when it could not be asked.
// It is a pure function of what was observed so that the words — which are the entire
// deliverable of th-150 — can be asserted without a hub.
func SKURefusal(skuName string, gpus int, skus []hub.RentalSKU, status *hub.RentalSKUStatus,
	now time.Time,
) *exit.Error {
	ask := orchestrator.MachineLabel(skuName, gpus)
	retry := rentNewCommand(skuName, gpus)
	said := fmt.Sprintf("NOTHING was rented and no other machine was substituted for %q", ask)
	switch {
	case status == nil:
		return exit.Named(exit.Validation, "rental.sku_unavailable",
			"no rental SKU %q is on offer right now — %s", ask, said).
			WithRemedy("the catalog is live provider inventory, so a name absent now may "+
				"return within minutes; `cozy rental new` alone lists what is offered "+
				"this minute (currently %s)", offeredNames(skus)).
			WithNext("cozy rental new")
	case !status.Known:
		return exit.Named(exit.Validation, "rental.sku_unknown",
			"Tensorhub sells no rental SKU named %q — %s", skuName, said).
			WithRemedy("choose one of the names it does sell: %s", offeredNames(skus)).
			WithNext("cozy rental new")
	case status.Offered:
		return exit.Named(exit.Capacity, "rental.sku_availability_changed",
			"%s became available after the catalog was read. Please retry the rental request.", ask).
			WithNext(retry, "cozy rental new")
	}
	message := fmt.Sprintf("Sorry, but our GPU providers have no inventory for %s right now.", ask)
	switch status.UnavailableReason {
	case "cooldown":
		message = fmt.Sprintf("%s is temporarily excluded after a recent boot failure.", ask)
		if status.RetryAfter != nil {
			message += " Retry after " + status.RetryAfter.UTC().Format(time.RFC3339) + "."
		} else {
			message += " Please try again shortly."
		}
		return exit.Named(exit.Capacity, "rental.sku_cooldown", "%s", message).
			WithNext(retry, "cozy rental new")
	case "retry_in_progress":
		return exit.Named(exit.Capacity, "rental.sku_retry_in_progress",
			"%s is temporarily unavailable while one retry after a recent boot failure is in progress. "+
				"Please try again once it finishes.", ask).
			WithNext(retry, "cozy rental new")
	}
	if status.LastSeenAt != nil {
		message += fmt.Sprintf(" %s was last available at %s (%s ago).", ask,
			status.LastSeenAt.UTC().Format(time.RFC3339),
			roughDuration(now.Sub(*status.LastSeenAt)))
	}
	next := []string{retry, "cozy rental new"}
	if status.SKU != nil && len(status.SKU.Widths) > 0 {
		message += fmt.Sprintf(" %s is offered now with %s GPU(s).", skuName, joinInts(status.SKU.Counts()))
		next = []string{retry, rentNewCommand(skuName, status.SKU.Widths[0].AcceleratorCount)}
	}
	return exit.Named(exit.Capacity, "rental.sku_out_of_stock",
		"%s Please try again later or rent a different GPU.", message).
		WithNext(next...)
}

// rentNewCommand is the `cozy rental new` line for one product at one count.
func rentNewCommand(skuName string, gpus int) string {
	if gpus > 1 {
		return fmt.Sprintf("cozy rental new %s --gpus %d", skuName, gpus)
	}
	return "cozy rental new " + skuName
}

func joinInts(values []int) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return strings.Join(parts, ", ")
}

// offeredNames is the live catalog as a reader can scan it, so a refusal shows the shape
// of the market it was refused against rather than asserting a bare absence.
func offeredNames(skus []hub.RentalSKU) string {
	products := hub.RentalProducts(skus)
	if len(products) == 0 {
		return "none"
	}
	names := make([]string, 0, len(products))
	for _, product := range products {
		name := product.Name
		if counts := product.Counts(); len(counts) > 1 || len(counts) == 1 && counts[0] != 1 {
			name += " (" + joinInts(counts) + " GPUs)"
		}
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) > 12 {
		return strings.Join(names[:12], ", ") +
			fmt.Sprintf(" and %d more", len(names)-12)
	}
	return strings.Join(names, ", ")
}

// acquire is the placement decision for a --rental request no rental holds a placement
// for (placement-economics.md, cl-165). The candidates are the fleet's ready rentals —
// the cl-132 exclusions still apply — and the catalog's purchasable products, each
// pinned to the lane its rung of the binding ladder selects (or the lane the request
// pinned itself) and priced from the live catalog; the model's published throughput rows
// give each its expected time and cost, and placement.prefer picks. --rental is
// permission AND intent to spend (owner ruling 2026-09-03). The chosen machine is pinned
// to THIS request alone, with the lane its rung names, in one write; a buy the hub
// refuses for stock drops that product and the choice repeats. A fitting rental whose
// worker has not attached yet is waited for, never bought around (cl-170) nor queued
// around (cl-174): the decision returns with no rental and the fleet's next observation
// re-asks. A refusal returns its record too, so every decision is durable.
func (m *managedRentals) acquire(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
	var none orchestrator.PlacementDecision
	m.mu.Lock()
	defer m.mu.Unlock()
	if !req.Rental || req.Worker != "" {
		return none, "", exit.Internalf("request %s is not an unassigned --rental request", req.ID)
	}
	resolver := NewResolver(m.store, m.ctx.Cfg, nil)
	needsAccelerator, problem := resolver.PrivateRentalNeedsAccelerator(req)
	if problem != nil {
		return none, "", problem
	}
	// CPU orchestration requests need their captured defaults to narrow the rental
	// choice (cl-210). Accelerator-owning requests retain only their own model slots;
	// unrelated captures must not inflate their residency or preparation selection.
	childModels, problem := resolver.UnpublishedChildModels(req)
	if req.InstallID == "" {
		childModels, problem = resolver.PublishedChildModels(m.ctx, req)
	}
	if problem != nil {
		return none, "", problem
	}
	req.Models = append(append([]records.ModelRef(nil), req.Models...), childModels...)
	needsAccelerator = needsAccelerator || len(childModels) > 0
	if problem := m.reconcileLocked(); problem != nil {
		return none, "", problem
	}
	skus, problem := m.catalogLocked()
	if problem != nil {
		return none, "", problem
	}
	bySKU := make(map[machineKey]hub.RentalSKU, len(skus))
	for _, sku := range skus {
		bySKU[machineKey{sku.Name, sku.AcceleratorCount}] = sku
	}
	decision := orchestrator.PlacementDecision{Tier: m.ctx.Cfg.PlacementPrefer,
		ConfigDigest: m.ctx.Cfg.Digest, Ladder: rental.Ladder(req.Models), Override: rental.Override(req.Models)}
	// ONE READING OF THE RELEASE for both halves of the decision: a machine already up and
	// a machine that would be bought are held to the same declared degrees (cl-179).
	constraints, constraintProblem := RentalConstraints(m.ctx, req)
	if constraintProblem != nil {
		return none, "", constraintProblem
	}
	if constraints.Working, problem = m.store.WorkingPeaks(req); problem != nil {
		return none, "", problem
	}
	attached, problem := m.attachedLocked(req, bySKU, needsAccelerator, constraints)
	if problem != nil {
		return none, "", problem
	}
	var purchases []orchestrator.PlacementCandidate
	if req.RequestedRental == "" {
		purchases = rental.Purchases(skus, req.Models, needsAccelerator, req.IsJob(), constraints)
	}
	var capped *exit.Error
	decision.Candidates = append(attached, purchases...)
	rows, problem := m.throughputLocked(req)
	if problem != nil {
		return none, "", problem
	}
	decision.Throughput = rental.Measure(decision.Candidates, rows, req.Models)
	for {
		i := rental.Place(decision.Tier, decision.Candidates)
		if w := rental.Attaching(decision.Candidates); w >= 0 &&
			(i < 0 || !decision.Candidates[i].Attached() || decision.Candidates[i].Ahead > 0) {
			// An idle attached rental is taken now; a buy, or a queue behind a busy one,
			// is not chosen over the machine attaching: the request parks unpinned and
			// goes to whichever is free first.
			//
			// The idle machine is asked for by NAME, not read off the tier's winner
			// (cl-185). A tier is free to score a purchase above a pod this fleet has
			// already bought, attached and left idle; waiting on a third machine because
			// of that arithmetic is the head-of-line shape cl-132 and cl-174 both forbid.
			idle := rental.IdleAttached(decision.Tier, decision.Candidates)
			if idle < 0 {
				rental.Wait(decision.Candidates, w)
				line, problem := m.lineLocked()
				return decision, line, problem
			}
			i = idle
		}
		if i < 0 {
			if req.RequestedRental != "" {
				return decision, "", exit.Named(exit.Conflict, "rental.selection_unavailable", "selected rental %s cannot accept this request: %s", req.RequestedRental, decision.Line())
			}
			return decision, "", rental.Refusal(req, decision, needsAccelerator, capped)
		}
		c := &decision.Candidates[i]
		rentalID := c.Rental
		if !c.Attached() {
			row, problem := m.buyLocked(req, *c, bySKU[machineKey{c.SKU, c.GPUs}])
			if problem != nil && problem.ErrName() == "rental.fleet_spend_cap" {
				// The hub's cap refused this product; a cheaper one may still fit.
				c.Verdict, capped = orchestrator.VerdictExcluded+problem.ErrName(), problem
				continue
			}
			if problem != nil {
				if problem.ErrName() != "rental.sku_out_of_stock" && problem.ErrName() != "rental.sku_unavailable" {
					return none, "", problem
				}
				c.Verdict = orchestrator.VerdictNoStock
				fmt.Fprintf(m.ctx.Out, "rentals: %s has no inventory; choosing again without it\n", c.Name())
				continue
			}
			rentalID, decision.Bought = row.ID, true
		}
		rental.Conclude(decision.Candidates, i)
		decision.RentalID, decision.Models = rentalID, c.Models
		if !decision.Bought && c.Ahead > 0 && req.RequestedRental == "" && !req.RetainWork {
			// Preparation may use this candidate, but its occupied seat is not an
			// assignment. The local queue can still take another ready rental;
			// dispatch records the chosen worker when it reserves a free seat.
			line, problem := m.lineLocked()
			return decision, line, problem
		}
		pinned, problem := m.store.PinRental(req.ID, rentalID, c.Models)
		if problem != nil {
			return none, "", problem
		}
		if !pinned {
			if !decision.Bought {
				return none, "", exit.New(exit.Canceled, "request %s settled before rental assignment", req.ID)
			}
			if _, releaseProblem := m.releaseLocked(rentalID); releaseProblem != nil {
				return none, "", releaseProblem
			}
			return none, "", exit.New(exit.Canceled,
				"request %s settled while rental %s was starting; the rental was released", req.ID, rentalID)
		}
		line, problem := m.lineLocked()
		return decision, line, problem
	}
}

// attachedLocked is every rental the fleet holds as a candidate: with the reason it
// cannot take this request, or, when it can, the attempts ahead of a new one. Nothing
// is silently dropped (cl-132): a decision reporting no attached candidate while the
// fleet holds two is the shape that read as waste live.
func (m *managedRentals) attachedLocked(req records.Request, bySKU map[machineKey]hub.RentalSKU,
	needsAccelerator bool, constraints rental.Constraints) ([]orchestrator.PlacementCandidate, *exit.Error) {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return nil, problem
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].HourlyRateUSDMicros != rows[j].HourlyRateUSDMicros {
			return rows[i].HourlyRateUSDMicros < rows[j].HourlyRateUSDMicros
		}
		return rows[i].ID < rows[j].ID
	})
	out := make([]orchestrator.PlacementCandidate, 0, len(rows))
	for _, row := range rows {
		if req.RequestedRental != "" && row.ID != req.RequestedRental {
			continue
		}
		c := orchestrator.PlacementCandidate{Rental: row.ID, Machine: row.MachineName, SKU: row.SKU,
			GPUs: row.AcceleratorCount, RateUSDMicrosPerHour: row.HourlyRateUSDMicros}
		sku, offered := bySKU[machineKey{row.SKU, row.AcceleratorCount}]
		if offered {
			c.RateUSDMicrosPerHour = sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
		}
		// Everything decidable from the rental ROW is settled by the chooser, in the one
		// order that keeps a transient state out of a permanent verdict (cl-185). What
		// is left are the questions only this host can answer.
		if rental.Standing(&c, req.Models, row, sku.VRAMGB, needsAccelerator, offered, req.IsJob(), constraints.Working) {
			if len(constraints.Requirements) > 0 || constraints.RequiresPython != "" {
				if problem := rentalCompatibility(m.ctx, row.ID, constraints); problem != nil {
					c.Verdict = orchestrator.VerdictExcluded + orchestrator.ExcludedBaseMismatch + ": " + problem.Message
					out = append(out, c)
					continue
				}
			}

			reason, problem := m.standingLocked(row, req)
			if problem != nil {
				return nil, problem
			}
			if reason != "" {
				c.Verdict = orchestrator.VerdictExcluded + reason
			} else {
				mode, held := m.owner.RentalStanding(row.ID, req.IsJob())
				queued, problem := m.store.RentalQueueAhead(row.ID, req.ID)
				if problem != nil {
					return nil, problem
				}
				c.Ahead = queued + held
				if mode != "" {
					c.Verdict = orchestrator.VerdictExcluded + mode
				}
			}
		}
		out = append(out, c)
	}
	return out, nil
}

// standingLocked is the fleet-side reason an ATTACHED rental cannot take any new
// placement, or "". Class, geometry and lifecycle are settled by the caller before this
// is asked, so every answer here is a fact about what the pod is already holding.
func (m *managedRentals) standingLocked(row records.Rental, req records.Request) (string, *exit.Error) {
	retained, problem := m.store.RentalHasRetainedJob(row.ID)
	if problem != nil {
		return "", problem
	}
	// Retained client jobs deliberately share this machine's durable work.
	// Active execution is fenced separately by RentalStanding and preparation.
	if retained && (!req.IsJob() || !req.RetainWork) {
		return orchestrator.ExcludedModeConflict, nil
	}
	spent, problem := orchestrator.RentalSpent(m.store, row)
	if problem != nil {
		return "", problem
	}
	if spent {
		return orchestrator.ExcludedSpent, nil
	}
	return "", nil
}

// throughputLocked reads the model's published throughput rows once per decision. A
// request binding no model has nothing to look up.
func (m *managedRentals) throughputLocked(req records.Request) ([]hub.ModelThroughput, *exit.Error) {
	if len(req.Models) == 0 {
		return nil, nil
	}
	ref, problem := hub.ParseRef(req.Models[0].Model)
	if problem != nil {
		return nil, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	return client(m.ctx).ModelThroughput(hctx, ref)
}

// buyLocked is one paid ask for the chosen product. The request's selection is pinned to
// that machine's lane BEFORE the ask, so the rental POST declares the lane the pod will
// hold (th-155), and the boot is named while it passes (cl-121).
func (m *managedRentals) buyLocked(req records.Request, c orchestrator.PlacementCandidate,
	sku hub.RentalSKU) (records.Rental, *exit.Error) {
	current, problem := m.store.RequestRow(req.ID)
	if problem != nil {
		return records.Rental{}, problem
	}
	if current == nil || settledRequest(current.State) {
		return records.Rental{}, exit.New(exit.Canceled, "request %s settled before rental acquisition", req.ID)
	}
	if problem := m.store.PinRequestModels(req.ID, c.Models); problem != nil {
		return records.Rental{}, problem
	}
	fmt.Fprintf(m.ctx.Out, "rentals: renting %s at %s (%s)\n", orchestrator.MachineLabel(sku.Name, sku.AcceleratorCount), skuRate(sku), pinText(c))
	m.owner.ObservePhase(req.ID, orchestrator.PhaseSample{Name: orchestrator.PhaseAcquiring})
	defer m.owner.ForgetPhase(req.ID)
	operationKey, problem := m.store.ManagedRentalOperationKey(req.ID)
	if problem != nil {
		return records.Rental{}, problem
	}
	row, _, _, problem := acquireRental(m.ctx, m.layout, m.store, sku.Name, sku.AcceleratorCount,
		operationKey, rental.AcquisitionReason(req),
		sku.PriceUSDMicrosPerHour, time.Time{}, req.ID,
		func(seen hub.Rental) {
			// A failure carried by a rental that is BACK in pending_acquisition is the
			// hub saying "that one did not work; I am buying again". The detail names
			// what refused, so a replan is visible AND attributable rather than being
			// 32 silent seconds inside a longer silence.
			retrying := seen.Failure != nil
			detail := seen.Detail
			if retrying && seen.Failure.Code != "" {
				detail = seen.Failure.Code
				if seen.Failure.ProviderHostID != "" {
					detail += " on " + seen.Failure.ProviderHostID
				}
			}
			if name := orchestrator.PhaseOfHubRental(seen.State, seen.ProviderState,
				seen.ContainerState, retrying); name != "" {
				m.owner.ObservePhase(req.ID, orchestrator.PhaseSample{
					Name: name, Machine: seen.Name, Detail: detail,
					Rental: &orchestrator.RentalProgress{
						AcceleratorModel:      seen.AcceleratorModel,
						AcceleratorCount:      seen.AcceleratorCount,
						HourlyRateUSDMicros:   seen.HourlyRateUSDMicros,
						BaseWorkerImageDigest: seen.BaseWorkerImageDigest,
						BaseWorkerImageTag:    seen.BaseWorkerImageTag,
						BaseWorkerProfile:     seen.BaseWorkerProfile,
					}})
			}
		})
	return row, problem
}

// pinText is a candidate's pin for a log line; the rung is named when the ladder chose.
func pinText(c orchestrator.PlacementCandidate) string {
	text := fmt.Sprintf("lane %s, fit %s", orNone(c.Lane), orNone(c.Fit))
	if c.Rung > 0 {
		return fmt.Sprintf("rung %d, %s", c.Rung, text)
	}
	return text
}

func (m *managedRentals) catalogLocked() ([]hub.RentalSKU, *exit.Error) {
	hctx, cancel := hub.Context()
	defer cancel()
	return client(m.ctx).RentalSKUs(hctx)
}

// release is the settlement hook: the orchestrator calls it as each request pinned to a
// rental settles, so a rental whose reason is spent goes at once and one that is merely
// idle is logged with its deadline. The sweep repeats the same observation from then on.
func (m *managedRentals) release(id string) (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return "", problem
	}
	if row == nil {
		return m.lineLocked()
	}
	return m.observeLocked(*row)
}

// Explicit transaction abandonment is not an idle observation. A manually held
// rental stays reserved; the existing releaseLocked guard checks every other
// owner before any managed rental reaches the provider DELETE.
func (m *managedRentals) releaseRetained(id string) (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return "", problem
	}
	if row == nil || row.ManagedRequestID == "" {
		return m.lineLocked()
	}
	if strings.TrimRight(row.Hub, "/") != client(m.ctx).Base() {
		return "", exit.Named(exit.Conflict, "rental.hub_mismatch", "retained rental belongs to a different Tensorhub authority")
	}
	return m.releaseLocked(id)
}

// releaseOrphaned resumes the policy after a daemon restart: every rental is reconciled
// with the hub and then observed as the sweep observes it, so a job's rental releases
// now and a warm one keeps only the unspent remainder of its grace.
func (m *managedRentals) releaseOrphaned() {
	m.mu.Lock()
	if problem := m.reconcileLocked(); problem != nil {
		fmt.Fprintf(m.ctx.Out, "rental reconciliation deferred: %s\n", problem.Message)
		m.mu.Unlock()
		return
	}
	m.sweepLocked()
	owner := m.owner
	m.mu.Unlock()
	if owner != nil {
		owner.RecoverLostWork()
	}
}

// hubReconcileCadence is how often the idle loop re-asks Tensorhub what each rental IS,
// rather than what this host last recorded about it.
//
// It is a POLL INTERVAL and not a deadline. Nothing is killed, released, or reclaimed
// because it elapsed; the only thing a tick does is ask a question, and every decision is
// still made from the answer.
//
// The sweep needs it because the daemon cannot learn a rental died any other way. Tensorhub
// owns provider reclaim, and a rental that fails AFTER it was serving reaches this host
// through no other channel — the worker control stream dying looks the same as a network
// blip, which is why it is not the signal. Every other reconcile is driven by a rental verb
// or by an acquisition (`status`, `admit`), and those are exactly the events a rental
// holding stranded work cannot cause: its requests are pinned, so they never reach
// `selectOrStart` and never ask the fleet anything. Until this, the loop read local records
// only, so a rental that failed while holding work was re-observed only if some UNRELATED
// work happened to arrive and poll the hub for its own reasons.
//
// Ten seconds is chosen against the dispatch retry cadence, not picked round: a pinned
// request re-asks its rental for a seat roughly every 20 s, so noticing at 10 s means the
// daemon never spends a whole retry believing in a machine the hub has already reclaimed.
// The cost is one GET per rental — a fleet is a handful of pods, not a datacenter.
const hubReconcileCadence = 10 * time.Second

// watch is the idle release's own loop. It re-reads the records at pollCadence — the
// resolution every rental verb already samples a rental at — and acts only on what they
// say; the grace is the debounce, and a sample that finds nothing to do costs a few local
// reads. It returns when quit closes.

func (m *managedRentals) watch(quit <-chan struct{}) {
	tick := time.NewTicker(pollCadence)
	defer tick.Stop()
	reconciled := time.Now()
	for {
		select {
		case <-quit:
			return
		case <-tick.C:
		}
		m.mu.Lock()
		if !m.closed {
			if time.Since(reconciled) >= hubReconcileCadence {
				reconciled = time.Now()
				if problem := m.reconcileLocked(); problem != nil {
					m.sayLocked("", "rental reconciliation deferred: "+problem.Message)
				}
			}
			m.sweepLocked()
		}
		owner := m.owner
		m.mu.Unlock()
		if m.installs != nil {
			m.installs.Wake()
		}
		if owner != nil {
			// OUTSIDE the lock, always: replanning released work asks the fleet for
			// capacity and re-enters this object. Under the lock it deadlocks the daemon,
			// measured on the first cut of this change. It runs every tick rather than on
			// a rental observation, because a rental whose record is GONE is observed by
			// nobody — the sweep reads the request side, which still has rows.
			owner.RecoverLostWork()
			// Hub recovery is an observed fleet change. Re-ask durable queued work outside
			// the fleet lock; selection calls back into this object.
			owner.WakeQueue()
		}
	}
}

func (m *managedRentals) sweepLocked() {
	m.sayUnrecordedLocked()
	rows, problem := m.store.Rentals()
	if problem != nil {
		m.sayLocked("", "idle release deferred: "+problem.Message)
		return
	}
	for _, row := range rows {
		if _, problem := m.observeLocked(row); problem != nil {
			m.sayLocked(row.ID, fmt.Sprintf("rental %s release deferred: %s; retrying every %s",
				row.ID, problem.Message, idleReleaseRetry))
		}
	}
}

// observeLocked is the whole idle-release decision for one rental, made from the records
// alone and acted on through the same paid path `cozy rental end` takes. A failed release
// is asked again after idleReleaseRetry, and each change of verdict is said once.
func (m *managedRentals) observeLocked(row records.Rental) (string, *exit.Error) {
	if m.closed || time.Now().Before(m.retryAt[row.ID]) {
		return m.lineLocked()
	}
	if base := client(m.ctx).Base(); strings.TrimRight(row.Hub, "/") != base {
		m.sayLocked(row.ID, fmt.Sprintf("rental %s was rented from %s, not %s; not idle-released here",
			row.ID, row.Hub, base))
		return m.lineLocked()
	}
	idle, problem := m.observeIdle(row)
	if problem != nil {
		return "", problem
	}
	due, eligible := idle.ReleaseAt()
	if !eligible {
		delete(m.said, row.ID)
		return m.lineLocked()
	}
	if now := time.Now(); now.Before(due) {
		m.sayLocked(row.ID, fmt.Sprintf("rental %s (%s) idle since %s; released at %s unless work arrives",
			row.ID, row.MachineName, idle.Since.UTC().Format("15:04:05"), due.UTC().Format("15:04:05")))
		return m.lineLocked()
	}
	line, problem := m.releaseIdleLocked(row.ID)
	if problem != nil {
		if m.retryAt == nil {
			m.retryAt = map[string]time.Time{}
		}
		m.retryAt[row.ID] = time.Now().Add(idleReleaseRetry)
		return "", problem
	}
	remaining, problem := m.store.RentalRow(row.ID)
	if problem != nil {
		return "", problem
	}
	if remaining != nil {
		return line, nil
	}
	m.forgetIdleLocked(row.ID)
	fmt.Fprintf(m.ctx.Out, "rental %s (%s) released after %s idle\n",
		row.ID, row.MachineName, time.Since(idle.Since).Round(time.Second))
	return line, nil
}

func (m *managedRentals) sayLocked(id, line string) {
	if m.said[id] == line {
		return
	}
	if m.said == nil {
		m.said = map[string]string{}
	}
	m.said[id] = line
	fmt.Fprintln(m.ctx.Out, line)
}

func (m *managedRentals) forgetIdleLocked(id string) {
	delete(m.said, id)
	delete(m.retryAt, id)
}

func (m *managedRentals) close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
}

// releaseLocked is the paid DELETE, re-observing the row under the lock first: the sweep
// and the settlement hook both arrive here, and only a pod with nothing queued, running,
// or owed on it goes. The owed re-check keeps the buy's rollback honest too: acquire
// releases a rental it just bought only because the buyer settled or routed elsewhere,
// which is exactly when the debt is gone.
func (m *managedRentals) releaseLocked(id string) (string, *exit.Error) {
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return "", problem
	}
	if row == nil {
		return m.lineLocked()
	}
	queued, running, problem := m.store.RentalRunCounts(id)
	if problem != nil {
		return "", problem
	}
	if queued != 0 || running != 0 {
		return m.lineLocked()
	}
	owed, problem := orchestrator.RentalOwedBy(m.store, *row)
	if problem != nil {
		return "", problem
	}
	if owed {
		return m.lineLocked()
	}
	return m.releasePaidLocked(*row)
}

// Idle expiry intentionally ignores retained custody: paused/failed files are
// not work. The capacity/purpose cleanup above retains its stronger custody gate.
func (m *managedRentals) releaseIdleLocked(id string) (string, *exit.Error) {
	claimed, problem := m.store.ClaimRentalIdleRelease(id, time.Now())
	if problem != nil {
		return "", problem
	}
	if !claimed {
		return m.lineLocked()
	}
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return "", problem
	}
	if row == nil {
		return m.lineLocked()
	}
	return m.releasePaidLocked(*row)
}

func (m *managedRentals) releasePaidLocked(row records.Rental) (string, *exit.Error) {
	id := row.ID
	operationKey, problem := m.store.RequestRentalRelease(id)
	if problem != nil {
		return "", problem
	}
	confirmed := row.State == hub.RentalReleased
	if operationKey != "" {
		operation, problem := m.store.RentalOperation(operationKey)
		if problem != nil {
			return "", problem
		}
		confirmed = confirmed || operation != nil && operation.State == hub.RentalReleased
	}
	if !confirmed && row.State != hub.RentalReleaseRequested {
		row.State = hub.RentalReleaseRequested
		if problem := m.store.RecordRental(row); problem != nil {
			return "", problem
		}
		if line, lineProblem := m.lineLocked(); lineProblem == nil {
			fmt.Fprintln(m.ctx.Out, line)
		}
	}
	if !confirmed {
		hctx, cancel := hub.Context()
		problem = client(m.ctx).Release(hctx, id, "")
		cancel()
		if problem != nil {
			if problem.Code == exit.NotFound {
				return "", rentalReleaseUnconfirmed(id)
			}
			return "", problem
		}
	}
	for !confirmed {
		hctx, cancel := hub.Context()
		remote, observed := client(m.ctx).Rental(hctx, id)
		cancel()
		switch {
		case observed == nil && remote.State == hub.RentalReleased:
			confirmed = true
		case observed == nil:
			row.State = remote.State
			copyRentalFailure(&row, remote)
			if update := m.store.RecordRental(row); update != nil {
				return "", update
			}
			time.Sleep(pollCadence)
		case observed.Code == exit.NotFound:
			return "", rentalReleaseUnconfirmed(id)
		case transient(observed):
			time.Sleep(pollCadence)
		default:
			return "", observed
		}
	}
	if m.owner != nil {
		m.owner.DetachRental(id)
	}
	if _, problem := rental.Forget(m.layout, m.store, id); problem != nil {
		return "", problem
	}
	if operationKey != "" {
		rental.ForgetPending(m.layout, operationKey)
	}
	return m.lineLocked()
}

// reconcileLocked converges this host's rental rows with the hub, and then asks the
// hub the question this host cannot answer from its own rows at all: what else does
// this account own? The two directions are not the same read and only one of them
// existed. Local-row reconciliation can correct a row; it can never notice a pod
// that has no row.
func (m *managedRentals) reconcileLocked() *exit.Error {
	if problem := m.reconcileRowsLocked(); problem != nil {
		if hub.Unanswered(problem) {
			// The census is as unknown as the rows: a later cached read must not
			// present the previous listing as current.
			m.listed, m.listingProblem, m.unrecorded, m.live = false, problem, nil, nil
		}
		return problem
	}
	m.reconcileListingLocked()
	return nil
}

// reconcileListingLocked refreshes the unrecorded set. A listing this hub does not
// publish, or cannot answer right now, leaves the set EMPTY and the reason recorded
// — never an assertion that there is nothing there.
func (m *managedRentals) reconcileListingLocked() {
	hctx, cancel := hub.Context()
	remote, listed, problem := client(m.ctx).Rentals(hctx)
	cancel()
	m.listed, m.listingProblem, m.unrecorded, m.live = listed, problem, nil, nil
	if problem != nil || !listed {
		return
	}
	for _, seen := range remote {
		if hub.RentalAbsent(seen.State) {
			continue
		}
		row, rowProblem := m.store.RentalRow(seen.ID)
		if rowProblem != nil {
			m.listingProblem = rowProblem
			m.unrecorded, m.live = nil, nil
			return
		}
		m.live = append(m.live, seen)
		if row != nil && strings.TrimRight(row.Hub, "/") == client(m.ctx).Base() {
			continue
		}
		m.unrecorded = append(m.unrecorded, seen)
	}
}

// sayUnrecordedLocked is the DAEMON's alarm, and it belongs to the sweep rather than
// to the reconcile because the reconcile also runs under a CLI verb, whose stdout is
// a machine document that a log line would corrupt. The daemon has no reader watching
// a board, so a machine billing outside its records has to reach the log by itself —
// once per machine, at the sweep's own cadence, stating the missing local activity.
func (m *managedRentals) sayUnrecordedLocked() {
	for _, seen := range m.unrecorded {
		m.sayLocked("hub:"+seen.ID, fmt.Sprintf(
			"rental %s (%s) is %s at %s and this host holds no record of it; activity is unknown to this controller",
			seen.ID, seen.Name, humanRentalState(seen.State), usdPerHour(seen.HourlyRateUSDMicros)))
	}
}

func (m *managedRentals) reconcileRowsLocked() *exit.Error {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return problem
	}
	operations, problem := m.store.ActiveRentalOperations()
	if problem != nil {
		return problem
	}
	pending := map[string]records.RentalOperation{}
	for _, operation := range operations {
		if operation.ManagedRequestID == "" && operation.RentalID != "" &&
			operation.Hub == client(m.ctx).Base() && operation.State != "attached" &&
			operation.State != hub.RentalFailed && operation.State != hub.RentalReleaseRequested {
			pending[operation.RentalID] = operation
		}
	}
	for _, row := range rows {
		if row.State == hub.RentalReleased {
			m.forgetIdleLocked(row.ID)
			if m.owner != nil {
				m.owner.DetachRental(row.ID)
			}
			if _, problem := rental.Forget(m.layout, m.store, row.ID); problem != nil {
				return problem
			}
			continue
		}
		if strings.TrimRight(row.Hub, "/") != client(m.ctx).Base() {
			continue
		}
		hctx, cancel := hub.Context()
		remote, observed := client(m.ctx).Rental(hctx, row.ID)
		cancel()
		if observed != nil {
			if observed.ErrName() == "rental.not_found" {
				return exit.Named(exit.Conflict, "rental.hub_record_missing",
					"Tensorhub no longer has rental %s, but this host still has its non-released record", row.ID).
					WithRemedy("reconcile Tensorhub with its provider before retrying; do not forget this rental or rent replacement capacity until its provider resource is confirmed released")
			}
			return observed
		}
		// Adopt the hub's reconciled billed rate (th-120): the burn this host
		// reports and caps on must be what the provider actually charges.
		if remote.HourlyRateUSDMicros > 0 {
			row.HourlyRateUSDMicros = remote.HourlyRateUSDMicros
		}
		if remote.State == hub.RentalReleased {
			m.forgetIdleLocked(row.ID)
			if m.owner != nil {
				m.owner.DetachRental(row.ID)
			}
			if _, problem := rental.Forget(m.layout, m.store, row.ID); problem != nil {
				return problem
			}
			continue
		}
		if operation, unfinished := pending[row.ID]; unfinished && remote.Attachable() {
			// A foreground acquisition may have completed or released this operation
			// while the GET was in flight. Its newer durable state wins.
			current, problem := m.store.RentalOperation(operation.Key)
			if problem != nil {
				return problem
			}
			if current == nil || current.State == "attached" || current.State == hub.RentalReleased ||
				current.State == hub.RentalFailed || current.State == hub.RentalReleaseRequested {
				continue
			}
			token, creator, problem := rental.RetainedAcquisitionCredentials(m.layout, *current)
			if problem != nil {
				return problem
			}
			if _, problem := finishRentalAttachment(m.layout, m.store, row, remote, operation.Key, token, creator); problem != nil {
				return problem
			}
			if m.owner != nil {
				// This method starts the existing reconnect loop asynchronously. No
				// owner/queue work runs while this goroutine holds the fleet lock.
				m.owner.ResumeRentalControl(row.ID)
			}
			continue
		}
		row.State = remote.State
		copyRentalFailure(&row, remote)
		if problem := m.store.RecordRental(row); problem != nil {
			return problem
		}
		m.detachLostRentalLocked(row)
	}
	return nil
}

// recoverLostWorkLocked hands a terminally failed rental's pinned work back to routing.
//
// This sweep is where the daemon LEARNS a rental died — it is the only place that reads
// the hub's verdict for a rental that was already serving — and until now it recorded that
// verdict and did nothing else. The requests pinned to the pod stayed pinned to it: never
// routed, because they named a machine no worker would ever answer for, and never
// released, because `RentalRunCounts` counted them and fenced the idle release. Meanwhile
// the fleet bought a second pod for a later request, served it, and released it while the
// stranded three still waited (observed live 2026-09-04, rental pr-183abac284d1e16f5f0a).
//
// detachLostRentalLocked drops the worker record of a rental the hub has failed, so no
// replan can choose the corpse: `rentalHeld` reads live worker records and a stale one
// would make `selectOrStart` decide the dead rental still holds the placement.
//
// Detaching is ALL that happens under the fleet lock. Handing the work back re-enters this
// object through the capacity question and deadlocks the daemon under it (measured), and it
// is the loop's request-side sweep that does it — a rental whose record is gone is observed
// by nobody, so the work has to be found from the side that still has rows.
func (m *managedRentals) detachLostRentalLocked(row records.Rental) {
	if row.State != hub.RentalFailed || m.owner == nil {
		return
	}
	m.owner.DetachRental(row.ID)
}

func (m *managedRentals) lineLocked() (string, *exit.Error) {
	count, burn, problem := m.totalsLocked()
	if problem != nil {
		return "", problem
	}
	machine := "machine"
	if count != 1 {
		machine = "machines"
	}
	line := fmt.Sprintf("rentals: %d remote %s running · %s", count, machine, usdPerHour(burn))
	return line, nil
}

// totalsLocked is what THE ACCOUNT is paying, as the hub lists it: every live
// rental, recorded here or not (cl-199).
func (m *managedRentals) totalsLocked() (int, int64, *exit.Error) {
	if m.listingProblem != nil {
		return 0, 0, m.listingProblem
	}
	if !m.listed {
		return 0, 0, exit.Named(exit.Unavailable, "rental.list_unavailable",
			"account rental census unavailable: this hub publishes no rental listing").
			WithRemedy("restore the Hub account-listing route before reading totals or acquiring a rental")
	}
	var burn int64
	for _, seen := range m.live {
		if seen.HourlyRateUSDMicros <= 0 {
			return 0, 0, exit.Named(exit.Unavailable, "rental.rate_unknown",
				"rental %s has no observed rate; account spend is unknown", seen.ID)
		}
		burn += seen.HourlyRateUSDMicros
	}
	return len(m.live), burn, nil
}

func usdPerHour(micros int64) string {
	return usdPerHourBare(micros) + "/hour"
}

// skuRate is a SKU's pre-spend rate the way a human must read it (th-126):
// the estimated total the pod will bill, decomposed into the GPU list rate and
// the spec-derived storage adder. A SKU whose hub itemizes no storage renders
// as the plain rate.
func skuRate(sku hub.RentalSKU) string {
	if sku.StorageUSDMicrosPerHour <= 0 {
		return usdPerHour(sku.PriceUSDMicrosPerHour)
	}
	return usdPerHour(sku.PriceUSDMicrosPerHour+sku.StorageUSDMicrosPerHour) +
		" (" + usdPerHourBare(sku.PriceUSDMicrosPerHour) + " gpu + " +
		usdPerHourBare(sku.StorageUSDMicrosPerHour) + " storage)"
}

// usdPerHourBare is the dollar figure alone, for a line that already says "per hour".
func usdPerHourBare(micros int64) string {
	// PRICES ARE READ IN PENNIES. Micro-dollar precision is how the provider quotes
	// and how we bill, but `$0.463504/hour` asks a reader to parse six decimals to
	// learn "about forty-six cents" -- the extra digits carry no decision. Money is
	// rounded here for DISPLAY only; every comparison, cap check and ledger entry
	// upstream still works in whole micros.
	if micros < 0 {
		return "-" + usdPerHourBare(-micros)
	}
	cents := (micros + 5_000) / 10_000
	// A rate that is real but smaller than a penny must not render as free: the CPU
	// storage adder is $0.00278/hour, and `$0.00` would say the wrong thing.
	if cents == 0 && micros > 0 {
		return "<$0.01"
	}
	return fmt.Sprintf("$%d.%02d", cents/100, cents%100)
}

func settledRequest(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// RentalConstraints reads the selected local install or published release's immutable
// requirements and interface. Missing facts refuse selection before spending.
func RentalConstraints(ctx *Context, req records.Request) (rental.Constraints, *exit.Error) {
	var out rental.Constraints
	var declared *launch.PackageInterface
	if req.InstallID != "" && strings.HasPrefix(req.Package, "local/") {
		_, store, problem := rentalStores(ctx)
		if problem != nil {
			return out, exit.Unavailablef("package requirements are unavailable for %s@%s", req.Package, req.Release)
		}
		defer store.Close()
		installed, problem := store.Install(req.InstallID)
		if problem != nil || installed == nil {
			return out, exit.Unavailablef("package requirements are unavailable for %s@%s", req.Package, req.Release)
		}
		python, problem := launch.EnvironmentPython(*installed)
		if problem != nil {
			return rental.Constraints{}, problem
		}
		selection, problem := install.ExecutionRequirements(context.Background(), filepath.Dir(filepath.Dir(python)),
			strings.TrimPrefix(installed.Package, "local/"), strings.Fields(installed.Extra))
		if problem != nil {
			return out, problem
		}
		out.Requirements, out.RequiresPython = selection.Requirements, selection.RequiresPython
		out.PythonVersion = installed.Python
		declared, _ = launch.ReadPackageInterface(launch.PackageInterfacePath(installed.Dir))
	} else {
		if req.Package == "" || req.Release == "" {
			return out, exit.Unavailablef("package requirements are unavailable for %s@%s", req.Package, req.Release)
		}
		ref, problem := hub.ParseRef(req.Package)
		if problem != nil {
			return out, exit.Unavailablef("package requirements are unavailable for %s@%s", req.Package, req.Release)
		}
		hctx, cancel := hub.Context()
		defer cancel()
		detail, problem := client(ctx).PackageRelease(hctx, ref, req.Release)
		if problem != nil {
			return out, exit.Unavailablef("package requirements are unavailable for %s@%s", req.Package, req.Release)
		}
		requirements, requiresPython, problem := detail.Constraints()
		if problem != nil {
			return out, exit.Unavailablef("package requirements are unavailable for %s@%s", req.Package, req.Release)
		}
		out.Requirements, out.RequiresPython = requirements, requiresPython
		out.PythonVersion = detail.PythonVersion
		declared, _ = launch.DecodePackageInterface(detail.PackageInterface)
	}
	// Rental admission is about the remote inventory. A client may have no Runtime
	// or a different installed Python set; local capture already bound its exact
	// interpreter above and published releases carry their own immutable selection.
	// Both sources apply the requested function's declared intersection. An unreadable
	// interface declares no width, so wide products remain excluded rather than guessed.
	if declared != nil {
		if entrypoint, e := declared.Function(req.Entrypoint); e == nil && entrypoint.Kind != "job" {
			out.Degrees = entrypoint.SequenceParallelDegrees()
		}
	}
	return out, nil
}

func rentalCompatibility(ctx *Context, id string, constraints rental.Constraints) *exit.Error {
	call, cancel := hub.Context()
	defer cancel()
	facts, problem := client(ctx).RentalImageInventory(call, id)
	if problem != nil {
		return problem
	}
	raw := facts.ImageInventory
	inventory, err := rental.ImageInventory(raw)
	if err != nil {
		return exit.Named(exit.Structural, "rental.image_inventory_invalid", "%s", err)
	}
	policy := [][]string{}
	if constraints.SupportedPythonMinors != nil {
		policy = append(policy, constraints.SupportedPythonMinors)
	}
	_, reason := launch.InventoryPython(inventory, constraints.RequiresPython, constraints.PythonVersion, policy...)
	if strings.HasPrefix(reason, "no available Python executor") && launch.ProvisionablePython(rental.ImagePythonCapabilities(raw), constraints.RequiresPython, constraints.PythonVersion, policy...) {
		reason = ""
	}
	if reason != "" {
		return exit.Named(exit.Conflict, "rental.dependency_mismatch", "rental %s: %s", id, reason)
	}
	return nil
}

func (m *managedRentals) observeIdle(row records.Rental) (rental.Idleness, *exit.Error) {
	return rental.ObserveIdle(m.store, row)
}
