package cli

import (
	"context"
	"fmt"
	"slices"
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
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// managedRentals serializes fleet admission, acknowledged keepalive and release.
// Fixed idle expiry counts real work on each machine, never retained bytes or
// an open controller connection. Durable records survive daemon restarts.
//
// mu guards this struct and each decision's read-then-write of the records. It is never
// held across Hub or provider I/O: an operation snapshots under it, asks with it
// released, and applies the answer under it again, so an answer that never comes holds
// up only the operation waiting for it.
type managedRentals struct {
	mu     sync.Mutex
	ctx    *Context
	layout home.Layout
	store  *records.Store
	owner  *orchestrator.Orchestrator
	// wakeQueue is installed by daemon setup. A rental becoming attachable is
	// an external capacity edge: queued machine executions must be re-asked
	// immediately rather than waiting for the next poll or a new CLI request.
	wakeQueue func()
	installs  *machines.Installs
	// forget closes the daemon's connection to a machine whose rental ended.
	forget func(string)
	// placeReleased places again the runs a rental proven gone left in the outbox.
	placeReleased func()
	// boot brings a newly attached worker boot to the Hub's target software before it takes
	// work: daemon setup installs it.
	boot func(string)
	// wakeMachine asks again about the runs a newly attached boot holds.
	wakeMachine func(string)
	// said holds the last line printed about each rental, so the fleet speaks once per change.
	said map[string]string
	// letGone holds the failed rentals this daemon has let go: a failure is acted on once, not
	// on every poll that repeats it.
	letGone map[string]bool
	closed  bool
	// census is each Tensorhub's account view, by origin. One daemon serves every hub;
	// each rental is reconciled, released and counted against the hub it was bought
	// from, with that hub's own credential.
	census map[string]*rentalCensus
	// disks is each rental's container disk as its Hub last reported it; a rental the
	// Hub has not reported is absent.
	disks map[string]int
	// buying is each managed purchase in flight, by operation key, as the machine it will
	// be: every other placement waits on it rather than buying around it.
	buying map[string]records.Rental
	// settling is each rental whose release is in flight; reconcile leaves it to that release.
	settling map[string]bool
}

// rentalCensus is one hub's answer to "what does this account own there".
type rentalCensus struct {
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
	// hubUnknown is every host row the hub answered `rental.not_found` for on the last
	// reconcile. A 404 is not proof of provider destruction, so the row is KEPT and shown;
	// it must never stop the reconcile of every other rental against the same hub.
	hubUnknown map[string]bool
}

// at is the fleet's context for one hub origin, with that origin's credential.
func (m *managedRentals) at(origin string) *Context { return m.ctx.forHub(origin) }

// atRental is the fleet's context on the hub one rental was bought from.
func (m *managedRentals) atRental(id string) *Context {
	row, problem := m.store.RentalRow(id)
	if problem != nil || row == nil {
		return m.ctx
	}
	return m.at(row.Hub)
}

// origin is the canonical spelling of a hub a record names; "" is the default hub.
func (m *managedRentals) origin(recorded string) string {
	return m.at(recorded).Cfg.HubURL
}

// onHub names a hub other than the default in a log line.
func (m *managedRentals) onHub(origin string) string {
	if origin == "" || m.origin(origin) == m.ctx.Cfg.HubURL {
		return ""
	}
	return " on " + m.ctx.Cfg.HubLabel(m.origin(origin))
}

func (m *managedRentals) censusLocked(origin string) *rentalCensus {
	origin = m.origin(origin)
	if m.census == nil {
		m.census = map[string]*rentalCensus{}
	}
	if m.census[origin] == nil {
		m.census[origin] = &rentalCensus{}
	}
	return m.census[origin]
}

// origins is every hub this host holds a rental or open acquisition on.
func (m *managedRentals) origins() ([]string, *exit.Error) {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return nil, problem
	}
	operations, problem := m.store.ActiveRentalOperations()
	if problem != nil {
		return nil, problem
	}
	seen := map[string]bool{}
	var origins []string
	add := func(recorded string) {
		if origin := m.origin(recorded); !seen[origin] {
			seen[origin] = true
			origins = append(origins, origin)
		}
	}
	for _, row := range rows {
		add(row.Hub)
	}
	for _, operation := range operations {
		add(operation.Hub)
	}
	sort.Strings(origins)
	return origins, nil
}

// Failed provider release is retried without changing the idle deadline.

// status is the fleet line of the request's hub, after reconciling that hub.
func (m *managedRentals) status(req records.Request) (string, *exit.Error) {
	if problem := m.reconcile(req.Hub); problem != nil {
		return "", problem
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lineLocked(req.Hub)
}

// machineKey names one product at one GPU count.
type machineKey struct {
	name string
	gpus int
}

func (m *managedRentals) admit(skuName string, gpus int, provider string) (string, hub.RentalSKU, *exit.Error) {
	origin := m.ctx.Cfg.HubURL
	line, problem := m.status(records.Request{Hub: origin})
	if problem != nil {
		return "", hub.RentalSKU{}, problem
	}
	hctx, cancel := hub.Context()
	skus, problem := client(m.at(origin)).RentalSKUs(hctx, provider)
	cancel()
	if problem != nil {
		return "", hub.RentalSKU{}, problem
	}
	if sku, found := hub.FindRentalSKU(skus, skuName, gpus); found {
		return line, sku, nil
	}
	// An explicit ask is HONOURED OR REFUSED, never widened to a neighbouring card or
	// count — and the refusal has to say so out loud (cl-132), AND say which absence it
	// hit (th-150).
	return "", hub.RentalSKU{}, m.refuseSKU(provider, skuName, gpus, skus)
}

// refuseSKU distinguishes unknown products, provider stock-outs, and
// temporary boot-failure exclusions. Reading status never acquires a rental.
func (m *managedRentals) refuseSKU(provider, skuName string, gpus int, skus []hub.RentalSKU) *exit.Error {
	hctx, cancel := hub.Context()
	status, problem := client(m.ctx).RentalSKUStatus(hctx, provider, skuName, gpus)
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
// worker has not attached yet — a purchase in flight included — is waited for, never
// bought around (cl-170) nor queued around (cl-174): the decision returns with no rental
// and the fleet's next observation re-asks. A refusal returns its record too, so every
// decision is durable. Every Hub read happens before the lock is taken.
func (m *managedRentals) acquire(req records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
	var none orchestrator.PlacementDecision
	if !req.Rental || req.Worker != "" {
		return none, "", exit.Internalf("request %s is not an unassigned --rental request", req.ID)
	}
	// Everything this decision reads or buys belongs to the request's hub.
	origin, scoped := m.origin(req.Hub), m.at(req.Hub)
	resolver := NewResolver(m.store, m.ctx.Cfg)
	needsAccelerator, problem := resolver.PrivateRentalNeedsAccelerator(req)
	if problem != nil {
		return none, "", problem
	}
	// CPU orchestration requests need their captured defaults to narrow the rental
	// choice (cl-210). A request sized by its own model slots retains only those;
	// unrelated captures must not inflate its residency or preparation selection. Its
	// own selection is made once and supersedes any callee default for the same slot.
	metadata, problem := rentalMetadata(scoped, req)
	if problem != nil {
		return none, "", problem
	}
	var childModels []records.ModelRef
	if req.InstallID == "" {
		childModels, problem = resolver.PublishedChildModels(scoped, metadata)
	} else {
		childModels, problem = resolver.UnpublishedChildModels(req)
	}
	if problem != nil {
		return none, "", problem
	}
	req.Models = records.OneSelectionPerSlot(req.Models, childModels)
	needsAccelerator = needsAccelerator || req.SizedByOwnModels() || len(childModels) > 0
	link, problem := m.store.MachineExecution(req.ID)
	if problem != nil {
		return none, "", problem
	}
	runtimeOwned := link != nil
	if problem := m.reconcile(origin); problem != nil {
		return none, "", problem
	}
	skus, problem := m.catalog(origin)
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
	constraints, constraintProblem := RentalConstraints(scoped, metadata)
	if constraintProblem != nil {
		return none, "", constraintProblem
	}
	if constraints.Working, problem = m.store.WorkingPeaks(req); problem != nil {
		return none, "", problem
	}
	throughput, problem := m.throughput(origin, req)
	if problem != nil {
		return none, "", problem
	}
	var purchases []orchestrator.PlacementCandidate
	if req.RequestedRental == "" {
		purchases = rental.Purchases(skus, req.Models, needsAccelerator, req.IsJob(), constraints)
	}
	var capped *exit.Error
	m.mu.Lock()
	defer m.mu.Unlock()
	for {
		// The fleet is read again on every pass: a refused buy ran with the lock released.
		attached, problem := m.attachedLocked(origin, req, runtimeOwned, bySKU, needsAccelerator, constraints)
		if problem != nil {
			return none, "", problem
		}
		decision.Candidates = append(attached, purchases...)
		decision.Throughput = rental.Measure(decision.Candidates, throughput, req.Models)
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
				line, problem := m.lineLocked(origin)
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
			purchase := &purchases[i-len(attached)]
			row, problem := m.buyLocked(req, *c, bySKU[machineKey{c.SKU, c.GPUs}])
			if problem != nil && problem.ErrName() == "rental.fleet_spend_cap" {
				// The hub's cap refused this product; a cheaper one may still fit.
				purchase.Verdict, capped = orchestrator.VerdictExcluded+problem.ErrName(), problem
				continue
			}
			if problem != nil {
				if problem.ErrName() != "rental.sku_out_of_stock" && problem.ErrName() != "rental.sku_unavailable" {
					return none, "", problem
				}
				purchase.Verdict = orchestrator.VerdictNoStock
				fmt.Fprintf(m.ctx.Out, "rentals: %s has no inventory; choosing again without it\n", c.Name())
				continue
			}
			rentalID, decision.Bought = row.ID, true
		}
		rental.Conclude(decision.Candidates, i)
		decision.RentalID, decision.Models = rentalID, c.Models
		if !decision.Bought && c.Ahead > 0 && req.RequestedRental == "" && !req.RetainWork && !runtimeOwned {
			// Preparation may use this candidate, but its occupied seat is not an
			// assignment. The local queue can still take another ready rental;
			// dispatch records the chosen worker when it reserves a free seat.
			line, problem := m.lineLocked(origin)
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
			_, release, problem := m.releaseLocked(rentalID)
			if problem == nil && release != nil {
				m.mu.Unlock()
				problem = m.settle(release)
				m.mu.Lock()
			}
			if problem != nil {
				return none, "", problem
			}
			return none, "", exit.New(exit.Canceled,
				"request %s settled while rental %s was starting; the rental was released", req.ID, rentalID)
		}
		line, problem := m.lineLocked(origin)
		return decision, line, problem
	}
}

// attachedLocked is every rental the fleet holds as a candidate: with the reason it
// cannot take this request, or, when it can, the attempts ahead of a new one. Nothing
// is silently dropped (cl-132): a decision reporting no attached candidate while the
// fleet holds two is the shape that read as waste live. A purchase in flight is a
// candidate too, before its pod has a row: the machine on its way.
func (m *managedRentals) attachedLocked(origin string, req records.Request, runtimeOwned bool, bySKU map[machineKey]hub.RentalSKU,
	needsAccelerator bool, constraints rental.Constraints) ([]orchestrator.PlacementCandidate, *exit.Error) {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return nil, problem
	}
	reserved, problem := m.reservationsLocked(rows)
	if problem != nil {
		return nil, problem
	}
	rows = append(rows, reserved...)
	// A purchase owns its pod until it has pinned its own request, even once attachable.
	busy, problem := m.inFlightLocked()
	if problem != nil {
		return nil, problem
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].HourlyRateUSDMicros != rows[j].HourlyRateUSDMicros {
			return rows[i].HourlyRateUSDMicros < rows[j].HourlyRateUSDMicros
		}
		return rows[i].ID < rows[j].ID
	})
	sourceBytes, problem := m.store.PlannedSourceBytes(req.ID)
	if problem != nil {
		return nil, problem
	}
	out := make([]orchestrator.PlacementCandidate, 0, len(rows))
	for _, row := range rows {
		if req.RequestedRental != "" && row.ID != req.RequestedRental {
			continue
		}
		// A pod serves only its own hub's packages and models.
		if m.origin(row.Hub) != origin {
			continue
		}
		c := orchestrator.PlacementCandidate{Rental: row.ID, Machine: row.MachineName, SKU: row.SKU,
			GPUs: row.AcceleratorCount, RateUSDMicrosPerHour: row.HourlyRateUSDMicros}
		if req.RentNew && row.ManagedRequestID != req.ID {
			c.Verdict = orchestrator.VerdictExcluded + "fresh_rental_requested"
			out = append(out, c)
			continue
		}
		sku, offered := bySKU[machineKey{row.SKU, row.AcceleratorCount}]
		if offered {
			c.RateUSDMicrosPerHour = sku.PriceUSDMicrosPerHour + sku.StorageUSDMicrosPerHour
		}
		// Everything decidable from the rental ROW is settled by the chooser, in the one
		// order that keeps a transient state out of a permanent verdict (cl-185). What
		// is left are the questions only this host can answer.
		retained, problem := m.store.RentalRetainedModelBytes(row.ID)
		if problem != nil {
			return nil, problem
		}
		disk := rental.Disk{HaveGB: m.disks[row.ID], RetainedBytes: retained, SourceBytes: sourceBytes}
		if rental.Standing(&c, req.Models, row, sku.VRAMGB, needsAccelerator, offered, req.IsJob(), constraints.Working, disk) {
			if busy[row.ID] {
				c.Verdict = orchestrator.VerdictAttaching
				out = append(out, c)
				continue
			}
			reason, problem := m.standingLocked(row, req, runtimeOwned)
			if problem != nil {
				return nil, problem
			}
			if reason != "" {
				c.Verdict = orchestrator.VerdictExcluded + reason
			} else {
				queued, problem := m.store.RentalQueueAhead(row.ID, req.ID)
				if problem != nil {
					return nil, problem
				}
				c.Ahead = queued
			}
		}
		out = append(out, c)
	}
	return out, nil
}

// reservationsLocked is each purchase in flight whose pod has no row yet.
func (m *managedRentals) reservationsLocked(rows []records.Rental) ([]records.Rental, *exit.Error) {
	have := make(map[string]bool, len(rows))
	for _, row := range rows {
		have[row.ID] = true
	}
	var out []records.Rental
	for key, machine := range m.buying {
		op, problem := m.store.RentalOperation(key)
		if problem != nil {
			return nil, problem
		}
		if op == nil || !have[op.RentalID] {
			out = append(out, machine)
		}
	}
	return out, nil
}

// standingLocked is the fleet-side reason an ATTACHED rental cannot take any new
// placement, or "". Class, geometry and lifecycle are settled by the caller before this
// is asked, so every answer here is a fact about what the pod is already holding.
func (m *managedRentals) standingLocked(row records.Rental, req records.Request, runtimeOwned bool) (string, *exit.Error) {
	if !runtimeOwned {
		// Work Creator still drives replaces the worker's whole desired state, which
		// would retire the executions Runtime owns there; it waits for those to end.
		// Their collected models and files are only bytes on the pod's disk.
		live, problem := m.store.RentalHasLiveMachineExecutions(row.ID)
		if problem != nil {
			return "", problem
		}
		retained, problem := m.store.RentalHasRetainedJob(row.ID)
		if problem != nil {
			return "", problem
		}
		if live || retained && (!req.IsJob() || !req.RetainWork) {
			return orchestrator.ExcludedModeConflict, nil
		}
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

// throughput reads the model's published throughput rows once per decision. A
// request binding no catalog model (none, or a provider source) has nothing to look up.
func (m *managedRentals) throughput(origin string, req records.Request) ([]hub.ModelThroughput, *exit.Error) {
	if len(req.Models) == 0 || req.Models[0].Model == "" {
		return nil, nil
	}
	ref, problem := hub.ParseRef(req.Models[0].Model)
	if problem != nil {
		return nil, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	return client(m.at(origin)).ModelThroughput(hctx, ref)
}

// buyLocked is one paid ask for the chosen product. Under the lock the request's
// selection is pinned to that machine's lane, so the rental POST declares the lane the
// pod will hold (th-155), and the paid operation is recorded and reserved: every other
// placement sees the machine on its way. The ask and the boot, named while it passes
// (cl-121), run with the lock released; a purchase the Hub never finishes holds only
// its own request.
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
	operationKey, problem := m.store.ManagedRentalOperationKey(req.ID)
	if problem != nil {
		return records.Rental{}, problem
	}
	purchase, problem := openRentalAcquisition(m.at(req.Hub), m.layout, m.store, sku.Name, sku.AcceleratorCount,
		operationKey, rental.AcquisitionReason(req), sku.PriceUSDMicrosPerHour, time.Time{}, req.ID)
	if problem != nil {
		return records.Rental{}, problem
	}
	sku.PriceUSDMicrosPerHour = purchase.rate
	if m.buying == nil {
		m.buying = map[string]records.Rental{}
	}
	m.buying[operationKey] = records.Rental{ID: operationKey, MachineName: purchase.machine, SKU: sku.Name,
		AcceleratorModel: sku.AcceleratorModel, AcceleratorCount: sku.AcceleratorCount,
		HourlyRateUSDMicros: sku.PriceUSDMicrosPerHour, ManagedRequestID: req.ID,
		State: "pending_acquisition", Hub: m.origin(req.Hub)}
	fmt.Fprintf(m.ctx.Out, "rentals: renting %s%s at %s (%s)\n", orchestrator.MachineLabel(sku.Name, sku.AcceleratorCount),
		machineShape(sku.VCPUCount, sku.MemoryGB), skuRate(sku), pinText(c))
	m.owner.ObservePhase(req.ID, orchestrator.PhaseSample{Name: orchestrator.PhaseAcquiring})
	m.mu.Unlock()
	m.owner.AwaitRental(req.ID, purchase.machine)
	row, bought, _, problem := purchase.complete(context.Background(), func(seen hub.Rental) {
		if sample, ok := orchestrator.RentalSample(seen); ok {
			m.owner.ObservePhase(req.ID, sample)
		}
	})
	m.owner.ForgetPhase(req.ID)
	m.mu.Lock()
	delete(m.buying, operationKey)
	if problem == nil {
		m.observeDiskLocked(row.ID, bought)
		if m.boot != nil {
			m.boot(row.ID)
		}
	}
	return row, problem
}

// observeDiskLocked keeps the container disk a Hub view reports for a rental.
func (m *managedRentals) observeDiskLocked(id string, seen hub.Rental) {
	if seen.ContainerDiskGB <= 0 {
		return
	}
	if m.disks == nil {
		m.disks = map[string]int{}
	}
	m.disks[id] = seen.ContainerDiskGB
}

// pinText is a candidate's pin for a log line; the rung is named when the ladder chose.
func pinText(c orchestrator.PlacementCandidate) string {
	text := fmt.Sprintf("lane %s, fit %s", orNone(c.Lane), orNone(c.Fit))
	if c.Rung > 0 {
		return fmt.Sprintf("rung %d, %s", c.Rung, text)
	}
	return text
}

func (m *managedRentals) catalog(origin string) ([]hub.RentalSKU, *exit.Error) {
	hctx, cancel := hub.Context()
	defer cancel()
	return client(m.at(origin)).RentalSKUs(hctx, "")
}

// releaseOrphaned resumes after a daemon restart: every rental is reconciled with the hub
// and lost work is recovered.
func (m *managedRentals) releaseOrphaned() {
	if problems := m.reconcileAll(); len(problems) > 0 {
		for origin, problem := range problems {
			fmt.Fprintf(m.ctx.Out, "rental reconciliation deferred%s: %s\n", m.onHub(origin), problem.Message)
		}
		return
	}
	m.report()
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

// refreshBooting re-reads, at the cadence a rental verb samples a provisioning rental, only
// the rentals the Hub is still acquiring: their boot is what the work pinned to them shows
// while it waits. A problem waits for the sweep, which names it.
func (m *managedRentals) refreshBooting() {
	origins, problem := m.origins()
	if problem != nil {
		return
	}
	for _, origin := range origins {
		_ = m.reconcileRows(origin, func(row records.Rental) bool { return acquiringState(row.State) })
	}
}

// acquiringState is a Hub rental state before the pod is ready.
func acquiringState(state string) bool {
	switch state {
	case "pending_acquisition", "acquiring", "booting":
		return true
	}
	return false
}

// watch is the idle release's own loop. It re-reads the records at pollCadence — the
// resolution every rental verb already samples a rental at — and acts only on what they
// say; the grace is the debounce, and a sample that finds nothing to do costs a few local
// reads. It returns when quit closes.

func (m *managedRentals) watch(quit <-chan struct{}) {
	tick := time.NewTicker(pollCadence)
	defer tick.Stop()
	reconciled := time.Now()
	// Re-asking the orchestrator can place work, and placement can buy and wait out a
	// pod's boot. It runs on its own goroutine, one ask at a time, so the sweep never
	// waits on a purchase.
	reask := make(chan struct{}, 1)
	defer close(reask)
	if m.owner != nil {
		go func() {
			for range reask {
				// OUTSIDE the lock, always: hub recovery is an observed fleet change, and
				// re-asking durable queued work calls back into this object.
				m.owner.WakeQueue()
			}
		}()
	}
	for {
		select {
		case <-quit:
			return
		case <-tick.C:
		}
		m.mu.Lock()
		closed := m.closed
		m.mu.Unlock()
		if !closed {
			if time.Since(reconciled) < hubReconcileCadence {
				m.refreshBooting()
			} else {
				reconciled = time.Now()
				problems := m.reconcileAll()
				m.mu.Lock()
				for origin, problem := range problems {
					m.sayLocked("reconcile:"+origin, "rental reconciliation deferred"+m.onHub(origin)+": "+problem.Message)
				}
				for origin := range m.census {
					if problems[origin] == nil {
						delete(m.said, "reconcile:"+origin)
					}
				}
				m.mu.Unlock()
			}
			m.report()
		}
		if m.installs != nil {
			m.installs.Wake()
		}
		select {
		case reask <- struct{}{}:
		default:
		}
	}
}

// report says, once per change, what the Hub knows that this host's records do not: rentals
// it owns that were never recorded here, and recorded rentals it no longer knows.
func (m *managedRentals) report() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sayUnrecordedLocked()
	rows, problem := m.store.Rentals()
	if problem != nil {
		m.sayLocked("", "rental census deferred: "+problem.Message)
		return
	}
	for _, row := range rows {
		if m.censusLocked(row.Hub).hubUnknown[row.ID] {
			settle := "kept until its provider release is confirmed"
			if row.State == hub.RentalFailed {
				settle = "it failed (provider absent) before the Hub lost it; `cozy rental end " + row.ID + "` forgets it"
			}
			m.sayLocked("hub-unknown:"+row.ID, fmt.Sprintf("rental %s is recorded on this host but unknown to %s; %s",
				row.ID, m.origin(row.Hub), settle))
		}
	}
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

func (m *managedRentals) close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
}

// releaseLocked decides the paid DELETE, re-observing the row under the lock first: the
// sweep and the settlement hook both arrive here, and only a pod with nothing queued,
// running, or owed on it goes. The owed re-check keeps the buy's rollback honest too:
// acquire releases a rental it just bought only because the buyer settled or routed
// elsewhere, which is exactly when the debt is gone. The caller settles the release.
func (m *managedRentals) releaseLocked(id string) (string, *pendingRelease, *exit.Error) {
	row, problem := m.store.RentalRow(id)
	if problem != nil || row == nil {
		return "", nil, problem
	}
	queued, running, problem := m.store.RentalRunCounts(id)
	if problem != nil {
		return "", nil, problem
	}
	owed := queued != 0 || running != 0
	if !owed {
		if owed, problem = orchestrator.RentalOwedBy(m.store, *row); problem != nil {
			return "", nil, problem
		}
	}
	var release *pendingRelease
	if !owed {
		if release, problem = m.beginReleaseLocked(*row); problem != nil {
			return "", nil, problem
		}
	}
	if release != nil {
		return "", release, nil
	}
	line, problem := m.lineLocked(row.Hub)
	return line, nil, problem
}

// pendingRelease is a release decided under the lock and settled without it.
type pendingRelease struct {
	row          records.Rental
	operationKey string
	confirmed    bool
}

// beginReleaseLocked records the intent to release a rental and takes it for this
// release; nil when another release already has it.
func (m *managedRentals) beginReleaseLocked(row records.Rental) (*pendingRelease, *exit.Error) {
	if m.settling[row.ID] {
		return nil, nil
	}
	operationKey, problem := m.store.RequestRentalRelease(row.ID)
	if problem != nil {
		return nil, problem
	}
	confirmed := row.State == hub.RentalReleased
	if operationKey != "" {
		operation, problem := m.store.RentalOperation(operationKey)
		if problem != nil {
			return nil, problem
		}
		confirmed = confirmed || operation != nil && operation.State == hub.RentalReleased
	}
	if !confirmed && row.State != hub.RentalReleaseRequested {
		row.State = hub.RentalReleaseRequested
		if problem := m.store.RecordRental(row); problem != nil {
			return nil, problem
		}
		if line, lineProblem := m.lineLocked(row.Hub); lineProblem == nil {
			fmt.Fprintln(m.ctx.Out, line)
		}
	}
	if m.settling == nil {
		m.settling = map[string]bool{}
	}
	m.settling[row.ID] = true
	return &pendingRelease{row: row, operationKey: operationKey, confirmed: confirmed}, nil
}

// settle carries out a release with no lock held: the DELETE to the hub the pod was
// bought from, with that hub's credential; the Hub's confirmation; then the local half.
func (m *managedRentals) settle(release *pendingRelease) *exit.Error {
	row, id, confirmed := release.row, release.row.ID, release.confirmed
	defer func() {
		m.mu.Lock()
		delete(m.settling, id)
		m.mu.Unlock()
	}()
	owner := client(m.at(row.Hub))
	if !confirmed {
		hctx, cancel := hub.Context()
		problem := owner.Release(hctx, id, "")
		cancel()
		if problem != nil {
			if problem.Code == exit.NotFound {
				return rentalReleaseUnconfirmed(id)
			}
			return problem
		}
	}
	for !confirmed {
		hctx, cancel := hub.Context()
		remote, observed := owner.Rental(hctx, id)
		cancel()
		switch {
		case observed == nil && remote.State == hub.RentalReleased:
			confirmed = true
		case observed == nil:
			row.State = remote.State
			copyRentalFailure(&row, remote)
			if update := m.store.RecordRental(row); update != nil {
				return update
			}
			time.Sleep(pollCadence)
		case observed.Code == exit.NotFound:
			return rentalReleaseUnconfirmed(id)
		case transient(observed):
			time.Sleep(pollCadence)
		default:
			return observed
		}
	}
	if m.forget != nil {
		m.forget(id)
	}
	if _, problem := rental.Forget(m.layout, m.store, id); problem != nil {
		return problem
	}
	if release.operationKey != "" {
		rental.ForgetPending(m.layout, release.operationKey)
	}
	return nil
}

func localRentalAttachable(row records.Rental) bool {
	return records.RentalReadyState(row.State) && row.Address != "" && row.CertPath != ""
}

// wakeQueueAsync re-enters the orchestrator outside the fleet lock. The rental
// reconciler runs while holding m.mu; waking synchronously would let placement
// call back into the fleet and deadlock.
func (m *managedRentals) wakeQueueAsync() {
	if m.wakeQueue != nil {
		go m.wakeQueue()
	}
}

// reconcile converges this host's rental rows on one hub with that hub, and then asks
// it the question this host cannot answer from its own rows at all: what else does this
// account own there? The two directions are not the same read and only one of them
// existed. Local-row reconciliation can correct a row; it can never notice a pod that
// has no row. The Hub is asked with the lock released; its answers are applied under
// the lock to the rows as they are by then.
func (m *managedRentals) reconcile(origin string) *exit.Error { return m.reconcileRows(origin, nil) }

// reconcileRows is reconcile, or with only, a re-read of just the rows it keeps: no rowless
// asks and no account listing.
func (m *managedRentals) reconcileRows(origin string, only func(records.Rental) bool) *exit.Error {
	origin = m.origin(origin)
	m.mu.Lock()
	rows, problem := m.reconcilableLocked(origin)
	var asks []records.RentalOperation
	if problem == nil && only == nil {
		asks, problem = m.rowlessAsksLocked(origin)
	}
	m.mu.Unlock()
	if problem != nil {
		return problem
	}
	if only != nil {
		rows = slices.DeleteFunc(rows, func(row records.Rental) bool { return !only(row) })
		if len(rows) == 0 {
			return nil
		}
	}
	owner := client(m.at(origin))
	views := make([]rentalView, 0, len(rows))
	var asked *exit.Error
	for _, row := range rows {
		if row.State == hub.RentalReleased {
			views = append(views, rentalView{id: row.ID, released: true})
			continue
		}
		hctx, cancel := hub.Context()
		remote, observed := owner.Rental(hctx, row.ID)
		cancel()
		if observed != nil && observed.ErrName() != "rental.not_found" {
			asked = observed
			break
		}
		views = append(views, rentalView{id: row.ID, remote: remote, unknown: observed != nil,
			released: observed == nil && remote.State == hub.RentalReleased})
	}
	// An ask whose answer was refused has no row to reconcile; the Hub still settles it.
	var gone []records.RentalOperation
	for _, op := range asks {
		if asked != nil {
			break
		}
		hctx, cancel := hub.Context()
		remote, observed := owner.RentalView(hctx, op.RentalID)
		cancel()
		switch {
		case observed == nil && hub.RentalAbsent(remote.State):
			gone = append(gone, op)
		case observed != nil && observed.Code != exit.NotFound:
			asked = observed
		}
	}
	var listing []hub.Rental
	var listed bool
	var listingProblem *exit.Error
	if asked == nil && only == nil {
		hctx, cancel := hub.Context()
		var stated hub.RentalListing
		stated, listingProblem = owner.RentalListing(hctx)
		listed = listingProblem == nil
		cancel()
		if listed {
			listing = stated.Rentals
			_ = m.store.StateHubBindingsRevision(origin, stated.BindingsRevision)
			_ = stateCredit(m.store, m.origin(origin), stated.Credit)
		}
	}
	m.mu.Lock()
	released, failed, rebooted, problem := m.applyRowsLocked(origin, views)
	if problem == nil {
		problem = m.settleAsksLocked(origin, gone)
	}
	census := m.censusLocked(origin)
	switch {
	case problem != nil:
	case asked != nil:
		problem = asked
		// Keep the last observed machines visible, but never present stale totals
		// as current, including after an authentication failure.
		census.listed, census.listingProblem = false, asked
	case only != nil:
	default:
		m.applyListingLocked(origin, listing, listed, listingProblem)
	}
	m.mu.Unlock()
	if detached := m.letGo(released, failed); problem == nil {
		problem = detached
	}
	m.reattach(rebooted)
	return problem
}

// reattach brings a newly attached boot to the target software first. Work bound to an old boot settles on its own;
// the rental, its custody, installs and queue continue on the new worker.
func (m *managedRentals) reattach(ids []string) {
	for _, id := range ids {
		if m.boot != nil {
			m.boot(id)
		}
		if m.wakeMachine != nil {
			m.wakeMachine(id)
		}
	}
	if len(ids) > 0 {
		if m.installs != nil {
			m.installs.Wake()
		}
		m.wakeQueueAsync()
	}
}

// rentalView is the Hub's answer about one rental row.
type rentalView struct {
	id     string
	remote hub.Rental
	// unknown is `rental.not_found`: not proof of provider destruction, so the row is
	// KEPT and shown, and never stops the reconcile of every other rental on the hub.
	unknown  bool
	released bool
}

// reconcileAll reconciles every hub this host holds rentals on. A hub that cannot
// be reconciled defers only its own rentals; every other hub still converges.
func (m *managedRentals) reconcileAll() map[string]*exit.Error {
	origins, problem := m.origins()
	if problem != nil {
		return map[string]*exit.Error{"": problem}
	}
	problems := map[string]*exit.Error{}
	for _, origin := range origins {
		if problem := m.reconcile(origin); problem != nil {
			problems[origin] = problem
		}
	}
	return problems
}

// reconcilableLocked is origin's rental rows, less those an operation in flight is
// already observing: a purchase or a release is the one reader of its own rental.
func (m *managedRentals) reconcilableLocked(origin string) ([]records.Rental, *exit.Error) {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return nil, problem
	}
	busy, problem := m.inFlightLocked()
	if problem != nil {
		return nil, problem
	}
	var out []records.Rental
	for _, row := range rows {
		if m.origin(row.Hub) == origin && !busy[row.ID] {
			out = append(out, row)
		}
	}
	return out, nil
}

// rowlessAsksLocked is origin's open paid asks that name a rental with no local row: an
// answer this host refused to record, or a row already gone. No row reconcile reads them.
func (m *managedRentals) rowlessAsksLocked(origin string) ([]records.RentalOperation, *exit.Error) {
	operations, problem := m.store.ActiveRentalOperations()
	if problem != nil {
		return nil, problem
	}
	busy, problem := m.inFlightLocked()
	if problem != nil {
		return nil, problem
	}
	var out []records.RentalOperation
	for _, op := range operations {
		if op.RentalID == "" || busy[op.RentalID] || m.origin(op.Hub) != origin {
			continue
		}
		row, problem := m.store.RentalRow(op.RentalID)
		if problem != nil {
			return nil, problem
		}
		if row == nil {
			out = append(out, op)
		}
	}
	return out, nil
}

// settleAsksLocked closes the rowless asks the Hub reported gone, unless a row or an
// operation in flight took them while the Hub was asked.
func (m *managedRentals) settleAsksLocked(origin string, gone []records.RentalOperation) *exit.Error {
	if len(gone) == 0 {
		return nil
	}
	current, problem := m.rowlessAsksLocked(origin)
	if problem != nil {
		return problem
	}
	for _, op := range gone {
		if !slices.ContainsFunc(current, func(each records.RentalOperation) bool { return each.Key == op.Key }) {
			continue
		}
		if problem := m.store.AdvanceRentalOperation(op.Key, op.RentalID, hub.RentalReleased); problem != nil {
			return problem
		}
		rental.ForgetPending(m.layout, op.Key)
	}
	return nil
}

// inFlightLocked is every rental a purchase or a release in flight holds.
func (m *managedRentals) inFlightLocked() (map[string]bool, *exit.Error) {
	busy := map[string]bool{}
	for id := range m.settling {
		busy[id] = true
	}
	for key := range m.buying {
		op, problem := m.store.RentalOperation(key)
		if problem != nil {
			return nil, problem
		}
		if op != nil && op.RentalID != "" {
			busy[op.RentalID] = true
		}
	}
	return busy, nil
}

// applyListingLocked replaces a census only with a complete successful answer.
// An unavailable listing retains its last rows as unverified, never current totals.
func (m *managedRentals) applyListingLocked(origin string, remote []hub.Rental, listed bool, problem *exit.Error) {
	census := m.censusLocked(origin)
	census.listed, census.listingProblem = listed, problem
	if problem != nil || !listed {
		return
	}
	var live, unrecorded []hub.Rental
	for _, seen := range remote {
		if hub.RentalAbsent(seen.State) {
			continue
		}
		row, rowProblem := m.store.RentalRow(seen.ID)
		if rowProblem != nil {
			census.listed, census.listingProblem = false, rowProblem
			return
		}
		live = append(live, seen)
		if row != nil && m.origin(row.Hub) == origin {
			continue
		}
		unrecorded = append(unrecorded, seen)
	}
	census.live, census.unrecorded = live, unrecorded
}

// sayUnrecordedLocked is the DAEMON's alarm, and it belongs to the sweep rather than
// to the reconcile because the reconcile also runs under a CLI verb, whose stdout is
// a machine document that a log line would corrupt. The daemon has no reader watching
// a board, so a machine billing outside its records has to reach the log by itself —
// once per machine, at the sweep's own cadence, stating the missing local activity.
func (m *managedRentals) sayUnrecordedLocked() {
	for origin, census := range m.census {
		for _, seen := range census.unrecorded {
			m.sayLocked("hub:"+seen.ID, fmt.Sprintf(
				"rental %s (%s) on %s is %s at %s and this host holds no record of it; activity is unknown to this controller",
				seen.ID, seen.Name, origin, humanRentalState(seen.State), usdPerHour(seen.HourlyRateUSDMicros)))
		}
	}
}

// applyRowsLocked records the Hub's answers on the rows as they are now; a row forgotten,
// or taken by an operation, while the Hub was asked is not this answer's to change. It
// returns the rentals the Hub released and failed, for letGo, and those it re-attested on a new boot.
func (m *managedRentals) applyRowsLocked(origin string, views []rentalView) (released, failed, rebooted []string, problem *exit.Error) {
	census := m.censusLocked(origin)
	census.hubUnknown = map[string]bool{}
	operations, problem := m.store.ActiveRentalOperations()
	if problem != nil {
		return nil, nil, nil, problem
	}
	pending := map[string]records.RentalOperation{}
	for _, operation := range operations {
		if operation.ManagedRequestID == "" && operation.RentalID != "" &&
			m.origin(operation.Hub) == origin && operation.State != "attached" &&
			operation.State != hub.RentalFailed && operation.State != hub.RentalReleaseRequested {
			pending[operation.RentalID] = operation
		}
	}
	busy, problem := m.inFlightLocked()
	if problem != nil {
		return nil, nil, nil, problem
	}
	for _, view := range views {
		current, problem := m.store.RentalRow(view.id)
		if problem != nil {
			return released, failed, rebooted, problem
		}
		if current == nil || busy[view.id] {
			continue
		}
		row, remote := *current, view.remote
		if view.unknown {
			census.hubUnknown[row.ID] = true
			continue
		}
		if view.released {
			// Released rows leave the fleet now; letGo forgets them.
			delete(m.said, row.ID)
			if row.State != hub.RentalReleased {
				row.State = hub.RentalReleased
				// Why it ended is the cause its unfinished runs settle with.
				if row.Failure.Code == "" {
					row.Failure.Code = remote.EndCause()
				}
				if problem := m.store.RecordRental(row); problem != nil {
					return released, failed, rebooted, problem
				}
			}
			released = append(released, row.ID)
			continue
		}
		m.observeDiskLocked(row.ID, remote)
		if m.owner != nil {
			m.owner.ObserveRental(remote)
		}
		// Adopt the hub's reconciled billed rate (th-120): the burn this host
		// reports and caps on must be what the provider actually charges.
		if remote.HourlyRateUSDMicros > 0 {
			row.HourlyRateUSDMicros = remote.HourlyRateUSDMicros
		}
		if operation, unfinished := pending[row.ID]; unfinished && remote.Attachable() {
			// A foreground acquisition may have completed or released this operation
			// while the GET was in flight. Its newer durable state wins.
			current, problem := m.store.RentalOperation(operation.Key)
			if problem != nil {
				return released, failed, rebooted, problem
			}
			if current == nil || current.State == "attached" || current.State == hub.RentalReleased ||
				current.State == hub.RentalFailed || current.State == hub.RentalReleaseRequested {
				continue
			}
			token, creator, problem := rental.RetainedAcquisitionCredentials(m.layout, *current)
			if problem != nil {
				return released, failed, rebooted, problem
			}
			if _, problem := finishRentalAttachment(m.layout, m.store, row, remote, operation.Key, token, creator); problem != nil {
				return released, failed, rebooted, problem
			}
			// The rental just crossed acquiring -> attachable: its boot is attached as a
			// rebooted one is, and parked machine executions are asked again.
			rebooted = append(rebooted, row.ID)
			continue
		}
		if remote.Attachable() && localRentalAttachable(row) && remote.WorkerID == row.ExpectedWorkerID &&
			(remote.WorkerBootID != row.ExpectedWorkerBootID || remote.Address != row.Address || remote.MediaAddress != row.MediaAddress) {
			if problem := rental.Reboot(m.layout, m.store, row.ID, remote); problem != nil {
				m.sayLocked("reboot:"+row.ID, "rental "+row.ID+" rebooted and is not re-attached: "+problem.Message)
			} else {
				fmt.Fprintf(m.ctx.Out, "rental %s (%s) rebooted; attaching its new worker\n", row.ID, row.MachineName)
				row.ExpectedWorkerBootID, row.Address, row.MediaAddress = remote.WorkerBootID, remote.Address, remote.MediaAddress
				rebooted = append(rebooted, row.ID)
			}
		}
		wasAttachable := localRentalAttachable(row)
		row.State = remote.State
		copyRentalFailure(&row, remote)
		// Written only when the Hub said something new: this runs every poll for every rental,
		// and an ended one's write settles its runs again.
		if row != *current {
			if problem := m.store.RecordRental(row); problem != nil {
				return released, failed, rebooted, problem
			}
		}
		if remote.Attachable() && !wasAttachable {
			// Reconcile changed the durable local rental to attachable. Wake
			// machineRuns after releasing the fleet lock so a pinned request
			// retries its existing execution without a second submission.
			m.wakeQueueAsync()
		}
		if row.State == hub.RentalFailed && !m.letGone[row.ID] {
			if m.letGone == nil {
				m.letGone = map[string]bool{}
			}
			m.letGone[row.ID] = true
			failed = append(failed, row.ID)
		}
	}
	return released, failed, rebooted, nil
}

// letGo detaches the workers of rentals the hub has failed or released, and forgets the
// released ones. It runs with the lock released: detaching waits for the worker to quiesce.
//
// This is where the daemon LEARNS a rental died — it is the only place that reads the
// hub's verdict for a rental that was already serving. A failed rental's worker record is
// dropped so no replan can choose the corpse: `rentalHeld` reads live worker records and a
// stale one would make `selectOrStart` decide the dead rental still holds the placement.
// Handing its pinned work back to routing is the loop's request-side sweep
// (RecoverLostWork): a rental whose record is gone is observed by nobody, so the work has
// to be found from the side that still has rows (observed live 2026-09-04, rental
// pr-183abac284d1e16f5f0a).
func (m *managedRentals) letGo(released, failed []string) *exit.Error {
	for _, id := range append(failed, released...) {
		if m.forget != nil {
			m.forget(id)
		}
		if m.owner != nil {
			m.owner.ForgetRental(id)
		}
	}
	for _, id := range released {
		if _, problem := rental.Forget(m.layout, m.store, id); problem != nil {
			return problem
		}
	}
	if len(released)+len(failed) > 0 && m.placeReleased != nil {
		m.placeReleased()
	}
	return nil
}

// lineLocked is one hub's fleet line: that account's machines and burn.
func (m *managedRentals) lineLocked(origin string) (string, *exit.Error) {
	count, burn, problem := m.totalsLocked(origin)
	if problem != nil {
		return "", problem
	}
	machine := "machine"
	if count != 1 {
		machine = "machines"
	}
	return fmt.Sprintf("rentals: %d remote %s running · %s%s", count, machine, usdPerHour(burn), m.onHub(origin)), nil
}

// totalsLocked is what THE ACCOUNT is paying, as the hub lists it: every live
// rental, recorded here or not (cl-199).
func (m *managedRentals) totalsLocked(origin string) (int, int64, *exit.Error) {
	census := m.censusLocked(origin)
	if census.listingProblem != nil {
		return 0, 0, census.listingProblem
	}
	if !census.listed {
		return 0, 0, exit.Named(exit.Unavailable, "rental.list_unavailable",
			"account rental census unavailable: this hub publishes no rental listing").
			WithRemedy("restore the Hub account-listing route before reading totals or acquiring a rental")
	}
	var burn int64
	for _, seen := range census.live {
		if seen.HourlyRateUSDMicros <= 0 {
			return 0, 0, exit.Named(exit.Unavailable, "rental.rate_unknown",
				"rental %s has no observed rate; account spend is unknown", seen.ID)
		}
		burn += costPerHour(seen.HourlyRateUSDMicros, seen.ComputeUSDMicrosPerHour, seen.StorageUSDMicrosPerHour)
	}
	return len(census.live), burn, nil
}

func usdPerHour(micros int64) string {
	return usdPerHourBare(micros) + "/hour"
}

// skuRate is a SKU's pre-spend rate the way a human must read it (th-126).
func skuRate(sku hub.RentalSKU) string {
	return rateBreakdown(sku.PriceUSDMicrosPerHour, sku.StorageUSDMicrosPerHour)
}

// rateBreakdown is an hourly cost the way a human reads it: the total the pod bills, split
// into the machine's compute and its disk's storage when the Hub itemizes storage.
func rateBreakdown(compute, storage int64) string {
	if storage <= 0 {
		return usdPerHour(compute)
	}
	return usdPerHour(compute+storage) + " (compute " + usdPerHourBare(compute) + " + storage " +
		usdPerHourBare(storage) + ")"
}

// costPerHour is what a rental costs an hour: the Hub's compute plus storage where it splits
// them, else its hourly rate (an older Hub's, or a fresh rental's compute quote alone).
func costPerHour(hourly, compute, storage int64) int64 {
	if compute > 0 {
		return compute + storage
	}
	return hourly
}

// machineShape names a machine's vCPUs and memory where the Hub states them.
func machineShape(vcpu, memoryGB int) string {
	if vcpu <= 0 || memoryGB <= 0 {
		return ""
	}
	return fmt.Sprintf(" (%d vCPU, %d GB)", vcpu, memoryGB)
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

// rentalMetadata reads the current release only to size a new rental. The returned copy
// is never the submitted request: a bare invocation stays unversioned, and its machine
// resolves the release from the selected Hub when it executes.
func rentalMetadata(ctx *Context, req records.Request) (records.Request, *exit.Error) {
	if req.Release != "" || req.Package == "" || strings.HasPrefix(req.Package, "local/") {
		return req, nil
	}
	ref, problem := hub.ParseRef(req.Package)
	if problem != nil {
		return req, problem
	}
	hctx, cancel := hub.Context()
	defer cancel()
	card, problem := client(ctx).PackageCard(hctx, ref)
	if problem != nil {
		return req, problem
	}
	req.Release, problem = newestPackageRelease(card.Releases)
	return req, problem
}

// RentalConstraints reads the selected local install or published release's immutable
// requirements and interface. Missing facts refuse selection before spending.
func RentalConstraints(ctx *Context, req records.Request) (rental.Constraints, *exit.Error) {
	var out rental.Constraints
	var problem *exit.Error
	if req, problem = rentalMetadata(ctx, req); problem != nil {
		return out, problem
	}
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
		selection, problem := install.InstalledRequirements(context.Background(), *installed)
		if problem != nil {
			return out, problem
		}
		out.Requirements, out.RequiresPython = selection.Requirements, selection.RequiresPython
		// A wheel closure can depend on its selected ABI/markers. Transferred
		// source instead lets the worker satisfy its authored Requires-Python.
		if installed.SourceKind != "local" {
			out.PythonVersion = installed.Python
		}
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

func (m *managedRentals) observeIdle(row records.Rental) (rental.Idleness, *exit.Error) {
	return rental.ObserveIdle(m.store, row)
}
