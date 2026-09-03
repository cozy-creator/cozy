package cli

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// The fleet, and its one reason to end a rental on its own. The owner's ruling: the main
// guard against over-spend is Creator reaping unused pods, so a controller that dies must
// not leave pods billing with no work being done. Every rental this daemon owns is under
// that rule — one bought for a request and one asked for with `cozy rental new` alike —
// because how a pod was acquired says nothing about whether it is doing anything.
//
// The observation is the decision: nothing queued for the pod, nothing running or owed on
// it, and that has been true since a fact this host recorded — the close of its last
// settled attempt, or the moment the hub first said `ready`. rentals.idle_release_s is only
// how long that must stay true. Its default, five minutes, is a few cold acquisitions: on
// the fleet proof a cold `--rental` answer took 34–56 s against 0.6 s warm, so an idle pod
// is worth keeping for a handful of those and no longer. "Owed" includes the buy itself
// (cl-113): a rental bought for a request is that request's debt from the moment the paid
// row exists — through the whole boot — until the request settles or durably routes to
// another machine, and the observation reads that debt from the rental row, never from
// the memory of the goroutine that bought it.
//
// managedRentals is deliberately small: one mutex serializes the local fleet ceiling, reuse
// decision, paid POST, and paid DELETE. The durable rental operation and request rows remain
// the crash-recovery authority; nothing here is remembered that the records do not hold.
type managedRentals struct {
	mu     sync.Mutex
	ctx    *Context
	layout home.Layout
	store  *records.Store
	owner  *orchestrator.Orchestrator
	// retryAt holds a rental whose release the hub refused or did not answer; said holds
	// the last line printed about each rental, so the sweep speaks once per change.
	retryAt map[string]time.Time
	said    map[string]string
	closed  bool
}

// idleReleaseRetry is how soon a release the hub did not confirm is asked again. A fifth
// of the grace: shorter than the grace, or a transient fault would add another whole
// grace of billing on top of the one the user accepted; not every sample, or a hub that
// is down would be asked at the sweep's cadence while the fleet lock is held for each ask.
func idleReleaseRetry(grace time.Duration) time.Duration { return grace / 5 }

// rentalIdleness is the one observation the idle release and `cozy rental` share.
type rentalIdleness struct {
	Queued, Running int
	// Owed marks a rental bought for a request that has not settled and has not been
	// durably routed to another machine (cl-113). The debt starts at the buy — before
	// the pod is ready, before dispatch pins — so a booting pod whose buyer is still
	// queued is busy, not idle.
	Owed bool
	// Since is the newest fact this host holds about the pod doing anything: the close of
	// its last settled attempt, or, for a rental that has never run, the moment this host
	// recorded the hub's `ready`. Zero while the pod is still booting — RentedAt is when it
	// was asked for, not when it began to exist — so a rental is never reaped mid-boot.
	Since time.Time
	// Spent says the rental's reason is over without waiting: a managed rental exists for
	// the request that bought it, and a job, or a request that never reached an attempt,
	// leaves nothing warm worth keeping. A manual rental exists because the user asked;
	// only idleness ends it.
	Spent bool
}

func (i rentalIdleness) busy() bool { return i.Queued > 0 || i.Running > 0 || i.Owed }

func observeRentalIdle(st *records.Store, row records.Rental) (rentalIdleness, *exit.Error) {
	var idle rentalIdleness
	var problem *exit.Error
	if idle.Queued, idle.Running, problem = st.RentalRunCounts(row.ID); problem != nil {
		return idle, problem
	}
	if row.ReadyAt != "" {
		ready, err := time.Parse(time.RFC3339Nano, row.ReadyAt)
		if err != nil {
			return idle, exit.Internalf("rental %s has an invalid ready timestamp: %s", row.ID, err)
		}
		idle.Since = ready
	}
	last, found, problem := st.RentalLastSettlement(row.ID)
	if problem != nil {
		return idle, problem
	}
	if found && last.ClosedAt.After(idle.Since) {
		idle.Since = last.ClosedAt
	}
	idle.Spent = row.ManagedRequestID != "" && (!found || last.Kind == "job" || last.ClosedAt.IsZero())
	if idle.Owed, problem = rentalOwedBy(st, row); problem != nil {
		return idle, problem
	}
	if idle.Owed {
		// The buyer has not settled: the rental's reason is live, not spent.
		idle.Spent = false
	}
	return idle, nil
}

// rentalOwedBy answers whether the request this rental was bought for still owes it a
// pin: not yet settled, and not durably routed to another machine. The debt is read from
// the durable rows alone — `rentals.managed_request_id` is written with the paid create —
// so it holds across a daemon restart and across the whole boot, when the request row
// still says worker=” because the pin is routing's output (cl-092) and routing has not
// run yet. Without this, a booting or freshly ready pod looks unowed to every observer
// and an idle sweep — or any fleet hygiene reading the same records — reaps a pod whose
// buyer is still queued for it (cl-113, observed live: rental pr-b192da1a released
// mid-boot while req-ff3f79f4 waited on it).
func rentalOwedBy(st *records.Store, row records.Rental) (bool, *exit.Error) {
	if row.ManagedRequestID == "" {
		return false, nil
	}
	owing, problem := st.RequestRow(row.ManagedRequestID)
	if problem != nil {
		return false, problem
	}
	return owing != nil && !settledRequest(owing.State) &&
		(owing.Worker == "" || owing.Worker == row.ID), nil
}

// releaseAt is when the idle release is due, or false while the rental is busy (queued,
// running, or owed by the request that bought it), still booting, or exempt because
// rentals.idle_release_s is zero. A rental whose ready_at is unset and which has never
// settled an attempt has a zero Since and is therefore never idle-released: a pod is
// never reaped mid-boot (th-105/cl-078) — a terminally failed acquisition ends through
// the hub's own `failed` state and reconciliation, not through this clock.
func (i rentalIdleness) releaseAt(grace time.Duration) (time.Time, bool) {
	if grace <= 0 || i.busy() || i.Since.IsZero() {
		return time.Time{}, false
	}
	if i.Spent {
		return i.Since, true
	}
	return i.Since.Add(grace), true
}

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

func (m *managedRentals) admit(skuName string) (string, hub.RentalSKU, *exit.Error) {
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
	for _, sku := range skus {
		if sku.Name == skuName {
			return line, sku, m.admitLocked(sku)
		}
	}
	return "", hub.RentalSKU{}, exit.Named(exit.Validation, "rental.sku_unavailable",
		"Tensorhub currently offers no rental SKU %q", skuName)
}

// acquire is the capacity decision for a --rental request no rental holds a placement
// for (residency-aware-routing.md §3.2, D4): among the ready rentals of the request's
// class the orchestrator's RankRentals puts the one whose store already holds the
// placement's manifests first, then the fewest missing bytes, then the most room — disk
// holdings ORDER the candidates, never veto. With no ready rental it BUYS a pod: --rental
// is permission AND intent to spend (owner ruling 2026-09-03), regardless of what local
// holds on disk. The chosen rental is pinned to THIS request alone — every other queued
// --rental request keeps routing over local and every rental by score, and takes its pin
// from dispatch (cl-092 step 4).
func (m *managedRentals) acquire(req records.Request) (orchestrator.RentalDecision, string, *exit.Error) {
	var none orchestrator.RentalDecision
	m.mu.Lock()
	defer m.mu.Unlock()
	if !req.Rental || req.Worker != "" {
		return none, "", exit.Internalf("request %s is not an unassigned --rental request", req.ID)
	}
	if problem := m.reconcileLocked(); problem != nil {
		return none, "", problem
	}
	needsCPU := !req.NeedsAccelerator
	skus, problem := m.catalogLocked()
	if problem != nil {
		return none, "", problem
	}
	rows, problem := m.store.Rentals()
	if problem != nil {
		return none, "", problem
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].HourlyRateUSDMicros != rows[j].HourlyRateUSDMicros {
			return rows[i].HourlyRateUSDMicros < rows[j].HourlyRateUSDMicros
		}
		return rows[i].ID < rows[j].ID
	})
	var ready []string
	for _, row := range rows {
		if (row.AcceleratorModel == "CPU") != needsCPU {
			continue
		}
		if row.State != hub.RentalReady && row.State != "attached" {
			continue
		}
		if row.Address == "" || row.CertPath == "" {
			continue
		}
		ready = append(ready, row.ID)
	}
	ready = m.owner.ModeCompatibleRentals(ready, req.IsJob())
	if len(ready) > 0 {
		ranked := m.owner.RankRentals(ready, req.Models)
		chosen := ranked[0].RentalID
		pinned, pinProblem := m.store.PinRental(req.ID, chosen)
		if pinProblem != nil {
			return none, "", pinProblem
		}
		if !pinned {
			return none, "", exit.New(exit.Canceled,
				"request %s settled before rental assignment", req.ID)
		}
		line, lineProblem := m.lineLocked()
		return orchestrator.RentalDecision{RentalID: chosen, Candidates: ranked}, line, lineProblem
	}

	sku, mismatch, found := rental.CheapestCompatibleSKU(skus, req.NeedsAccelerator,
		releaseConstraints(m.ctx, req))
	if !found && mismatch != "" {
		// Refused BEFORE the paid ask, in the pod's own vocabulary. Publication stays
		// base-independent: the release is published and simply unqualified here.
		return none, "", exit.Named(exit.Unavailable, "rental.package_base_incompatible",
			"no rentable machine can run %s@%s — %s", req.Package, req.Release, mismatch).
			WithRemedy("publish a release whose requirements one of Tensorhub's offered base images satisfies")
	}
	if !found {
		return none, "", exit.Named(exit.Capacity, "rental.no_skus",
			"Tensorhub currently offers no compatible rental SKU")
	}
	if problem := m.admitLocked(sku); problem != nil {
		return none, "", problem
	}
	current, currentProblem := m.store.RequestRow(req.ID)
	if currentProblem != nil {
		return none, "", currentProblem
	}
	if current == nil || settledRequest(current.State) {
		return none, "", exit.New(exit.Canceled, "request %s settled before rental acquisition", req.ID)
	}
	fmt.Fprintf(m.ctx.Out, "rentals: renting %s at %s\n", sku.Name, skuRate(sku))
	row, _, _, problem := acquireRental(m.ctx, m.layout, m.store, sku.Name,
		"managed-rental-"+req.ID, "",
		sku.PriceUSDMicrosPerHour, sku.StorageUSDMicrosPerHour,
		m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros, time.Time{}, req.ID)
	if problem != nil {
		return none, "", problem
	}
	pinned, problem := m.store.PinRental(req.ID, row.ID)
	if problem != nil {
		return none, "", problem
	}
	if !pinned {
		_, releaseProblem := m.releaseLocked(row.ID)
		if releaseProblem != nil {
			return none, "", releaseProblem
		}
		return none, "", exit.New(exit.Canceled,
			"request %s settled while rental %s was starting; the rental was released", req.ID, row.ID)
	}
	line, problem := m.lineLocked()
	return orchestrator.RentalDecision{RentalID: row.ID, Bought: true}, line, problem
}

func (m *managedRentals) catalogLocked() ([]hub.RentalSKU, *exit.Error) {
	hctx, cancel := hub.Context()
	defer cancel()
	return client(m.ctx).RentalSKUs(hctx)
}

func (m *managedRentals) admitLocked(sku hub.RentalSKU) *exit.Error {
	cap := m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros
	if cap <= 0 {
		return exit.Named(exit.Usage, "rental.spend_cap_required",
			"rentals.max_hourly_spend_usd must be positive before Creator rents a pod")
	}
	_, burn, problem := m.totalsLocked()
	if problem != nil {
		return problem
	}
	// The cap is a SPEND cap and burn is billed-truth money (th-120), so the
	// figure admitted is the estimated TOTAL the pod will bill — the GPU rate
	// plus the SKU's storage adder — never the GPU rate alone (th-126).
	if burn > cap || sku.PriceUSDMicrosPerHour+sku.StorageUSDMicrosPerHour > cap-burn {
		return exit.Named(exit.Capacity, "rental.fleet_spend_cap",
			"rental %s at %s would exceed %s",
			sku.Name, skuRate(sku), usdPerHour(cap)).
			WithRemedy("raise rentals.max_hourly_spend_usd or end another rental")
	}
	return nil
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

// releaseOrphaned resumes the policy after a daemon restart: every rental is reconciled
// with the hub and then observed as the sweep observes it, so a job's rental releases
// now and a warm one keeps only the unspent remainder of its grace.
func (m *managedRentals) releaseOrphaned() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		fmt.Fprintf(m.ctx.Out, "rental reconciliation deferred: %s\n", problem.Message)
		return
	}
	m.sweepLocked()
}

// watch is the idle release's own loop. It re-reads the records at pollCadence — the
// resolution every rental verb already samples a rental at — and acts only on what they
// say; the grace is the debounce, and a sample that finds nothing to do costs a few local
// reads. It returns when quit closes, or at once when idle release is configured off.
func (m *managedRentals) watch(quit <-chan struct{}) {
	tick := time.NewTicker(pollCadence)
	defer tick.Stop()
	for {
		select {
		case <-quit:
			return
		case <-tick.C:
		}
		m.mu.Lock()
		if !m.closed {
			m.sweepLocked()
		}
		owner := m.owner
		m.mu.Unlock()
		if owner != nil {
			// Hub recovery is an observed fleet change. Re-ask durable queued work outside
			// the fleet lock; selection calls back into this object.
			owner.WakeQueue()
		}
	}
}

func (m *managedRentals) sweepLocked() {
	rows, problem := m.store.Rentals()
	if problem != nil {
		m.sayLocked("", "idle release deferred: "+problem.Message)
		return
	}
	for _, row := range rows {
		if _, problem := m.observeLocked(row); problem != nil {
			m.sayLocked(row.ID, fmt.Sprintf("rental %s release deferred: %s; retrying every %s",
				row.ID, problem.Message, idleReleaseRetry(m.ctx.Cfg.RentalsIdleRelease)))
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
	idle, problem := observeRentalIdle(m.store, row)
	if problem != nil {
		return "", problem
	}
	due, eligible := idle.releaseAt(m.ctx.Cfg.RentalsIdleRelease)
	if !eligible {
		delete(m.said, row.ID)
		return m.lineLocked()
	}
	if now := time.Now(); now.Before(due) {
		m.sayLocked(row.ID, fmt.Sprintf("rental %s (%s) idle since %s; released at %s unless work arrives",
			row.ID, row.MachineName, idle.Since.UTC().Format("15:04:05"), due.UTC().Format("15:04:05")))
		return m.lineLocked()
	}
	line, problem := m.releaseLocked(row.ID)
	if problem != nil {
		if m.retryAt == nil {
			m.retryAt = map[string]time.Time{}
		}
		m.retryAt[row.ID] = time.Now().Add(idleReleaseRetry(m.ctx.Cfg.RentalsIdleRelease))
		return "", problem
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
	owed, problem := rentalOwedBy(m.store, *row)
	if problem != nil {
		return "", problem
	}
	if owed {
		return m.lineLocked()
	}
	operationKey, problem := m.store.RequestRentalRelease(id)
	if problem != nil {
		return "", problem
	}
	if row.State != hub.RentalReleaseRequested {
		row.State = hub.RentalReleaseRequested
		if problem := m.store.RecordRental(*row); problem != nil {
			return "", problem
		}
		if line, lineProblem := m.lineLocked(); lineProblem == nil {
			fmt.Fprintln(m.ctx.Out, line)
		}
	}
	hctx, cancel := hub.Context()
	problem = client(m.ctx).Release(hctx, id, "")
	cancel()
	if problem != nil && problem.Code != exit.NotFound {
		return "", problem
	}
	for problem == nil {
		hctx, cancel = hub.Context()
		remote, observed := client(m.ctx).Rental(hctx, id)
		cancel()
		switch {
		case observed == nil && remote.State == hub.RentalReleased:
			problem = exit.New(exit.NotFound, "rental released")
		case observed == nil:
			row.State = remote.State
			copyRentalFailure(row, remote)
			if update := m.store.RecordRental(*row); update != nil {
				return "", update
			}
			time.Sleep(pollCadence)
		case observed.Code == exit.NotFound:
			problem = observed
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

func (m *managedRentals) reconcileLocked() *exit.Error {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return problem
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
		row.State = remote.State
		copyRentalFailure(&row, remote)
		if problem := m.store.RecordRental(row); problem != nil {
			return problem
		}
	}
	return nil
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
	return fmt.Sprintf("rentals: %d remote %s running · %s of %s",
		count, machine, usdPerHour(burn), usdPerHour(m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros)), nil
}

func (m *managedRentals) totalsLocked() (int, int64, *exit.Error) {
	count, burn, problem := m.store.RentalFleetTotals()
	if problem != nil || burn < 0 {
		return 0, 0, problem
	}
	return count, burn, nil
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
	whole, fraction := micros/1_000_000, micros%1_000_000
	decimal := fmt.Sprintf("%06d", fraction)
	for len(decimal) > 2 && decimal[len(decimal)-1] == '0' {
		decimal = decimal[:len(decimal)-1]
	}
	return fmt.Sprintf("$%d.%s", whole, decimal)
}

func settledRequest(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

// releaseConstraints reads the release's own immutable requirements so the SKU choice
// above can decline a base that already contradicts them. It is ADVISORY: a package this
// host cannot name, or a hub that will not answer, yields no constraints and therefore no
// refusal — the pod remains the authority on whether the package runs (th-075).
func releaseConstraints(ctx *Context, req records.Request) rental.Constraints {
	if req.Package == "" || req.Release == "" {
		return rental.Constraints{}
	}
	ref, problem := hub.ParseRef(req.Package)
	if problem != nil {
		return rental.Constraints{}
	}
	hctx, cancel := hub.Context()
	defer cancel()
	detail, problem := client(ctx).PackageRelease(hctx, ref, req.Release)
	if problem != nil {
		return rental.Constraints{}
	}
	requirements, requiresPython, problem := detail.Constraints()
	if problem != nil {
		return rental.Constraints{}
	}
	return rental.Constraints{Requirements: requirements, RequiresPython: requiresPython}
}
