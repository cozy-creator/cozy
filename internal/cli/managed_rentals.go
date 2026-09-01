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
// is worth keeping for a handful of those and no longer.
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

func (i rentalIdleness) busy() bool { return i.Queued > 0 || i.Running > 0 }

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
	return idle, nil
}

// releaseAt is when the idle release is due, or false while the rental is busy, still
// booting, or exempt because rentals.idle_release_s is zero.
func (i rentalIdleness) releaseAt(grace time.Duration) (time.Time, bool) {
	if grace <= 0 || i.busy() || i.Since.IsZero() {
		return time.Time{}, false
	}
	if i.Spent {
		return i.Since, true
	}
	return i.Since.Add(grace), true
}

func (m *managedRentals) status() (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		return "", problem
	}
	return m.lineLocked()
}

func (m *managedRentals) admit(skuName string) (string, int64, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		return "", 0, problem
	}
	line, problem := m.lineLocked()
	if problem != nil {
		return "", 0, problem
	}
	skus, problem := m.catalogLocked()
	if problem != nil {
		return "", 0, problem
	}
	for _, sku := range skus {
		if sku.Name == skuName {
			return line, sku.PriceUSDMicrosPerHour, m.admitLocked(sku)
		}
	}
	return "", 0, exit.Named(exit.Validation, "rental.sku_unavailable",
		"Tensorhub currently offers no rental SKU %q", skuName)
}

func (m *managedRentals) acquire(req records.Request) (string, string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !req.Rental || req.Worker != "" {
		return "", "", exit.Internalf("request %s is not an unassigned --rental request", req.ID)
	}
	if problem := m.reconcileLocked(); problem != nil {
		return "", "", problem
	}
	needsCPU := !req.NeedsAccelerator
	skus, problem := m.catalogLocked()
	if problem != nil {
		return "", "", problem
	}
	rows, problem := m.store.Rentals()
	if problem != nil {
		return "", "", problem
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].HourlyRateUSDMicros != rows[j].HourlyRateUSDMicros {
			return rows[i].HourlyRateUSDMicros < rows[j].HourlyRateUSDMicros
		}
		return rows[i].ID < rows[j].ID
	})
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
		if batchProblem := m.store.AssignManagedRentalClass(row.ID, needsCPU); batchProblem != nil {
			return "", "", batchProblem
		}
		assigned, assignProblem := m.store.AssignManagedRental(req.ID, row.ID)
		if assignProblem != nil {
			return "", "", assignProblem
		}
		if !assigned {
			return "", "", exit.New(exit.Canceled,
				"request %s settled before rental assignment", req.ID)
		}
		line, lineProblem := m.lineLocked()
		return row.ID, line, lineProblem
	}

	sku, mismatch, found := rental.CheapestCompatibleSKU(skus, req.NeedsAccelerator,
		releaseConstraints(m.ctx, req))
	if !found && mismatch != "" {
		// Refused BEFORE the paid ask, in the pod's own vocabulary. Publication stays
		// base-independent: the release is published and simply unqualified here.
		return "", "", exit.Named(exit.Unavailable, "rental.package_base_incompatible",
			"no rentable machine can run %s@%s — %s", req.Package, req.Release, mismatch).
			WithRemedy("publish a release whose requirements one of Tensorhub's offered base images satisfies")
	}
	if !found {
		return "", "", exit.Named(exit.Capacity, "rental.no_skus",
			"Tensorhub currently offers no compatible rental SKU")
	}
	if problem := m.admitLocked(sku); problem != nil {
		return "", "", problem
	}
	current, currentProblem := m.store.RequestRow(req.ID)
	if currentProblem != nil {
		return "", "", currentProblem
	}
	if current == nil || settledRequest(current.State) {
		return "", "", exit.New(exit.Canceled, "request %s settled before rental acquisition", req.ID)
	}
	fmt.Fprintf(m.ctx.Out, "rentals: renting %s at %s\n", sku.Name, usdPerHour(sku.PriceUSDMicrosPerHour))
	row, _, _, problem := acquireRental(m.ctx, m.layout, m.store, sku.Name,
		"managed-rental-"+req.ID, "",
		sku.PriceUSDMicrosPerHour, m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros, time.Time{}, req.ID)
	if problem != nil {
		return "", "", problem
	}
	if batchProblem := m.store.AssignManagedRentalClass(row.ID, needsCPU); batchProblem != nil {
		return "", "", batchProblem
	}
	assigned, problem := m.store.AssignManagedRental(req.ID, row.ID)
	if problem != nil {
		return "", "", problem
	}
	if !assigned {
		_, releaseProblem := m.releaseLocked(row.ID)
		if releaseProblem != nil {
			return "", "", releaseProblem
		}
		return "", "", exit.New(exit.Canceled,
			"request %s settled while rental %s was starting; the rental was released", req.ID, row.ID)
	}
	line, problem := m.lineLocked()
	return row.ID, line, problem
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
	if burn > cap || sku.PriceUSDMicrosPerHour > cap-burn {
		return exit.Named(exit.Capacity, "rental.fleet_spend_cap",
			"rental %s at %s would exceed %s",
			sku.Name, usdPerHour(sku.PriceUSDMicrosPerHour), usdPerHour(cap)).
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
	if m.ctx.Cfg.RentalsIdleRelease <= 0 {
		return
	}
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
		m.mu.Unlock()
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
// and the settlement hook both arrive here, and only a pod with nothing pinned to it goes.
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
		if remote.HourlyRateUSDMicros != row.HourlyRateUSDMicros {
			return exit.Named(exit.Conflict, "rental.hourly_rate_changed",
				"rental %s changed its locked Cozy retail rate", row.ID)
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
	whole, fraction := micros/1_000_000, micros%1_000_000
	decimal := fmt.Sprintf("%06d", fraction)
	for len(decimal) > 2 && decimal[len(decimal)-1] == '0' {
		decimal = decimal[:len(decimal)-1]
	}
	return fmt.Sprintf("$%d.%s/hour", whole, decimal)
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
