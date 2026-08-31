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

// managedRentals is deliberately small: one mutex serializes the local fleet
// ceiling, reuse decision, and paid POST. The durable rental operation and
// request rows remain the crash-recovery authority.
type managedRentals struct {
	mu           sync.Mutex
	ctx          *Context
	layout       home.Layout
	store        *records.Store
	owner        *orchestrator.Orchestrator
	idleGrace    time.Duration
	idleSequence uint64
	idleTimers   map[string]idleRelease
	closed       bool
}

type idleRelease struct {
	sequence uint64
	timer    *time.Timer
}

const managedRentalIdleGrace = 5 * time.Minute

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
	needsCPU := len(req.Models) == 0 && req.JobGPUCount == 0
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
		if !containsString(req.AcceptableWheelhouseManifestDigests,
			row.WheelhouseManifestDigest) {
			continue
		}
		queued, running, countsProblem := m.store.RentalRunCounts(row.ID)
		if countsProblem != nil {
			return "", "", countsProblem
		}
		if queued != 0 || running != 0 {
			continue
		}
		assigned, assignProblem := m.store.AssignManagedRental(req.ID, row.ID)
		if assignProblem != nil {
			return "", "", assignProblem
		}
		if !assigned {
			return "", "", exit.New(exit.Canceled,
				"request %s settled before rental assignment", req.ID)
		}
		m.cancelIdleReleaseLocked(row.ID)
		line, lineProblem := m.lineLocked()
		return row.ID, line, lineProblem
	}

	skus, problem := m.catalogLocked()
	if problem != nil {
		return "", "", problem
	}
	compatible := skus[:0]
	for _, sku := range skus {
		if (sku.AcceleratorModel == "CPU") == needsCPU {
			compatible = append(compatible, sku)
		}
	}
	skus = compatible
	if len(skus) == 0 {
		return "", "", exit.Named(exit.Capacity, "rental.no_skus",
			"Tensorhub currently offers no compatible rental SKU")
	}
	sort.Slice(skus, func(i, j int) bool {
		if skus[i].PriceUSDMicrosPerHour != skus[j].PriceUSDMicrosPerHour {
			return skus[i].PriceUSDMicrosPerHour < skus[j].PriceUSDMicrosPerHour
		}
		return skus[i].Name < skus[j].Name
	})
	sku := skus[0]
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
	row, _, _, problem := acquireRental(m.ctx, m.layout, m.store, sku.Name, "",
		"managed-rental-"+req.ID, "",
		sku.PriceUSDMicrosPerHour, m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros, time.Time{}, req.ID,
		req.AcceptableWheelhouseManifestDigests)
	if problem != nil {
		return "", "", problem
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

func containsString(values []string, wanted string) bool {
	index := sort.SearchStrings(values, wanted)
	return wanted != "" && index < len(values) && values[index] == wanted
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

func (m *managedRentals) release(id string) (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.releaseWhenIdleLocked(id)
}

// releaseOrphaned resumes the post-terminal policy after a daemon restart: jobs and
// pre-attempt failures release now, while a warm serving rental keeps only the unspent
// remainder of its original grace. An owed or running request is left for recovery.
func (m *managedRentals) releaseOrphaned() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		fmt.Fprintf(m.ctx.Out, "managed rental reconciliation deferred: %s\n", problem.Message)
		return
	}
	rows, problem := m.store.Rentals()
	if problem != nil {
		fmt.Fprintf(m.ctx.Out, "managed rental reconciliation deferred: %s\n", problem.Message)
		return
	}
	for _, row := range rows {
		if row.ManagedRequestID == "" {
			continue
		}
		queued, running, countsProblem := m.store.RentalRunCounts(row.ID)
		if countsProblem != nil || queued != 0 || running != 0 {
			continue
		}
		line, releaseProblem := m.releaseWhenIdleLocked(row.ID)
		if releaseProblem != nil {
			fmt.Fprintf(m.ctx.Out, "managed rental %s release deferred: %s\n", row.ID, releaseProblem.Message)
			continue
		}
		fmt.Fprintln(m.ctx.Out, line)
	}
}

func (m *managedRentals) releaseWhenIdleLocked(id string) (string, *exit.Error) {
	if m.closed {
		return m.lineLocked()
	}
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return "", problem
	}
	if row == nil || row.ManagedRequestID == "" {
		return m.lineLocked()
	}
	queued, running, problem := m.store.RentalRunCounts(id)
	if problem != nil {
		return "", problem
	}
	if queued != 0 || running != 0 {
		return m.lineLocked()
	}
	last, found, problem := m.store.RentalLastSettlement(id)
	if problem != nil {
		return "", problem
	}
	if found && last.Kind != "job" && !last.ClosedAt.IsZero() {
		deadline := last.ClosedAt.Add(m.grace())
		if time.Now().Before(deadline) {
			m.scheduleIdleReleaseLocked(id, deadline)
			return m.lineLocked()
		}
	}
	return m.releaseLocked(id)
}

func (m *managedRentals) grace() time.Duration {
	if m.idleGrace > 0 {
		return m.idleGrace
	}
	return managedRentalIdleGrace
}

func (m *managedRentals) scheduleIdleReleaseLocked(id string, deadline time.Time) {
	if m.closed {
		return
	}
	m.cancelIdleReleaseLocked(id)
	if m.idleTimers == nil {
		m.idleTimers = map[string]idleRelease{}
	}
	m.idleSequence++
	sequence := m.idleSequence
	timer := time.AfterFunc(time.Until(deadline), func() { m.releaseIdle(id, sequence) })
	m.idleTimers[id] = idleRelease{sequence: sequence, timer: timer}
}

func (m *managedRentals) cancelIdleReleaseLocked(id string) {
	if pending, ok := m.idleTimers[id]; ok {
		pending.timer.Stop()
		delete(m.idleTimers, id)
	}
}

func (m *managedRentals) releaseIdle(id string, sequence uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return
	}
	pending, ok := m.idleTimers[id]
	if !ok || pending.sequence != sequence {
		return
	}
	delete(m.idleTimers, id)
	line, problem := m.releaseWhenIdleLocked(id)
	if problem != nil {
		fmt.Fprintf(m.ctx.Out, "managed rental %s release deferred: %s\n", id, problem.Message)
		return
	}
	if line != "" {
		fmt.Fprintln(m.ctx.Out, line)
	}
}

func (m *managedRentals) close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	for id, pending := range m.idleTimers {
		pending.timer.Stop()
		delete(m.idleTimers, id)
	}
}

func (m *managedRentals) releaseLocked(id string) (string, *exit.Error) {
	row, problem := m.store.RentalRow(id)
	if problem != nil {
		return "", problem
	}
	if row == nil || row.ManagedRequestID == "" {
		return m.lineLocked()
	}
	queued, running, problem := m.store.RentalRunCounts(id)
	if problem != nil {
		return "", problem
	}
	if queued != 0 || running != 0 {
		return m.lineLocked()
	}
	m.cancelIdleReleaseLocked(id)
	operationKey, problem := m.store.RequestRentalRelease(id)
	if problem != nil {
		return "", problem
	}
	row.State = hub.RentalReleaseRequested
	if problem := m.store.RecordRental(*row); problem != nil {
		return "", problem
	}
	if line, lineProblem := m.lineLocked(); lineProblem == nil {
		fmt.Fprintln(m.ctx.Out, line)
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
			m.cancelIdleReleaseLocked(row.ID)
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
			if observed.Code == exit.NotFound {
				m.cancelIdleReleaseLocked(row.ID)
				if m.owner != nil {
					m.owner.DetachRental(row.ID)
				}
				if _, problem := rental.Forget(m.layout, m.store, row.ID); problem != nil {
					return problem
				}
				continue
			}
			return observed
		}
		if remote.HourlyRateUSDMicros != row.HourlyRateUSDMicros {
			return exit.Named(exit.Conflict, "rental.hourly_rate_changed",
				"rental %s changed its locked Cozy retail rate", row.ID)
		}
		if remote.State == hub.RentalReleased {
			m.cancelIdleReleaseLocked(row.ID)
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
