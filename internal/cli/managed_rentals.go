package cli

import (
	"fmt"
	"math"
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
	mu     sync.Mutex
	ctx    *Context
	layout home.Layout
	store  *records.Store
	owner  *orchestrator.Orchestrator
}

func (m *managedRentals) status() (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if problem := m.reconcileLocked(); problem != nil {
		return "", problem
	}
	return m.lineLocked()
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
		if row.State != hub.RentalReady && row.State != "attached" {
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
		line, lineProblem := m.lineLocked()
		return row.ID, line, lineProblem
	}

	cap := m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros
	if cap <= 0 {
		return "", "", exit.Named(exit.Usage, "rental.spend_cap_required",
			"rentals.max_hourly_spend_usd must be positive before Creator rents a pod")
	}
	hctx, cancel := hub.Context()
	skus, problem := client(m.ctx).RentalSKUs(hctx)
	cancel()
	if problem != nil {
		return "", "", problem
	}
	if len(skus) == 0 {
		return "", "", exit.Named(exit.Capacity, "rental.no_skus",
			"Tensorhub currently offers no rental SKU")
	}
	sort.Slice(skus, func(i, j int) bool {
		if skus[i].PriceUSDMicrosPerHour != skus[j].PriceUSDMicrosPerHour {
			return skus[i].PriceUSDMicrosPerHour < skus[j].PriceUSDMicrosPerHour
		}
		return skus[i].Name < skus[j].Name
	})
	_, burn, problem := m.totalsLocked()
	if problem != nil {
		return "", "", problem
	}
	sku := skus[0]
	if burn > cap || sku.PriceUSDMicrosPerHour > cap-burn {
		return "", "", exit.Named(exit.Capacity, "rental.fleet_spend_cap",
			"the cheapest rental %s at %s would exceed %s",
			sku.Name, usdPerHour(sku.PriceUSDMicrosPerHour), usdPerHour(cap)).
			WithRemedy("raise rentals.max_hourly_spend_usd or end another rental")
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
		"managed-rental-"+req.ID, "cozy run --rental "+req.Package, time.Time{}, req.ID)
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

func (m *managedRentals) release(id string) (string, *exit.Error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.releaseLocked(id)
}

// releaseOrphaned closes the only crash window after terminal acknowledgement:
// the request is settled, but the daemon died before its managed pod disappeared.
// An owed or running origin request is left for normal recovery.
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
		origin, originProblem := m.store.RequestRow(row.ManagedRequestID)
		if originProblem != nil || origin != nil && !settledRequest(origin.State) {
			continue
		}
		line, releaseProblem := m.releaseLocked(row.ID)
		if releaseProblem != nil {
			fmt.Fprintf(m.ctx.Out, "managed rental %s release deferred: %s\n", row.ID, releaseProblem.Message)
			continue
		}
		fmt.Fprintln(m.ctx.Out, line)
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
	problem = client(m.ctx).Release(hctx, id, "cozy run --rental queue drained")
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
	return fmt.Sprintf("rentals: %d remote machines running · %s of %s",
		count, usdPerHour(burn), usdPerHour(m.ctx.Cfg.RentalsMaxHourlySpendUSDMicros)), nil
}

func (m *managedRentals) totalsLocked() (int, int64, *exit.Error) {
	rows, problem := m.store.Rentals()
	if problem != nil {
		return 0, 0, problem
	}
	var burn int64
	for _, row := range rows {
		if row.HourlyRateUSDMicros <= 0 || burn > math.MaxInt64-row.HourlyRateUSDMicros {
			return 0, 0, exit.Internalf("rental fleet hourly burn is invalid")
		}
		burn += row.HourlyRateUSDMicros
	}
	return len(rows), burn, nil
}

func usdPerHour(micros int64) string {
	whole, fraction := micros/1_000_000, micros%1_000_000
	if fraction == 0 {
		return fmt.Sprintf("$%d/hour", whole)
	}
	decimal := strings.TrimRight(fmt.Sprintf("%06d", fraction), "0")
	return fmt.Sprintf("$%d.%s/hour", whole, decimal)
}

func settledRequest(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}
