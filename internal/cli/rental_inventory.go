package cli

import (
	"encoding/json"
	"slices"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// readRentalInventory is one hub's reconciled fleet, or with allHubs every hub's this
// host holds rentals on. A listing of one hub still counts this host's live rentals
// on the others, so switching hubs never hides a machine that is billing.
func readRentalInventory(st *records.Store, fleet *managedRentals, origin string, allHubs, reconcile bool) (api.RentalInventory, *exit.Error) {
	// Keep account totals and the corresponding unrecorded set on one observation.
	fleet.mu.Lock()
	defer fleet.mu.Unlock()
	origin = fleet.origin(origin)
	if !allHubs {
		result, problem := fleet.inventoryLocked(st, origin, reconcile)
		if problem != nil {
			return result, problem
		}
		result.OtherHubs, problem = otherHubRentals(st, fleet, origin)
		return result.Current(), problem
	}
	origins, problem := fleet.originsLocked()
	if problem != nil {
		return api.RentalInventory{}, problem
	}
	if !slices.Contains(origins, origin) {
		origins = append(origins, origin)
	}
	var merged api.RentalInventory
	for _, each := range origins {
		result, problem := fleet.inventoryLocked(st, each, reconcile)
		if problem != nil {
			return merged, problem
		}
		if result.HubUnanswered != nil && merged.HubUnanswered == nil {
			merged.HubUnanswered = result.HubUnanswered
		}
		merged.MachinesRunning += result.MachinesRunning
		merged.HourlySpendUSDMicros += result.HourlySpendUSDMicros
		merged.IdleReleaseSeconds = result.IdleReleaseSeconds
		merged.Rentals = append(merged.Rentals, result.Rentals...)
		merged.Unrecorded = append(merged.Unrecorded, result.Unrecorded...)
		merged.Pending = append(merged.Pending, result.Pending...)
	}
	return merged.Current(), nil
}

// otherHubRentals counts this host's live rental records on hubs other than origin.
func otherHubRentals(st *records.Store, fleet *managedRentals, origin string) ([]api.HubRentals, *exit.Error) {
	rows, problem := st.Rentals()
	if problem != nil {
		return nil, problem
	}
	counts := map[string]int{}
	for _, row := range rows {
		if each := fleet.origin(row.Hub); each != origin && !hub.RentalAbsent(row.State) {
			counts[each]++
		}
	}
	var out []api.HubRentals
	for each, count := range counts {
		out = append(out, api.HubRentals{Hub: each, Rentals: count})
	}
	slices.SortFunc(out, func(a, b api.HubRentals) int { return strings.Compare(a.Hub, b.Hub) })
	return out, nil
}

func (fleet *managedRentals) inventoryLocked(st *records.Store, origin string, reconcile bool) (api.RentalInventory, *exit.Error) {
	var result api.RentalInventory
	if reconcile {
		if problem := fleet.reconcileLocked(origin); problem != nil && !hub.Unanswered(problem) {
			return result, problem
		}
	}
	// totalsLocked propagates a failed Hub census and refuses when the Hub has
	// no listing route. A Hub that did not answer at all yields this host's own
	// records marked as such, never totals: the caller must still fail.
	count, burn, problem := fleet.totalsLocked(origin)
	switch {
	case hub.Unanswered(problem):
		result.HubUnanswered = problem
	case problem != nil:
		return result, problem
	default:
		result.MachinesRunning, result.HourlySpendUSDMicros = count, burn
	}
	census := fleet.censusLocked(origin)
	result.IdleReleaseSeconds = int64(rental.IdleTimeout / time.Second)
	rows, problem := st.Rentals()
	if problem != nil {
		return result, problem
	}
	boughtFor, problem := st.RentalProvenance()
	if problem != nil {
		return result, problem
	}
	live := make(map[string]hub.Rental, len(census.live))
	for _, seen := range census.live {
		live[seen.ID] = seen
	}
	attached := make(map[string]bool, len(rows))
	for _, row := range rows {
		attached[row.ID] = true
		if fleet.origin(row.Hub) != origin {
			continue
		}
		idle, problem := fleet.observeIdle(row)
		if problem != nil {
			return result, problem
		}
		summary := api.RentalSummary{
			ID: row.ID, MachineName: row.MachineName, SKU: row.SKU, State: row.State,
			AcceleratorModel: row.AcceleratorModel, AcceleratorCount: row.AcceleratorCount,
			HourlyRateUSDMicros: row.HourlyRateUSDMicros,
			Address:             row.Address, MediaAddress: row.MediaAddress, Hub: row.Hub,
			RentedAt: row.RentedAt, ReadyAt: row.ReadyAt, BoughtFor: boughtFor[row.ID],
			Activity: &api.RentalActivity{Running: idle.Running, Queued: idle.Queued},
			Failure: hub.RentalFailure{
				Code: row.Failure.Code, BaseWorkerImageDigest: row.Failure.BaseWorkerImageDigest,
				Provider: row.Failure.Provider, ProviderResourceID: row.Failure.ProviderResourceID,
				ProviderHostID: row.Failure.ProviderHostID, ProviderState: row.Failure.ProviderState,
				ContainerState: row.Failure.ContainerState,
			},
			BaseWorkerImageDigest: live[row.ID].BaseWorkerImageDigest,
			BaseWorkerImageTag:    live[row.ID].BaseWorkerImageTag,
			HubUnknown:            census.hubUnknown[row.ID],
		}
		if idle.PendingPreparation > 0 {
			summary.Activity = nil
		}
		if due, eligible := idle.ReleaseAt(); eligible {
			summary.Activity.IdleSince = idle.Since.UTC().Format(time.RFC3339)
			summary.Activity.ReleaseDue = due.UTC().Format(time.RFC3339)
		}
		result.Rentals = append(result.Rentals, summary)
	}
	hubNamed := map[string]bool{}
	for _, seen := range census.unrecorded {
		hubNamed[seen.ID], hubNamed[seen.Name] = true, true
		result.Unrecorded = append(result.Unrecorded, api.RentalSummary{
			ID: seen.ID, MachineName: seen.Name, State: seen.State,
			AcceleratorModel: seen.AcceleratorModel, AcceleratorCount: seen.AcceleratorCount,
			HourlyRateUSDMicros: seen.HourlyRateUSDMicros, Address: seen.Address,
			MediaAddress: seen.MediaAddress, Hub: origin, RentedAt: seen.CreatedAt,
			ProviderState: seen.ProviderState, ContainerState: seen.ContainerState,
			BaseWorkerImageDigest: seen.BaseWorkerImageDigest, BaseWorkerImageTag: seen.BaseWorkerImageTag,
		})
	}
	open, problem := st.ActiveRentalOperations()
	if problem != nil {
		return result, problem
	}
	for _, op := range open {
		if attached[op.RentalID] || fleet.origin(op.Hub) != origin {
			continue
		}
		var request hub.RentalRequest
		_ = json.Unmarshal(op.RequestBody, &request)
		if hubNamed[op.RentalID] || hubNamed[request.Name] {
			continue
		}
		result.Pending = append(result.Pending, api.RentalSummary{
			ID: op.RentalID, MachineName: either(request.Name, op.Key), SKU: request.SKU,
			State: op.State, HourlyRateUSDMicros: op.HourlyRateUSDMicros,
			Hub: op.Hub, RentedAt: op.CreatedAt, BoughtFor: op.ManagedRequestID, Operation: op.Key,
		})
	}
	return result, nil
}
