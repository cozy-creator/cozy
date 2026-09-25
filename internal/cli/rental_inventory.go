package cli

import (
	"encoding/json"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func readRentalInventory(st *records.Store, fleet *managedRentals, reconcile bool) (api.RentalInventory, *exit.Error) {
	// Keep account totals and the corresponding unrecorded set on one observation.
	fleet.mu.Lock()
	defer fleet.mu.Unlock()
	var result api.RentalInventory
	if reconcile {
		if problem := fleet.reconcileLocked(); problem != nil && !hub.Unanswered(problem) {
			return result, problem
		}
	}
	// totalsLocked propagates a failed Hub census and refuses when the Hub has
	// no listing route. A Hub that did not answer at all yields this host's own
	// records marked as such, never totals: the caller must still fail.
	count, burn, problem := fleet.totalsLocked()
	switch {
	case hub.Unanswered(problem):
		result.HubUnanswered = problem
	case problem != nil:
		return result, problem
	default:
		result.MachinesRunning, result.HourlySpendUSDMicros = count, burn
	}
	result.IdleReleaseSeconds = int64(rental.IdleTimeout / time.Second)
	rows, problem := st.Rentals()
	if problem != nil {
		return result, problem
	}
	boughtFor, problem := st.RentalProvenance()
	if problem != nil {
		return result, problem
	}
	live := make(map[string]hub.Rental, len(fleet.live))
	for _, seen := range fleet.live {
		live[seen.ID] = seen
	}
	attached := make(map[string]bool, len(rows))
	for _, row := range rows {
		attached[row.ID] = true
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
		}
		if idle.PendingPreparation > fleet.preparing[row.ID] {
			summary.Activity = nil
		}
		if due, eligible := idle.ReleaseAt(); eligible {
			summary.Activity.IdleSince = idle.Since.UTC().Format(time.RFC3339)
			summary.Activity.ReleaseDue = due.UTC().Format(time.RFC3339)
		}
		result.Rentals = append(result.Rentals, summary)
	}
	hubNamed := map[string]bool{}
	for _, seen := range fleet.unrecorded {
		hubNamed[seen.ID], hubNamed[seen.Name] = true, true
		result.Unrecorded = append(result.Unrecorded, api.RentalSummary{
			ID: seen.ID, MachineName: seen.Name, State: seen.State,
			AcceleratorModel: seen.AcceleratorModel, AcceleratorCount: seen.AcceleratorCount,
			HourlyRateUSDMicros: seen.HourlyRateUSDMicros, Address: seen.Address,
			MediaAddress: seen.MediaAddress, Hub: fleet.ctx.Cfg.HubURL, RentedAt: seen.CreatedAt,
			ProviderState: seen.ProviderState, ContainerState: seen.ContainerState,
			BaseWorkerImageDigest: seen.BaseWorkerImageDigest, BaseWorkerImageTag: seen.BaseWorkerImageTag,
		})
	}
	open, problem := st.ActiveRentalOperations()
	if problem != nil {
		return result, problem
	}
	for _, op := range open {
		if attached[op.RentalID] {
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
	return result.Current(), nil
}
