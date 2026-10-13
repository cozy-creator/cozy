package cli

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// readRentalInventory is one hub's reconciled fleet, or with allHubs every known
// account's fleet. A listing of one hub still counts this host's live rentals
// on the others, so switching hubs never hides a machine that is billing. The hubs are
// asked before the fleet lock is taken, so a purchase or release in flight never holds
// the listing. Each ready machine's own Status is read beside the hubs, at the identity they
// last listed; a hub not yet listed is read first.
func readRentalInventory(ctx context.Context, st *records.Store, fleet *managedRentals, origin string, allHubs, reconcile bool) (api.RentalInventory, *exit.Error) {
	origin = fleet.origin(origin)
	origins := []string{origin}
	if allHubs {
		held, problem := rentalInventoryOrigins(fleet.ctx, st, origin)
		if problem != nil {
			return api.RentalInventory{}, problem
		}
		// A hub whose last answer named a live rental stays asked: it may still bill.
		fleet.mu.Lock()
		for known, census := range fleet.census {
			if len(census.live) > 0 && !slices.Contains(held, known) {
				held = append(held, known)
			}
		}
		fleet.mu.Unlock()
		slices.Sort(held)
		origins = held
	}
	// Without reconcile the listing serves what the fleet's own loop and the last listing read,
	// but a hub never asked since this daemon started is asked now: no answer is not an empty fleet.
	fleet.mu.Lock()
	unobserved := slices.DeleteFunc(slices.Clone(origins), func(each string) bool { return fleet.censusLocked(each).observed })
	fleet.mu.Unlock()
	ask := unobserved
	if reconcile {
		ask = origins
	}
	var machines sync.WaitGroup
	if reconcile && len(unobserved) == 0 {
		machines.Go(func() { fleet.readMachines(ctx) })
	}
	asked := fleet.reconcileEach(ask)
	if len(unobserved) > 0 {
		fleet.readMachines(ctx)
	}
	machines.Wait()
	// Keep account totals and the corresponding unrecorded set on one observation.
	fleet.mu.Lock()
	defer fleet.mu.Unlock()
	if !allHubs {
		result, problem := fleet.inventoryLocked(st, origin, asked[origin], false)
		if problem != nil {
			return result, problem
		}
		result.OtherHubs, problem = otherHubRentals(st, fleet, origin)
		return result.Current(), problem
	}
	// One hub that cannot be read never hides another: its problem is named, its rows
	// are this host's records marked unverified, and every other hub is listed as usual.
	var merged api.RentalInventory
	for _, each := range origins {
		result, problem := fleet.inventoryLocked(st, each, asked[each], true)
		if problem != nil {
			return merged, problem
		}
		if result.HubUnanswered != nil {
			merged.UnreadableHubs = append(merged.UnreadableHubs, api.HubProblem{Hub: each, Error: result.HubUnanswered})
			for _, rows := range [][]api.RentalSummary{result.Rentals, result.Unrecorded, result.Pending} {
				for i := range rows {
					rows[i].Unverified = true
				}
			}
		}
		merged.MachinesRunning += result.MachinesRunning
		merged.HourlySpendUSDMicros += result.HourlySpendUSDMicros
		merged.Rentals = append(merged.Rentals, result.Rentals...)
		merged.Unrecorded = append(merged.Unrecorded, result.Unrecorded...)
		merged.Pending = append(merged.Pending, result.Pending...)
	}
	return merged.Current(), nil
}

// rentalInventoryOrigins is every hub an every-hub listing asks, from what holds now: the
// asking command's hub, every hub this machine is signed in to (another client's rentals
// need no local record), and every hub this host holds a rental or acquisition on. A hub
// name never adds a hub, so a daemon's start-up copy of config.yaml cannot keep a removed
// one listed.
func rentalInventoryOrigins(ctx *Context, st *records.Store, asked string) ([]string, *exit.Error) {
	seen := map[string]bool{asked: true}
	for _, origin := range accountauth.KnownOrigins(ctx.Cfg.Home) {
		seen[origin] = true
	}
	if ctx.Cfg.StaticToken() {
		seen[ctx.Cfg.ConfiguredHubURL] = true
	}
	if st != nil {
		rows, problem := st.Rentals()
		if problem != nil {
			return nil, problem
		}
		operations, problem := st.ActiveRentalOperations()
		if problem != nil {
			return nil, problem
		}
		for _, row := range rows {
			seen[ctx.forHub(row.Hub).Cfg.HubURL] = true
		}
		for _, operation := range operations {
			seen[ctx.forHub(operation.Hub).Cfg.HubURL] = true
		}
	}
	origins := make([]string, 0, len(seen))
	for origin := range seen {
		origins = append(origins, origin)
	}
	slices.Sort(origins)
	return origins, nil
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

// inventoryLocked is one hub's fleet; asked is what reconciling it just now met, if it
// was reconciled. isolated keeps any problem asking that hub in HubUnanswered, with this
// host's records, so an every-hub read can continue past it.
func (fleet *managedRentals) inventoryLocked(st *records.Store, origin string, asked *exit.Error, isolated bool) (api.RentalInventory, *exit.Error) {
	var result api.RentalInventory
	if asked != nil && !hub.Unanswered(asked) {
		if !isolated {
			return result, asked
		}
		result.HubUnanswered = asked
	}
	// totalsLocked propagates a failed Hub census and refuses when the Hub has
	// no listing route. A Hub that did not answer at all yields this host's own
	// records marked as such, never totals: the caller must still fail.
	count, burn, problem := fleet.totalsLocked(origin)
	switch {
	case result.HubUnanswered != nil:
	case hub.Unanswered(problem) || problem != nil && isolated:
		result.HubUnanswered = problem
	case problem != nil:
		return result, problem
	default:
		result.MachinesRunning, result.HourlySpendUSDMicros = count, burn
	}
	census := fleet.censusLocked(origin)
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
		if fleet.origin(row.Hub) != origin {
			continue
		}
		// A row the Hub answered 404 for stays listed: that is not proof the pod is gone.
		seen, current := live[row.ID]
		if !current && result.HubUnanswered == nil && !census.hubUnknown[row.ID] {
			continue
		}
		attached[row.ID] = true
		queued, running, problem := st.RentalWorkCounts(row.ID, row.ManagedRequestID)
		if problem != nil {
			return result, problem
		}
		summary := api.RentalSummary{
			Provider: live[row.ID].Provider, ProviderMachineID: live[row.ID].ProviderMachineID, ProviderResourceID: live[row.ID].ProviderResourceID,
			ID: row.ID, MachineName: row.DisplayName(), RentalAlias: row.MachineName, SKU: row.SKU, State: row.State,
			AcceleratorModel: row.AcceleratorModel, AcceleratorCount: row.AcceleratorCount,
			HourlyRateUSDMicros: row.HourlyRateUSDMicros,
			SpendUSDMicros:      live[row.ID].SpendUSDMicros, SpendBasis: live[row.ID].SpendBasis,
			Address: row.Address, MediaAddress: row.MediaAddress, Hub: row.Hub,
			RentedAt: row.RentedAt, ReadyAt: row.ReadyAt, BoughtFor: boughtFor[row.ID],
			Activity: &api.RentalActivity{Running: running, Queued: queued},
			Failure: hub.RentalFailure{
				Code: row.Failure.Code, BaseWorkerImageDigest: row.Failure.BaseWorkerImageDigest,
				Provider: row.Failure.Provider, ProviderResourceID: row.Failure.ProviderResourceID,
				ProviderHostID: row.Failure.ProviderHostID, ProviderState: row.Failure.ProviderState,
				ContainerState: row.Failure.ContainerState,
			},
			Boot:                  live[row.ID].Boot,
			BaseWorkerImageDigest: live[row.ID].BaseWorkerImageDigest,
			BaseWorkerImageTag:    live[row.ID].BaseWorkerImageTag,
			HubUnknown:            census.hubUnknown[row.ID],
			UnreachableSince:      live[row.ID].UnreachableSince,
		}
		if current {
			summary.MachineName, summary.MachineAliases = seen.DisplayName(), seen.MachineAliases
			summary.RentalAlias, summary.State = seen.Name, seen.State
			summary.Address, summary.MediaAddress = seen.Address, seen.MediaAddress
			summary.AcceleratorModel, summary.AcceleratorCount = seen.AcceleratorModel, seen.AcceleratorCount
			summary.HourlyRateUSDMicros = seen.HourlyRateUSDMicros
			summary.Failure = hub.RentalFailure{}
			if seen.Failure != nil {
				summary.Failure = *seen.Failure
			}
		}
		if read, ok := fleet.statuses[row.ID]; ok && current && seen.State == hub.RentalReady {
			if summary.MachineProblem = read.problem; read.problem == nil {
				summary.Machine = &read.status
				if due := read.status.IdleDeadlineUnixMS; due > 0 {
					summary.ReleaseDue = time.UnixMilli(due).UTC().Format(time.RFC3339Nano)
				}
			}
		}
		costOf(&summary, live[row.ID])
		if update, problem := st.RuntimeUpdate(row.ID); problem != nil {
			return result, problem
		} else if update != nil && update.Active() && current && seen.State == hub.RentalReady && seen.WorkerBootID == update.BootID {
			summary.RuntimeUpdate = update.State
		}
		result.Rentals = append(result.Rentals, summary)
	}
	hubNamed := map[string]bool{}
	for _, seen := range census.unrecorded {
		hubNamed[seen.ID], hubNamed[seen.Name] = true, true
		if attached[seen.ID] {
			continue
		}
		result.Unrecorded = append(result.Unrecorded, api.RentalSummary{
			Provider: seen.Provider, ProviderMachineID: seen.ProviderMachineID, ProviderResourceID: seen.ProviderResourceID,
			ID: seen.ID, MachineName: seen.DisplayName(), RentalAlias: seen.Name, MachineAliases: seen.MachineAliases, State: seen.State,
			AcceleratorModel: seen.AcceleratorModel, AcceleratorCount: seen.AcceleratorCount,
			HourlyRateUSDMicros: seen.HourlyRateUSDMicros, Address: seen.Address,
			SpendUSDMicros: seen.SpendUSDMicros, SpendBasis: seen.SpendBasis,
			MediaAddress: seen.MediaAddress, Hub: origin, RentedAt: seen.CreatedAt,
			Boot:                  seen.Boot,
			BaseWorkerImageDigest: seen.BaseWorkerImageDigest, BaseWorkerImageTag: seen.BaseWorkerImageTag,
			UnreachableSince: seen.UnreachableSince,
		})
		costOf(&result.Unrecorded[len(result.Unrecorded)-1], seen)
	}
	open, problem := st.ActiveRentalOperations()
	if problem != nil {
		return result, problem
	}
	for _, op := range open {
		if op.State == "attached" || attached[op.RentalID] || fleet.origin(op.Hub) != origin {
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

// costOf carries the Hub's split of a rental's hourly cost and its machine's shape.
func costOf(summary *api.RentalSummary, seen hub.Rental) {
	summary.ComputeUSDMicrosPerHour, summary.StorageUSDMicrosPerHour = seen.ComputeUSDMicrosPerHour, seen.StorageUSDMicrosPerHour
	summary.VCPUCount, summary.MemoryGB = seen.VCPUCount, seen.MemoryGB
}
