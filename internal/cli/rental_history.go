package cli

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// pastRental is one rental of a hub's history, with what this computer's records hold of it.
type pastRental struct {
	hub          string
	rental       hub.Rental
	started, end time.Time // end is now while it lives
	work         records.RentalWork
	recorded     bool
}

// rentalHistory is `cozy rental list --all` (every rental, live and ended) and `--ended`: one
// listing read per hub this host is signed in to and one grouped read of this computer's
// records, live first, then the latest ended. A hub that cannot answer is named, never
// silently left out; a watching board re-reads the hubs at most every pollCadence.
func rentalHistory(ctx *Context, endedOnly bool) (func(context.Context) (output.List, *exit.Error), func(), *exit.Error) {
	store, _ := existingRecords(ctx)
	done := func() {
		if store != nil {
			store.Close()
		}
	}
	origins, problem := rentalInventoryOrigins(ctx, store)
	if problem != nil {
		done()
		return nil, nil, problem
	}
	var rentals []pastRental
	var unanswered []string
	var read time.Time
	return func(context.Context) (output.List, *exit.Error) {
		if time.Since(read) >= pollCadence {
			if rentals, unanswered, problem = readRentalHistory(ctx, origins, endedOnly); problem != nil {
				return output.List{}, problem
			}
			read = time.Now()
		}
		work := map[string]records.RentalWork{}
		if store != nil {
			if work, problem = store.RentalWorkByID(); problem != nil {
				return output.List{}, problem
			}
		}
		now := time.Now()
		for i := range rentals {
			rentals[i].work, rentals[i].recorded = work[rentals[i].rental.ID]
			if !hub.RentalAbsent(rentals[i].rental.State) {
				rentals[i].end = now
			}
		}
		ctx.exitCode = min(len(unanswered), 1)
		return renderRentalHistory(rentals, unanswered, len(origins) > 1, endedOnly), nil
	}, done, nil
}

// readRentalHistory reads every hub's rentals at once, the latest ended first.
func readRentalHistory(ctx *Context, origins []string, endedOnly bool) ([]pastRental, []string, *exit.Error) {
	answers := make([][]hub.Rental, len(origins))
	problems := make([]*exit.Error, len(origins))
	var wg sync.WaitGroup
	for i, origin := range origins {
		scoped := ctx.forHub(origin)
		if len(origins) > 1 && !scoped.Cfg.HubToken.Present() && !accountauth.New(scoped.Cfg).CredentialPresent() {
			continue
		}
		wg.Go(func() {
			call, cancel := hub.Context()
			defer cancel()
			answers[i], problems[i] = client(scoped).RentalHistory(call, "")
		})
	}
	wg.Wait()
	var rentals []pastRental
	var unanswered []string
	now := time.Now()
	for i, origin := range origins {
		if problems[i] != nil && len(origins) == 1 {
			return nil, nil, problems[i]
		}
		if problems[i] != nil {
			unanswered = append(unanswered, fmt.Sprintf("hub %s did not answer: %s; its rentals are not shown",
				ctx.Cfg.HubLabel(origin), problems[i].Message))
		}
		for _, r := range answers[i] {
			if endedOnly && !hub.RentalAbsent(r.State) {
				continue
			}
			started, _ := time.Parse(time.RFC3339Nano, r.CreatedAt)
			end, _ := time.Parse(time.RFC3339Nano, r.EndedAt)
			if !hub.RentalAbsent(r.State) {
				end = now
			}
			rentals = append(rentals, pastRental{hub: ctx.Cfg.HubLabel(origin), rental: r, started: started, end: end})
		}
	}
	at := func(p pastRental) time.Time {
		if p.end.IsZero() {
			return p.started
		}
		return p.end
	}
	slices.SortStableFunc(rentals, func(a, b pastRental) int {
		return cmp.Or(at(b).Compare(at(a)), b.started.Compare(a.started))
	})
	return rentals, unanswered, nil
}

func renderRentalHistory(rentals []pastRental, unanswered []string, hubs, endedOnly bool) output.List {
	fields := []string{"machine", "sku", "gpus", "started", "ended", "lifetime", "runs", "spent", "ended by"}
	if hubs {
		fields = slices.Insert(fields, 3, "hub")
	}
	list := output.List{Name: "rental_history", Fields: fields,
		AllFields: []string{"machine", "rental", "sku", "gpus", "accelerator", "hub", "state", "started", "ended", "lifetime",
			"runs", "failed", "canceled", "$/hour", "spent", "ended by", "provider", "provider machine", "provider resource"},
		TypedFields: []string{"machine", "rental_id", "sku", "gpus", "hub", "state", "rented_at", "ended_at", "lifetime_s",
			"runs_succeeded", "spend_usd_micros", "spend_basis", "release_cause"},
		TypedAllFields: []string{"machine", "rental_id", "sku", "gpus", "accelerator", "accelerator_count", "hub", "state",
			"rented_at", "ended_at", "lifetime_s", "runs_succeeded", "runs_failed", "runs_canceled", "hourly_rate_usd_micros",
			"spend_usd_micros", "spend_basis", "release_cause", "provider", "provider_machine_id", "provider_resource_id"},
		Trail: []string{"RUNS counts the runs this computer sent that succeeded; - where it holds no record of the rental."},
		Next:  []string{"cozy rental show <machine>"}}
	label := "Rentals"
	if endedOnly {
		list.Name, label = "ended_rentals", "Ended rentals"
	}
	var lifetime time.Duration
	var spend []api.RentalSummary
	live, total := 0, records.RentalWork{}
	for _, p := range rentals {
		r := p.rental
		summary := api.RentalSummary{SpendUSDMicros: r.SpendUSDMicros, SpendBasis: r.SpendBasis}
		spend = append(spend, summary)
		row := map[string]string{"machine": r.Name, "rental": r.ID, "gpus": gpuCell(r.AcceleratorModel, r.AcceleratorCount),
			"accelerator": acceleratorLabel(r.AcceleratorModel, r.AcceleratorCount), "hub": p.hub,
			"state": humanRentalState(r.State), "started": clock(p.started), "ended": "live", "ended by": r.EndCause(),
			"$/hour": rentalHourlyRate(costPerHour(r.HourlyRateUSDMicros, r.ComputeUSDMicrosPerHour, r.StorageUSDMicrosPerHour)),
			"spent":  rentalSpend(summary), "provider": r.Provider, "provider machine": r.ProviderMachineID,
			"provider resource": r.ProviderResourceID}
		typed := map[string]any{"machine": r.Name, "rental_id": r.ID, "hub": p.hub, "state": r.State,
			"accelerator_count": r.AcceleratorCount, "hourly_rate_usd_micros": r.HourlyRateUSDMicros}
		if gpus, ok := gpuCount(r.AcceleratorModel, r.AcceleratorCount); ok {
			typed["gpus"] = gpus
		}
		if hub.RentalAbsent(r.State) {
			row["ended"] = clock(p.end)
		} else {
			live++
		}
		if !p.started.IsZero() && !p.end.IsZero() {
			d := p.end.Sub(p.started)
			lifetime += d
			row["lifetime"], typed["lifetime_s"] = brief(d), int64(d.Seconds())
		}
		if w := p.work; p.recorded {
			total.Succeeded, total.Failed, total.Canceled = total.Succeeded+w.Succeeded, total.Failed+w.Failed, total.Canceled+w.Canceled
			row["sku"], row["runs"], row["failed"], row["canceled"] = w.SKU, strconv.Itoa(w.Succeeded), strconv.Itoa(w.Failed), strconv.Itoa(w.Canceled)
			typed["runs_succeeded"], typed["runs_failed"], typed["runs_canceled"] = w.Succeeded, w.Failed, w.Canceled
		}
		for key, value := range map[string]string{"sku": row["sku"], "release_cause": r.EndCause(), "rented_at": r.CreatedAt,
			"ended_at": r.EndedAt, "accelerator": r.AcceleratorModel, "provider": r.Provider,
			"provider_machine_id": r.ProviderMachineID, "provider_resource_id": r.ProviderResourceID} {
			if value != "" {
				typed[key] = value
			}
		}
		spendFields(typed, summary)
		list.Rows, list.TypedRows = append(list.Rows, row), append(list.TypedRows, typed)
	}
	accrued, spendFacts := accruedSpend(api.RentalInventory{Rentals: spend})
	count := strconv.Itoa(len(rentals))
	if !endedOnly {
		count += fmt.Sprintf(" (%d live)", live)
	}
	list.Lead = append([]string{fmt.Sprintf("%s: %s · lifetime %s · %d runs from this computer%s",
		label, count, brief(lifetime), total.Succeeded, accrued)}, unanswered...)
	list.Aggregates = append([]output.Field{{K: "rentals", V: jsonFact{len(rentals)}}, {K: "live_rentals", V: jsonFact{live}},
		{K: "lifetime_s", V: jsonFact{int64(lifetime.Seconds())}}, {K: "runs_succeeded", V: jsonFact{total.Succeeded}},
		{K: "runs_failed", V: jsonFact{total.Failed}}, {K: "runs_canceled", V: jsonFact{total.Canceled}}}, spendFacts...)
	return list
}

// clock is a moment as this computer's wall clock reads it, to the minute.
func clock(at time.Time) string {
	if at.IsZero() {
		return ""
	}
	return at.Local().Format("2006-01-02 15:04")
}
