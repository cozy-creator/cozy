package rental

import (
	"context"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rentalid"
)

// LiveResolver reads provider identity and lifecycle only from the owning Hub.
// Local records discover authorities and retain operation/key custody, not inventory.
type LiveResolver struct {
	Store   *records.Store
	Origins func() ([]string, *exit.Error)
	Client  func(string) *hub.Client
}

type Live struct {
	Hub    string
	Rental hub.Rental
}

func (r LiveResolver) Resolve(ctx context.Context, typed string) (Live, *exit.Error) {
	typed = strings.TrimSpace(typed)
	if r.Store != nil {
		row, problem := r.Store.RentalRow(typed)
		if problem != nil {
			return Live{}, problem
		}
		if row != nil {
			return r.Read(ctx, typed)
		}
	}
	origins, problem := r.Origins()
	if problem != nil {
		return Live{}, problem
	}
	var named, exact []Live
	for _, origin := range origins {
		rows, problem := r.Client(origin).Rentals(ctx)
		if problem != nil {
			return Live{}, problem
		}
		for _, row := range rows {
			if row.ID == typed {
				exact = append(exact, Live{Hub: origin, Rental: row})
			}
			if !hub.RentalAbsent(row.State) && row.MatchesName(typed) {
				named = append(named, Live{Hub: origin, Rental: row})
			}
		}
	}
	selected := named
	if len(exact) > 0 {
		selected = exact
	}
	if len(selected) > 1 {
		return Live{}, exit.Named(exit.Conflict, "rental.machine_name_ambiguous", "machine %q identifies multiple live rentals; use a rental ID", typed)
	}
	if len(selected) == 0 {
		// A portable string is not proof that it is an ID. Only an affirmative
		// Hub read can resolve an unlisted ID; a name's 404 never proves absence.
		if rentalid.Valid(typed) {
			if live, problem := r.Read(ctx, typed); problem == nil || problem.Code != exit.NotFound {
				return live, problem
			}
		}
		return Live{}, exit.Named(exit.NotFound, "rental.selection_unavailable", "%q does not name a live rental on the queried Hubs", typed)
	}
	// The inventory establishes identity; the ID read supplies the current endpoint
	// and catches a rental ending between selection and action.
	remote, problem := r.Client(selected[0].Hub).Rental(ctx, selected[0].Rental.ID)
	return Live{Hub: selected[0].Hub, Rental: remote}, problem
}

func (r LiveResolver) Read(ctx context.Context, id string) (Live, *exit.Error) {
	if r.Store != nil {
		row, problem := r.Store.RentalRow(id)
		if problem != nil {
			return Live{}, problem
		}
		origin := ""
		if row != nil {
			origin = row.Hub
		}
		if origin == "" {
			operations, problem := r.Store.RentalOperations()
			if problem != nil {
				return Live{}, problem
			}
			for _, operation := range operations {
				if operation.RentalID == id {
					origin = operation.Hub
					break
				}
			}
		}
		if origin != "" {
			remote, problem := r.Client(origin).Rental(ctx, id)
			return Live{Hub: origin, Rental: remote}, problem
		}
	}
	origins, problem := r.Origins()
	if problem != nil {
		return Live{}, problem
	}
	for _, origin := range origins {
		remote, problem := r.Client(origin).Rental(ctx, id)
		if problem == nil {
			return Live{Hub: origin, Rental: remote}, nil
		}
		if problem.Code != exit.NotFound {
			return Live{}, problem
		}
	}
	return Live{}, exit.Named(exit.NotFound, "rental.not_found", "no Hub returned rental %s", id)
}

// Record is a disposable projection for activity/history joins. Every live selector
// and connection reads the Hub again, so removing this projection loses no identity.
func (r LiveResolver) Record(live Live) (*records.Rental, *exit.Error) {
	var row records.Rental
	if r.Store != nil {
		prior, problem := r.Store.RentalRow(live.Rental.ID)
		if problem != nil {
			return nil, problem
		}
		if prior != nil {
			row = *prior
		}
	}
	remote := live.Rental
	row.ID, row.Hub = remote.ID, live.Hub
	row.StableMachineName, row.State = remote.StableMachineName, remote.State
	row.AcceleratorModel, row.AcceleratorCount = remote.AcceleratorModel, remote.AcceleratorCount
	row.HourlyRateUSDMicros = remote.HourlyRateUSDMicros
	row.Address, row.MediaAddress = remote.Address, remote.MediaAddress
	row.CertificatePEM = remote.CertPEM
	row.ExpectedWorkerID, row.ExpectedWorkerBootID = remote.WorkerID, remote.WorkerBootID
	row.Failure = records.RentalFailure{}
	if remote.Failure != nil {
		row.Failure.Code, row.Failure.ContainerState = remote.Failure.Code, remote.Failure.ContainerState
	}
	// An ended rental's cause is the one its runs settle with: who released it, else its failure.
	if cause := remote.EndCause(); cause != "" {
		row.Failure.Code = cause
	}
	if row.RentedAt == "" {
		row.RentedAt = remote.CreatedAt
	}
	if r.Store != nil {
		if problem := r.Store.ObserveRental(row); problem != nil {
			return nil, problem
		}
	}
	return &row, nil
}
