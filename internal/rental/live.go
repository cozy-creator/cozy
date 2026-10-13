package rental

import (
	"cmp"
	"context"
	"slices"
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

// Resolve finds the live rental typed names, an ID or a machine name, at the Hub that owns
// it. A rental this host recorded is read at its own Hub alone, and a name it recorded is asked
// of that Hub first; any other name is asked of each Hub in Origins' order, the selected one
// first, until one names it. A Hub that cannot answer stops a selection only when no Hub
// named the rental: a local-hub command never depends on another Hub.
func (r LiveResolver) Resolve(ctx context.Context, typed string) (Live, *exit.Error) {
	typed = strings.TrimSpace(typed)
	origins, problem := r.Origins()
	if problem != nil {
		return Live{}, problem
	}
	hint := ""
	if r.Store != nil {
		row, problem := r.Store.RentalRow(typed)
		if problem != nil {
			return Live{}, problem
		}
		if row != nil {
			return r.Read(ctx, typed)
		}
		if named, _ := r.Store.RentalByMachine(typed); named != nil && named.Hub != "" {
			hint = named.Hub
			origins = append([]string{hint}, slices.DeleteFunc(origins, func(o string) bool { return o == hint })...)
		}
	}
	var unanswered *exit.Error
	for _, origin := range origins {
		selected, problem := r.named(ctx, origin, typed)
		// A hub this computer holds no login for lists none of its rentals, unless it recorded one there.
		if problem != nil && problem.ErrName() == SignedOut && origin != hint {
			continue
		}
		if problem != nil && problem.ErrName() != "rental.machine_name_ambiguous" {
			unanswered = cmp.Or(unanswered, problem)
			continue
		}
		if problem != nil {
			return Live{}, problem
		}
		if selected != nil {
			// The listing establishes identity; the ID read supplies the current endpoint and
			// catches a rental ending between selection and action. Selection never judges
			// whether a rental can be attached: ending one the Hub describes badly must work.
			remote, problem := r.Client(origin).RentalView(ctx, selected.ID)
			return Live{Hub: origin, Rental: remote}, problem
		}
	}
	// A portable string is not proof that it is an ID. Only an affirmative Hub read can
	// resolve an unlisted ID; a name's 404 never proves absence.
	if rentalid.Valid(typed) {
		if live, problem := r.Read(ctx, typed); problem == nil || problem.Code != exit.NotFound {
			return live, problem
		}
	}
	if unanswered != nil {
		return Live{}, unanswered
	}
	return Live{}, exit.Named(exit.NotFound, "rental.selection_unavailable", "no live rental is named %q on any hub this computer is signed in to", typed)
}

// SignedOut is the refusal of a hub this computer holds no login for.
const SignedOut = "auth.machine_key_missing"

// named is the one rental of origin's listing whose ID is typed, else the one live rental
// it names; nil when it names none.
func (r LiveResolver) named(ctx context.Context, origin, typed string) (*hub.Rental, *exit.Error) {
	rows, problem := r.Client(origin).Rentals(ctx)
	if problem != nil {
		return nil, problem
	}
	var named []hub.Rental
	for _, row := range rows {
		if row.ID == typed {
			return &row, nil
		}
		if !hub.RentalAbsent(row.State) && row.MatchesName(typed) {
			named = append(named, row)
		}
	}
	if len(named) > 1 {
		return nil, exit.Named(exit.Conflict, "rental.machine_name_ambiguous", "machine %q identifies multiple live rentals; use a rental ID", typed)
	}
	if len(named) == 0 {
		return nil, nil
	}
	return &named[0], nil
}

// Read is the rental with this ID at the Hub that owns it: its recorded Hub, else the first of
// Origins that knows it. Hubs that cannot answer are passed over while another may.
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
			remote, problem := r.Client(origin).RentalView(ctx, id)
			return Live{Hub: origin, Rental: remote}, problem
		}
	}
	origins, problem := r.Origins()
	if problem != nil {
		return Live{}, problem
	}
	var unanswered *exit.Error
	for _, origin := range origins {
		remote, problem := r.Client(origin).RentalView(ctx, id)
		if problem == nil {
			return Live{Hub: origin, Rental: remote}, nil
		}
		if problem.Code != exit.NotFound && problem.ErrName() != SignedOut {
			unanswered = cmp.Or(unanswered, problem)
		}
	}
	if unanswered != nil {
		return Live{}, unanswered
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
