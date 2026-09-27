package api

import (
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (s *Server) validateRequestedRental(id, hub string) *exit.Error {
	if id == "" {
		return nil
	}
	row, problem := s.store.RentalRow(id)
	if problem != nil {
		return problem
	}
	if row == nil || records.RentalTerminalState(row.State) {
		return exit.Named(exit.NotFound, "rental.selection_unavailable", "selected rental %s is unavailable", id)
	}
	if origin, invalid := config.HubOrigin(row.Hub); invalid == nil && origin != hub {
		return rentalHubMismatch(id, row.Hub, hub)
	}
	return nil
}

// rentalHubMismatch refuses one command whose work was resolved on a different hub
// from the rental it names. Nothing else is affected.
func rentalHubMismatch(id, rentalHub, selected string) *exit.Error {
	return exit.Named(exit.Conflict, "rental.hub_mismatch",
		"rental %s belongs to Tensorhub %s; this command addressed %s", id, rentalHub, selected).
		WithRemedy("repeat it with --tensorhub=%s", rentalHub)
}
