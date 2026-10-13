package api

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// validateRequestedRental refuses work for a rental its authority says is absent: Calcifer's
// (its record, else its Hub), or without one this host's record.
func (s *Server) validateRequestedRental(id string) *exit.Error {
	if id == "" {
		return nil
	}
	if s.rentals != nil {
		_, problem := s.rentals(id)
		return problem
	}
	row, problem := s.store.RentalRow(id)
	if problem != nil {
		return problem
	}
	if row == nil || records.RentalTerminalState(row.State) {
		return exit.Named(exit.NotFound, "rental.selection_unavailable", "selected rental %s is unavailable", id)
	}
	return nil
}
