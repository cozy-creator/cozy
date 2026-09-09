package api

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (s *Server) validateRequestedRental(id string) *exit.Error {
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
	return nil
}
