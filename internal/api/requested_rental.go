package api

import (
	"github.com/cozy-creator/cozy/internal/exit"
)

func (s *Server) validateRequestedRental(id string) *exit.Error {
	if id == "" {
		return nil
	}
	if s.rentals == nil {
		return exit.Unavailablef("live rental authority is unavailable")
	}
	_, problem := s.rentals(id)
	return problem
}
