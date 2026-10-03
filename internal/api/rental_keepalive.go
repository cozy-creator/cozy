package api

import (
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RentalKeepaliveResult is the deadline the rental's machine holds after one explicit reset.
type RentalKeepaliveResult struct {
	Rental               string `json:"rental"`
	WorkerID             string `json:"worker_id"`
	WorkerBootID         string `json:"worker_boot_id"`
	AcknowledgedAtUnixMS int64  `json:"acknowledged_at_unix_ms"`
	IdleDeadlineUnixMS   int64  `json:"idle_deadline_unix_ms"`
}

func (s *Server) keepRentalAlive(w http.ResponseWriter, r *http.Request) {
	if s.rentalKeepalive == nil {
		s.refuseTyped(w, r, exit.Unavailablef("rental keepalive is unavailable"))
		return
	}
	result, problem := s.rentalKeepalive(r.Context(), r.PathValue("rental_id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, result)
}
