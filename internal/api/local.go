package api

import (
	"net/http"

	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// claimRental attaches the daemon to one already-provisioned private worker.
// Provider acquisition remains Tensorhub's responsibility.
func (s *Server) claimRental(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("rental_id")
	instance, pkg, change, problem := s.orchestrator.EnsureRental(id)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	status := http.StatusAccepted
	if change == orchestrator.ChangeNone {
		status = http.StatusOK
	}
	s.ok(w, r, status, map[string]any{
		"instance_id": instance, "package": pkg, "change": string(change),
	})
}
