package api

import (
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

func (s *Server) refreshPackage(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	if s.packages == nil {
		return "", false, false, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	installID, editable, changed, problem = s.packages.RefreshEditable(pkg)
	if problem != nil || !changed {
		return
	}
	_, problem = s.orchestrator.UnloadIdleLocalPackage(pkg)
	return
}

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

// detachRental closes the daemon's control loop before the CLI removes this rental's
// pinned certificate and signing key. An absent slot is an idempotent no-op.
func (s *Server) detachRental(w http.ResponseWriter, r *http.Request) {
	changed := s.orchestrator.DetachRental(r.PathValue("rental_id"))
	s.ok(w, r, http.StatusOK, map[string]any{"changed": changed})
}
