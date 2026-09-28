package api

import (
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
)

func (s *Server) refreshPackage(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	if s.packages == nil {
		return "", false, false, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	return s.packages.RefreshEditable(pkg)
}

// detachRental drops the daemon's kept connection to one rented machine before the CLI
// removes its pinned certificate and signing key.
func (s *Server) detachRental(w http.ResponseWriter, r *http.Request) {
	if s.machineExecutions != nil {
		s.machineExecutions.Forget(r.PathValue("rental_id"))
	}
	s.ok(w, r, http.StatusOK, map[string]any{"changed": true})
}
