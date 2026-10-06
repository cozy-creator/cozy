package api

import (
	"github.com/cozy-creator/cozy/internal/exit"
)

func (s *Server) refreshPackage(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	if s.packages == nil {
		return "", false, false, exit.Unavailablef("this Cozy daemon resolves no packages")
	}
	return s.packages.RefreshEditable(pkg)
}
