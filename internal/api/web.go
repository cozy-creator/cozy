package api

import "net/http"

func (s *Server) webUI(w http.ResponseWriter, r *http.Request) {
	if s.web == nil {
		s.refuse(w, r, http.StatusNotFound, "web_unavailable",
			"this build carries no localhost web UI", "install a complete Cozy Creator release")
		return
	}
	s.web.ServeHTTP(w, r)
}
