package api

import "net/http"

func (s *Server) webUI(w http.ResponseWriter, r *http.Request) {
	if s.web == nil {
		s.refuse(w, r, http.StatusNotFound, "web_unavailable",
			"this build carries no localhost web UI", "install a complete Cozy Creator release")
		return
	}
	w.Header().Set("Content-Security-Policy",
		"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "+
			"connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
	s.web.ServeHTTP(w, r)
}
