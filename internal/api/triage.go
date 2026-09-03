package api

import "net/http"

// attemptTriage serves one attempt's kept triage bundle out of its own row (cl-116).
// The bundle was verified against the terminal's reference before it was stored, so
// this is a read of an accepted fact, not a file. An attempt that kept none — or an
// unknown key — is one 404: an opaque key never confirms which half was absent.
func (s *Server) attemptTriage(w http.ResponseWriter, r *http.Request) {
	subject, bundle, problem := s.store.TriageBundle(r.PathValue("attempt_key"))
	if problem != nil {
		s.refuse(w, r, http.StatusInternalServerError, "triage_unreadable", problem.Message, "")
		return
	}
	if len(bundle) == 0 {
		s.refuse(w, r, http.StatusNotFound, "triage_not_kept",
			"no kept triage bundle for that attempt", "")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("X-Triage-Subject", subject)
	_, _ = w.Write(bundle)
}
