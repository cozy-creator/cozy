package api

import "net/http"

func (s *Server) retryJobPublication(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	if problem := s.orchestrator.RetryModelTransferPublication(row.ID, requestActor(r)); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	current, problem := s.store.RequestRow(row.ID)
	if problem != nil || current == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal", "publication retry could not be read back", "")
		return
	}
	s.ok(w, r, http.StatusAccepted, s.jobStateOf(*current))
}
