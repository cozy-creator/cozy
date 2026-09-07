package api

import "net/http"

func (s *Server) pauseJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	if problem := s.orchestrator.PauseRequest(row.ID, requestActor(r)); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	updated, problem := s.store.RequestRow(row.ID)
	if problem != nil || updated == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal", "paused request cannot be read", "")
		return
	}
	status := http.StatusAccepted
	if updated.State == "paused" {
		status = http.StatusOK
	}
	s.ok(w, r, status, s.jobStateOf(*updated))
}

func (s *Server) resumeJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	s.shutdownAdmission.RLock()
	if s.shuttingDown {
		s.shutdownAdmission.RUnlock()
		s.refuse(w, r, http.StatusServiceUnavailable, "daemon_shutting_down", "the daemon is shutting down", "")
		return
	}
	problem := s.orchestrator.ResumeRequest(row.ID, requestActor(r))
	s.shutdownAdmission.RUnlock()
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	updated, problem := s.store.RequestRow(row.ID)
	if problem != nil || updated == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal", "resumed request cannot be read", "")
		return
	}
	s.ok(w, r, http.StatusAccepted, s.jobStateOf(*updated))
}
