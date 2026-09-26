package api

import (
	"context"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// An idempotency replay returns the accepted request. It never replaces that
// request's installation or compares a newly synced source tree to old code.
func (s *Server) verifyJobReplayInstall(ctx context.Context, sub JobSubmission, recorded records.Request) *exit.Error {
	if sub.InstallID == "" || sub.InstallID == recorded.InstallID {
		return nil
	}
	prior, problem := s.store.Install(recorded.InstallID)
	if problem != nil {
		return problem
	}
	next, problem := s.store.Install(sub.InstallID)
	if problem != nil {
		return problem
	}
	if prior == nil || next == nil {
		return exit.Named(exit.Conflict, "request.replay_install_unavailable", "the supplied and recorded package snapshots must both be available to verify replay")
	}
	if prior.Package != next.Package || prior.Version != next.Version {
		return exit.Named(exit.Conflict, "request.replay_install_changed", "this idempotency key names a different immutable package snapshot; submit a new request to run the edited revision")
	}
	return nil
}

func (s *Server) pauseJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	if s.machineJobControl(w, r, row, "pause", "") {
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
	if s.machineJobControl(w, r, row, "resume", "") {
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
