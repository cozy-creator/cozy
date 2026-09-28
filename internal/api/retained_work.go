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
	s.refuseTyped(w, r, notOnAMachine("pause"))
}

// notOnAMachine refuses control of work no machine executes: the daemon's own model
// transfer only cancels.
func notOnAMachine(action string) *exit.Error {
	return exit.Named(exit.Conflict, "request.not_on_a_machine", "only work a machine executes can %s", action)
}

func (s *Server) resumeJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	if s.machineJobControl(w, r, row, "resume", "") {
		return
	}
	s.refuseTyped(w, r, notOnAMachine("resume"))
}
