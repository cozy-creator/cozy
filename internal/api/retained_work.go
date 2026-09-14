package api

import (
	"bytes"
	"context"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// Fresh snapshots may have different local install IDs with identical immutable
// contents. An idempotency replay accepts that equivalence, but it must never
// silently replace newly supplied code/dependencies with the prior request's code.
func (s *Server) verifyJobReplayInstall(ctx context.Context, sub JobSubmission, recorded records.Request) *exit.Error {
	if sub.InstallID == "" || sub.InstallID == recorded.InstallID {
		return nil
	}
	if recorded.LocalPackageDigest != "" && s.packages != nil {
		link, problem := s.store.MachineExecution(recorded.ID)
		if problem != nil {
			return problem
		}
		if link != nil {
			// Completed snapshot trees may be reclaimed. The durable submitted
			// capture still proves whether the new intake is the identical code.
			revision, problem := s.packages.PrepareLocal(ctx, sub.InstallID)
			if problem != nil {
				return problem
			}
			if revision.Digest != recorded.LocalPackageDigest || revision.Package != recorded.Package || revision.Release != recorded.Release {
				return exit.Named(exit.Conflict, "request.replay_install_changed", "this idempotency key names a different captured package; submit a new request to run the edited revision")
			}
			if len(link.Submission) > 0 {
				capture, problem := localpackage.CaptureExecution(sub.InstallID, revision, s.store.ChildBindings, s.packages.LocalRevision)
				if problem != nil {
					return problem
				}
				var submitted pb.MachineExecutionSubmit
				if proto.Unmarshal(link.Submission, &submitted) != nil || !bytes.Equal(capture.Digest, submitted.CaptureDigest) {
					return exit.Named(exit.Conflict, "request.replay_install_changed", "this idempotency key names different captured dependencies; submit a new request to run the edited revision")
				}
			}
			return nil
		}
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
	if prior.Package != next.Package || prior.Version != next.Version || prior.SourceDigest != next.SourceDigest ||
		prior.LockDigest != next.LockDigest || prior.PackageInterface != next.PackageInterface || prior.Closure != next.Closure ||
		prior.Platform != next.Platform || prior.Extra != next.Extra || prior.PlacementSetDigest != next.PlacementSetDigest {
		return exit.Named(exit.Conflict, "request.replay_install_changed", "this idempotency key names a different immutable package snapshot; submit a new request to run the edited revision")
	}
	return nil
}

func (s *Server) pauseJob(w http.ResponseWriter, r *http.Request) {
	row, ok := s.jobRow(w, r)
	if !ok {
		return
	}
	if s.machineJobControl(w, r, row, "pause") {
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
	if s.machineJobControl(w, r, row, "resume") {
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
