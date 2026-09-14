package api

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type MachineExecutions interface {
	Refresh(context.Context, records.Request) *exit.Error
	Control(context.Context, records.Request, string) *exit.Error
}

// This is a client observation, not an execution or custody receipt of its own.
type MachineExecutionView struct {
	Accepted         bool   `json:"accepted"`
	Machine          string `json:"machine"`
	Collected        bool   `json:"collected"`
	ObservationError string `json:"observation_error,omitempty"`
}

func (s *Server) refreshMachineExecution(ctx context.Context, request records.Request) *exit.Error {
	link, problem := s.store.MachineExecution(request.ID)
	if problem != nil || link == nil || len(link.Receipt) == 0 || link.Collected && len(link.PendingControl) == 0 {
		return problem
	}
	if s.machineExecutions == nil {
		return exit.Unavailablef("this client cannot observe Runtime-owned execution")
	}
	return s.machineExecutions.Refresh(ctx, request)
}

func (s *Server) machineJobState(row records.Request, link *records.MachineExecution) JobState {
	machine := row.Machine
	if machine == "" {
		machine = link.MachineID
	}
	view := &MachineExecutionView{Accepted: len(link.Receipt) > 0, Machine: machine, Collected: link.Collected}
	retaining, retentionProblem := s.store.MachineExecutionOwesWork(row.ID)
	if retentionProblem != nil {
		retaining = true
		view.ObservationError = retentionProblem.Message
	}
	state := JobState{
		Number: row.Number, JobID: row.ID, Status: contractStatus(row.State), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal), Attempts: int(row.Ordinal),
		RetainWork: row.RetainWork, Retaining: retaining,
		CreatedAt: row.CreatedAt, EventsURL: "/v1/requests/" + row.ID + "/events", Outputs: []MediaRef{},
		MachineExecution: view,
	}
	state.OutputExport = s.outputExportOf(row.ID)
	if outputs, problem := s.store.VisibleOutputs(row.ID); problem == nil {
		for _, output := range outputs {
			state.Outputs = append(state.Outputs, MediaRef{OutputID: output.OutputID, MediaID: output.MediaID, URL: "/v1/media/" + output.MediaID, MimeType: output.MimeType, Length: output.Length, Digest: output.Digest})
		}
	}
	if !view.Accepted {
		state.Stage = "waiting for durable machine acceptance"
		if len(link.Submission) > 0 && state.Status == "blocked" {
			state.Status = "queued"
			state.Stage = "reconciling durable machine acceptance"
		}
		if events, problem := s.store.EventsAfter(row.ID, 0, 256); problem == nil {
			for _, event := range events {
				if event.Type == "request.blocked" || event.Type == "request.failed" {
					state.ErrorType, _ = event.Payload["error_type"].(string)
					state.Error, _ = event.Payload["error"].(string)
				}
			}
		}
	}
	var submission pb.MachineExecutionSubmit
	if proto.Unmarshal(link.Submission, &submission) == nil {
		state.RetryBudget = int64(submission.MaxAttempts)
	}
	if len(link.Outcome) > 0 {
		var outcome pb.AttemptOutcome
		var body pb.AttemptOutcomeBody
		if proto.Unmarshal(link.Outcome, &outcome) == nil && canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body) == nil {
			if body.Cause != nil && body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
				state.Error = body.SafeMessage
				state.ErrorType = body.Cause.Code.String()
			}
			if view.Collected && body.Result != nil && len(body.Result.InlineResult) > 0 {
				state.Result = json.RawMessage(body.Result.InlineResult)
			}
			if !view.Collected {
				view.ObservationError = "execution finished; result collection has not established recipient custody"
			}
		}
	}
	return state
}

func (s *Server) machineJobControl(w http.ResponseWriter, r *http.Request, row records.Request, action string) bool {
	owned, problem := s.controlMachineExecution(r.Context(), row, action)
	if !owned {
		return false
	}
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return true
	}
	if latest, e := s.store.RequestRow(row.ID); e == nil && latest != nil {
		latest.Number = row.Number
		row = *latest
	}
	s.ok(w, r, http.StatusAccepted, s.jobStateOf(row))
	return true
}

func (s *Server) machineRequestControl(w http.ResponseWriter, r *http.Request, row records.Request, action string) bool {
	owned, problem := s.controlMachineExecution(r.Context(), row, action)
	if !owned {
		return false
	}
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return true
	}
	if latest, e := s.store.RequestRow(row.ID); e == nil && latest != nil {
		latest.Number = row.Number
		row = *latest
	}
	s.ok(w, r, http.StatusAccepted, s.lifecycleOf(row))
	return true
}

func (s *Server) controlMachineExecution(ctx context.Context, row records.Request, action string) (bool, *exit.Error) {
	link, problem := s.store.MachineExecution(row.ID)
	if problem != nil {
		return true, problem
	}
	if link == nil {
		return false, nil
	}
	if action == "cancel" && len(link.Receipt) == 0 {
		_, problem = s.store.CancelMachineBeforeAcceptance(row.ID)
	} else if s.machineExecutions == nil {
		problem = exit.Unavailablef("this client cannot control Runtime-owned execution")
	} else {
		problem = s.machineExecutions.Control(ctx, row, action)
	}
	return true, problem
}
