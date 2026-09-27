package api

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"slices"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type MachineExecutions interface {
	Refresh(context.Context, records.Request) *exit.Error
	Control(context.Context, records.Request, string) *exit.Error
	// Withdraw stops a canceled request's submission work that has not reached Runtime.
	Withdraw(string)
}

// This is a client observation, not an execution or custody receipt of its own.
type MachineExecutionView struct {
	Accepted         bool   `json:"accepted"`
	Machine          string `json:"machine"`
	Collected        bool   `json:"collected"`
	ObservationError string `json:"observation_error,omitempty"`
	// Retained says why the finished result stays with the machine: nothing on this host
	// can receive it, so no observation waits for it.
	Retained string `json:"retained,omitempty"`
}

// RetainedOutput is an output held on the machine that produced it, in this host's custody.
type RetainedOutput struct {
	Output   string `json:"output"`
	Machine  string `json:"machine"`
	Manifest string `json:"manifest,omitempty"`
}

func (s *Server) refreshMachineExecution(ctx context.Context, request records.Request) *exit.Error {
	link, problem := s.store.MachineExecution(request.ID)
	if problem != nil || link == nil || len(link.Receipt) == 0 || link.Collected && len(link.PendingControl) == 0 {
		return problem
	}
	if lost, problem := s.store.MachineExecutionLost(request.ID); problem != nil || lost {
		return problem
	}
	if retained, problem := s.store.MachineResultRetained(request.ID); problem != nil || retained != "" {
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
	if !link.Collected {
		view.Retained, _ = s.store.MachineResultRetained(row.ID)
	}
	var retained []RetainedOutput
	if holds, problem := s.store.MachineModelRetentions(row.ID); problem == nil && len(holds) > 0 {
		holder := machine
		if rented, _ := s.store.RentalRow(link.MachineID); row.Machine == "" && rented != nil && rented.MachineName != "" {
			holder = rented.MachineName
		}
		for _, hold := range holds {
			if artifact, _ := records.DecodeModelArtifact(hold.Artifact); hold.State == "held" && artifact != nil {
				retained = append(retained, RetainedOutput{Output: artifact.OutputSlot, Machine: holder, Manifest: artifact.Manifest.Digest})
			}
		}
	}
	state := JobState{
		Number: row.Number, JobID: row.ID, Status: s.publicStatusOf(row), Package: row.Package,
		Function: row.Entrypoint, Attempt: uint64(row.Ordinal), Attempts: int(row.Ordinal),
		RetainWork: row.RetainWork, Retaining: retaining, RetainedOutputs: retained,
		RetryAvailable: s.store.RetainedRetryAvailable(row),
		CreatedAt:      row.CreatedAt, EventsURL: "/v1/requests/" + row.ID + "/events", Outputs: []MediaRef{},
		MachineExecution: view,
	}
	state.OutputExport = s.outputExportOf(row.ID)
	if row.State == "blocked" && state.Status == "failed" {
		state.StoppedEventID = s.store.StoppedEventID(row)
	}
	if row.State == "failed" || row.State == "blocked" {
		state.ErrorType, state.ErrorCode, state.Error, _ = s.store.SettledFailure(row.ID)
	}
	if outputs, problem := s.store.VisibleOutputs(row.ID); problem == nil {
		for _, output := range outputs {
			state.Outputs = append(state.Outputs, MediaRef{OutputID: output.OutputID, MediaID: output.MediaID, URL: "/v1/media/" + output.MediaID, MimeType: output.MimeType, Length: output.Length, Digest: output.Digest})
		}
	}
	if row.ModelTransfer != nil {
		if transfer, problem := s.store.ModelTransferOf(row.ID); problem == nil && transfer != nil {
			state.ModelDestination, state.ModelOutputs = transfer.Destination, transfer.Checkpoints
			if transfer.State == "failed" {
				state.ErrorType, state.Error = transfer.ErrorCode, transfer.SafeError
			}
		}
	}
	if !view.Accepted {
		if state.Status == "queued" {
			state.Stage = "waiting for durable machine acceptance"
			if len(link.Submission) > 0 {
				state.Stage = "reconciling durable machine acceptance"
			}
		}
		if events, problem := s.store.EventsAfter(row.ID, 0, 256); problem == nil {
			for _, event := range events {
				if event.Type == "request.blocked" || event.Type == "request.failed" {
					state.ErrorType, _ = event.Payload["error_type"].(string)
					state.ErrorCode, _ = event.Payload["error_code"].(string)
					state.Error, _ = event.Payload["error"].(string)
				}
			}
		}
	}
	if view.Accepted && state.Status == "in_progress" {
		// No live fanout carries Runtime-owned progress; the imported sample does.
		if value, problem := s.store.LatestMachineProgress(row.ID, row.Ordinal); problem == nil && value != nil {
			state.Progress = value
			state.Stage, _ = value["stage"].(string)
		}
	}
	var submission pb.MachineExecutionSubmit
	if proto.Unmarshal(link.Submission, &submission) == nil {
		state.RetryBudget = int64(submission.MaxAttempts)
	}
	var terminal *pb.AttemptOutcomeBody
	if len(link.Outcome) > 0 {
		var outcome pb.AttemptOutcome
		var body pb.AttemptOutcomeBody
		if proto.Unmarshal(link.Outcome, &outcome) == nil && canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body) == nil {
			terminal = &body
			if state.ErrorType == "" && body.Cause != nil && body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
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
	if intervals, problem := s.store.MachineExecutionIntervals(row.ID); problem != nil {
		view.ObservationError = problem.Message
	} else {
		if state.StoppedEventID != 0 {
			current, present := max(row.Ordinal, 1), false
			for i := range intervals {
				if intervals[i].Attempt == current {
					present = true
					if intervals[i].FinishedAt == "" {
						intervals[i].FinishedAt = s.store.StoppedEventAt(row)
					}
				}
			}
			if !present {
				intervals = append(intervals, records.MachineExecutionInterval{Attempt: current, FinishedAt: s.store.StoppedEventAt(row)})
			}
		}
		state.ExecutionMS = machineExecutionMS(row, link, intervals, terminal, time.Now().UnixMilli())
	}
	return state
}

// Machine attempts belong to Runtime. Count each admission-to-outcome interval,
// just as local attempts count dispatch-to-close. Paused/retry gaps and delayed
// result collection are outside those intervals. An active interval alone uses
// the current clock; a terminal interval never grows when read back later.
func machineExecutionMS(row records.Request, link *records.MachineExecution, intervals []records.MachineExecutionInterval, terminal *pb.AttemptOutcomeBody, now int64) int64 {
	var receipt pb.MachineExecutionReceipt
	if proto.Unmarshal(link.Receipt, &receipt) != nil || receipt.RequestId != row.ID ||
		receipt.AcceptedAtMs == 0 || receipt.AcceptedAtMs > math.MaxInt64 {
		return 0
	}
	type span struct{ start, end int64 }
	spans := map[int64]span{1: {start: int64(receipt.AcceptedAtMs)}}
	for _, interval := range intervals {
		if interval.Attempt <= 0 {
			continue
		}
		value := spans[interval.Attempt]
		if interval.Attempt > 1 {
			if stamp := parseStamp(interval.ResumedAt); !stamp.IsZero() {
				value.start = stamp.UnixMilli()
			}
		}
		if stamp := parseStamp(interval.FinishedAt); !stamp.IsZero() {
			value.end = stamp.UnixMilli()
		}
		spans[interval.Attempt] = value
	}
	current := max(row.Ordinal, 1)
	if _, exists := spans[current]; !exists {
		spans[current] = span{}
	}
	var total int64
	for attempt, value := range spans {
		if attempt == current && (value.start == 0 || value.end == 0) {
			// Older observations may have retained the terminal without its event
			// page. The outcome's own measured runtime is the bounded fallback.
			if terminal != nil && terminal.AttemptOrdinal == uint64(attempt) && terminal.Metrics != nil &&
				terminal.Metrics.RuntimeMs <= math.MaxInt64 && !slices.Contains(terminal.Metrics.UnverifiedFields, "runtime_ms") {
				total += int64(terminal.Metrics.RuntimeMs)
				continue
			}
			switch row.State {
			case "submitted", "queued", "dispatching", "pausing", "canceling":
				value.end = now
			}
		}
		if value.start > 0 && value.end > value.start {
			total += value.end - value.start
		}
	}
	return total
}

func (s *Server) machineJobControl(w http.ResponseWriter, r *http.Request, row records.Request, action, actor string) bool {
	owned, problem := s.controlMachineExecution(r.Context(), row, action, actor)
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

func (s *Server) machineRequestControl(w http.ResponseWriter, r *http.Request, row records.Request, action, actor string) bool {
	owned, problem := s.controlMachineExecution(r.Context(), row, action, actor)
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

func (s *Server) controlMachineExecution(ctx context.Context, row records.Request, action, actor string) (bool, *exit.Error) {
	link, problem := s.store.MachineExecution(row.ID)
	if problem != nil {
		return true, problem
	}
	if link == nil {
		return false, nil
	}
	if action == "cancel" && len(link.Receipt) == 0 {
		_, problem = s.store.CancelMachineBeforeAcceptance(row.ID)
		if problem == nil && s.machineExecutions != nil {
			s.machineExecutions.Withdraw(row.ID)
		}
	} else if s.machineExecutions == nil {
		problem = exit.Unavailablef("this client cannot control Runtime-owned execution")
	} else {
		problem = s.machineExecutions.Control(ctx, row, action)
	}
	// Machine execution controls used to leave no durable actor when they came
	// through the machine-owned route. That made a cancellation look like it
	// came from an observer disconnect. Record the successful explicit control
	// at the API boundary; a watcher never reaches this handler.
	if problem == nil && action == "cancel" {
		if actor == "" {
			actor = "an unnamed api client"
		}
		if e := s.store.AppendEvent(row.ID, "request.cancel_requested", int64(row.Ordinal), map[string]any{
			"actor": actor, "source": "machine_control",
		}); e != nil {
			return true, e
		}
	}
	return true, problem
}
