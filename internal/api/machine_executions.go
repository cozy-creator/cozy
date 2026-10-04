package api

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

type MachineExecutions interface {
	Refresh(context.Context, records.Request) *exit.Error
	// Cancel control only wakes delivery of recorded intent; it must not wait
	// for remote I/O. Pause and resume retain synchronous control semantics.
	Control(context.Context, records.Request, string) *exit.Error
	// Withdraw stops a canceled request's submission work that has not reached Runtime.
	Withdraw(string)
	// Status is one machine's picture as it reports it (cozy.machine.v1 Status).
	Status(ctx context.Context, machine string) (MachineStatus, *exit.Error)
	// MachineLog is one log a machine keeps, at most its newest tailBytes when nonzero.
	MachineLog(ctx context.Context, machine, log string, tailBytes uint64) (MachineLog, *exit.Error)
	// Forget drops this daemon's kept connection to a machine, before its credentials go.
	Forget(machine string)
	// ForgetPackage tells every machine this daemon knows that a package's releases or owner
	// bindings changed, so its next run reads them once.
	ForgetPackage(ctx context.Context, pkg string) ForgottenPackage
}

// ForgottenPackage names the machines that dropped what they read of a package, and why any
// other known machine could not be told.
type ForgottenPackage struct {
	Package  string   `json:"package"`
	Machines []string `json:"machines"`
	Notes    []string `json:"notes,omitempty"`
}

func (s *Server) forgetPackage(w http.ResponseWriter, r *http.Request) {
	if s.machineExecutions == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this Cozy daemon runs no machines"))
		return
	}
	var body struct {
		Package string `json:"package"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
	if err := decoder.Decode(&body); err != nil || strings.Count(body.Package, "/") != 1 {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "name one package as org/name", "")
		return
	}
	s.ok(w, r, http.StatusOK, s.machineExecutions.ForgetPackage(r.Context(), body.Package))
}

// MachineStatus is what one machine reports of itself: identity, software, GPUs, the
// owner's live runs and held environments, disk and its idle deadline (0: none).
type MachineStatus struct {
	WorkerID           string               `json:"worker_id"`
	BootID             string               `json:"boot_id"`
	Agent              string               `json:"agent"`
	Phase              string               `json:"phase"`
	Runtime            string               `json:"runtime,omitempty"`
	TensorFS           string               `json:"tensorfs,omitempty"`
	Capabilities       []string             `json:"capabilities"`
	GPUs               []MachineGPU         `json:"gpus"`
	Runs               []MachineRun         `json:"runs"`
	Environments       []MachineEnvironment `json:"environments"`
	DiskTotalBytes     uint64               `json:"disk_total_bytes,omitempty"`
	DiskFreeBytes      uint64               `json:"disk_free_bytes,omitempty"`
	IdleDeadlineUnixMS int64                `json:"idle_deadline_unix_ms,omitempty"`
}
type MachineGPU struct {
	Index       uint32 `json:"index"`
	Name        string `json:"name"`
	MemoryBytes uint64 `json:"memory_bytes"`
	Driver      string `json:"driver"`
}
type MachineRun struct {
	ID     string `json:"id"`
	Number uint64 `json:"number"`
	State  string `json:"state"`
}
type MachineEnvironment struct {
	Installation string `json:"installation"`
	Package      string `json:"package"`
	Release      string `json:"release"`
}

func (s *Server) machineStatus(w http.ResponseWriter, r *http.Request) {
	if s.machineExecutions == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this Cozy daemon runs no machines"))
		return
	}
	status, problem := s.machineExecutions.Status(r.Context(), r.PathValue("machine"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, status)
}

// MachineLog is one log a machine keeps, oldest line first. Unavailable says why the machine
// could not answer, for a machine whose agent predates the read: a note, never a failure.
type MachineLog struct {
	Log         string `json:"log"`
	Text        string `json:"text"`
	Unavailable string `json:"unavailable,omitempty"`
}

func (s *Server) machineLog(w http.ResponseWriter, r *http.Request) {
	if s.machineExecutions == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this Cozy daemon runs no machines"))
		return
	}
	var tail uint64
	if text := r.URL.Query().Get("tail_bytes"); text != "" {
		n, err := strconv.ParseUint(text, 10, 64)
		if err != nil {
			s.refuseTyped(w, r, exit.Usagef("tail_bytes is a byte count, not %q", text))
			return
		}
		tail = n
	}
	log, problem := s.machineExecutions.MachineLog(r.Context(), r.PathValue("machine"), r.PathValue("log"), tail)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, log)
}

// This is a client observation, not an execution or custody receipt of its own.
type MachineExecutionView struct {
	AbandonedLocally bool   `json:"abandoned_locally,omitempty"`
	Accepted         bool   `json:"accepted"`
	Machine          string `json:"machine"`
	Collected        bool   `json:"collected"`
	ObservationError string `json:"observation_error,omitempty"`
	// Retained says why the finished result stays with the machine: nothing on this host
	// can receive it, so no observation waits for it.
	Retained string `json:"retained,omitempty"`
	// CollectionRefused names why collecting a finished result ended (a destination that
	// refuses its files, bytes its machine no longer holds); only `cozy run watch` tries again.
	CollectionRefused string `json:"collection_refused,omitempty"`
	// Worker and Number name the run on its machine, as the machine's receipt does; Number
	// is 0 from a Runtime older than wire 66.
	Worker string `json:"worker,omitempty"`
	Number uint64 `json:"number,omitempty"`
}

// RetainedOutput is an output held on the machine that produced it, in this host's custody,
// with each upload of it to a private checkpoint.
type RetainedOutput struct {
	Output   string                 `json:"output"`
	Machine  string                 `json:"machine"`
	Manifest string                 `json:"manifest,omitempty"`
	Uploads  []records.OutputUpload `json:"uploads,omitempty"`
}

// providerRefusal is TensorFS reporting that a provider refused a machine's source call for
// lack of authentication.
var providerRefusal = regexp.MustCompile(`CREDENTIAL_REQUIRED|origin answered HTTP 40[13]\b`)

func (s *Server) refreshMachineExecution(ctx context.Context, request records.Request) *exit.Error {
	link, problem := s.store.MachineExecution(request.ID)
	if problem != nil || link == nil || link.Abandoned || len(link.Receipt) == 0 || link.Collected && len(link.PendingControl) == 0 {
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
	view := &MachineExecutionView{Accepted: len(link.Receipt) > 0, Machine: machine, Collected: link.Collected, AbandonedLocally: link.Abandoned}
	v1run, _ := s.store.RunV1(row.ID)
	var receipt pb.MachineExecutionReceipt
	if accepted := records.RunV1State(link); v1run && accepted != nil {
		view.Worker, view.Number = link.MachineID, accepted.Number
	} else if !v1run && proto.Unmarshal(link.Receipt, &receipt) == nil {
		view.Worker, view.Number = receipt.WorkerId, receipt.Number
	}
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
		uploads, _ := s.store.OutputUploads(row.ID)
		for _, hold := range holds {
			if artifact, _ := records.DecodeModelArtifact(hold.Artifact); hold.State == "held" && artifact != nil {
				output := RetainedOutput{Output: artifact.OutputSlot, Machine: holder, Manifest: artifact.Manifest.Digest}
				for _, upload := range uploads {
					if upload.Output == artifact.OutputSlot {
						output.Uploads = append(output.Uploads, upload)
					}
				}
				retained = append(retained, output)
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
	if row.State == "failed" || row.State == "blocked" || row.State == "abandoned" {
		state.ErrorType, state.ErrorCode, state.Error, _ = s.store.SettledFailure(row.ID)
	}
	// The run's output log as this client holds it; its result is the fold once collected.
	state.Output, _ = s.store.Output(row.ID, row.State)
	if outputs, problem := s.store.VisibleOutputs(row.ID); problem == nil {
		for _, output := range outputs {
			state.Outputs = append(state.Outputs, MediaRef{OutputID: output.OutputID, MimeType: output.MimeType, Length: output.Length, Digest: output.Digest, Path: output.Path})
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
				if event.Type == "request.blocked" || event.Type == "run.failed" {
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
	var terminal *pb.AttemptOutcomeBody
	if outcome := records.RunV1Outcome(link); v1run && outcome != nil {
		// A cozy.machine.v1 run's outcome carries its result and typed reason.
		if state.ErrorType == "" && outcome.Status != "succeeded" && outcome.Reason != nil {
			state.ErrorType, state.Error = outcome.Reason.Code, outcome.Reason.Message
		}
		if view.Collected && len(outcome.Result) > 0 {
			state.Result = json.RawMessage(outcome.Result)
		}
		if code, message, problem := s.store.MachineCollectionRefusal(row.ID); !view.Collected && problem == nil && code != "" {
			view.CollectionRefused, view.ObservationError = code, message
		}
	} else if len(link.Outcome) > 0 && !v1run {
		var outcome pb.AttemptOutcome
		var body pb.AttemptOutcomeBody
		if proto.Unmarshal(link.Outcome, &outcome) == nil && canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body) == nil {
			terminal = &body
			if state.ErrorType == "" && body.Cause != nil && body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
				state.Error = body.SafeMessage
				state.ErrorType = strings.TrimPrefix(body.Cause.Code.String(), "CAUSE_CODE_")
				if providerRefusal.MatchString(body.SafeMessage) {
					state.ErrorType = "model_source.auth_required"
					state.Error += "; the provider requires authentication: set huggingface_token or civitai_token in the daemon config"
				}
			}
			if view.Collected && body.Result != nil && len(body.Result.InlineResult) > 0 {
				state.Result = json.RawMessage(body.Result.InlineResult)
			}
			if !view.Collected {
				view.ObservationError = "execution finished; result collection has not established recipient custody"
				if code, message, problem := s.store.MachineCollectionRefusal(row.ID); problem == nil && code != "" {
					view.CollectionRefused, view.ObservationError = code, message
				}
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
		if !v1run {
			state.AttemptWallMS = machineAttemptWallMS(row, link, intervals, terminal, time.Now().UnixMilli())
		}
	}
	if measured, known, problem := s.store.MachineRunExecutionMS(row.ID, max(row.Ordinal, 1), state.Status == "in_progress"); problem != nil {
		view.ObservationError = problem.Message
	} else {
		state.ExecutionMS, state.ExecutionKnown = measured, known
	}
	if state.Status == "canceled" {
		// A canceled run says WHO (cl-108), as a job without a machine does.
		if actor, _, _, problem := s.store.CancelAttribution(row.ID); problem == nil {
			state.CanceledBy = actor
		}
	}
	return state
}

// Machine attempts belong to Runtime. Count each admission-to-outcome interval,
// just as local attempts count dispatch-to-close. Paused/retry gaps and delayed
// result collection are outside those intervals. An active interval alone uses
// the current clock; a terminal interval never grows when read back later.
func machineAttemptWallMS(row records.Request, link *records.MachineExecution, intervals []records.MachineExecutionInterval, terminal *pb.AttemptOutcomeBody, now int64) int64 {
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
	if link.Abandoned {
		if action == "cancel" {
			return true, nil
		}
		return true, exit.Named(exit.Conflict, "request.abandoned", "this run was abandoned locally and cannot be resumed")
	}
	if action == "cancel" && records.Settled(row.State) && !row.RetainWork {
		// Already terminal and holding nothing a cancel would release: the same answer
		// again, without asking the machine. A completed run's collection releases what it
		// holds; any other ended run its Runtime still retains is released by this cancel.
		owed, problem := s.store.MachineExecutionOwesWork(row.ID)
		if problem != nil || !owed || row.State == "succeeded" {
			return true, problem
		}
	}
	if action == "cancel" {
		if actor == "" {
			actor = "an unnamed api client"
		}
		if _, problem := s.store.RequestMachineCancellation(row.ID, actor); problem != nil {
			return true, problem
		}
		// Control's cancel arm only wakes the existing observer. The durable
		// request survives a closing daemon or an unavailable machine.
		if s.machineExecutions != nil {
			_ = s.machineExecutions.Control(ctx, row, action)
		}
		return true, nil
	}
	if s.machineExecutions == nil {
		return true, exit.Unavailablef("this client cannot control Runtime-owned execution")
	}
	return true, s.machineExecutions.Control(ctx, row, action)
}
