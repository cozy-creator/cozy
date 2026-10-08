package api

import (
	"context"
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/archive"
	"math"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/records"
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
	Warm               []MachineWarmMember  `json:"warm,omitempty"`
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
	Level        string `json:"level,omitempty"` // what it holds now: installed … gpu
}

// MachineWarmMember is one function of the caller's warm set: the level asked, the level it
// holds now, and why that is lower.
type MachineWarmMember struct {
	Package    string `json:"package"`
	Release    string `json:"release,omitempty"`
	Entrypoint string `json:"entrypoint"`
	Level      string `json:"level"`
	Holds      string `json:"holds"`
	HeldBack   string `json:"held_back,omitempty"`
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

// ReleaseDescription is a package's newest release at the request's selected Hub and its interface,
// as that machine installed it (describe/1).
type ReleaseDescription struct {
	Release   string          `json:"release"`
	Interface json.RawMessage `json:"interface"`
}

type releaseDescriber interface {
	Describe(ctx context.Context, machine, pkg, hub string) (ReleaseDescription, *exit.Error)
}

// describeRelease asks one machine to install a package's newest release at the request's selected Hub and
// name it, so a client that never installed the package types its run with no Hub read.
func (s *Server) describeRelease(w http.ResponseWriter, r *http.Request) {
	describer, ok := s.machineExecutions.(releaseDescriber)
	if !ok {
		s.refuseTyped(w, r, exit.Named(exit.Unavailable, "machine.describe_unsupported", "this Cozy daemon describes no releases"))
		return
	}
	var body struct {
		Package string `json:"package"`
	}
	if json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&body) != nil || body.Package == "" {
		s.refuseTyped(w, r, exit.New(exit.Validation, "a description names one package"))
		return
	}
	hub, problem := s.hubOf(r)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	described, problem := describer.Describe(r.Context(), r.PathValue("machine"), body.Package, hub)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, described)
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

// RetainedOutput is an output held on the machine that produced it, in this host's custody.
type RetainedOutput struct {
	Output   string `json:"output"`
	Machine  string `json:"machine"`
	Manifest string `json:"manifest,omitempty"`
}

// providerRefusal is TensorFS reporting that a provider refused a machine's source call for
// lack of authentication.
var providerRefusal = regexp.MustCompile(`CREDENTIAL_REQUIRED|origin answered HTTP 40[13]\b`)

func (s *Server) refreshMachineExecution(ctx context.Context, request records.Request) *exit.Error {
	link, problem := s.store.MachineExecution(request.ID)
	if problem != nil || link == nil || link.Abandoned || len(link.Receipt) == 0 || link.Collected && len(link.PendingControl) == 0 {
		return problem
	}
	// A run whose rental is known to have ended settles here: nothing waits on its machine.
	if problem := s.store.ReconcileEndedMachineExecution(request.ID); problem != nil {
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

// machineWord is the machine a run names: its recorded word, else its machine, a rental
// reached as an explicit endpoint (a foreground --rental run) by that rental's name.
func (s *Server) machineWord(row records.Request, link *records.MachineExecution) string {
	if row.Machine != "" {
		return row.Machine
	}
	if machineendpoint.IsName(link.MachineID) {
		if ep, _ := s.store.RequestMachineEndpoint(row.ID, link.MachineID); ep != nil && ep.WorkspaceID != "" {
			if rented, _ := s.store.RentalRow(ep.WorkspaceID); rented != nil && rented.MachineName != "" {
				return rented.MachineName
			}
		}
	}
	return link.MachineID
}

func (s *Server) machineJobState(row records.Request, link *records.MachineExecution) JobState {
	machine := s.machineWord(row, link)
	view := &MachineExecutionView{Accepted: len(link.Receipt) > 0, Machine: machine, Collected: link.Collected, AbandonedLocally: link.Abandoned}
	v1run, _ := s.store.RunV1(row.ID)
	if accepted := records.RunV1State(link); v1run && accepted != nil {
		view.Worker, view.Number = link.MachineID, accepted.Number
	} else if !v1run {
		if receipt, err := archive.ReadReceipt(link.Receipt); err == nil {
			view.Worker, view.Number = receipt.WorkerID, receipt.Number
		}
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
	var terminal *archive.Terminal
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
		if body, err := archive.ReadOutcome(link.Outcome); err == nil {
			terminal = body
			if state.ErrorType == "" && body.Cause != "" && body.Status != 1 {
				state.Error = body.Message
				state.ErrorType = body.Cause
				if providerRefusal.MatchString(body.Message) {
					state.ErrorType = "model_source.auth_required"
					state.Error += "; the provider requires authentication: set huggingface_token or civitai_token in the daemon config"
				}
			}
			if view.Collected && len(body.Result) > 0 {
				state.Result = json.RawMessage(body.Result)
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
	if v1run {
		state.AttemptWallMS = s.runV1WallMS(row, time.Now())
	}
	if measured, known, problem := s.store.MachineRunExecutionMS(row.ID, max(row.Ordinal, 1), state.Status == "in_progress"); problem != nil {
		view.ObservationError = problem.Message
	} else {
		state.ExecutionMS, state.ExecutionKnown = measured, known
	}
	if outcome := records.RunV1Outcome(link); v1run && outcome != nil && outcome.ExecutionMs > 0 && outcome.ExecutionMs <= math.MaxInt64 {
		// A cozy.machine.v1 machine measures the run's own running time, every attempt summed.
		state.ExecutionMS, state.ExecutionKnown = int64(outcome.ExecutionMs), true
	}
	if state.Status == "canceled" {
		// A canceled run says WHO (cl-108), as a job without a machine does.
		if actor, _, _, problem := s.store.CancelAttribution(row.ID); problem == nil {
			state.CanceledBy = actor
		}
	}
	return state
}

// runV1WallMS is an unsettled cozy.machine.v1 run's time on its machine so far: from its start
// to now, or to its pause while it rests paused; 0 before it started. A settled run's wall_ms
// and execution_ms say the rest.
func (s *Server) runV1WallMS(row records.Request, now time.Time) int64 {
	if records.Settled(row.State) {
		return 0
	}
	started, ended, paused, problem := s.store.RunV1Span(row.ID)
	if problem != nil || started.IsZero() {
		return 0
	}
	switch {
	case !ended.IsZero():
		now = ended
	case row.State == "paused" && !paused.IsZero():
		now = paused
	}
	return max(now.Sub(started).Milliseconds(), 0)
}

// Machine attempts belong to Runtime. Count each admission-to-outcome interval,
// just as local attempts count dispatch-to-close. Paused/retry gaps and delayed
// result collection are outside those intervals. An active interval alone uses
// the current clock; a terminal interval never grows when read back later.
func machineAttemptWallMS(row records.Request, link *records.MachineExecution, intervals []records.MachineExecutionInterval, terminal *archive.Terminal, now int64) int64 {
	receipt, err := archive.ReadReceipt(link.Receipt)
	if err != nil || receipt.RequestID != row.ID ||
		receipt.AcceptedMS == 0 || receipt.AcceptedMS > math.MaxInt64 {
		return 0
	}
	type span struct{ start, end int64 }
	spans := map[int64]span{1: {start: int64(receipt.AcceptedMS)}}
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
			if terminal != nil && terminal.Attempt == uint64(attempt) &&
				terminal.RuntimeMS <= math.MaxInt64 && !slices.Contains(terminal.Unverified, "runtime_ms") {
				total += int64(terminal.RuntimeMS)
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
