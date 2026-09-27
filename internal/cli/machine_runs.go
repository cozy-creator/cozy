package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"
)

type publishedPreparation struct {
	*pb.DesiredPlacementSet
	InstalledPackage   *pb.InstalledPackage
	LockedRequirements []byte
	Retained           bool // the machine answered from its preparation of these exact inputs
}

func retainWorkerInstallation(connection *machineConnection, expected localpackage.Installation, installed *pb.InstalledPackage) *exit.Error {
	if installed == nil || installed.InstallationId != expected.ID || installed.Package != expected.Package || installed.Release != expected.Release || len(installed.PackageInterface) == 0 {
		return exit.New(exit.Validation, "worker did not return the selected installation and interface")
	}
	connection.installed[expected.ID] = proto.Clone(installed).(*pb.InstalledPackage)
	return nil
}

// machineDestinationRuntimeFloor is the first Runtime that publishes a job root's weights
// outputs to its destination; an older one silently ignores the destination.
const machineDestinationRuntimeFloor = "0.18.52"

// machineRuns is a client transport and observer. Stopping it closes connections
// and upload/observation work; it never sends an execution cancellation.
type machineRuns struct {
	ctx       context.Context
	cancel    context.CancelFunc
	context   *Context
	layout    home.Layout
	store     *records.Store
	resolver  *Resolver
	fleet     *managedRentals
	mu        sync.Mutex
	running   map[string]bool
	placed    map[string]string // the last placement decision recorded per waiting run
	machines  *machines.Resolver
	observers sync.Map // one collection/control lock per observed request
	updates   *rentalRuntimeUpdates
	// ownerReads paces owner finalization reads of publications a machine cannot settle.
	ownerReads ownerReads
	// submitting stops each request's submission work in flight (upload, preparation,
	// staging) once its cancel is durable; guarded by mu.
	submitting map[string]context.CancelFunc
}

func newMachineRuns(ctx *Context, layout home.Layout, store *records.Store, resolver *Resolver, fleet *managedRentals, found *machines.Resolver) *machineRuns {
	background, cancel := context.WithCancel(context.Background())
	return &machineRuns{ctx: background, cancel: cancel, context: ctx, layout: layout, store: store, resolver: resolver, fleet: fleet, machines: found, running: map[string]bool{}, placed: map[string]string{}, submitting: map[string]context.CancelFunc{}}
}

func (m *machineRuns) Start(request records.Request) *exit.Error {
	m.mu.Lock()
	if m.ctx.Err() != nil {
		m.mu.Unlock()
		return exit.Named(exit.Unavailable, "daemon.closing", "the client observer has disconnected")
	}
	if m.running[request.ID] {
		m.mu.Unlock()
		return nil
	}
	m.running[request.ID] = true
	m.mu.Unlock()
	go func() {
		defer func() { m.mu.Lock(); delete(m.running, request.ID); delete(m.placed, request.ID); m.mu.Unlock() }()
		lastError, delay := "", time.Second
		for m.ctx.Err() == nil {
			current, problem := m.store.RequestRow(request.ID)
			if problem != nil || current == nil {
				return
			}
			if owed, problem := m.machineWorkOwed(request.ID); problem != nil || !owed {
				return
			}
			if current.State == "refused" {
				return
			}
			link, problem := m.store.MachineExecution(request.ID)
			if problem != nil || link == nil {
				return
			}
			if len(link.Receipt) == 0 && (records.Settled(current.State) || records.RetainedState(current.State)) &&
				!(current.State == "canceled" && len(link.Submission) == 0) {
				// Restart resumes observation, not withdrawn or failed execution
				// intent. A receipt can reconcile a real outcome; absent one, do
				// not turn a stopped request into a new Submit RPC.
				return
			}
			if len(link.Receipt) == 0 {
				if current.State == "canceled" && len(link.Submission) == 0 {
					if link.MachineID != "" {
						var connection *machineConnection
						connection, problem = m.connect(m.ctx, link.MachineID, m.runHolder(*current, "releasing its inputs"))
						if problem == nil {
							problem = m.releaseMachineInputs(m.ctx, *current, connection)
							connection.Close()
						}
						if problem == nil {
							return
						}
					} else {
						return
					}
				} else {
					problem = m.submit(*current, link)
				}
			} else {
				if link.CancelRequested {
					problem = m.control(m.ctx, *current, "cancel", true)
				} else {
					problem = m.Refresh(m.ctx, *current)
					if problem != nil && problem.ErrName() == "machine_execution.result_custody_required" {
						return // the result stays with the machine; nothing here can collect it
					}
				}
				if problem == nil {
					observed, e := m.store.MachineExecution(request.ID)
					// A publication only its owner can settle keeps the execution observed.
					awaiting, awaitingProblem := m.store.MachinePublicationsAwaitingOwner(request.ID)
					if e == nil && observed != nil && observed.Collected && awaitingProblem == nil && len(awaiting) == 0 {
						return
					}
					if e == nil && observed != nil && len(observed.PendingControl) == 0 {
						var state pb.MachineExecutionState
						if proto.Unmarshal(observed.ObservedState, &state) == nil && state.State == "canceled" {
							if owed, problem := m.machineWorkOwed(request.ID); problem == nil && !owed {
								return
							}
						}
					}
				}
			}
			// A wait that repeats itself (no machine yet, one still booting) is asked less
			// often; anything else is observed again at once.
			if problem != nil && problem.Message == lastError {
				delay = min(2*delay, 5*time.Second)
			} else {
				delay = time.Second
			}
			if problem != nil && problem.Message != lastError && m.ctx.Err() == nil {
				fmt.Fprintf(m.context.Out, "machine execution %s: %s\n", request.ID, problem.Message)
				lastError = problem.Message
				// submit may have durably frozen/transmitted its offer after this
				// loop read link. Only a fresh journal read can prove it was unsent.
				// A machine lost mid-preparation released the run; it is placed again. A
				// canceled run is neither failed nor parked: the next pass releases its inputs.
				latest, readProblem := m.store.MachineExecution(request.ID)
				row, rowProblem := m.store.RequestRow(request.ID)
				unsent := readProblem == nil && rowProblem == nil && latest != nil && row != nil && row.State != "canceled" && len(latest.Submission) == 0
				if unsent && problem.Code != exit.Unavailable && problem.Code != exit.Deadline && latest.MachineID == link.MachineID {
					_, _ = m.store.FailQueuedRequest(request.ID, records.QueuedFailure(problem))
					return
				}
				if unsent {
					// A run still waiting to reach its machine says why, as a queued run does.
					_ = m.store.AppendEvent(request.ID, "request.parked", 0, map[string]any{"reason": problem.Message, "wait": orchestrator.WaitRental})
				}
			}
			select {
			case <-m.ctx.Done():
				return
			case <-time.After(delay):
			}
		}
	}()
	return nil
}

func (m *machineRuns) Resume() {
	if problem := m.store.ReconcileEndedMachineExecutions(); problem != nil {
		fmt.Fprintf(m.context.Out, "machine execution loss recovery: %s\n", problem.Message)
		return
	}
	links, problem := m.store.MachineExecutions()
	if problem != nil {
		fmt.Fprintf(m.context.Out, "machine execution observation recovery: %s\n", problem.Message)
		return
	}
	for _, link := range links {
		if awaiting, problem := m.store.MachinePublicationsAwaitingOwner(link.RequestID); link.Collected && len(link.PendingControl) == 0 && (problem != nil || len(awaiting) == 0) {
			continue
		}
		request, problem := m.store.RequestRow(link.RequestID)
		if problem == nil && request != nil {
			_ = m.Start(*request)
		}
	}
}

// Withdraw stops the submission work in flight for a request whose cancel is durable. The
// observer's next pass releases whatever of it reached the machine.
func (m *machineRuns) Withdraw(request string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if stop := m.submitting[request]; stop != nil {
		stop()
	}
}

// machineWorkOwed is whether the execution still owes work or custody, or holds a sent
// publication only its owner can settle.
func (m *machineRuns) machineWorkOwed(request string) (bool, *exit.Error) {
	if owed, problem := m.store.MachineExecutionOwesWork(request); problem != nil || owed {
		return owed, problem
	}
	awaiting, problem := m.store.MachinePublicationsAwaitingOwner(request)
	return len(awaiting) > 0, problem
}

func (m *machineRuns) submit(request records.Request, link *records.MachineExecution) (out *exit.Error) {
	ctx, stop := context.WithCancel(m.ctx)
	m.mu.Lock()
	m.submitting[request.ID] = stop
	m.mu.Unlock()
	defer func() {
		m.mu.Lock()
		delete(m.submitting, request.ID)
		m.mu.Unlock()
		stop()
	}()
	_, deadline, problem := m.store.RequestExecutionTiming(request.ID)
	if problem != nil {
		return problem
	}
	request.DeadlineUnixMS = deadline
	if link.MachineID == "" {
		machine := machines.Placement(request)
		if request.Rental {
			if machine == "" {
				// The placement decision pins the machine and, for a model ladder, the
				// lanes that machine takes. Runtime alone decides its devices.
				decision, line, problem := m.fleet.acquire(request)
				m.recordPlacement(request, decision, line)
				if problem != nil {
					return problem
				}
				machine = decision.RentalID
				if machine == "" {
					return exit.Unavailablef("waiting to select a machine for execution")
				}
			}
			if pinned, problem := m.store.PinRental(request.ID, machine, nil); problem != nil || !pinned {
				if problem != nil {
					return problem
				}
				return exit.New(exit.Canceled, "submission stopped before machine selection")
			}
		}
		if problem := m.store.LinkMachineExecution(request.ID, machine); problem != nil {
			return problem
		}
		link.MachineID = machine
	}
	if request.Rental {
		current, problem := m.store.RequestRow(request.ID)
		if problem != nil || current == nil {
			return problem
		}
		request.Worker, request.Models, request.PlanID = link.MachineID, current.Models, current.PlanID
	}

	if len(link.Submission) == 0 && m.updates != nil {
		if problem := m.updates.preflight(ctx, request, link.MachineID); problem != nil {
			return problem
		}
	}
	began := time.Now()
	connection, problem := m.connect(ctx, link.MachineID, m.runHolder(request, "preparing its submission"))
	if problem != nil {
		return problem
	}
	defer connection.Close()
	if len(link.Submission) == 0 {
		m.submissionStage(request.ID, "connect", link.MachineID, began)
	}
	// Only a new submission needs the current contract; observation, collection,
	// cancellation and release reach any peer.
	if problem := connection.ValidateNewWork(); problem != nil {
		return problem
	}
	if len(link.Submission) == 0 {
		if request, problem = m.pinToMachine(ctx, request, connection); problem != nil {
			return problem
		}
	}
	submission := &pb.MachineExecutionSubmit{}
	if len(link.Submission) > 0 {
		if err := proto.Unmarshal(link.Submission, submission); err != nil {
			return exit.Internalf("recorded machine submission is unreadable: %s", err)
		}
		if submission.ExpectedExecutionWorkspaceId == "" {
			// A submission recorded before workspace fencing defaults to the machine's
			// current workspace; the receipt then pins the one that accepted it.
			workspace, problem := currentExecutionWorkspace(ctx, connection)
			if problem != nil {
				return problem
			}
			submission.ExpectedExecutionWorkspaceId = workspace.ExecutionWorkspaceId
		}
	} else {
		authorization, problem := m.publicationAuthorization(ctx, request.ID, link.MachineID, connection)
		if problem != nil {
			return problem
		}
		var built *pb.MachineExecutionSubmit
		if request.LocalInstallationID == "" {
			built, problem = m.publishedSubmission(ctx, request, connection)
			if problem != nil {
				return problem
			}
		} else {
			capture, problem := m.resolver.CaptureMachineExecution(request)
			if problem != nil {
				return problem
			}
			var placements *pb.DesiredPlacementSet
			for index, revision := range capture.Installations {
				if problem := connection.prepare(ctx, request.ID, revision); problem != nil {
					return problem
				}
				capture.Installations[index].PackageInterface = connection.installed[revision.ID].PackageInterface
				set, problem := connection.prepareModels(ctx, request, revision)
				if problem != nil {
					return problem
				}
				if revision.ID == request.LocalInstallationID {
					placements = set
				}
			}
			var graph pb.MachineExecutionCapture
			if err := canonical.Unmarshal(capture.Canonical, &graph); err != nil {
				return exit.Internalf("cannot read accepted installation graph: %s", err)
			}
			for i, installed := range graph.InstalledPackages {
				graph.InstalledPackages[i] = connection.installed[installed.InstallationId]
			}
			var encodeErr error
			capture.Canonical, capture.Digest, encodeErr = canonical.Identity(&graph)
			if encodeErr != nil {
				return exit.Internalf("cannot retain worker installation metadata: %s", encodeErr)
			}
			origin, problem := connection.modelDefaultOrigin(ctx)
			if problem != nil {
				return problem
			}
			began := time.Now()
			var detail string
			capture, detail, problem = m.resolver.captureMachineModelDefaults(capture, request, origin)
			if problem != nil {
				return problem
			}
			m.submissionStage(request.ID, "model_defaults", detail, began)
			installed := connection.installed[request.LocalInstallationID]
			if installed == nil {
				return exit.New(exit.Conflict, "machine root installation is unavailable")
			}
			surface, problem := launch.DecodePackageInterface(installed.PackageInterface)
			if problem != nil {
				return problem
			}
			job, problem := surface.Function(request.Entrypoint)
			if problem != nil {
				return problem
			}
			if (job.Kind == "job") != request.IsJob() {
				return exit.New(exit.Conflict, "installed callable changed its kind")
			}
			if job.Kind == "job" {
				request.PlanID = job.DescriptorID
			} else if request, problem = m.bindServingPlan(request, installed.InstallationId, placements); problem != nil {
				return problem
			}
			began = time.Now()
			byteInputs, problem := m.stageMachineInputs(ctx, request, connection)
			if problem != nil {
				return problem
			}
			if len(byteInputs) > 0 {
				m.submissionStage(request.ID, "inputs", fmt.Sprintf("%d input(s)", len(byteInputs)), began)
			}
			if job.Kind == "job" {
				var plan *orchestrator.JobPlan
				if plan, problem = machineJobPlan(ctx, connection, request, installed.InstallationId, job); problem != nil {
					return problem
				}
				built, problem = orchestrator.MachineJobSubmission(request, capture, plan, byteInputs)
			} else {
				built, problem = orchestrator.MachineServingSubmission(request, capture, placements, byteInputs)
			}
			if problem != nil {
				return problem
			}

		}
		if connection.WireMinor < pb.ModelDefaultGPUCountWireMinor {
			var capture pb.MachineExecutionCapture
			if err := canonical.Unmarshal(built.CaptureCanonicalBytes, &capture); err != nil {
				return exit.Internalf("cannot read captured model preferences: %s", err)
			}
			for _, row := range capture.ModelDefaults {
				for _, rung := range row.Rungs {
					if rung.Gpus > 0 {
						return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "this call has an explicit model GPU group count and requires Runtime protocol63")
					}
				}
			}
		}
		built.PublicationAuthorizationId = authorization
		built.PreparedState.WireMinor = min(built.PreparedState.WireMinor, connection.WireMinor)
		if problem := m.freezeMachineSubmission(ctx, connection, request.ID, link.MachineID, built); problem != nil {
			return problem
		}
		submission = built
	}
	if _, problem := m.resolver.capturedResultInterface(request); problem != nil {
		return problem
	}
	began = time.Now()
	if problem := m.sendMachineSubmission(ctx, connection, request.ID, submission); problem != nil {
		return problem
	}
	m.submissionStage(request.ID, "submit", "", began)
	if m.fleet != nil && m.fleet.owner != nil {
		m.fleet.owner.ForgetPhase(request.ID) // Runtime reports the run from here on
	}
	return m.releaseMachineInputs(ctx, request, connection)
}

// pinToMachine fixes each model ladder's rung for the devices this machine measured. A
// rental's placement decision already pinned its rung from the width it bought.
func (m *machineRuns) pinToMachine(ctx context.Context, request records.Request, connection *machineConnection) (records.Request, *exit.Error) {
	pinned := true
	for _, model := range request.Models {
		pinned = pinned && model.Pinned()
	}
	if pinned {
		return request, nil
	}
	workspace, err := connection.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: connection.Claim})
	if err != nil {
		return request, machineTransport(err)
	}
	accelerator := ""
	if len(workspace.Devices) > 0 {
		accelerator = workspace.Devices[0].Name
	}
	models, _, ok := rental.Pin(request.Models, accelerator, len(workspace.Devices))
	if !ok {
		return request, exit.Named(exit.Structural, "machine_execution.model_rung_unavailable",
			"no rung of this call's model ladder fits %d× %q on %s", len(workspace.Devices), accelerator, connection.Name)
	}
	if problem := m.store.PinMachineModels(request.ID, models); problem != nil {
		return request, problem
	}
	request.Models = models
	return request, nil
}

// recordPlacement makes a waiting run's placement decision durable, as a queued run's is:
// the fleet line and the decision record, each once per change.
func (m *machineRuns) recordPlacement(request records.Request, decision orchestrator.PlacementDecision, line string) {
	m.mu.Lock()
	news := m.placed[request.ID] != decision.Line()+"\x00"+line
	m.placed[request.ID] = decision.Line() + "\x00" + line
	m.mu.Unlock()
	if !news {
		return
	}
	if line != "" {
		_ = m.store.AppendEvent(request.ID, "request.rentals", 0, map[string]any{"line": line})
	}
	if len(decision.Candidates) > 0 {
		m.fleet.owner.LogPlacement(request, decision)
	}
}

// freezeMachineSubmission persists authenticated journal identity with the exact
// offer before any Submit RPC. Only a never-transmitted submission may discover it.
func (m *machineRuns) freezeMachineSubmission(ctx context.Context, connection *machineConnection, requestID, machine string, submission *pb.MachineExecutionSubmit) *exit.Error {
	workspace, problem := currentExecutionWorkspace(ctx, connection)
	if problem != nil {
		return problem
	}
	if problem := exactExecutionGPUs(submission, workspace); problem != nil {
		return problem
	}
	submission.ExpectedExecutionWorkspaceId = workspace.ExecutionWorkspaceId
	if problem := m.store.RecordMachineSubmission(requestID, submission); problem != nil || workspace.SourceCredentials || len(m.resolver.SourceCredentials()) == 0 {
		return problem
	}
	return m.store.AppendEvent(requestID, "request.warning", 0, map[string]any{"message": "this machine's Runtime cannot receive your Hugging Face or Civitai credential, so a gated source fails there; " + machines.RuntimeUpdate(machine)})
}

// exactExecutionGPUs sends a counted group only to a Runtime that runs it exactly. Without
// that capability a count covering the whole machine is already Runtime's own width; a
// narrower one cannot be honoured and refuses.
func exactExecutionGPUs(submission *pb.MachineExecutionSubmit, workspace *pb.MachineExecutionWorkspace) *exit.Error {
	set := submission.PreparedState.GetPlacementSet()
	if set.GetExecutionGpus() == 0 || workspace.ExactExecutionGpus {
		return nil
	}
	if devices := len(workspace.Devices); devices == 0 || int(set.ExecutionGpus) < devices {
		return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required",
			"this call runs on exactly %d GPUs of the machine, and its Runtime cannot hold a call to an exact GPU count; update the rental Runtime", set.ExecutionGpus)
	}
	set.ExecutionGpus = 0
	return nil
}

func currentExecutionWorkspace(ctx context.Context, connection *machineConnection) (*pb.MachineExecutionWorkspace, *exit.Error) {
	workspace, err := connection.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: connection.Claim})
	if err != nil {
		return nil, machineTransport(err)
	}
	if workspace == nil || workspace.WorkerId != connection.Claim.WorkerId || workspace.WorkerBootId == "" ||
		(connection.Claim.WorkerBootId != "" && workspace.WorkerBootId != connection.Claim.WorkerBootId) ||
		workspace.ExecutionWorkspaceId == "" || len(workspace.ExecutionWorkspaceId) > 256 {
		return nil, exit.New(exit.Conflict, "machine returned an invalid execution workspace identity")
	}
	return workspace, nil
}

func (m *machineRuns) sendMachineSubmission(ctx context.Context, connection *machineConnection, requestID string, frozen *pb.MachineExecutionSubmit) *exit.Error {
	if frozen.ExpectedExecutionWorkspaceId == "" {
		return exit.Named(exit.Conflict, "machine_execution.workspace_required", "recorded submission has no workspace identity; its acceptance cannot safely be retried")
	}
	// The owner's provider credentials ride each transmission only, never the frozen record.
	submission := proto.Clone(frozen).(*pb.MachineExecutionSubmit)
	submission.SourceCredentials = m.resolver.SourceCredentials()
	submission.Claim = connection.Claim
	submission.Offer.WorkerBootId = connection.Claim.WorkerBootId
	submission.Offer.RecordOwnerEpoch = connection.Claim.RecordOwnerEpoch
	var trailer metadata.MD
	receipt, err := connection.Host.SubmitMachineExecution(ctx, submission, grpc.Trailer(&trailer))
	if err != nil {
		for _, code := range trailer.Get("cozy-error-code") {
			if code == "execution_workspace_changed" || code == "execution_workspace_required" {
				return exit.Named(exit.Conflict, "machine_execution.workspace_changed", "execution workspace no longer matches the frozen submission; prior acceptance remains unresolved")
			}
		}
		for _, code := range trailer.Get("cozy-error-code") {
			if code == "execution_submission_refused" {
				problem := machineTransport(err)
				if recordProblem := m.store.RefuseMachineSubmission(requestID, problem.ErrName(), problem.Message); recordProblem != nil {
					return recordProblem
				}
				break
			}
		}
		return machineTransport(err)
	}
	if receipt == nil || receipt.WorkerId != connection.Claim.WorkerId || receipt.WorkerBootId == "" {
		return exit.New(exit.Conflict, "execution was accepted by an unexpected worker")
	}
	if problem := m.store.AcceptMachineExecution(requestID, receipt); problem != nil {
		return problem
	}
	return nil
}

// supplySourceCredentials hands Runtime the owner's provider credentials again, by the identical
// resubmission it answers with the same receipt: after a restart lost them, or before a resume.
func (m *machineRuns) supplySourceCredentials(ctx context.Context, connection *machineConnection, link *records.MachineExecution, workspace string) *exit.Error {
	var submission pb.MachineExecutionSubmit
	if err := proto.Unmarshal(link.Submission, &submission); err != nil {
		return exit.Internalf("recorded machine submission is unreadable: %s", err)
	}
	if submission.ExpectedExecutionWorkspaceId == "" {
		submission.ExpectedExecutionWorkspaceId = workspace
	}
	return m.sendMachineSubmission(ctx, connection, link.RequestID, &submission)
}

func machineTransport(err error) *exit.Error { return machines.Transport(err) }

func (m *machineRuns) executionConnection(ctx context.Context, request records.Request) (*machineConnection, *records.MachineExecution, *pb.MachineExecutionQuery, *exit.Error) {
	if lost, problem := m.store.MachineExecutionLost(request.ID); problem != nil || lost {
		if problem != nil {
			return nil, nil, nil, problem
		}
		return nil, nil, nil, exit.Named(exit.Conflict, "machine_execution.state_lost", "execution's rented machine was confirmed destroyed")
	}
	link, problem := m.store.MachineExecution(request.ID)
	if problem != nil {
		return nil, nil, nil, problem
	}
	var receipt pb.MachineExecutionReceipt
	if link == nil || len(link.Receipt) == 0 || proto.Unmarshal(link.Receipt, &receipt) != nil {
		return nil, nil, nil, exit.Unavailablef("waiting for durable machine acceptance")
	}
	connection, problem := m.connect(ctx, link.MachineID, m.runHolder(request, "reading or collecting its execution"))
	if problem != nil {
		return nil, nil, nil, problem
	}
	query := &pb.MachineExecutionQuery{Claim: connection.Claim, RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}
	return connection, link, query, nil
}

// runHolder names a run and what it is doing on its machine.
func (m *machineRuns) runHolder(request records.Request, doing string) string {
	if numbered, problem := m.store.RequestByReference(request.ID); problem == nil && numbered != nil && numbered.Number > 0 {
		return fmt.Sprintf("run %d %s", numbered.Number, doing)
	}
	return "run " + request.ID + " " + doing
}

func (m *machineRuns) Refresh(parent context.Context, request records.Request) *exit.Error {
	ctx, done := m.observation(request.ID).observe(parent)
	defer done()
	problem := m.refresh(ctx, request)
	if problem != nil && errors.Is(context.Cause(ctx), errObservationYielded) {
		return nil // the control that took the turn observes the execution itself
	}
	return problem
}

var errObservationYielded = errors.New("machine observation yielded to a control")

// observation orders one request's observation, collection and control. Observing only
// reads Runtime's journal and resumes wherever it stopped, as after a restart, so it yields
// to a control: a cancel never queues behind another reader's dial and event pages.
type observation struct {
	turn     sync.Mutex
	mu       sync.Mutex
	yield    context.CancelCauseFunc
	controls int
}

func (m *machineRuns) observation(request string) *observation {
	value, _ := m.observers.LoadOrStore(request, &observation{})
	return value.(*observation)
}

func (o *observation) observe(parent context.Context) (context.Context, func()) {
	o.turn.Lock()
	ctx, cancel := context.WithCancelCause(parent)
	o.mu.Lock()
	o.yield = cancel
	if o.controls > 0 {
		cancel(errObservationYielded)
	}
	o.mu.Unlock()
	return ctx, func() {
		o.mu.Lock()
		o.yield = nil
		o.mu.Unlock()
		cancel(nil)
		o.turn.Unlock()
	}
}

// control takes the turn at once: an observation in flight stops, and none starts while a
// control waits.
func (o *observation) control() func() {
	o.mu.Lock()
	o.controls++
	if o.yield != nil {
		o.yield(errObservationYielded)
	}
	o.mu.Unlock()
	o.turn.Lock()
	o.mu.Lock()
	o.controls--
	o.mu.Unlock()
	return o.turn.Unlock
}

// refresh observes and, once finished, collects one execution. Collection moves result
// bytes whose size no clock can predict, so the work is bounded by progress: it ends only
// when no step completes and no byte arrives for the stall budget.
func (m *machineRuns) refresh(parent context.Context, request records.Request) *exit.Error {
	progress := &transfer.Progress{}
	ctx, cancel := progress.Context(parent)
	defer cancel()
	connection, link, query, problem := m.executionConnection(ctx, request)
	if problem != nil {
		return problem
	}
	defer connection.Close()
	return m.observeOn(ctx, progress, request, connection, link, query)
}

func (m *machineRuns) observeOn(ctx context.Context, progress *transfer.Progress, request records.Request, connection *machineConnection, link *records.MachineExecution, query *pb.MachineExecutionQuery) *exit.Error {
	connection.progress = progress
	if len(link.PendingControl) > 0 {
		if _, problem := m.flushMachineControl(ctx, connection, link); problem != nil {
			return problem
		}
	}
	var trailer metadata.MD
	state, err := connection.Host.GetMachineExecution(ctx, query, grpc.Trailer(&trailer))
	if err != nil {
		if slices.Contains(trailer.Get("cozy-error-code"), "execution_workspace_changed") {
			return m.store.LoseMachineExecution(request.ID, link.MachineID,
				"the machine no longer holds this run's execution workspace (its worker restarted or the workspace was replaced); the run failed alone and the machine keeps its other work")
		}
		return machineTransport(err)
	}
	if len(state.AwaitingSourceCredentials) > 0 {
		if problem := m.supplySourceCredentials(ctx, connection, link, query.ExpectedExecutionWorkspaceId); problem != nil {
			return problem
		}
	}
	progress.Advance(1)
	cursor := uint64(link.RemoteCursor)
	for {
		page, err := connection.Host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: cursor, Limit: 128})
		if err != nil {
			return machineTransport(err)
		}
		progress.Advance(1)
		if problem := m.store.ObserveMachineExecution(request.ID, state, page); problem != nil {
			return problem
		}
		if page.NextAfter < page.HeadSequence && page.NextAfter <= cursor {
			return exit.New(exit.Conflict, "machine event cursor made no progress")
		}
		cursor = page.NextAfter
		if page.NextAfter >= page.HeadSequence {
			break
		}
	}
	if problem := m.reconcilePublications(ctx, request.ID, request.Hub, connection, query); problem != nil {
		return problem
	}
	if problem := m.releaseMachineInputs(ctx, request, connection); problem != nil {
		return problem
	}
	if state.Collected || state.State == "canceled" || state.State != "succeeded" && state.State != "failed" {
		return nil
	}
	outcome, err := connection.Host.CollectMachineExecution(ctx, &pb.MachineExecutionCollect{Execution: query, AttemptOrdinal: state.AttemptOrdinal})
	if err != nil {
		// Another observer may have completed the same collection after our
		// status read. Its ACK releases the original root hold, so re-read the
		// authority before treating that now-stale collect as a custody failure.
		latest, readError := connection.Host.GetMachineExecution(ctx, query)
		if readError == nil && latest.Collected && latest.AttemptOrdinal == state.AttemptOrdinal {
			return m.store.ObserveMachineExecution(request.ID, latest, &pb.MachineExecutionEventPage{NextAfter: cursor, HeadSequence: max(cursor, latest.Sequence)})
		}
		return machineTransport(err)
	}
	progress.Advance(1)
	var body pb.AttemptOutcomeBody
	if err := canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body); err != nil {
		return exit.New(exit.Conflict, "machine outcome is not its canonical document")
	}
	plan, planProblem := m.planMachineFiles(request, outcome, &body)
	modelPlan, modelProblem := m.planMachineModels(request, connection, &body)
	if len(link.Outcome) == 0 {
		var warnings []string
		if plan != nil {
			warnings = append(warnings, plan.warnings...)
		}
		if modelPlan != nil {
			warnings = append(warnings, modelPlan.warnings...)
		}
		seen := map[string]bool{}
		for _, warning := range warnings {
			if seen[warning] {
				continue // a result position both collectors read warns once
			}
			seen[warning] = true
			if problem := m.store.AppendEvent(request.ID, "request.warning", int64(outcome.AttemptOrdinal), map[string]any{"message": warning}); problem != nil {
				return problem
			}
		}
	}
	if problem := m.store.RecordMachineOutcome(request.ID, outcome); problem != nil {
		return problem
	}
	if ref := body.TriageBundle; ref != nil {
		// Evidence, not custody: a bundle the machine cannot hand over leaves the run as it is.
		if bundle, problem := readMachineTriage(ctx, connection, query, outcome.AttemptOrdinal, ref); problem != nil {
			fmt.Fprintf(m.context.Out, "machine execution %s: triage bundle not kept: %s\n", request.ID, problem.Message)
		} else if problem := m.store.RecordMachineTriage(request.ID, int64(outcome.AttemptOrdinal), bundle); problem != nil {
			return problem
		}
	}
	if modelProblem != nil {
		return modelProblem
	}
	written := writtenBytes(&body)
	models, problem := m.collectMachineModels(ctx, request, connection, outcome, modelPlan, written)
	if problem != nil {
		return problem
	}
	if planProblem != nil {
		return planProblem
	}
	files, problem := m.collectMachineFiles(ctx, request, connection, plan)
	if problem != nil {
		return problem
	}
	if problem := m.retainMachineWeights(ctx, request, connection, outcome, &body, modelPlan, written); problem != nil {
		return m.retainedResult(request, outcome, problem)
	}
	if len(body.GetOutputManifest().GetOutputs()) > 0 && !files || body.GetResult().GetResultBlob() != nil || body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED && request.ChildArtifacts && !models && !files {
		return m.retainedResult(request, outcome, exit.Named(exit.Unavailable, "machine_execution.result_custody_required",
			"execution finished; referenced output bytes remain retained on the machine: this Creator has no recipient for them"))
	}
	if body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED && request.ModelTransfer != nil {
		if problem := m.store.SettleMachineDestination(request.ID, fmt.Sprintf(
			"the machine uploaded no checkpoint to %s; its Runtime predates machine destinations. Update the rental's Runtime to %s or newer and run again",
			request.ModelTransfer.Destination, machineDestinationRuntimeFloor)); problem != nil {
			return problem
		}
	}
	ack := &pb.AttemptOutcomeAck{RequestId: outcome.RequestId, AttemptOrdinal: outcome.AttemptOrdinal, InvocationSpecDigest: outcome.InvocationSpecDigest, OutcomeId: outcome.OutcomeId, OutcomeDigest: outcome.OutcomeDigest}
	collected, err := connection.Host.AcknowledgeMachineExecutionCollection(ctx, &pb.MachineExecutionCollectionAck{Execution: query, Outcome: ack})
	if err != nil {
		return machineTransport(err)
	}
	return m.store.ObserveMachineExecution(request.ID, collected, &pb.MachineExecutionEventPage{NextAfter: cursor, HeadSequence: max(cursor, collected.Sequence)})
}

// retainedResult records, once, that an outcome's result stays with the machine because
// nothing on this host can receive it. The run is settled for its clients and no
// observation retries a transfer that nobody drives; a newer Creator retries on restart.
func (m *machineRuns) retainedResult(request records.Request, outcome *pb.AttemptOutcome, problem *exit.Error) *exit.Error {
	if problem.ErrName() != "machine_execution.result_custody_required" {
		return problem
	}
	reason, readProblem := m.store.MachineResultRetained(request.ID)
	if readProblem != nil {
		return readProblem
	}
	if reason != problem.Message {
		if recordProblem := m.store.AppendEvent(request.ID, "machine.result_retained", int64(outcome.AttemptOrdinal),
			map[string]any{"reason": problem.Message}); recordProblem != nil {
			return recordProblem
		}
	}
	return problem
}

// Control sends the command on its own connection the moment it is durable; observing the
// execution afterwards reuses that connection.
func (m *machineRuns) Control(parent context.Context, request records.Request, action string) *exit.Error {
	return m.control(parent, request, action, false)
}

// control with requested is the observer delivering a recorded cancel request: one the
// API's own control finished meanwhile needs only observing.
func (m *machineRuns) control(parent context.Context, request records.Request, action string, requested bool) *exit.Error {
	defer m.observation(request.ID).control()()
	progress := &transfer.Progress{}
	ctx, cancel := progress.Context(parent)
	defer cancel()
	value := map[string]pb.MachineExecutionAction{"pause": pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_PAUSE, "resume": pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RESUME, "cancel": pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}[action]
	if value == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_UNSPECIFIED {
		return exit.New(exit.Validation, "unknown machine execution control")
	}
	connection, link, query, problem := m.executionConnection(ctx, request)
	if problem != nil {
		return problem
	}
	defer connection.Close()
	if requested && !link.CancelRequested {
		return m.observeOn(ctx, progress, request, connection, link, query)
	}
	if len(link.PendingControl) > 0 {
		previous, problem := m.flushMachineControl(ctx, connection, link)
		if problem != nil {
			return problem
		}
		if previous == value {
			return m.observeAfterControl(ctx, progress, request, connection, query)
		}
	}
	state, err := connection.Host.GetMachineExecution(ctx, query)
	if err != nil {
		return machineTransport(err)
	}
	command := &pb.MachineExecutionControl{Execution: query, CommandId: records.NewID("control"), ExpectedGeneration: state.Generation, Action: value}
	if problem := m.store.RecordMachineControl(request.ID, command); problem != nil {
		return problem
	}
	if problem := m.observeAfterControl(ctx, progress, request, connection, query); problem != nil {
		return problem
	}
	return m.Start(request)
}

// observeAfterControl flushes the recorded control and imports what it changed.
func (m *machineRuns) observeAfterControl(ctx context.Context, progress *transfer.Progress, request records.Request, connection *machineConnection, query *pb.MachineExecutionQuery) *exit.Error {
	link, problem := m.store.MachineExecution(request.ID)
	if problem != nil {
		return problem
	}
	if link == nil {
		return exit.New(exit.Conflict, "machine execution link disappeared during control")
	}
	return m.observeOn(ctx, progress, request, connection, link, query)
}

func (m *machineRuns) flushMachineControl(ctx context.Context, connection *machineConnection, link *records.MachineExecution) (pb.MachineExecutionAction, *exit.Error) {
	var command pb.MachineExecutionControl
	if proto.Unmarshal(link.PendingControl, &command) != nil || command.Execution == nil {
		return 0, exit.Internalf("recorded machine control is unreadable")
	}
	command.Execution.Claim = connection.Claim
	if command.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RESUME && len(m.resolver.SourceCredentials()) > 0 {
		if problem := m.supplySourceCredentials(ctx, connection, link, command.Execution.ExpectedExecutionWorkspaceId); problem != nil {
			return command.Action, problem
		}
	}
	if command.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL {
		if problem := m.releaseMachineFiles(ctx, link.RequestID, connection); problem != nil {
			return command.Action, problem
		}
		if problem := m.releaseMachineModels(ctx, link.RequestID, connection); problem != nil {
			return command.Action, problem
		}
	}
	var trailer metadata.MD
	if _, err := connection.Host.ControlMachineExecution(ctx, &command, grpc.Trailer(&trailer)); err != nil {
		for _, code := range trailer.Get("cozy-error-code") {
			if code == "execution_generation_stale" {
				if problem := m.store.RejectMachineControl(link.RequestID, link.PendingControl); problem != nil {
					return command.Action, problem
				}
				return command.Action, exit.Named(exit.Conflict, "machine_execution.generation_changed", "execution changed before the control was accepted; retry the control against its current state")
			}
		}
		return command.Action, machineTransport(err)
	}
	return command.Action, m.store.CompleteMachineControl(link.RequestID, link.PendingControl)
}

func validateMachinePrepared(result *pb.DesiredPlacementSet) *exit.Error {
	if result == nil || !bytes.Equal(canonical.Digest(result.PlacementSetCanonicalBytes), result.PlacementSetDigest) {
		return exit.New(exit.Conflict, "machine preparation returned a different document identity")
	}
	if _, err := canonical.Read(result.PlacementSetCanonicalBytes, &pb.PlacementSet{}); err != nil {
		return exit.New(exit.Conflict, "machine preparation returned an invalid placement document")
	}
	return nil
}

func readMachinePreparedSet(stream grpc.ServerStreamingClient[pb.PrepareEvent], observe func(*pb.PrepareEvent)) (*pb.DesiredPlacementSet, *exit.Error) {
	event, problem := readMachinePreparationEvent(stream, observe)
	if problem != nil {
		return nil, problem
	}
	return event.PlacementSet, nil
}

// readMachinePreparationEvent reads a preparation to its verified end, handing each event
// to `observe` (nil for none) so a waiting run can show what its machine is doing.
func readMachinePreparationEvent(stream grpc.ServerStreamingClient[pb.PrepareEvent], observe func(*pb.PrepareEvent)) (*pb.PrepareEvent, *exit.Error) {
	for {
		event, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				return nil, exit.Unavailablef("machine preparation ended before verified completion")
			}
			if problem := orchestrator.RuntimeRequirementTrailer(stream.Trailer()); problem != nil {
				return nil, problem
			}
			return nil, machineTransport(err)
		}
		if observe != nil {
			observe(event)
		}
		switch event.Stage {
		case pb.PrepareStage_PREPARE_STAGE_REFUSED:
			problem := orchestrator.RuntimeRequirementEvent(event)
			if problem == nil && event.SafeCode == "package_environment_dependency_base_conflict" {
				problem = exit.Named(exit.Structural, "machine_execution.runtime_requirement", "%s", event.SafeDetail)
			} else if problem == nil {
				problem = exit.Named(exit.Conflict, "machine_execution.prepare_refused", "%s: %s", event.SafeCode, event.SafeDetail)
			}
			return event, problem.WithCause(event.SafeCode)
		case pb.PrepareStage_PREPARE_STAGE_PREPARED:
			if event.InstalledPackage != nil {
				return event, nil
			}
			return event, validateMachinePrepared(event.PlacementSet)
		}
	}
}

// Retain an ordinary JSON error summary in the observation API, not raw RPC
// envelopes or sensitive capture contents.
func machineObservationError(problem *exit.Error) json.RawMessage {
	if problem == nil {
		return nil
	}
	raw, _ := json.Marshal(map[string]string{"code": problem.ErrName(), "message": problem.Message})
	return raw
}

// readMachineTriage reads one attempt's retained triage bundle through the Host, the same
// guarded connection its outcome came over.
func readMachineTriage(ctx context.Context, connection *machineConnection, query *pb.MachineExecutionQuery, attempt uint64, ref *pb.TriageBundleRef) ([]byte, *exit.Error) {
	if !connection.SupportsTriage {
		return nil, exit.Named(exit.Structural, "machine_execution.triage_unsupported", "this machine's Host reads no triage bundle")
	}
	triage, err := connection.Host.ReadMachineExecutionTriage(ctx, &pb.MachineExecutionTriageQuery{Execution: query, AttemptOrdinal: attempt})
	if err != nil {
		return nil, machineTransport(err)
	}
	data := triage.BundleCanonicalBytes
	if uint64(len(data)) != ref.Length || !bytes.Equal(canonical.Digest(data), ref.WriteReceiptDigest) {
		return nil, exit.New(exit.Conflict, "triage bundle does not match its terminal")
	}
	return data, nil
}
