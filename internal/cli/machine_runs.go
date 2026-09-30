package cli

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

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

// machineOutputLogRuntimeFloor is the first Runtime that reports a run's outputs as they are
// made (worker wire 65); an older machine's runs are refused with this floor.
const machineOutputLogRuntimeFloor = "0.18.73"

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
	uploading sync.Map // retained-output uploads in progress, by operation
	// awaited names the runs whose last observation ended on the machine's next event.
	awaited map[string]bool
	updates *rentalRuntimeUpdates
	// ownerReads paces owner finalization reads of publications a machine cannot settle.
	ownerReads ownerReads
	// submitting stops each request's submission work in flight (upload, preparation,
	// staging) once its cancel is durable; guarded by mu.
	submitting map[string]context.CancelFunc
}

func newMachineRuns(ctx *Context, layout home.Layout, store *records.Store, resolver *Resolver, fleet *managedRentals, found *machines.Resolver) *machineRuns {
	background, cancel := context.WithCancel(context.Background())
	return &machineRuns{ctx: background, cancel: cancel, context: ctx, layout: layout, store: store, resolver: resolver, fleet: fleet, machines: found, running: map[string]bool{}, awaited: map[string]bool{}, placed: map[string]string{}, submitting: map[string]context.CancelFunc{}}
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
		defer func() {
			m.mu.Lock()
			delete(m.running, request.ID)
			delete(m.placed, request.ID)
			m.mu.Unlock()
			// A control can be persisted after the final observation, while its
			// Start still sees this goroutine running. Read after withdrawing our
			// running marker so either that Start or this handoff owns the wakeup.
			if m.ctx.Err() != nil {
				return
			}
			link, problem := m.store.MachineExecution(request.ID)
			if problem != nil || link == nil || link.Abandoned || !link.Collected || len(link.Receipt) == 0 || !link.CancelRequested && len(link.PendingControl) == 0 {
				return
			}
			if owed, problem := m.machineWorkOwed(request.ID); problem != nil || !owed {
				return
			}
			current, problem := m.store.RequestRow(request.ID)
			if problem == nil && current != nil {
				_ = m.Start(*current)
			}
		}()
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
			if problem != nil || link == nil || link.Abandoned {
				return
			}
			if len(link.Receipt) == 0 && len(link.Submission) == 0 && link.CancelRequested && current.State == "canceling" {
				// An older cozy left this cancel waiting on a machine that never accepted the run.
				if _, problem := m.store.RequestMachineCancellation(request.ID, ""); problem != nil {
					return
				}
				continue
			}
			if len(link.Receipt) == 0 && (records.Settled(current.State) || records.RetainedState(current.State)) &&
				current.State != "canceled" {
				// Restart resumes observation, not withdrawn or failed execution
				// intent. A receipt can reconcile a real outcome; absent one, do
				// not turn a stopped request into a new Submit RPC.
				return
			}
			if len(link.Receipt) == 0 {
				if link.CancelRequested && len(link.Submission) > 0 && !link.SubmissionClosed {
					problem = m.closePendingSubmission(m.ctx, *current, link)
				} else if current.State == "canceled" {
					// Canceled before any machine accepted it: nothing is submitted again. Only
					// input bytes staged on its machine are owed, and only while there are any.
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
					// An accepted submission is observed at once, not on the next clock.
					problem = m.submit(*current, link)
					m.mu.Lock()
					m.awaited[request.ID] = problem == nil
					m.mu.Unlock()
				}
			} else {
				if link.CancelRequested {
					problem = m.control(m.ctx, *current, "cancel", true)
				} else {
					problem = m.follow(m.ctx, *current)
					if problem != nil && problem.ErrName() == "machine_execution.result_custody_required" {
						return // the result stays with the machine; nothing here can collect it
					}
				}
				if problem == nil {
					observed, e := m.store.MachineExecution(request.ID)
					// A durable cancellation stays observed through its native release
					// acknowledgement, even when the original failed result was collected.
					awaiting, awaitingProblem := m.store.MachinePublicationsAwaitingOwner(request.ID)
					if e == nil && observed != nil && observed.Collected && !observed.CancelRequested && len(observed.PendingControl) == 0 && awaitingProblem == nil && len(awaiting) == 0 {
						return
					}
					if e == nil && observed != nil && !observed.CancelRequested && len(observed.PendingControl) == 0 {
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
			m.mu.Lock()
			awaited := m.awaited[request.ID]
			delete(m.awaited, request.ID)
			m.mu.Unlock()
			if problem != nil && problem.Message == lastError {
				delay = min(2*delay, 5*time.Second)
			} else if problem == nil && awaited {
				delay = 0 // the machine answered with its next event
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
				if row != nil && !records.Settled(row.State) && (unsent || latest != nil && len(latest.Receipt) == 0) {
					// Unsent and acceptance-unknown work both expose what reconciliation awaits.
					parked := map[string]any{"reason": problem.Message, "wait": orchestrator.WaitRental}
					if row.Machine != "" {
						parked["waiting_on"] = row.Machine
					}
					_ = m.store.AppendEvent(request.ID, "request.parked", 0, parked)
				}
			}
			if problem != nil && (problem.ErrName() == "machine_execution.closure_unavailable" || problem.ErrName() == "machine_execution.workspace_required") {
				// A retired peer cannot acquire this capability by retrying a work/control
				// call. Preserve uncertainty; an explicit reconnect after upgrade resumes it.
				return
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

// closePendingSubmission resolves cancellation without retransmitting work.
// A missing receipt is uncertainty; a durable closed key is a nonacceptance proof.
func (m *machineRuns) closePendingSubmission(ctx context.Context, request records.Request, link *records.MachineExecution) *exit.Error {
	if latest, problem := m.store.MachineExecution(request.ID); problem != nil {
		return problem
	} else if latest != nil && latest.Abandoned {
		return exit.Named(exit.Conflict, "request.abandoned", "local submission intent was abandoned")
	}
	var frozen pb.MachineExecutionSubmit
	if proto.Unmarshal(link.Submission, &frozen) != nil || frozen.Offer == nil || frozen.ExpectedExecutionWorkspaceId == "" {
		return exit.Named(exit.Conflict, "machine_execution.workspace_required", "pending cancellation has no intact submission identity")
	}
	connection, problem := m.connect(ctx, link.MachineID, m.runHolder(request, "closing its pending submission"))
	if problem != nil {
		return problem
	}
	defer connection.Close()
	closed, problem := closeSubmission(ctx, connection, &frozen)
	if problem != nil {
		if problem.ErrName() == "machine_execution.workspace_changed" {
			return m.loseSubmissionWorkspace(request.ID, link.MachineID)
		}
		return problem
	}
	if closed.Receipt != nil {
		return m.store.AcceptMachineExecution(request.ID, closed.Receipt)
	}
	return m.store.CompleteMachineSubmissionClosure(request.ID, closed)
}

func closeSubmission(ctx context.Context, connection *machineConnection, frozen *pb.MachineExecutionSubmit) (*pb.MachineSubmissionClosure, *exit.Error) {
	var header, trailer metadata.MD
	closed, err := connection.Host.CloseMachineSubmission(ctx, &pb.MachineSubmissionClose{Claim: connection.Claim,
		SubmissionId: frozen.SubmissionId, RequestId: frozen.Offer.RequestId, ExpectedExecutionWorkspaceId: frozen.ExpectedExecutionWorkspaceId}, grpc.Header(&header), grpc.Trailer(&trailer))
	// A journal snapshot can carry an old workspace while the Runtime is booting or
	// being replaced. Its metadata is the authority for this phase; do not turn the
	// snapshot's closure error into a permanent workspace/capability verdict.
	if runtimeUnavailable(header, trailer) {
		return nil, waitForRuntime()
	}
	if executionWorkspaceChanged(err, trailer) {
		return nil, exit.Named(exit.Conflict, "machine_execution.workspace_changed", "the machine no longer holds the frozen submission's execution workspace")
	}
	if submissionClosureUnavailable(err) {
		return nil, pendingClosureUnavailable(status.Convert(err).Message())
	}
	if err != nil {
		return nil, machineTransport(err)
	}
	if closed == nil || closed.SubmissionId != frozen.SubmissionId || closed.RequestId != frozen.Offer.RequestId || closed.ExecutionWorkspaceId != frozen.ExpectedExecutionWorkspaceId {
		return nil, exit.New(exit.Conflict, "machine returned another submission's closure")
	}
	if closed.Receipt != nil {
		if closed.Receipt.WorkerId != connection.Claim.WorkerId {
			return nil, exit.New(exit.Conflict, "submission receipt names another machine")
		}
	}
	return closed, nil
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
		if awaiting, problem := m.store.MachinePublicationsAwaitingOwner(link.RequestID); link.Collected && !link.CancelRequested && len(link.PendingControl) == 0 && (problem != nil || len(awaiting) == 0) {
			continue
		}
		request, problem := m.store.RequestRow(link.RequestID)
		if problem == nil && request != nil {
			_ = m.Start(*request)
		}
	}
	uploads, problem := m.store.UnfinishedOutputUploads()
	if problem != nil {
		fmt.Fprintf(m.context.Out, "output upload recovery: %s\n", problem.Message)
		return
	}
	for request, pending := range uploads {
		for _, upload := range pending {
			m.startUpload(request, upload)
		}
	}
}

// Withdraw stops local submission/observation for a durable cancel or abandonment.
// The observer's next pass delivers cancellation for any accepted execution.
func (m *machineRuns) Withdraw(request string) {
	m.mu.Lock()
	stop := m.submitting[request]
	m.mu.Unlock()
	if stop != nil {
		stop()
	}
	if link, problem := m.store.MachineExecution(request); problem == nil && link != nil && (link.Abandoned || link.CancelRequested) {
		observation := m.observation(request)
		observation.mu.Lock()
		if observation.yield != nil {
			observation.yield(errObservationYielded)
		}
		observation.mu.Unlock()
	}
}

// machineWorkOwed is whether the execution still owes work or custody, or holds a sent
// publication only its owner can settle.
func (m *machineRuns) machineWorkOwed(request string) (bool, *exit.Error) {
	if link, problem := m.store.MachineExecution(request); problem != nil || link != nil && link.Abandoned {
		return false, problem
	}
	if owed, problem := m.store.MachineExecutionOwesWork(request); problem != nil || owed {
		return owed, problem
	}
	awaiting, problem := m.store.MachinePublicationsAwaitingOwner(request)
	return len(awaiting) > 0, problem
}

func (m *machineRuns) submit(request records.Request, link *records.MachineExecution) (out *exit.Error) {
	var frozen pb.MachineExecutionSubmit
	replaying := len(link.Submission) > 0
	if replaying {
		if err := proto.Unmarshal(link.Submission, &frozen); err != nil {
			return exit.Named(exit.Validation, "machine_execution.submission_invalid", "recorded machine submission is unreadable: %s", err)
		}
		if frozen.Offer == nil || frozen.Offer.RequestId != request.ID || frozen.SubmissionId == "" || link.MachineID == "" {
			return exit.Named(exit.Validation, "machine_execution.submission_invalid", "recorded machine submission has incomplete identity; its bytes and acceptance remain unchanged")
		}
		if frozen.ExpectedExecutionWorkspaceId == "" {
			return exit.Named(exit.Conflict, "machine_execution.workspace_required", "recorded submission has no workspace identity; preserve it until its machine is reconciled or retired")
		}
	}
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
	if latest, problem := m.store.MachineExecution(request.ID); problem != nil {
		return problem
	} else if latest != nil && latest.Abandoned {
		return exit.Named(exit.Conflict, "request.abandoned", "local submission intent was abandoned")
	}
	_, deadline, problem := m.store.RequestExecutionTiming(request.ID)
	if problem != nil {
		return problem
	}
	request.DeadlineUnixMS = deadline
	if link.MachineID == "" {
		machine := machines.Placement(request)
		if request.Rental {
			if machine == "" {
				// A named rental is where the run goes: no fit is computed for it.
				machine = request.RequestedRental
			}
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
	var connection *machineConnection
	if len(link.Submission) > 0 {
		// Ownership authenticates replay even after the current Hub login expires.
		// The frozen Hub remains operation scope; Runtime checks its durable
		// acceptance before deciding whether a fresh grant is relevant.
		origin := frozen.Hub
		if frozen.ReleaseRoot != nil {
			origin = frozen.ReleaseRoot.Hub
		}
		connection, problem = m.connect(ctx, link.MachineID, m.runHolder(request, "replaying its submission"))
		if connection != nil {
			connection.Hub = origin
		}
	} else {
		connection, problem = m.connectFor(ctx, request, link.MachineID, "preparing its submission")
	}
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
	workspace, problem := newExecutionWorkspace(ctx, connection)
	if problem != nil {
		if len(link.Submission) > 0 && problem.ErrName() == "machine.submission_closure_required" {
			return pendingClosureUnavailable("the connected machine does not support the required submission-closure contract")
		}
		return problem
	}
	connection.KeepWorkspace(workspace)
	if request.RequiresModelOverrides() && (connection.WireMinor < 70 || !workspace.ModelOverrides) {
		code := exit.Structural
		if len(link.Submission) > 0 {
			code = exit.Unavailable // an already transmitted offer retains its uncertainty
		}
		return exit.Named(code, "model_overrides.unavailable",
			"this machine's agent or Runtime cannot apply the requested model overrides; no submission was sent").
			WithRemedy("update the machine agent and Runtime before retrying this request")
	}
	rooted := len(link.Submission) == 0 && m.releaseRoot(request)
	prepared := false // the root's unpublished installation is on the machine this pass
	if rooted {
		workspace, problem := m.workspace(ctx, connection)
		if problem != nil {
			return problem
		}
		// Unpublished code reaches the machine before its submission is recorded, as a
		// capture's does: an upload refused or canceled leaves the run unsent.
		if request.LocalInstallationID != "" {
			revision, problem := m.capturedRevision(request)
			if problem == nil {
				problem = connection.prepare(ctx, request.ID, revision)
			}
			if problem != nil {
				return problem
			}
			prepared = true
		}
		built, problem := m.releaseRootSubmission(ctx, request, connection)
		if problem != nil {
			return problem
		}
		built.ExpectedExecutionWorkspaceId, built.OwnerMemo = workspace.ExecutionWorkspaceId, true
		if built.PublicationAuthorizationId, problem = m.publicationAuthorization(ctx, request.ID, link.MachineID, connection); problem != nil {
			return problem
		}
		if problem := m.store.RecordMachineSubmission(request.ID, built); problem != nil {
			return problem
		}
		link.Submission, _ = proto.Marshal(built)
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
	} else {
		authorization, problem := m.publicationAuthorization(ctx, request.ID, link.MachineID, connection)
		if problem != nil {
			return problem
		}
		var built *pb.MachineExecutionSubmit
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
		built.PublicationAuthorizationId, built.OwnerMemo, built.Account, built.Hub = authorization, true, m.runAccount(request), connection.Hub
		if problem := m.freezeMachineSubmission(ctx, connection, request.ID, link.MachineID, built); problem != nil {
			return problem
		}
		submission = built
	}
	if !replaying {
		if _, problem := m.resolver.capturedResultInterface(request); problem != nil {
			return problem
		}
	}
	began = time.Now()
	if submission.ReleaseRoot != nil {
		if !replaying && submission.ReleaseRoot.InstallationId != "" && !prepared {
			revision, problem := m.capturedRevision(request)
			if problem == nil {
				problem = connection.prepare(ctx, request.ID, revision)
			}
			if problem != nil {
				// The frozen submission may have reached the machine on an earlier pass.
				// Resolve its acceptance before treating a new preparation refusal as final.
				if refusedPreparation(problem) {
					return m.settleSubmissionRefusal(ctx, connection, request.ID, problem)
				}
				return problem
			}
		}
		problem := m.sendReleaseRoot(ctx, request, connection, submission)
		if replaying && problem != nil && problem.ErrName() == "machine_execution.installation_absent" && submission.ReleaseRoot.InstallationId != "" {
			// Only the machine's specific missing-installation reply permits source
			// recovery. An already accepted root replays without local preparation.
			revision, prepareProblem := m.capturedRevision(request)
			if prepareProblem == nil {
				prepareProblem = connection.prepare(ctx, request.ID, revision)
			}
			if prepareProblem != nil {
				return prepareProblem
			}
			problem = m.sendReleaseRoot(ctx, request, connection, submission)
		}
		if problem != nil {
			return problem
		}
	} else if problem := m.sendMachineSubmission(ctx, connection, request.ID, submission); problem != nil {
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
	deferred := capturedModelChoices(request.Models)
	materialized := slices.DeleteFunc(append([]records.ModelRef(nil), request.Models...), func(model records.ModelRef) bool {
		return model.Choice && model.Callable != ""
	})
	pinned := true
	for _, model := range materialized {
		pinned = pinned && model.Pinned()
	}
	if pinned {
		return request, nil
	}
	workspace, problem := newExecutionWorkspace(ctx, connection)
	if problem != nil {
		return request, problem
	}
	accelerator := ""
	if len(workspace.Devices) > 0 {
		accelerator = workspace.Devices[0].Name
	}
	models, _, ok := rental.Pin(materialized, accelerator, len(workspace.Devices))
	if !ok {
		return request, exit.Named(exit.Structural, "machine_execution.model_rung_unavailable",
			"no rung of this call's model ladder fits %d× %q on %s", len(workspace.Devices), accelerator, connection.Name)
	}
	models = append(models, deferred...)
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
	workspace, problem := newExecutionWorkspace(ctx, connection)
	if problem != nil {
		return problem
	}
	submission.ExpectedExecutionWorkspaceId = workspace.ExecutionWorkspaceId
	return m.store.RecordMachineSubmission(requestID, submission)
}

// New submissions require durable reconciliation through the complete machine route.
// Retained work keeps its frozen identity until authoritative reconciliation.
func newExecutionWorkspace(ctx context.Context, connection *machineConnection) (*pb.MachineExecutionWorkspace, *exit.Error) {
	workspace, available, problem := executionWorkspaceReply(ctx, connection)
	if problem != nil {
		return nil, problem
	}
	if !available {
		return nil, waitForRuntime()
	}
	if !workspace.SubmissionClose {
		return nil, exit.Named(exit.Structural, "machine.submission_closure_required",
			"this machine does not support durable submission closure; no submission was sent").
			WithRemedy("use a machine image with a closure-capable machine agent and Runtime, or update this machine before submitting again")
	}
	if connection.closureWorkspace == workspace.ExecutionWorkspaceId {
		return workspace, nil
	}
	// Older machine agents can forward a newer Runtime's capability bit while omitting the
	// closure route. Exercise that route with an unused identity before freezing
	// any real offer. Closing this key cannot create or cancel an execution.
	probe := records.NewID("closure-probe")
	var header, trailer metadata.MD
	closed, err := connection.Host.CloseMachineSubmission(ctx, &pb.MachineSubmissionClose{
		Claim: connection.Claim, SubmissionId: probe, RequestId: probe,
		ExpectedExecutionWorkspaceId: workspace.ExecutionWorkspaceId,
	}, grpc.Header(&header), grpc.Trailer(&trailer))
	if runtimeUnavailable(header, trailer) {
		return nil, waitForRuntime()
	}
	if executionWorkspaceChanged(err, trailer) {
		connection.KeepWorkspace(nil)
		return nil, exit.Named(exit.Unavailable, "machine_execution.workspace_changed", "the machine replaced its workspace before submission; checking its new workspace")
	}
	if err != nil {
		problem := machineTransport(err)
		if submissionClosureUnavailable(err) {
			problem = exit.Named(exit.Structural, "machine.submission_closure_required",
				"this machine cannot reconcile submissions through its machine agent and Runtime; no submission was sent: %s", status.Convert(err).Message()).
				WithRemedy("use a machine image with a closure-capable machine agent and Runtime")
		}
		return nil, problem
	}
	if closed == nil || closed.RequestId != probe || closed.SubmissionId != probe ||
		closed.ExecutionWorkspaceId != workspace.ExecutionWorkspaceId || closed.Receipt != nil {
		return nil, exit.New(exit.Conflict, "machine did not close the unused submission probe; no submission was sent")
	}
	connection.closureWorkspace = workspace.ExecutionWorkspaceId
	return workspace, nil
}

func pendingClosureUnavailable(detail string) *exit.Error {
	return exit.Named(exit.Unavailable, "machine_execution.closure_unavailable",
		"acceptance remains unresolved: this machine requires an agent and Runtime with durable submission closure (%s); upgrade it, then run `cozy down` and `cozy up` to reconnect", detail).
		WithRemedy("upgrade the machine, then run `cozy down` and `cozy up` to reconnect; the frozen submission and cancellation intent are retained")
}

func submissionClosureUnavailable(err error) bool {
	return status.Code(err) == codes.Unimplemented ||
		status.Code(err) == codes.FailedPrecondition && strings.HasPrefix(status.Convert(err).Message(), pb.CapabilityUnavailableCode+":")
}

func executionWorkspaceChanged(err error, trailer metadata.MD) bool {
	return err != nil && (slices.Contains(trailer.Get("cozy-error-code"), "execution_workspace_changed") ||
		strings.HasPrefix(status.Convert(err).Message(), "execution_workspace_changed:"))
}

func (m *machineRuns) loseSubmissionWorkspace(requestID, machine string) *exit.Error {
	return m.store.LoseMachineExecution(requestID, machine,
		"the machine no longer holds this run's frozen execution workspace; its prior acceptance cannot be recovered and the submission will not be sent to the replacement journal")
}

func runtimeUnavailable(values ...metadata.MD) bool {
	for _, value := range values {
		if slices.Contains(value.Get("cozy-runtime-available"), "false") {
			return true
		}
	}
	return false
}

func waitForRuntime() *exit.Error {
	return exit.Named(exit.Unavailable, "machine.runtime_unavailable",
		"the machine Runtime is temporarily unavailable; waiting for live submission capabilities")
}

func currentExecutionWorkspace(ctx context.Context, connection *machineConnection) (*pb.MachineExecutionWorkspace, *exit.Error) {
	workspace, _, problem := executionWorkspaceReply(ctx, connection)
	return workspace, problem
}

// Journal snapshots identify retained work, but do not describe a live Runtime's
// mutation capabilities. Observation can use them; new admission must wait.
func executionWorkspaceReply(ctx context.Context, connection *machineConnection) (*pb.MachineExecutionWorkspace, bool, *exit.Error) {
	var header, trailer metadata.MD
	workspace, err := connection.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: connection.Claim}, grpc.Header(&header), grpc.Trailer(&trailer))
	available := !runtimeUnavailable(header, trailer)
	if err != nil {
		if !available {
			return nil, false, waitForRuntime()
		}
		return nil, true, machineTransport(err)
	}
	if workspace == nil || workspace.WorkerId != connection.Claim.WorkerId || workspace.WorkerBootId == "" ||
		(connection.Claim.WorkerBootId != "" && workspace.WorkerBootId != connection.Claim.WorkerBootId) ||
		workspace.ExecutionWorkspaceId == "" || len(workspace.ExecutionWorkspaceId) > 256 {
		if !available {
			return nil, false, waitForRuntime()
		}
		return nil, true, exit.New(exit.Conflict, "machine returned an invalid execution workspace identity")
	}
	if !workspace.RunOutputLog {
		if !available {
			return nil, false, waitForRuntime()
		}
		return nil, true, exit.Named(exit.Structural, "machine.runtime_update_required",
			"this machine's Runtime predates the run output log; runs need Runtime %s or newer", machineOutputLogRuntimeFloor).
			WithRemedy("update the machine's Runtime: `cozy rental update <rental>`, or `cozy machine install` for this computer")
	}
	return workspace, available, nil
}

func (m *machineRuns) sendMachineSubmission(ctx context.Context, connection *machineConnection, requestID string, frozen *pb.MachineExecutionSubmit) *exit.Error {
	if frozen.Hub != "" && connection.WireMinor < 69 {
		return exit.Named(exit.Unavailable, "machine_execution.hub_scope_unavailable", "this machine cannot preserve the frozen submission's explicit Hub scope; update its agent and Runtime; acceptance remains unresolved")
	}
	if latest, problem := m.store.MachineExecution(requestID); problem != nil {
		return problem
	} else if latest != nil && latest.Abandoned {
		return exit.Named(exit.Conflict, "request.abandoned", "local submission intent was abandoned")
	}
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
		return m.submissionRefused(ctx, connection, requestID, trailer, err)
	}
	if receipt == nil || receipt.WorkerId != connection.Claim.WorkerId || receipt.WorkerBootId == "" {
		return exit.New(exit.Conflict, "execution was accepted by an unexpected worker")
	}
	if problem := m.store.AcceptMachineExecution(requestID, receipt); problem != nil {
		return problem
	}
	return nil
}

// A refusal alone does not prove nonacceptance: close the frozen key before ending
// the run, or recover the accepted receipt if another transmission already committed.
func (m *machineRuns) settleSubmissionRefusal(ctx context.Context, connection *machineConnection, requestID string, problem *exit.Error) *exit.Error {
	if problem.Code == exit.Unavailable || problem.Code == exit.Deadline {
		return problem
	}
	link, recordProblem := m.store.MachineExecution(requestID)
	if recordProblem != nil {
		return recordProblem
	}
	var frozen pb.MachineExecutionSubmit
	if link == nil || proto.Unmarshal(link.Submission, &frozen) != nil || frozen.Offer == nil || frozen.ExpectedExecutionWorkspaceId == "" {
		return exit.New(exit.Conflict, "refused submission has no retained identity")
	}
	closed, closeProblem := closeSubmission(ctx, connection, &frozen)
	if closeProblem != nil {
		if closeProblem.ErrName() == "machine_execution.workspace_changed" {
			return m.loseSubmissionWorkspace(requestID, link.MachineID)
		}
		return unresolvedSubmission(problem, closeProblem)
	}
	if closed.Receipt != nil {
		return m.store.AcceptMachineExecution(requestID, closed.Receipt)
	}
	if recordProblem := m.store.RefuseMachineSubmission(requestID, problem.ErrName(), problem.Message, closed); recordProblem != nil {
		return recordProblem
	}
	return problem
}

// Keep the actual refusal alongside reconciliation's failure. In particular, a missing
// closure RPC is not proof that this frozen submission was never accepted.
func unresolvedSubmission(refused, reconciliation *exit.Error) *exit.Error {
	problem := *reconciliation
	problem.Message = fmt.Sprintf("submission refused: %s; %s", refused.Message, reconciliation.Message)
	problem.Details = map[string]any{
		"submission_refusal_code": refused.ErrName(), "submission_refusal": refused.Message,
		"reconciliation_code": reconciliation.ErrName(), "reconciliation": reconciliation.Message,
	}
	return &problem
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

// refusedPreparation is a machine's refusal to prepare that retrying cannot change.
func refusedPreparation(problem *exit.Error) bool {
	switch problem.ErrName() {
	case "machine_execution.prepare_refused", "machine_execution.runtime_requirement", "machine_execution.package_requirement":
		return true
	}
	return false
}

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
	if link != nil && link.Abandoned {
		return nil, nil, nil, exit.Named(exit.Conflict, "request.abandoned", "this run was abandoned locally; no further remote control or observation is started")
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
	return m.runName(request) + " " + doing
}

func (m *machineRuns) runName(request records.Request) string {
	if numbered, problem := m.store.RequestByReference(request.ID); problem == nil && numbered != nil && numbered.Number > 0 {
		return fmt.Sprintf("run %d", numbered.Number)
	}
	return "run " + request.ID
}

// Refresh brings a run's record up to date for a reader. A run the observer is following
// already is: the observer holds its machine's next event, and the reader takes the record,
// unless its collection is refused.
func (m *machineRuns) Refresh(parent context.Context, request records.Request) *exit.Error {
	m.mu.Lock()
	following := m.running[request.ID]
	m.mu.Unlock()
	if following && !m.collectionRefused(request.ID) {
		return nil
	}
	return m.follow(parent, request)
}

// collectionRefused is whether a finished result waits on its owner. A reader asking about
// it tries the collection again at once: the cause may be fixed since the observer tried.
func (m *machineRuns) collectionRefused(id string) bool {
	link, problem := m.store.MachineExecution(id)
	if problem != nil || link == nil || link.Collected {
		return false
	}
	code, _, problem := m.store.MachineCollectionRefusal(id)
	return problem == nil && code != ""
}

func (m *machineRuns) follow(parent context.Context, request records.Request) *exit.Error {
	ctx, done := m.observation(request.ID).observe(parent)
	defer done()
	problem := m.refresh(ctx, request)
	if problem != nil && errors.Is(context.Cause(ctx), errObservationYielded) {
		return nil // the control that took the turn observes the execution itself
	}
	if problem != nil && problem.Code != exit.Unavailable && problem.Code != exit.Deadline && problem.Code != exit.Conflict {
		// A finished result this host refuses (a destination it cannot write, say) waits
		// on its owner; the record says why, and a caller waiting for the result wakes.
		if recordProblem := m.store.RefuseMachineCollection(request.ID, problem); recordProblem != nil {
			return recordProblem
		}
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
	return m.observeOn(ctx, progress, request, connection, link, query, true)
}

func (m *machineRuns) observeOn(ctx context.Context, progress *transfer.Progress, request records.Request, connection *machineConnection, link *records.MachineExecution, query *pb.MachineExecutionQuery, wait bool) *exit.Error {
	latest, problem := m.store.MachineExecution(request.ID)
	if problem != nil {
		return problem
	}
	if latest != nil && latest.Abandoned {
		return nil
	}
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
	reread := cursor // where the state was last read again
	var terminal *pb.AttemptOutcome
	for {
		page, err := connection.Host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: cursor, Limit: 128})
		if err != nil {
			return machineTransport(err)
		}
		progress.Advance(1)
		// A product becomes an event only once its bytes are here.
		held, problem := m.holdProducts(ctx, request, connection, page, cursor)
		if problem != nil {
			return problem
		}
		for _, event := range page.Events {
			if event.GetKind() == "outcome" && event.GetOutcome() != nil && event.Sequence > cursor {
				terminal = event.Outcome
			}
		}
		if problem := m.store.ObserveMachinePage(request.ID, state, page, held); problem != nil {
			return problem
		}
		if problem := m.answerMemoLookups(ctx, request, connection, query, page); problem != nil {
			return problem
		}
		if page.NextAfter < page.HeadSequence && page.NextAfter <= cursor {
			return exit.New(exit.Conflict, "machine event cursor made no progress")
		}
		cursor = page.NextAfter
		if page.NextAfter < page.HeadSequence {
			continue
		}
		// The pages ran past the state read: the machine journaled more meanwhile, as a run's
		// last products and its outcome land together. Read the state again and page on to
		// it: nothing is collected before every entry up to its outcome is recorded. Pages that
		// find nothing past the state read again end it.
		if state.Sequence >= cursor || cursor == reread {
			break
		}
		if state, err = connection.Host.GetMachineExecution(ctx, query); err != nil {
			return machineTransport(err)
		}
		reread = cursor
	}
	if problem := m.reconcilePublications(ctx, request.ID, request.Hub, connection, query); problem != nil {
		return problem
	}
	if state.Collected || state.State != "succeeded" && state.State != "failed" && state.State != "canceled" {
		if problem := m.releaseMachineInputs(ctx, request, connection); problem != nil {
			return problem
		}
		if wait && !state.Collected && !machineEnded(state.State) {
			m.awaitEvents(ctx, connection, request.ID, query, cursor)
		}
		return nil
	}
	// The log's terminal entry carries the outcome: the result is the fold of its products.
	outcome := terminal
	if outcome == nil {
		var problem *exit.Error
		if outcome, problem = m.terminalEntry(ctx, connection, query, cursor, state.AttemptOrdinal); problem != nil {
			return problem
		}
	}
	progress.Advance(1)
	var body pb.AttemptOutcomeBody
	if err := canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body); err != nil {
		return exit.New(exit.Conflict, "machine outcome is not its canonical document")
	}
	modelPlan, modelProblem := m.planMachineModels(request, connection, &body)
	if len(link.Outcome) == 0 {
		var warnings []string
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
	// Evidence, not custody: the triage bundle is read beside the result, and a bundle the
	// machine cannot hand over leaves the run as it is.
	var bundle []byte
	var triageProblem *exit.Error
	var triage sync.WaitGroup
	if ref := body.TriageBundle; ref != nil {
		triage.Add(1)
		go func() {
			defer triage.Done()
			bundle, triageProblem = readMachineTriage(ctx, connection, query, outcome.AttemptOrdinal, ref)
		}()
	}
	written := writtenBytes(&body)
	models, problem := false, modelProblem
	if problem == nil {
		models, problem = m.collectMachineModels(ctx, request, connection, outcome, modelPlan, written)
	}
	if problem == nil {
		problem = m.exportProducts(ctx, connection, query, request)
	}
	triage.Wait()
	if body.TriageBundle != nil && triageProblem != nil {
		fmt.Fprintf(m.context.Out, "machine execution %s: triage bundle not kept: %s\n", request.ID, triageProblem.Message)
	} else if body.TriageBundle != nil {
		if recorded := m.store.RecordMachineTriage(request.ID, int64(outcome.AttemptOrdinal), bundle); recorded != nil {
			return recorded
		}
	}
	if problem != nil {
		return problem
	}
	if problem := m.retainMachineWeights(ctx, request, connection, outcome, &body, modelPlan, written); problem != nil {
		return m.retainedResult(request, outcome, problem)
	}
	if body.GetResult().GetResultBlob() != nil || body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED && request.ChildArtifacts && !models && len(body.GetOutputManifest().GetOutputs()) == 0 {
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
	if problem := m.store.ObserveMachineExecution(request.ID, collected, &pb.MachineExecutionEventPage{NextAfter: cursor, HeadSequence: max(cursor, collected.Sequence)}); problem != nil {
		return problem
	}
	// The run is complete for its caller; its inputs are released after, not before.
	return m.releaseMachineInputs(ctx, request, connection)
}

// terminalEntry reads the log's terminal entry again: an earlier observation recorded it and
// stopped before the outcome was kept. It is among the log's last few entries.
func (m *machineRuns) terminalEntry(ctx context.Context, connection *machineConnection, query *pb.MachineExecutionQuery, cursor, attempt uint64) (*pb.AttemptOutcome, *exit.Error) {
	page, err := connection.Host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: cursor - min(cursor, 64), Limit: 256})
	if err != nil {
		return nil, machineTransport(err)
	}
	var found *pb.AttemptOutcome
	for _, event := range page.Events {
		if event.GetKind() == "outcome" && event.GetOutcome().GetAttemptOrdinal() == attempt {
			found = event.Outcome
		}
	}
	if found == nil {
		return nil, exit.New(exit.Conflict, "the machine's log has no terminal entry carrying its outcome")
	}
	return found, nil
}

// awaitEvents holds one events read open until the machine records the next event, so the
// observer asks again at once instead of on a clock. A control that takes the turn ends it.
func (m *machineRuns) awaitEvents(ctx context.Context, connection *machineConnection, request string, query *pb.MachineExecutionQuery, cursor uint64) {
	if _, err := connection.Host.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: cursor, Limit: 1, Wait: true}); err == nil {
		m.mu.Lock()
		m.awaited[request] = true
		m.mu.Unlock()
	}
}

// workspace is the machine's execution workspace and capabilities, read once per claimed
// connection; a submission its journal refuses as replaced reads it again.
func (m *machineRuns) workspace(ctx context.Context, connection *machineConnection) (*pb.MachineExecutionWorkspace, *exit.Error) {
	if held := connection.Workspace(); held != nil {
		return held, nil
	}
	workspace, problem := currentExecutionWorkspace(ctx, connection)
	if problem == nil {
		connection.KeepWorkspace(workspace)
	}
	return workspace, problem
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
		if recordProblem := m.store.AppendEvent(request.ID, records.MachineResultRetainedType, int64(outcome.AttemptOrdinal),
			map[string]any{"reason": problem.Message}); recordProblem != nil {
			return recordProblem
		}
	}
	return problem
}

// Control wakes delivery of a durable cancel without waiting on its machine.
// Pause/resume retain their synchronous generation-checked control path.
func (m *machineRuns) Control(parent context.Context, request records.Request, action string) *exit.Error {
	if action == "cancel" {
		m.Withdraw(request.ID)
		return m.Start(request)
	}
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
		return m.observeOn(ctx, progress, request, connection, link, query, false)
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

// observeAfterControl flushes the recorded control and imports available changes.
// The background observer waits for completion; a control acknowledgement never does.
func (m *machineRuns) observeAfterControl(ctx context.Context, progress *transfer.Progress, request records.Request, connection *machineConnection, query *pb.MachineExecutionQuery) *exit.Error {
	link, problem := m.store.MachineExecution(request.ID)
	if problem != nil {
		return problem
	}
	if link == nil {
		return exit.New(exit.Conflict, "machine execution link disappeared during control")
	}
	return m.observeOn(ctx, progress, request, connection, link, query, false)
}

func (m *machineRuns) flushMachineControl(ctx context.Context, connection *machineConnection, link *records.MachineExecution) (pb.MachineExecutionAction, *exit.Error) {
	latest, problem := m.store.MachineExecution(link.RequestID)
	if problem != nil {
		return 0, problem
	}
	if latest != nil && latest.Abandoned {
		return 0, exit.Named(exit.Conflict, "request.abandoned", "pending control was retained as evidence after local abandonment")
	}
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
		// A canceled run keeps its products: its log holds them until the terminal is acked.
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

// readMachineTriage reads one attempt's retained triage bundle through the Host, the same
// guarded connection its outcome came over.
func readMachineTriage(ctx context.Context, connection *machineConnection, query *pb.MachineExecutionQuery, attempt uint64, ref *pb.TriageBundleRef) ([]byte, *exit.Error) {
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
