package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type machineExecutionClient interface {
	GetMachineExecutionWorkspace(context.Context, *pb.MachineExecutionWorkspaceQuery, ...grpc.CallOption) (*pb.MachineExecutionWorkspace, error)
	SubmitMachineExecution(context.Context, *pb.MachineExecutionSubmit, ...grpc.CallOption) (*pb.MachineExecutionReceipt, error)
	GetMachineExecution(context.Context, *pb.MachineExecutionQuery, ...grpc.CallOption) (*pb.MachineExecutionState, error)
	ListMachineExecutionEvents(context.Context, *pb.MachineExecutionEventsQuery, ...grpc.CallOption) (*pb.MachineExecutionEventPage, error)
	ControlMachineExecution(context.Context, *pb.MachineExecutionControl, ...grpc.CallOption) (*pb.MachineExecutionState, error)
	CollectMachineExecution(context.Context, *pb.MachineExecutionCollect, ...grpc.CallOption) (*pb.AttemptOutcome, error)
	AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck, ...grpc.CallOption) (*pb.MachineExecutionState, error)
}

type publishedPreparation struct {
	*pb.DesiredPlacementSet
	InstalledPackage   *pb.InstalledPackage
	LockedRequirements []byte
}

func retainWorkerInstallation(connection *machineConnection, expected localpackage.Installation, installed *pb.InstalledPackage) *exit.Error {
	if installed == nil || installed.InstallationId != expected.ID || installed.Package != expected.Package || installed.Release != expected.Release || len(installed.PackageInterface) == 0 {
		return exit.New(exit.Validation, "worker did not return the selected installation and interface")
	}
	connection.installed[expected.ID] = proto.Clone(installed).(*pb.InstalledPackage)
	return nil
}

type machineConnection struct {
	installed          map[string]*pb.InstalledPackage
	importInputTree    func(context.Context) (grpc.ClientStreamingClient[pb.InputTreeImportFrame, pb.NativeByteRetentionResult], error)
	connection         *machineClientConnection
	client             machineExecutionClient
	claim              *pb.Claim
	prepare            func(context.Context, string, localpackage.Installation) *exit.Error
	prepareModels      func(context.Context, records.Request, localpackage.Installation) *exit.Error
	modelDefaultOrigin func(context.Context) (string, *exit.Error)
	preparePublished   func(context.Context, records.Request) (*publishedPreparation, *exit.Error)
	wireMinor          uint32
	publicOrigin       string // renter-authenticated Hub facts name the public byte endpoint
	certificateDigest  []byte
	retainModel        func(context.Context, *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error)
	releaseModel       func(context.Context, *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error)
	retainBytes        func(context.Context, *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error)
	releaseBytes       func(context.Context, *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error)
	readBytes          func(context.Context, *pb.NativeByteRetentionRequest, *pb.Ref) (machineByteStream, error)
	progress           *transfer.Progress
}

type machineClientConnection struct {
	*grpc.ClientConn
	release func()
}

func (c *machineClientConnection) Close() error {
	err := c.ClientConn.Close()
	if c.release != nil {
		c.release()
	}
	return err
}

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
	localMu   sync.Mutex
	localPID  int
	observers sync.Map // one collection/control lock per observed request
	updates   *rentalRuntimeUpdates
}

func newMachineRuns(ctx *Context, layout home.Layout, store *records.Store, resolver *Resolver, fleet *managedRentals) *machineRuns {
	background, cancel := context.WithCancel(context.Background())
	return &machineRuns{ctx: background, cancel: cancel, context: ctx, layout: layout, store: store, resolver: resolver, fleet: fleet, running: map[string]bool{}}
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
		defer func() { m.mu.Lock(); delete(m.running, request.ID); m.mu.Unlock() }()
		lastError := ""
		for m.ctx.Err() == nil {
			current, problem := m.store.RequestRow(request.ID)
			if problem != nil || current == nil {
				return
			}
			if owed, problem := m.store.MachineExecutionOwesWork(request.ID); problem != nil || !owed {
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
						connection, problem = m.connect(m.ctx, link.MachineID)
						if problem == nil {
							problem = m.releaseMachineInputs(m.ctx, *current, connection)
							connection.connection.Close()
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
					problem = m.Control(m.ctx, *current, "cancel")
				} else {
					problem = m.Refresh(m.ctx, *current)
				}
				if problem == nil {
					observed, e := m.store.MachineExecution(request.ID)
					if e == nil && observed != nil && observed.Collected {
						return
					}
					if e == nil && observed != nil && len(observed.PendingControl) == 0 {
						var state pb.MachineExecutionState
						if proto.Unmarshal(observed.ObservedState, &state) == nil && state.State == "canceled" {
							if owed, problem := m.store.MachineExecutionOwesWork(request.ID); problem == nil && !owed {
								return
							}
						}
					}
				}
			}
			if problem != nil && problem.Message != lastError && m.ctx.Err() == nil {
				fmt.Fprintf(m.context.Out, "machine execution %s: %s\n", request.ID, problem.Message)
				lastError = problem.Message
				// submit may have durably frozen/transmitted its offer after this
				// loop read link. Only a fresh journal read can prove it was unsent.
				latest, readProblem := m.store.MachineExecution(request.ID)
				if readProblem == nil && latest != nil && problem.Code != exit.Unavailable && problem.Code != exit.Deadline && len(latest.Submission) == 0 {
					_, _ = m.store.FailQueuedRequest(request.ID, map[string]any{"error_type": problem.ErrName(), "error": problem.Message})
					return
				}
			}
			select {
			case <-m.ctx.Done():
				return
			case <-time.After(time.Second):
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
		if link.Collected && len(link.PendingControl) == 0 {
			continue
		}
		request, problem := m.store.RequestRow(link.RequestID)
		if problem == nil && request != nil {
			_ = m.Start(*request)
		}
	}
}

func (m *machineRuns) submit(request records.Request, link *records.MachineExecution) (out *exit.Error) {
	_, deadline, problem := m.store.RequestExecutionTiming(request.ID)
	if problem != nil {
		return problem
	}
	request.DeadlineUnixMS = deadline
	if link.MachineID == "" {
		machine := "local"
		if request.Rental {
			machine = request.Worker
			if machine == "" {
				machine = request.RequestedRental
			}
			if machine == "" {
				decision, _, problem := m.fleet.acquire(request)
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

	if len(link.Submission) == 0 && m.updates != nil {
		if problem := m.updates.preflight(m.ctx, request, link.MachineID); problem != nil {
			return problem
		}
	}
	connection, problem := m.connect(m.ctx, link.MachineID)
	if problem != nil {
		return problem
	}
	defer connection.connection.Close()
	if connection.wireMinor < pb.WorkspaceFencedExecutionWireMinor {
		return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "workspace-fenced execution requires Runtime protocol %d", pb.WorkspaceFencedExecutionWireMinor)
	}
	if request.Rental && connection.wireMinor < pb.RentalExecutionAdmissionWireMinor {
		return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required",
			"this request uses detached execution, which requires Runtime protocol %d with typed execution lookup for safe idle shutdown; this worker reports %d. Update the rental Runtime; existing result collection remains compatible",
			pb.RentalExecutionAdmissionWireMinor, connection.wireMinor)
	}
	if len(request.Models) > 0 {
		if connection.wireMinor < pb.NativeRootInputsWireMinor {
			return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "durable root Model inputs require Runtime protocol 55")
		}
		if !request.Rental {
			if problem := m.resolver.EnsureLocalModels(request.Hub, request.Models); problem != nil {
				return problem
			}
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
			workspace, problem := currentExecutionWorkspace(m.ctx, connection)
			if problem != nil {
				return problem
			}
			submission.ExpectedExecutionWorkspaceId = workspace
		}
		if submission.PublicationAuthorizationId != "" && connection.wireMinor < 52 {
			return exit.Named(exit.Structural, "publication.worker_upgrade_required", "the frozen publication authorization requires actual Runtime protocol 52")
		}
	} else {
		authorization, problem := m.publicationAuthorization(m.ctx, request.ID, link.MachineID, connection)
		if problem != nil {
			return problem
		}
		var built *pb.MachineExecutionSubmit
		if request.LocalInstallationID == "" {
			built, problem = m.publishedSubmission(m.ctx, request, connection)
			if problem != nil {
				return problem
			}
		} else {
			capture, problem := m.resolver.CaptureMachineExecution(request)
			if problem != nil {
				return problem
			}
			for index, revision := range capture.Installations {
				if problem := connection.prepare(m.ctx, request.ID, revision); problem != nil {
					return problem
				}
				capture.Installations[index].PackageInterface = connection.installed[revision.ID].PackageInterface
				if connection.prepareModels != nil {
					if problem := connection.prepareModels(m.ctx, request, revision); problem != nil {
						return problem
					}
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
			if connection.wireMinor >= pb.CapturedModelDefaultsWireMinor {
				origin := ""
				if connection.modelDefaultOrigin != nil {
					origin, problem = connection.modelDefaultOrigin(m.ctx)
					if problem != nil {
						return problem
					}
				}
				capture, problem = m.resolver.captureMachineModelDefaults(capture, request, origin)
				if problem != nil {
					return problem
				}
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
			if job.Kind != "job" {
				return exit.New(exit.Conflict, "installed callable is not a job")
			}
			request.PlanID = job.DescriptorID
			plan := machineJobPlan(request, installed.InstallationId, job)
			byteInputs, problem := m.stageMachineInputs(m.ctx, request, connection)
			if problem != nil {
				return problem
			}
			built, problem = orchestrator.MachineJobSubmission(request, capture, plan, byteInputs)
			if problem != nil {
				return problem
			}

		}
		if connection.wireMinor < pb.ModelDefaultGPUCountWireMinor {
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
		built.PreparedState.WireMinor = min(built.PreparedState.WireMinor, connection.wireMinor)
		if problem := m.freezeMachineSubmission(m.ctx, connection, request.ID, built); problem != nil {
			return problem
		}
		submission = built
	}
	if _, problem := m.resolver.capturedResultInterface(request); problem != nil {
		return problem
	}
	if request.LocalInstallationID == "" && connection.wireMinor < pb.PublishedMachineCaptureWireMinor {
		return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "published machine execution requires Runtime protocol 54")
	}
	if problem := m.sendMachineSubmission(m.ctx, connection, request.ID, submission); problem != nil {
		return problem
	}
	return m.releaseMachineInputs(m.ctx, request, connection)
}

// freezeMachineSubmission persists authenticated journal identity with the exact
// offer before any Submit RPC. Only a never-transmitted submission may discover it.
func (m *machineRuns) freezeMachineSubmission(ctx context.Context, connection *machineConnection, requestID string, submission *pb.MachineExecutionSubmit) *exit.Error {
	if connection.wireMinor < pb.WorkspaceFencedExecutionWireMinor {
		return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "workspace-fenced execution requires Runtime protocol %d", pb.WorkspaceFencedExecutionWireMinor)
	}
	workspace, problem := currentExecutionWorkspace(ctx, connection)
	if problem != nil {
		return problem
	}
	submission.ExpectedExecutionWorkspaceId = workspace
	return m.store.RecordMachineSubmission(requestID, submission)
}

func currentExecutionWorkspace(ctx context.Context, connection *machineConnection) (string, *exit.Error) {
	workspace, err := connection.client.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{Claim: connection.claim})
	if err != nil {
		return "", machineTransport(err)
	}
	if workspace == nil || workspace.WorkerId != connection.claim.WorkerId || workspace.WorkerBootId == "" ||
		(connection.claim.WorkerBootId != "" && workspace.WorkerBootId != connection.claim.WorkerBootId) ||
		workspace.ExecutionWorkspaceId == "" || len(workspace.ExecutionWorkspaceId) > 256 {
		return "", exit.New(exit.Conflict, "machine returned an invalid execution workspace identity")
	}
	return workspace.ExecutionWorkspaceId, nil
}

func (m *machineRuns) sendMachineSubmission(ctx context.Context, connection *machineConnection, requestID string, submission *pb.MachineExecutionSubmit) *exit.Error {
	if connection.wireMinor < pb.WorkspaceFencedExecutionWireMinor {
		return exit.Named(exit.Structural, "machine_execution.worker_upgrade_required", "workspace-fenced execution requires Runtime protocol %d", pb.WorkspaceFencedExecutionWireMinor)
	}
	if submission.ExpectedExecutionWorkspaceId == "" {
		return exit.Named(exit.Conflict, "machine_execution.workspace_required", "recorded submission has no workspace identity; its acceptance cannot safely be retried")
	}
	submission.Claim = connection.claim
	submission.Offer.WorkerBootId = connection.claim.WorkerBootId
	submission.Offer.RecordOwnerEpoch = connection.claim.RecordOwnerEpoch
	var trailer metadata.MD
	receipt, err := connection.client.SubmitMachineExecution(ctx, submission, grpc.Trailer(&trailer))
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
	if receipt == nil || receipt.WorkerId != connection.claim.WorkerId || receipt.WorkerBootId == "" {
		return exit.New(exit.Conflict, "execution was accepted by an unexpected worker")
	}
	if problem := m.store.AcceptMachineExecution(requestID, receipt); problem != nil {
		return problem
	}
	return nil
}

func machineTransport(err error) *exit.Error {
	code := status.Code(err)
	if code == codes.Unimplemented {
		return exit.Named(exit.Unavailable, "machine_execution.worker_upgrade_required", "worker does not implement workspace-fenced execution; worker protocol 59 is required")
	}
	if code == codes.Unavailable || code == codes.DeadlineExceeded || code == codes.Canceled || code == codes.ResourceExhausted || code == codes.Aborted {
		return exit.Named(exit.Unavailable, "machine_execution.transport_unavailable", "machine execution observation is unavailable: %s", status.Convert(err).Message())
	}
	return exit.Named(exit.Conflict, "machine_execution.refused", "Runtime refused machine execution: %s", status.Convert(err).Message())
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
	var receipt pb.MachineExecutionReceipt
	if link == nil || len(link.Receipt) == 0 || proto.Unmarshal(link.Receipt, &receipt) != nil {
		return nil, nil, nil, exit.Unavailablef("waiting for durable machine acceptance")
	}
	connection, problem := m.connect(ctx, link.MachineID)
	if problem != nil {
		return nil, nil, nil, problem
	}
	query := &pb.MachineExecutionQuery{Claim: connection.claim, RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId}
	return connection, link, query, nil
}

func (m *machineRuns) Refresh(ctx context.Context, request records.Request) *exit.Error {
	lock := m.observationLock(request.ID)
	lock.Lock()
	defer lock.Unlock()
	return m.refresh(ctx, request)
}

func (m *machineRuns) observationLock(request string) *sync.Mutex {
	lock, _ := m.observers.LoadOrStore(request, &sync.Mutex{})
	return lock.(*sync.Mutex)
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
	defer connection.connection.Close()
	connection.progress = progress
	if problem := m.releaseMachineInputs(ctx, request, connection); problem != nil {
		return problem
	}
	if len(link.PendingControl) > 0 {
		if _, problem := m.flushMachineControl(ctx, connection, link); problem != nil {
			return problem
		}
	}
	state, err := connection.client.GetMachineExecution(ctx, query)
	if err != nil {
		return machineTransport(err)
	}
	progress.Advance(1)
	cursor := uint64(link.RemoteCursor)
	for {
		page, err := connection.client.ListMachineExecutionEvents(ctx, &pb.MachineExecutionEventsQuery{Execution: query, After: cursor, Limit: 128})
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
	if state.Collected || state.State == "canceled" || state.State != "succeeded" && state.State != "failed" {
		return nil
	}
	outcome, err := connection.client.CollectMachineExecution(ctx, &pb.MachineExecutionCollect{Execution: query, AttemptOrdinal: state.AttemptOrdinal})
	if err != nil {
		// Another observer may have completed the same collection after our
		// status read. Its ACK releases the original root hold, so re-read the
		// authority before treating that now-stale collect as a custody failure.
		latest, readError := connection.client.GetMachineExecution(ctx, query)
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
	if len(link.Outcome) == 0 && plan != nil {
		for _, warning := range plan.warnings {
			if problem := m.store.AppendEvent(request.ID, "request.warning", int64(outcome.AttemptOrdinal), map[string]any{"message": warning}); problem != nil {
				return problem
			}
		}
	}
	if problem := m.store.RecordMachineOutcome(request.ID, outcome); problem != nil {
		return problem
	}
	models, problem := m.collectMachineModels(ctx, request, connection, outcome, &body)
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
	if len(body.WeightsReceipts) > 0 && !models || len(body.GetOutputManifest().GetOutputs()) > 0 && !files || body.GetResult().GetResultBlob() != nil || body.Status == pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED && request.ChildArtifacts && !models && !files {
		return exit.Named(exit.Unavailable, "machine_execution.result_custody_required", "execution finished; referenced output bytes remain retained on the machine until recipient custody is established")
	}
	ack := &pb.AttemptOutcomeAck{RequestId: outcome.RequestId, AttemptOrdinal: outcome.AttemptOrdinal, InvocationSpecDigest: outcome.InvocationSpecDigest, OutcomeId: outcome.OutcomeId, OutcomeDigest: outcome.OutcomeDigest}
	collected, err := connection.client.AcknowledgeMachineExecutionCollection(ctx, &pb.MachineExecutionCollectionAck{Execution: query, Outcome: ack})
	if err != nil {
		return machineTransport(err)
	}
	return m.store.ObserveMachineExecution(request.ID, collected, &pb.MachineExecutionEventPage{NextAfter: cursor, HeadSequence: max(cursor, collected.Sequence)})
}

func (m *machineRuns) Control(parent context.Context, request records.Request, action string) *exit.Error {
	lock := m.observationLock(request.ID)
	lock.Lock()
	defer lock.Unlock()
	ctx, cancel := (&transfer.Progress{}).Context(parent)
	defer cancel()
	value := map[string]pb.MachineExecutionAction{"pause": pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_PAUSE, "resume": pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RESUME, "cancel": pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL}[action]
	if value == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_UNSPECIFIED {
		return exit.New(exit.Validation, "unknown machine execution control")
	}
	connection, link, query, problem := m.executionConnection(ctx, request)
	if problem != nil {
		return problem
	}
	defer connection.connection.Close()
	if len(link.PendingControl) > 0 {
		previous, problem := m.flushMachineControl(ctx, connection, link)
		if problem != nil {
			return problem
		}
		if previous == value {
			return m.refresh(parent, request)
		}
	}
	state, err := connection.client.GetMachineExecution(ctx, query)
	if err != nil {
		return machineTransport(err)
	}
	command := &pb.MachineExecutionControl{Execution: query, CommandId: records.NewID("control"), ExpectedGeneration: state.Generation, Action: value}
	if problem := m.store.RecordMachineControl(request.ID, command); problem != nil {
		return problem
	}
	if problem := m.refresh(parent, request); problem != nil {
		return problem
	}
	return m.Start(request)
}

func (m *machineRuns) flushMachineControl(ctx context.Context, connection *machineConnection, link *records.MachineExecution) (pb.MachineExecutionAction, *exit.Error) {
	var command pb.MachineExecutionControl
	if proto.Unmarshal(link.PendingControl, &command) != nil || command.Execution == nil {
		return 0, exit.Internalf("recorded machine control is unreadable")
	}
	command.Execution.Claim = connection.claim
	if command.Action == pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL {
		if problem := m.releaseMachineFiles(ctx, link.RequestID, connection); problem != nil {
			return command.Action, problem
		}
		if problem := m.releaseMachineModels(ctx, link.RequestID, connection); problem != nil {
			return command.Action, problem
		}
	}
	var trailer metadata.MD
	if _, err := connection.client.ControlMachineExecution(ctx, &command, grpc.Trailer(&trailer)); err != nil {
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

func readMachinePreparation(stream grpc.ServerStreamingClient[pb.PrepareEvent]) *exit.Error {
	_, problem := readMachinePreparedSet(stream)
	return problem
}

func readMachinePreparedSet(stream grpc.ServerStreamingClient[pb.PrepareEvent]) (*pb.DesiredPlacementSet, *exit.Error) {
	event, problem := readMachinePreparationEvent(stream)
	if problem != nil {
		return nil, problem
	}
	return event.PlacementSet, nil
}

func readMachinePreparationEvent(stream grpc.ServerStreamingClient[pb.PrepareEvent]) (*pb.PrepareEvent, *exit.Error) {
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
		switch event.Stage {
		case pb.PrepareStage_PREPARE_STAGE_REFUSED:
			if problem := orchestrator.RuntimeRequirementEvent(event); problem != nil {
				return event, problem
			}
			if event.SafeCode == "package_environment_dependency_base_conflict" {
				return event, exit.Named(exit.Structural, "machine_execution.runtime_requirement", "%s", event.SafeDetail)
			}
			return event, exit.Named(exit.Conflict, "machine_execution.prepare_refused", "%s: %s", event.SafeCode, event.SafeDetail)
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
