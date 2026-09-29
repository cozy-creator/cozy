package cli

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// machineConnection is one claimed machine, local or rented, and what this submission has
// prepared on it. Every call below is the same PodHost call whichever machine answers it.
type machineConnection struct {
	*machines.Machine
	runs       *machineRuns
	installed  map[string]*pb.InstalledPackage
	placements map[string]*pb.DesiredPlacementSet // each captured revision's code-only placement
	progress   *transfer.Progress
}

// connect opens a machine for work that reads no hub; holder is what the caller is doing
// there, which a rental's maintenance refusal names.
func (m *machineRuns) connect(ctx context.Context, name, holder string) (*machineConnection, *exit.Error) {
	return m.connectAt(ctx, name, "", holder, true)
}

// connectFor opens the machine a run executes on, for the run's hub: a run of an install
// reads its release at the hub the install came from.
func (m *machineRuns) connectFor(ctx context.Context, request records.Request, name, doing string) (*machineConnection, *exit.Error) {
	origin := request.Hub
	// A captured local program can start without a Hub account or network. Explicit
	// publication, catalog models and published code need delegated execution access.
	localCode := request.LocalInstallationID != "" || strings.HasPrefix(request.Package, "local/")
	if localCode && !request.IsJob() && request.ModelTransfer == nil && len(request.Models) == 0 {
		consent, problem := m.store.RequestPublicationRepositories(request.ID)
		if problem != nil {
			return nil, problem
		}
		if len(consent) == 0 {
			origin = ""
		}
	}
	return m.connectAt(ctx, name, origin, m.runHolder(request, doing), true)
}

// connectAtHub opens a machine for a published release read at hub.
func (m *machineRuns) connectAtHub(ctx context.Context, name, hub, holder string) (*machineConnection, *exit.Error) {
	return m.connectAt(ctx, name, hub, holder, true)
}

// adoptLocalHost migrates the embedded Host once no accepted run is active.
func (m *machineRuns) adoptLocalHost(ctx context.Context) {
	if !m.machines.Host.Outdated() {
		return
	}
	active, problem := m.store.ActiveRequests()
	if problem != nil {
		return
	}
	idle := true
	for _, request := range active {
		if link, problem := m.store.MachineExecution(request.ID); request.Worker == machines.Local && problem == nil && link != nil && len(link.Submission) > 0 {
			idle = false
			break
		}
	}
	if adopted, problem := m.machines.Host.Adopt(ctx, idle); problem != nil {
		fmt.Fprintf(m.context.Out, "the machine keeps its Host: %s\n", problem.Message)
	} else if adopted {
		m.machines.Forget(machines.Local)
	}
}

func (m *machineRuns) connectAt(ctx context.Context, name, hub, holder string, named bool) (*machineConnection, *exit.Error) {
	if machines.IsLocal(name) {
		m.adoptLocalHost(ctx)
	}
	machine, problem := m.machines.DialAt(ctx, name, hub, holder, named)
	if problem != nil {
		return nil, problem
	}
	return &machineConnection{Machine: machine, runs: m, installed: map[string]*pb.InstalledPackage{}, placements: map[string]*pb.DesiredPlacementSet{}}, nil
}

// artifactCommands numbers this host's retained-artifact commands; the machine keys a
// command's replay by it.
var artifactCommands atomic.Uint64

// artifactTransfer sends one retained-artifact command over the machine connection: a
// closure page or one granted object upload, authorized by this connection's claim.
func (c *machineConnection) artifactTransfer(ctx context.Context, command *pb.NativeArtifactTransfer) (*pb.NativeArtifactTransferStatus, *exit.Error) {
	command.CommandId = artifactCommands.Add(1)
	status, err := c.Host.NativeArtifactTransfer(ctx, &pb.NativeArtifactTransferCall{Claim: c.Claim, Request: command})
	if err != nil {
		return nil, machineTransport(err)
	}
	if problem := publication.ArtifactTransferRefusal(status.SafeCode, status.SafeDetail); problem != nil {
		return nil, problem
	}
	return status, nil
}

func (c *machineConnection) retainModel(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
	return c.Host.RetainDerivedResult(ctx, &pb.DerivedRetentionCall{Claim: c.Claim, Request: request})
}

func (c *machineConnection) releaseModel(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
	return c.Host.ReleaseDerivedRetention(ctx, &pb.DerivedRetentionCall{Claim: c.Claim, Request: request})
}

func (c *machineConnection) importInputTree(ctx context.Context) (grpc.ClientStreamingClient[pb.InputTreeImportFrame, pb.NativeByteRetentionResult], error) {
	return c.Host.ImportInputTree(ctx)
}

func (c *machineConnection) readBytes(ctx context.Context, source *pb.NativeByteRetentionRequest, object *pb.Ref, offset uint64) (machineByteStream, error) {
	return c.Host.ReadByteTreeObject(ctx, &pb.NativeByteReadCall{Claim: c.Claim, Source: source, Object: object, Offset: offset})
}

// prepare installs one captured revision: its wheels move from this client to the machine
// through the Host's verified upload, then the Host prepares its environment.
func (c *machineConnection) prepare(ctx context.Context, request string, revision localpackage.Installation) *exit.Error {
	m := c.runs
	baseOperation := machinePackageOperation(request, revision)
	transfer, problem := m.store.MachinePackageTransfer(request, c.Claim.WorkerBootId, revision.ID)
	if problem != nil {
		return problem
	}
	operation := transfer.Operation
	if operation == "" {
		operation = baseOperation
	}
	if operation != baseOperation {
		prefix := baseOperation + ".repair-"
		sequence, err := strconv.Atoi(strings.TrimPrefix(operation, prefix))
		if !strings.HasPrefix(operation, prefix) || err != nil || sequence < 1 || sequence > transfer.Completed {
			return exit.New(exit.Conflict, "captured package transfer names another operation")
		}
	}
	selected, problem := orchestrator.LocalPackageSelection(operation, revision)
	if problem != nil {
		return problem
	}
	if !transfer.Uploaded {
		// A machine that already holds this installation (an unchanged editable package run
		// again) reopens it: nothing is uploaded. Any other answer is an upload.
		if stream, err := c.Host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: c.Claim, LocalPackageSet: selected}); err == nil {
			if event, problem := readMachinePreparationEvent(stream, nil); problem == nil {
				c.placements[revision.ID] = event.PlacementSet
				return retainWorkerInstallation(c, revision, event.InstalledPackage)
			}
		}
		if transfer.Operation != operation || transfer.Uploaded {
			if problem := m.store.AppendEvent(request, "machine.package_upload_started", 0, map[string]any{"worker_boot_id": c.Claim.WorkerBootId, "revision": revision.ID, "operation_id": operation}); problem != nil {
				return problem
			}
		}
		if problem := uploadMachinePackage(ctx, c.Host, c.Claim, operation, revision); problem != nil {
			return problem
		}
		if problem := m.store.AppendEvent(request, "machine.package_uploaded", 0, map[string]any{"worker_boot_id": c.Claim.WorkerBootId, "revision": revision.ID, "operation_id": operation}); problem != nil {
			return problem
		}
	}
	stream, err := c.Host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: c.Claim, LocalPackageSet: selected})
	if err != nil {
		return machineTransport(err)
	}
	event, problem := readMachinePreparationEvent(stream, m.preparationPhase(request, revision.Package).observe)
	if problem != nil {
		return problem
	}
	c.placements[revision.ID] = event.PlacementSet
	return retainWorkerInstallation(c, revision, event.InstalledPackage)
}

// prepareModels lands a captured revision's exact model inputs through the Host's download
// set. Only inference supplies these; installing the captured code above is independent of
// any model, including unused child defaults.
func (c *machineConnection) prepareModels(ctx context.Context, request records.Request, revision localpackage.Installation) (*pb.DesiredPlacementSet, *exit.Error) {
	models := orchestrator.PrivateRevisionModelRefs(request, revision.Package)
	if len(models) == 0 {
		if request.IsJob() || revision.ID != request.LocalInstallationID {
			return nil, nil
		}
		// A model-free inference root runs on the placement its code preparation made.
		return c.placements[revision.ID], nil
	}
	transfer, problem := c.runs.store.MachinePackageTransfer(request.ID, c.Claim.WorkerBootId, revision.ID)
	if problem != nil {
		return nil, problem
	}
	operation := transfer.Operation
	if operation == "" {
		operation = machinePackageOperation(request.ID, revision)
	}
	downloads, problem := rental.DownloadSet(nil, models)
	if problem != nil {
		return nil, problem
	}
	selected := &pb.DesiredPrivatePlacementSet{OperationId: operation, InstallationId: revision.ID, DownloadDelegation: downloads}
	stream, err := c.Host.PreparePrivatePlacement(ctx, &pb.PreparePrivatePlacementCall{
		Claim: c.Claim, PrivatePlacementSet: selected,
	})
	if err != nil {
		return nil, machineTransport(err)
	}
	return readMachinePreparedSet(stream, c.runs.preparationPhase(request.ID, revision.Package).observe)
}

// Prewarm prepares a published package, its selected models, or models alone on a machine
// without running anything: `cozy package install` and `cozy model download` for any
// machine. bootID, when set, is the worker lifetime the selection was queued for.
func (m *machineRuns) Prewarm(ctx context.Context, machine, hub, bootID, pkg, release string, models []*pb.DownloadModelRef) *exit.Error {
	connection, problem := m.connectAtHub(ctx, machine, hub, "installing "+either(pkg, "models"))
	if problem != nil {
		return problem
	}
	defer connection.Close()
	if bootID != "" && connection.Claim.WorkerBootId != bootID {
		return exit.Unavailablef("the rental's worker restarted before preparation; the installation is claimed again on its new boot")
	}
	if problem := connection.ValidateNewWork(); problem != nil {
		return problem
	}
	var packages []*pb.DownloadPackageRef
	call := &pb.PreparePackageSetCall{Claim: connection.Claim, Hub: connection.Hub}
	if pkg != "" {
		packages = append(packages, &pb.DownloadPackageRef{Package: pkg, Release: release})
	}
	downloads, problem := rental.DownloadSet(packages, models)
	if problem != nil {
		return problem
	}
	call.PackageSet = &pb.DesiredPackageSet{DownloadDelegation: downloads}
	stream, err := connection.Host.PreparePackageSet(ctx, call)
	if err != nil {
		return machineTransport(err)
	}
	_, problem = readMachinePreparationEvent(stream, nil)
	return problem
}

// Forget drops the kept connection to a machine, so a rental's credentials can be removed.
func (m *machineRuns) Forget(machine string) { m.machines.Forget(machine) }

// Describe asks the machine for a published release's interface, as it reads it at its own Hub.
func (m *machineRuns) Describe(ctx context.Context, machine, hub, pkg, release string) (api.DescribedRelease, *exit.Error) {
	connection, problem := m.connectAtHub(ctx, machine, hub, "describing "+pkg)
	if problem != nil {
		return api.DescribedRelease{}, problem
	}
	defer connection.Close()
	workspace, err := connection.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{
		Claim: connection.Claim, Describe: &pb.PackageSelection{Package: pkg, Release: release, Hub: connection.Hub}})
	if err != nil {
		return api.DescribedRelease{}, machineTransport(err)
	}
	described := workspace.GetDescribedRelease()
	if described == nil {
		return api.DescribedRelease{}, exit.Named(exit.Structural, "machine_execution.worker_upgrade_required",
			"this machine's Runtime describes no release; %s", machines.RuntimeUpdate(machine))
	}
	if described.Package != pkg || described.Release == "" || release != "" && described.Release != release {
		return api.DescribedRelease{}, exit.New(exit.Conflict, "the machine described another release than %s", pkg)
	}
	keepReleaseInterface(m.layout.Root, pkg, described.Release, described.PackageInterface, nil)
	return api.DescribedRelease{Package: described.Package, Release: described.Release, PackageInterface: described.PackageInterface}, nil
}

// ForgetPackage tells each machine this daemon knows (this computer's while it runs, every
// ready rental) that a package's releases or owner bindings changed: each keeps what it read
// of the package at its Hub, and reads it once more on its next run.
func (m *machineRuns) ForgetPackage(ctx context.Context, pkg string) api.ForgottenPackage {
	out := api.ForgottenPackage{Package: pkg, Machines: []string{}}
	var names []string
	if status, problem := m.machines.Host.Status(); problem == nil && status.Running {
		names = append(names, machines.Local)
	}
	rentals, problem := m.store.Rentals()
	if problem != nil {
		out.Notes = append(out.Notes, "rentals were not told: "+problem.Message)
	}
	for _, row := range rentals {
		if row.State == "ready" && row.Address != "" {
			names = append(names, row.ID)
		}
	}
	for _, name := range names {
		connection, problem := m.connect(ctx, name, "forgetting "+pkg)
		if problem == nil {
			_, err := connection.Host.ForgetPackage(ctx, &pb.ForgetPackageCall{Claim: connection.Claim, Package: pkg})
			connection.Close()
			if status.Code(err) == codes.Unimplemented {
				continue // an older machine keeps no package cache: it reads the package every run
			}
			if err != nil {
				problem = machineTransport(err)
			}
		}
		if problem != nil {
			out.Notes = append(out.Notes, fmt.Sprintf("%s keeps what it read of %s until it restarts: %s", name, pkg, problem.Message))
			continue
		}
		out.Machines = append(out.Machines, name)
	}
	return out
}

// PruneOperationCache frees one machine's unused cached operation results through its Host.
func (m *machineRuns) PruneOperationCache(ctx context.Context, machine string) (uint32, uint64, bool, *exit.Error) {
	connection, problem := m.connect(ctx, machine, "pruning its operation cache")
	if problem != nil {
		return 0, 0, false, problem
	}
	defer connection.Close()
	result, err := connection.Host.PruneOperationCache(ctx, &pb.PruneOperationCacheCall{Claim: connection.Claim})
	if err != nil {
		return 0, 0, false, machineTransport(err)
	}
	if result == nil {
		return 0, 0, false, exit.Named(exit.Structural, "operation.prune_reply_absent", "Host returned no cache pruning observation")
	}
	return result.RemovedEntries, result.ReclaimedBytes, result.StoreBusy, nil
}

// preparationPhase shows a waiting run what its machine's preparation is doing: resolving,
// downloading its models, installing its package. Each ended stage stays on the run.
func (m *machineRuns) preparationPhase(request, label string) *machinePreparation {
	return &machinePreparation{machines: m, request: request, label: label}
}

type machinePreparation struct {
	machines       *machineRuns
	request, label string
	stage          pb.PrepareStage
	began          time.Time
	last           *pb.PrepareEvent
	// retained: the Host answered with the preparation it already holds for these exact
	// inputs, its only event PREPARED.
	retained bool
}

func (p *machinePreparation) observe(event *pb.PrepareEvent) {
	p.machines.fleet.owner.ObservePrepareEvent(p.request, "", "", event)
	if event.Stage == p.stage {
		p.last = event
		return
	}
	if payload := orchestrator.PrepareStagePayload(p.label, p.stage, p.began, p.last); payload != nil {
		payload["detail"] = p.label
		_ = p.machines.store.AppendEvent(p.request, "request.preparing", 0, payload)
	}
	p.retained = p.stage == pb.PrepareStage_PREPARE_STAGE_UNSPECIFIED && event.Stage == pb.PrepareStage_PREPARE_STAGE_PREPARED
	p.stage, p.began, p.last = event.Stage, time.Now(), event
}

func machinePackageOperation(request string, revision localpackage.Installation) string {
	return request + "." + revision.ID
}

// The ordinary Host upload has durable verified prefixes. Private code moves
// directly from this client to its machine, without a package repository.
func uploadMachinePackage(ctx context.Context, host pb.PodHostClient, claim *pb.Claim, operation string, revision localpackage.Installation) *exit.Error {
	for _, file := range revision.Files {
		digest, err := canonical.Raw(file.Digest)
		if err != nil && file.Kind != "source" {
			return exit.New(exit.Conflict, "captured wheel identity is invalid")
		}
		header := &pb.LocalPackageUploadHeader{Claim: claim, OperationId: operation,
			File: &pb.LocalPackageFileRef{Digest: digest, Length: uint64(file.Length), Filename: file.Filename}}
		if problem := localpackage.UploadFile(ctx, host, header, file.Path, nil); problem != nil {
			return problem
		}
	}
	return nil
}
