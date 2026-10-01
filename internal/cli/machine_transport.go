package cli

import (
	"archive/tar"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/publication"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
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
func (m *machineRuns) connect(ctx context.Context, name string, holder orchestrator.Holder) (*machineConnection, *exit.Error) {
	return m.connectAt(ctx, name, "", holder, true)
}

// connectFor opens the machine a run executes on, for the run's hub: a run of an install
// reads its release at the hub the install came from.
func (m *machineRuns) connectFor(ctx context.Context, request records.Request, name, doing string) (*machineConnection, *exit.Error) {
	origin := request.Hub
	// A captured local program can start without a Hub account or network. Explicit
	// publication, catalog models and published code need delegated execution access.
	localCode := request.LocalInstallationID != "" || strings.HasPrefix(request.Package, "local/")
	if localCode && request.ModelTransfer == nil && len(request.Models) == 0 {
		consent, problem := m.store.RequestPublicationRepositories(request.ID)
		if problem != nil {
			return nil, problem
		}
		if len(consent) == 0 {
			scope, problem := m.capturedHubScope(request)
			if problem != nil {
				return nil, problem
			}
			if !scope.required {
				if scope.possible && origin != "" && client(m.context.forHub(origin)).CredentialIdentity() != "" {
					if connected, problem := m.connectAt(ctx, name, origin, m.runHolder(request, doing), true); problem == nil {
						return connected, nil
					}
				}
				// Optional authority could not be attached. Empty scope cannot borrow
				// a retained Hub grant, even when the same machine still holds one.
				origin = ""
			}
		}
	}
	return m.connectAt(ctx, name, origin, m.runHolder(request, doing), true)
}

type capturedHubUse struct{ required, possible bool }

// Captured Model slots may receive local artifacts, so they establish only
// possible catalog use. Account-index dependencies require delegated authority.
// Explicit model selections and publication are handled by connectFor.
func (m *machineRuns) capturedHubScope(request records.Request) (capturedHubUse, *exit.Error) {
	use := capturedHubUse{}
	if request.LocalInstallationID == "" {
		return use, nil
	}
	capture, problem := m.resolver.CaptureMachineExecution(request)
	if problem != nil {
		return use, problem
	}
	var document pb.MachineExecutionCapture
	if err := canonical.Unmarshal(capture.Canonical, &document); err != nil {
		return use, exit.New(exit.Validation, "captured installation graph is unreadable: %s", err)
	}
	selected := map[string]map[string]bool{request.LocalInstallationID: {request.Entrypoint: true}}
	for _, binding := range document.Bindings {
		if selected[binding.CalleeInstallationId] == nil {
			selected[binding.CalleeInstallationId] = map[string]bool{}
		}
		selected[binding.CalleeInstallationId][binding.Entrypoint] = true
	}
	for _, installation := range capture.Installations {
		face, problem := launch.DecodePackageInterface(installation.PackageInterface)
		if problem != nil {
			return use, problem
		}
		for name := range selected[installation.ID] {
			callable, problem := face.Function(name)
			if problem != nil {
				return use, problem
			}
			use.possible = use.possible || len(callable.Models) > 0
		}
		if installation.SourceArchive == "" {
			continue
		}
		found := false
		for _, file := range installation.Files {
			if file.Filename != installation.SourceArchive {
				continue
			}
			found = true
			uses, problem := capturedSourceAccountIndex(file.Path)
			if problem != nil {
				return use, problem
			}
			if uses {
				use.required = true
				return use, nil
			}
		}
		if !found {
			return use, exit.Named(exit.Validation, "machine_execution.source_metadata_missing", "captured source archive is missing")
		}
	}
	return use, nil
}

func capturedSourceAccountIndex(path string) (bool, *exit.Error) {
	file, err := os.Open(path)
	if err != nil {
		return false, exit.Named(exit.Validation, "machine_execution.source_metadata_missing", "cannot read captured source archive: %s", err)
	}
	defer file.Close()
	// The file itself, not a reader over it: members before pyproject.toml (vendored wheels
	// sort first) are skipped by seeking instead of read.
	archive := tar.NewReader(file)
	for count := 0; count <= packagepublish.MaxSourceFiles; count++ {
		header, err := archive.Next()
		if err == io.EOF {
			return false, exit.Named(exit.Validation, "machine_execution.source_metadata_missing", "captured source has no pyproject.toml")
		}
		if err != nil {
			return false, exit.Named(exit.Validation, "machine_execution.source_metadata_invalid", "captured source archive is unreadable: %s", err)
		}
		if header.Name != "pyproject.toml" {
			continue
		}
		limit := packagepublish.SourceFileLimit(header.Name)
		if header.Typeflag != tar.TypeReg || header.Size <= 0 || header.Size > limit {
			return false, exit.Named(exit.Validation, "machine_execution.source_metadata_invalid", "captured pyproject.toml is not a bounded regular file")
		}
		raw, err := io.ReadAll(io.LimitReader(archive, limit+1))
		if err != nil {
			return false, exit.Named(exit.Validation, "machine_execution.source_metadata_invalid", "captured pyproject.toml is unreadable: %s", err)
		}
		return packagepublish.UsesAccountIndexDocument(raw)
	}
	return false, exit.Named(exit.Validation, "machine_execution.source_metadata_invalid", "captured source archive has too many members")
}

// connectAtHub opens a machine for a published release read at hub.
func (m *machineRuns) connectAtHub(ctx context.Context, name, hub string, holder orchestrator.Holder) (*machineConnection, *exit.Error) {
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

func (m *machineRuns) connectAt(ctx context.Context, name, hub string, holder orchestrator.Holder, named bool) (*machineConnection, *exit.Error) {
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
	// Wire 69 carries the selected Hub on this operation; older peers would drop it.
	if c.Hub != "" && c.WireMinor < 69 {
		return exit.Named(exit.Structural, "machine.local_package_hub_required", "this machine must support explicit Hub selection for unpublished package dependencies; update its agent and Runtime")
	}
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
		if stream, err := c.Host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: c.Claim, LocalPackageSet: selected, Hub: c.Hub}); err == nil {
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
	stream, err := c.Host.PrepareLocalPackage(ctx, &pb.PrepareLocalPackageCall{Claim: c.Claim, LocalPackageSet: selected, Hub: c.Hub})
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
	var choices []*pb.ModelChoice
	if revision.ID == request.LocalInstallationID {
		own := request.OwnModels()
		if slices.ContainsFunc(own, func(model records.ModelRef) bool { return len(model.Adapters) > 0 || model.Source != "" }) {
			if c.WireMinor < 70 {
				return nil, exit.Named(exit.Structural, "model_overrides.private_preparation_unavailable",
					"this machine's agent cannot forward private model overrides; update its agent and Runtime")
			}
			var problem *exit.Error
			if choices, problem = orchestrator.PrivateModelChoices(request); problem != nil {
				return nil, problem
			}
		}
	}
	if len(models) == 0 && len(choices) == 0 {
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
	if len(choices) > 0 {
		selected.ModelChoices, selected.SourceCredentials = choices, c.runs.resolver.SourceCredentials()
		selected.Hub, selected.Owner = c.Hub, c.runs.runAccount(request)
	}
	stream, err := c.Host.PreparePrivatePlacement(ctx, &pb.PreparePrivatePlacementCall{
		Claim: c.Claim, PrivatePlacementSet: selected,
	})
	if err != nil {
		return nil, machineTransport(err)
	}
	return readMachinePreparedSet(stream, c.runs.preparationPhase(request.ID, revision.Package).observe)
}

// Prewarm lands one journaled download or installation on its machine without running
// anything, folding the machine's per-model bytes into the operation's live phase. It
// answers the bytes last observed. A worker boot the operation was not claimed on refuses.
func (m *machineRuns) Prewarm(ctx context.Context, o records.Operation) ([]records.ModelProgress, *exit.Error) {
	models := orchestrator.DownloadModelRefs(o.Install.Models)
	if len(models) != len(o.Install.Models) {
		return nil, exit.New(exit.Validation, "the installation contains non-downloadable model selections")
	}
	connection, problem := m.connectAtHub(ctx, o.Machine, o.Install.Hub, orchestrator.Holder{Number: o.Number, What: o.Kind + " " + o.Target()})
	if problem != nil {
		return nil, problem
	}
	defer connection.Close()
	if o.BootID != "" && connection.Claim.WorkerBootId != o.BootID {
		return nil, exit.Unavailablef("the rental's worker restarted before preparation; the installation is claimed again on its new boot")
	}
	if problem := connection.ValidateNewWork(); problem != nil {
		return nil, problem
	}
	var packages []*pb.DownloadPackageRef
	if o.Install.Package != "" {
		packages = append(packages, &pb.DownloadPackageRef{Package: o.Install.Package, Release: o.Install.Release})
	}
	downloads, problem := rental.DownloadSet(packages, models)
	if problem != nil {
		return nil, problem
	}
	observed := &operationProgress{owner: m.fleet.owner, id: o.ID, machine: connection.Name, label: o.Target()}
	defer m.fleet.owner.ForgetPhase(o.ID)
	// The operation names its requester: a reconnect attaches to the same work, and only
	// its owner's cancel detaches it.
	stream, err := connection.Host.PreparePackageSet(ctx, &pb.PreparePackageSetCall{Claim: connection.Claim, Hub: connection.Hub,
		PackageSet: &pb.DesiredPackageSet{DownloadDelegation: downloads}, OperationId: o.ID})
	if err == nil {
		_, problem = readMachinePreparationEvent(stream, observed.observe)
	} else {
		problem = machineTransport(err)
	}
	if errors.Is(context.Cause(ctx), machines.ErrCanceled) {
		return observed.last, m.cancelPreparation(connection, o, observed)
	}
	return observed.last, problem
}

// cancelPreparation detaches a canceled operation from its machine. The machine stops the
// transfer once no other requester needs it; one that predates cancellation finishes it.
func (m *machineRuns) cancelPreparation(connection *machineConnection, o records.Operation, observed *operationProgress) *exit.Error {
	name := o.Machine
	if row, problem := m.store.RentalRow(o.Machine); problem == nil && row != nil && row.MachineName != "" {
		name = row.MachineName
	}
	result, err := connection.Host.CancelPreparation(m.ctx, &pb.CancelPreparationCall{Claim: connection.Claim, OperationId: o.ID})
	switch {
	case status.Code(err) == codes.Unimplemented || status.Code(err) == codes.FailedPrecondition &&
		strings.HasPrefix(status.Convert(err).Message(), pb.CapabilityUnavailableCode+":"):
		return exit.Named(exit.Canceled, "machine.cancel_unavailable",
			"%s predates cancellation: it finishes the transfer it started; landed bytes stay", name)
	case err != nil:
		return exit.Named(exit.Canceled, "machine.cancel_unconfirmed", "%s was not told to stop: %s", name, machineTransport(err).Message)
	}
	observed.observe(&pb.PrepareEvent{ModelProgress: result.GetModelProgress()})
	return exit.Named(exit.Canceled, "operation.canceled",
		"%s stops the transfer once nothing else needs it; landed bytes stay, so the same download resumes", name)
}

// operationProgress is one operation's live phase and the model bytes it last saw.
type operationProgress struct {
	owner              *orchestrator.Orchestrator
	id, machine, label string
	last               []records.ModelProgress
}

func (p *operationProgress) observe(event *pb.PrepareEvent) {
	p.owner.ObservePrepareEvent(p.id, p.machine, p.label, event)
	if len(event.GetModelProgress()) == 0 {
		return
	}
	p.last = p.last[:0]
	for _, row := range event.GetModelProgress() {
		model := row.GetModel()
		p.last = append(p.last, records.ModelProgress{Model: model.GetModel(), Release: model.GetRelease(), Lane: model.GetLane(),
			Manifest: model.GetManifest(), Moved: row.GetTransferredBytes(), Total: row.GetTotalBytes()})
	}
}

// Forget drops the kept connection to a machine, so a rental's credentials can be removed.
func (m *machineRuns) Forget(machine string) { m.machines.Forget(machine) }

// Describe asks the machine for a published release's interface, as it reads it at its own Hub.
func (m *machineRuns) Describe(ctx context.Context, machine, hub, pkg, release string) (api.DescribedRelease, *exit.Error) {
	connection, problem := m.connectAtHub(ctx, machine, hub, orchestrator.Holder{What: "describing " + pkg})
	if problem != nil {
		return api.DescribedRelease{}, problem
	}
	defer connection.Close()
	var header, trailer metadata.MD
	workspace, err := connection.Host.GetMachineExecutionWorkspace(ctx, &pb.MachineExecutionWorkspaceQuery{
		Claim: connection.Claim, Describe: &pb.PackageSelection{Package: pkg, Release: release, Hub: connection.Hub}}, grpc.Header(&header), grpc.Trailer(&trailer))
	if runtimeUnavailable(header, trailer) {
		return api.DescribedRelease{}, waitForRuntime()
	}
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

// Software reads what one machine runs from its DescribeMachine (wire 66). A machine that
// predates it answers UNIMPLEMENTED, which is said rather than guessed.
func (m *machineRuns) Software(ctx context.Context, machine string) (api.MachineSoftware, *exit.Error) {
	connection, problem := m.connect(ctx, machine, orchestrator.Holder{What: "reading its software"})
	if problem != nil {
		return api.MachineSoftware{}, problem
	}
	defer connection.Close()
	described, err := connection.Host.DescribeMachine(ctx, &pb.DescribeMachineQuery{Claim: connection.Claim})
	if status.Code(err) == codes.Unimplemented {
		return api.MachineSoftware{}, exit.Named(exit.Structural, "machine.software_unreported",
			"this machine's agent predates software reporting")
	}
	if err != nil {
		return api.MachineSoftware{}, machineTransport(err)
	}
	return api.MachineSoftware{Agent: described.GetHost().GetVersion(), Runtime: described.GetRuntime().GetVersion(),
		TensorFS: described.GetRuntime().GetTensorfsVersion(), RuntimeAbsent: described.GetRuntimeAbsent()}, nil
}

// machineLogs maps the log names clients use to the logs machines keep.
var machineLogs = map[string]pb.MachineLog{"tensorfs": pb.MachineLog_MACHINE_LOG_TENSORFS_TRANSPORT}

// MachineLog reads one log a machine keeps (wire 72). A machine whose agent predates the
// read answers a note in Unavailable, never a failure.
func (m *machineRuns) MachineLog(ctx context.Context, machine, log string, tailBytes uint64) (api.MachineLog, *exit.Error) {
	kept, ok := machineLogs[log]
	if !ok {
		return api.MachineLog{}, exit.Named(exit.NotFound, "machine.log_unknown", "machines keep no log %q", log)
	}
	connection, problem := m.connect(ctx, machine, orchestrator.Holder{What: "reading its logs"})
	if problem != nil {
		return api.MachineLog{}, problem
	}
	defer connection.Close()
	out := api.MachineLog{Log: log}
	stream, err := connection.Host.ReadMachineLog(ctx, &pb.MachineLogQuery{Claim: connection.Claim, Log: kept, TailBytes: tailBytes})
	var text strings.Builder
	for err == nil {
		var chunk *pb.MachineLogChunk
		if chunk, err = stream.Recv(); err == nil {
			text.Write(chunk.GetData())
		}
	}
	switch {
	case errors.Is(err, io.EOF):
		out.Text = text.String()
		return out, nil
	case status.Code(err) == codes.Unimplemented:
		out.Unavailable = "this machine's agent predates reading its logs; a machine started on a newer agent has them"
		return out, nil
	}
	return api.MachineLog{}, machineTransport(err)
}

// ForgetPackage tells each machine this daemon knows (this computer's while it runs, every
// ready rental) that a package's releases or owner bindings, or a model's releases, changed:
// each keeps what it read of that name at its Hub, across Runtime restarts, and reads it once
// more on its next run.
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
		connection, problem := m.connect(ctx, name, orchestrator.Holder{What: "forgetting " + pkg})
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
			out.Notes = append(out.Notes, fmt.Sprintf("%s keeps what it read of %s until it is told or stopped: %s", name, pkg, problem.Message))
			continue
		}
		out.Machines = append(out.Machines, name)
	}
	return out
}

// PruneOperationCache frees one machine's unused cached operation results through its Host.
func (m *machineRuns) PruneOperationCache(ctx context.Context, machine string) (uint32, uint64, bool, *exit.Error) {
	connection, problem := m.connect(ctx, machine, orchestrator.Holder{What: "pruning its operation cache"})
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
