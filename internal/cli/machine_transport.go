package cli

import (
	"archive/tar"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	machinepb "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/machinev1"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
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

// ValidateEndpoint proves target TLS, controller authority and workspace before admission.
func (m *machineRuns) ValidateEndpoint(ctx context.Context, ep *machineendpoint.Endpoint) *exit.Error {
	if ep == nil {
		return exit.New(exit.Validation, "explicit machine endpoint is absent")
	}
	// A cozy.machine.v1 machine answers Status to a key it authorizes: that is the check.
	v1, problem := m.machines.DialEndpointV1(*ep)
	if problem != nil {
		return problem
	}
	frame, err := v1.Status(ctx)
	v1.Close()
	switch {
	case err == nil && frame.WorkerId != ep.WorkerID:
		return exit.New(exit.Conflict, "the explicit endpoint is machine %s, not %s", frame.WorkerId, ep.WorkerID)
	case err == nil:
		return nil
	case status.Code(err) == codes.Unauthenticated || status.Code(err) == codes.PermissionDenied:
		return exit.Named(exit.Credential, "machine.endpoint_unauthorized", "the machine does not authorize this host's key: %s", status.Convert(err).Message())
	case status.Code(err) != codes.Unimplemented:
		return machines.Transport(err)
	}
	connection, problem := m.machines.DialEndpoint(ctx, *ep)
	if problem != nil {
		return problem
	}
	defer connection.Close()
	if !connection.Seen.Workspace.Load().RunOutputLog || !connection.Seen.Workspace.Load().SubmissionClose {
		return exit.Named(exit.Structural, "machine.endpoint_capability_unavailable", "explicit endpoint lacks durable execution output or submission closure")
	}
	return nil
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
func (m *machineRuns) connectAtHub(ctx context.Context, name, hub, holder string) (*machineConnection, *exit.Error) {
	return m.connectAt(ctx, name, hub, holder, true)
}

func (m *machineRuns) connectAt(ctx context.Context, name, hub, holder string, named bool) (*machineConnection, *exit.Error) {
	machine, problem := m.machines.DialAt(ctx, name, hub, holder, named)
	if problem != nil {
		return nil, problem
	}
	return &machineConnection{Machine: machine, runs: m, installed: map[string]*pb.InstalledPackage{}, placements: map[string]*pb.DesiredPlacementSet{}}, nil
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

// Prewarm makes one installation on its machine without running anything: `cozy package
// install`, `cozy model download` and `cozy model upload`. A cozy.machine.v1 machine takes it
// as a warm run; any other prepares it over worker.v1, which takes no uploads.
func (m *machineRuns) Prewarm(ctx context.Context, row records.RentalInstall, report func(machines.InstallProgress)) (json.RawMessage, *exit.Error) {
	result, problem := m.prewarmV1(ctx, row, report)
	if problem != errNotV1 {
		return result, problem
	}
	selection := row.Selection
	if selection.Destination != "" {
		return nil, exit.Named(exit.Structural, "machine.upload_unsupported", "this machine takes no model uploads; %s", machines.RuntimeUpdate(row.RentalID))
	}
	models := orchestrator.DownloadModelRefs(selection.Models)
	if len(models) != len(selection.Models) {
		return nil, exit.New(exit.Validation, "rental installation contains non-downloadable model selections")
	}
	return nil, m.prewarmWorker(ctx, row.RentalID, selection.Hub, row.WorkerBootID, selection.Package, selection.Release, models, report)
}

// prewarmWorker is Prewarm over worker.v1. bootID, when set, is the worker lifetime the
// selection was queued for.
func (m *machineRuns) prewarmWorker(ctx context.Context, machine, hub, bootID, pkg, release string, models []*pb.DownloadModelRef, report func(machines.InstallProgress)) *exit.Error {
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
	_, problem = readMachinePreparationEvent(stream, func(event *pb.PrepareEvent) {
		report(machines.InstallProgress{Stage: strings.ToLower(strings.TrimPrefix(event.Stage.String(), "PREPARE_STAGE_")),
			TotalBytes: event.TotalBytes, TransferredBytes: event.TransferredBytes})
	})
	return problem
}

// Forget drops the kept connection to a machine, so a rental's credentials can be removed.
func (m *machineRuns) Forget(machine string) { m.machines.Forget(machine) }

// Status reads one machine's picture over cozy.machine.v1 as its owner.
func (m *machineRuns) Status(ctx context.Context, machine string) (api.MachineStatus, *exit.Error) {
	connection, problem := m.machines.DialV1(ctx, machine, "reading its status")
	if problem != nil {
		return api.MachineStatus{}, problem
	}
	defer connection.Close()
	frame, err := connection.Status(ctx)
	if err != nil {
		return api.MachineStatus{}, machines.Transport(err)
	}
	return statusOf(frame), nil
}

// machineLogs maps the log names clients use to the logs machines keep: the name
// cozy.machine.v1 Read takes, and worker.v1's.
var machineLogs = map[string]struct {
	name string
	kept pb.MachineLog
}{"tensorfs": {"tensorfs-transport", pb.MachineLog_MACHINE_LOG_TENSORFS_TRANSPORT}}

// MachineLog reads one log a machine keeps, with cozy.machine.v1 Read. A machine serving no
// v1 (the Go agent) is read on worker.v1 (wire 72); one predating that answers a note in
// Unavailable, never a failure.
func (m *machineRuns) MachineLog(ctx context.Context, machine, log string, tailBytes uint64) (api.MachineLog, *exit.Error) {
	kept, ok := machineLogs[log]
	if !ok {
		return api.MachineLog{}, exit.Named(exit.NotFound, "machine.log_unknown", "machines keep no log %q", log)
	}
	out := api.MachineLog{Log: log}
	var text strings.Builder
	v1, problem := m.machines.DialV1(ctx, machine, "reading its logs")
	if problem != nil {
		return api.MachineLog{}, problem
	}
	_, _, err := v1.ReadLog(ctx, kept.name, tailBytes, &text)
	v1.Close()
	if err == nil {
		out.Text = text.String()
		return out, nil
	}
	if status.Code(err) != codes.Unimplemented {
		return api.MachineLog{}, machines.Transport(err)
	}
	connection, problem := m.connect(ctx, machine, "reading its logs")
	if problem != nil {
		return api.MachineLog{}, problem
	}
	defer connection.Close()
	text.Reset()
	stream, err := connection.Host.ReadMachineLog(ctx, &pb.MachineLogQuery{Claim: connection.Claim, Log: kept.kept, TailBytes: tailBytes})
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
	case status.Code(err) == codes.Unimplemented && !errors.As(err, new(*machinev1.Newer)):
		out.Unavailable = "this machine's agent predates reading its logs; a machine started on a newer agent has them"
		return out, nil
	}
	return api.MachineLog{}, machineTransport(err)
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

// statusOf is a machine's Status frame as the daemon's API answers it.
func statusOf(frame *machinepb.StatusFrame) api.MachineStatus {
	out := api.MachineStatus{WorkerID: frame.GetWorkerId(), BootID: frame.GetBootId(), Agent: frame.GetVersion(),
		Phase: frame.GetPhase(), Runtime: frame.GetRuntime(), TensorFS: frame.GetTensorfs(),
		Capabilities: frame.GetCapabilities(), IdleDeadlineUnixMS: frame.GetIdleDeadlineUnixMs(),
		GPUs: []api.MachineGPU{}, Runs: []api.MachineRun{}, Environments: []api.MachineEnvironment{},
		DiskTotalBytes: frame.GetDisk().GetTotalBytes(), DiskFreeBytes: frame.GetDisk().GetFreeBytes()}
	for _, gpu := range frame.GetGpus() {
		out.GPUs = append(out.GPUs, api.MachineGPU{Index: gpu.GetIndex(), Name: gpu.GetName(), MemoryBytes: gpu.GetMemoryBytes(), Driver: gpu.GetDriver()})
	}
	for _, run := range frame.GetRuns() {
		out.Runs = append(out.Runs, api.MachineRun{ID: run.GetId(), Number: run.GetNumber(), State: run.GetState()})
	}
	for _, env := range frame.GetEnvironments() {
		out.Environments = append(out.Environments, api.MachineEnvironment{Installation: env.GetInstallation(), Package: env.GetPackage(),
			Release: env.GetRelease(), Level: env.GetLevel()})
	}
	for _, item := range frame.GetWarm() {
		level := strings.ToLower(strings.TrimPrefix(item.GetLevel().String(), "WARM_LEVEL_"))
		out.Warm = append(out.Warm, api.MachineWarmMember{Package: either(item.GetRelease().GetPackage(), item.GetInstallation()),
			Release: item.GetRelease().GetRelease(), Entrypoint: item.GetEntrypoint(), Level: level, Holds: item.GetHolds(), HeldBack: item.GetHeldBack()})
	}
	return out
}
