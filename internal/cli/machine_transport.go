package cli

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

// machineConnection is one claimed machine, local or rented, and what this submission has
// prepared on it. Every call below is the same PodHost call whichever machine answers it.
type machineConnection struct {
	*machines.Machine
	runs         *machineRuns
	installed    map[string]*pb.InstalledPackage
	placements   map[string]*pb.DesiredPlacementSet // each captured revision's code-only placement
	publicOrigin string                             // renter-authenticated Hub facts name the public byte endpoint
	progress     *transfer.Progress
}

// connect opens the machine a run executes on; holder is what the run is doing there,
// which a rental's maintenance refusal names.
func (m *machineRuns) connect(ctx context.Context, name, holder string) (*machineConnection, *exit.Error) {
	machine, problem := m.machines.Dial(ctx, name, holder)
	if problem != nil {
		return nil, problem
	}
	return &machineConnection{Machine: machine, runs: m, installed: map[string]*pb.InstalledPackage{}, placements: map[string]*pb.DesiredPlacementSet{}}, nil
}

func (c *machineConnection) modelDefaultOrigin(ctx context.Context) (string, *exit.Error) {
	return c.PublicOrigin(ctx)
}

func (c *machineConnection) preparePublished(ctx context.Context, request records.Request) (*publishedPreparation, *exit.Error) {
	ref := &pb.DownloadPackageRef{Package: request.Package, Release: request.Release}
	facts, problem := c.PrepareFacts(ctx, ref)
	if problem != nil {
		return nil, problem
	}
	c.publicOrigin = rental.PublicOrigin(facts.LockedRequirements, request.Package)
	// The package's exact model bindings ride its own download set, so TensorFS holds
	// each checkpoint before execution admits it. A published callee's bindings ride
	// the callee's preparation (capturePublishedDependencies).
	downloads, problem := rental.DownloadSet([]*pb.DownloadPackageRef{ref}, orchestrator.DownloadModelRefs(request.PreparedModels()))
	if problem != nil {
		return nil, problem
	}
	stream, err := c.Host.PreparePackageSet(ctx, &pb.PreparePackageSetCall{
		Claim: c.Claim, SupportsModelMaterializationRecovery: true, PackageSet: &pb.DesiredPackageSet{DownloadDelegation: downloads},
		Application: facts.Application, ModelSlotPaths: facts.ModelSlotPaths,
		PythonRequires: facts.PythonRequires, PythonVersion: facts.PythonVersion, ImageInventory: facts.ImageInventory, LockedRequirements: facts.LockedRequirements,
		PackageInterface: facts.PackageInterface,
	})
	if err != nil {
		return nil, machineTransport(err)
	}
	observed := c.runs.preparationPhase(request.ID, request.Package+"@"+request.Release)
	event, problem := readMachinePreparationEvent(stream, observed.observe)
	if problem != nil {
		return nil, problem
	}
	return &publishedPreparation{DesiredPlacementSet: event.PlacementSet, InstalledPackage: event.InstalledPackage, LockedRequirements: facts.LockedRequirements, Retained: observed.retained}, nil
}

func (c *machineConnection) retainModel(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
	return c.Host.RetainDerivedResult(ctx, &pb.DerivedRetentionCall{Claim: c.Claim, Request: request})
}

func (c *machineConnection) releaseModel(ctx context.Context, request *pb.DerivedRetentionRequest) (*pb.DerivedRetentionResult, error) {
	return c.Host.ReleaseDerivedRetention(ctx, &pb.DerivedRetentionCall{Claim: c.Claim, Request: request})
}

func (c *machineConnection) retainBytes(ctx context.Context, request *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error) {
	return c.Host.RetainByteTree(ctx, &pb.NativeByteRetentionCall{Claim: c.Claim, Request: request})
}

func (c *machineConnection) releaseBytes(ctx context.Context, request *pb.NativeByteRetentionRequest) (*pb.NativeByteRetentionResult, error) {
	return c.Host.ReleaseByteTree(ctx, &pb.NativeByteRetentionCall{Claim: c.Claim, Request: request})
}

func (c *machineConnection) importInputTree(ctx context.Context) (grpc.ClientStreamingClient[pb.InputTreeImportFrame, pb.NativeByteRetentionResult], error) {
	return c.Host.ImportInputTree(ctx)
}

func (c *machineConnection) readBytes(ctx context.Context, source *pb.NativeByteRetentionRequest, object *pb.Ref) (machineByteStream, error) {
	return c.Host.ReadByteTreeObject(ctx, &pb.NativeByteReadCall{Claim: c.Claim, Source: source, Object: object})
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
		Claim: c.Claim, SupportsModelMaterializationRecovery: true, PrivatePlacementSet: selected,
	})
	if err != nil {
		return nil, machineTransport(err)
	}
	return readMachinePreparedSet(stream, c.runs.preparationPhase(request.ID, revision.Package).observe)
}

// Prewarm prepares a published package, its selected models, or models alone on a machine
// without running anything: `cozy package install` and `cozy model download` for any
// machine. bootID, when set, is the worker lifetime the selection was queued for.
func (m *machineRuns) Prewarm(ctx context.Context, machine, bootID, pkg, release string, models []*pb.DownloadModelRef) *exit.Error {
	connection, problem := m.connect(ctx, machine, "installing "+either(pkg, "models"))
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
	call := &pb.PreparePackageSetCall{Claim: connection.Claim, SupportsModelMaterializationRecovery: true}
	if pkg != "" {
		ref := &pb.DownloadPackageRef{Package: pkg, Release: release}
		facts, problem := connection.PrepareFacts(ctx, ref)
		if problem != nil {
			return problem
		}
		packages = append(packages, ref)
		call.Application, call.ModelSlotPaths, call.ImageInventory = facts.Application, facts.ModelSlotPaths, facts.ImageInventory
		call.PythonRequires, call.PythonVersion, call.LockedRequirements = facts.PythonRequires, facts.PythonVersion, facts.LockedRequirements
		call.PackageInterface = facts.PackageInterface
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
