package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// A published inference root reaches its machine as one message: the release, the callable,
// the payload, the caller's explicit Model choices and its inputs. The machine installs what
// it lacks, resolves every slot for its own devices, prepares and mints the offer
// (worker-protocol release_roots); nothing here reads a ladder or prepares a package.

// releaseRoot answers whether a request is submitted by its release: a published serving
// root whose Models are only the caller's own choices.
func releaseRoot(request records.Request) bool {
	// A job's only transfer a release root carries is its `--upload-to` destination.
	transfer := request.ModelTransfer != nil && (!request.IsJob() || request.ModelTransfer.HasAcquisition() ||
		request.ModelTransfer.Destination == "")
	if request.LocalInstallationID != "" || strings.HasPrefix(request.Package, "local/") ||
		request.Trees != "" || transfer {
		return false
	}
	for _, model := range request.Models {
		if len(model.Adapters) > 0 || !model.Choice {
			return false
		}
	}
	return true
}

func (m *machineRuns) releaseRootSubmission(ctx context.Context, request records.Request, connection *machineConnection, workspace *pb.MachineExecutionWorkspace) (*pb.MachineExecutionSubmit, *exit.Error) {
	facts, problem := m.releaseFacts(ctx, request.Hub, connection, request.Package, request.Release)
	if problem != nil {
		return nil, problem
	}
	edges, problem := m.releaseEdges(ctx, request.Hub, connection, request.Package, request.Release)
	if problem != nil {
		return nil, problem
	}
	origin := rental.PublicOrigin(facts.LockedRequirements, request.Package)
	if origin == "" {
		if held, ok := m.origins.Load(connection.Name); ok {
			origin = held.(string)
		} else if origin, problem = connection.PublicOrigin(ctx); problem != nil {
			return nil, problem
		} else {
			m.origins.Store(connection.Name, origin)
		}
	}
	root := &pb.ReleaseRoot{Package: request.Package, Release: request.Release, Entrypoint: request.Entrypoint,
		Callees: edges, DeadlineUnixMs: uint64(max(request.DeadlineUnixMS, 0)), AttentionKernel: request.AttentionKernel,
		CatalogOrigin: origin}
	if request.IsJob() {
		root.Job, root.PublicationGrant = true, home.ScratchRepo(request.Org, request.ID)
		if request.ModelTransfer != nil {
			root.WeightsDestination = request.ModelTransfer.Destination
		}
	}
	for _, model := range request.Models {
		parameter := model.Slot[strings.LastIndex(model.Slot, ".")+1:]
		if model.Source != "" {
			root.Models = append(root.Models, &pb.ModelChoice{Parameter: parameter, Source: model.Source,
				Profiles: model.Profiles})
			continue
		}
		choice := &pb.ModelChoice{Parameter: parameter,
			Repository: model.Model, Release: model.Release, Lane: model.Lane}
		if model.Manifest != "" {
			digest, err := canonical.Raw(model.Manifest)
			if err != nil {
				return nil, exit.Named(exit.Validation, "model.manifest_invalid", "%s names no exact manifest", model.Slot)
			}
			choice.Manifest = &pb.Ref{Digest: digest, Length: uint64(max(model.ManifestLength, 0))}
		}
		root.Models = append(root.Models, choice)
	}
	sort.Slice(root.Models, func(i, j int) bool { return root.Models[i].Parameter < root.Models[j].Parameter })
	if request.Capture != "" {
		root.Capture = &pb.ActivationCapture{}
		if err := json.Unmarshal([]byte(request.Capture), root.Capture); err != nil {
			return nil, exit.New(exit.Validation, "recorded capture options are invalid")
		}
	}
	began := time.Now()
	access, problem := m.stageMachineInputs(ctx, request, connection)
	if problem != nil {
		return nil, problem
	}
	if root.InputAccess = access; len(access) > 0 {
		m.submissionStage(request.ID, "inputs", fmt.Sprintf("%d input(s)", len(access)), began)
	}
	for _, asset := range request.Assets {
		root.Inputs = append(root.Inputs, &pb.InputBinding{InputId: asset.FieldPath, Digest: asset.Digest,
			Length: uint64(asset.Length), KindMime: asset.MediaType, Order: asset.Order})
	}
	return &pb.MachineExecutionSubmit{SubmissionId: records.MachineSubmissionID(request.IdemKey),
		Offer: &pb.AttemptOffer{RequestId: request.ID}, PayloadCanonicalBytes: request.Payload,
		ReleaseRoot: root}, nil
}

// sendReleaseRoot replays the one frozen submission until the machine answers a receipt. The
// machine answers while it prepares with its progress, and names any release it holds no
// installation of, whose facts then ride the same submission.
func (m *machineRuns) sendReleaseRoot(ctx context.Context, request records.Request, connection *machineConnection, frozen *pb.MachineExecutionSubmit) *exit.Error {
	var installs []*pb.DeferredInstallation
	progress := ""
	for {
		submission := proto.Clone(frozen).(*pb.MachineExecutionSubmit)
		submission.SourceCredentials, submission.Claim = m.resolver.SourceCredentials(), connection.Claim
		submission.Offer.WorkerBootId, submission.Offer.RecordOwnerEpoch = connection.Claim.WorkerBootId, connection.Claim.RecordOwnerEpoch
		submission.ReleaseRoot.Installations = installs
		var trailer metadata.MD
		receipt, err := connection.Host.SubmitMachineExecution(ctx, submission, grpc.Trailer(&trailer))
		if err == nil {
			if receipt == nil || receipt.WorkerId != connection.Claim.WorkerId || receipt.WorkerBootId == "" {
				return exit.New(exit.Conflict, "execution was accepted by an unexpected worker")
			}
			return m.store.AcceptMachineExecution(request.ID, receipt)
		}
		codeOf := trailer.Get("cozy-error-code")
		switch {
		case slices.Contains(codeOf, "release_root_preparing"):
			if message := status.Convert(err).Message(); message != progress {
				progress = message
				_ = m.store.AppendEvent(request.ID, "request.preparing", 0, map[string]any{"stage": "machine", "detail": message})
			}
		case status.Code(err) == codes.DeadlineExceeded && ctx.Err() == nil:
			// The Host bounds one admission; the machine keeps preparing: ask again.
		case slices.Contains(codeOf, "release_root_installations_absent"):
			absent := trailer.Get("cozy-absent-release")
			if len(absent) == 0 || len(installs) > 0 {
				return exit.Named(exit.Conflict, "machine_execution.install_facts_refused",
					"the machine still lacks install facts for %s", strings.Join(absent, ", "))
			}
			for _, key := range absent {
				pkg, release, _ := strings.Cut(key, "@")
				row, problem := m.deferredInstallation(ctx, request.Hub, connection, pkg, release)
				if problem != nil {
					return problem
				}
				installs = append(installs, row)
			}
		default:
			if slices.Contains(codeOf, "execution_workspace_changed") {
				connection.KeepWorkspace(nil)
			}
			return m.submissionRefused(request.ID, trailer, err)
		}
	}
}

// deferredInstallation is what a machine installs a release from: its hub facts.
func (m *machineRuns) deferredInstallation(ctx context.Context, origin string, connection *machineConnection, pkg, release string) (*pb.DeferredInstallation, *exit.Error) {
	facts, problem := m.releaseFacts(ctx, origin, connection, pkg, release)
	if problem != nil {
		return nil, problem
	}
	downloads, problem := rental.DownloadSet([]*pb.DownloadPackageRef{{Package: pkg, Release: release}}, nil)
	if problem != nil {
		return nil, problem
	}
	return &pb.DeferredInstallation{Key: pkg + "@" + release, Package: pkg, Release: release,
		Preparation: &pb.PreparePackageSetRequest{DownloadDelegation: downloads, Application: facts.Application,
			ModelSlotPaths: facts.ModelSlotPaths, ImageInventory: facts.ImageInventory,
			LockedRequirements: facts.LockedRequirements, PythonRequires: facts.PythonRequires,
			PythonVersion: facts.PythonVersion, PackageInterface: facts.PackageInterface}}, nil
}

// releaseEdges is the root's published callee closure: each release's own lock names the
// Tensorhub packages it calls. Every document read here is immutable and kept.
func (m *machineRuns) releaseEdges(ctx context.Context, origin string, connection *machineConnection, pkg, release string) ([]*pb.ReleaseEdge, *exit.Error) {
	var edges []*pb.ReleaseEdge
	seen := map[string]bool{}
	var walk func(string, string) *exit.Error
	walk = func(pkg, release string) *exit.Error {
		key := pkg + "@" + release
		if seen[key] {
			return nil
		}
		if seen[key] = true; len(seen) > 128 {
			return exit.New(exit.Validation, "published callable closure exceeds its bound")
		}
		ref, problem := hub.ParseRef(pkg)
		if problem != nil {
			return problem
		}
		plan, problem := m.resolver.catalog(origin).PackageDownloads(ctx, ref, release)
		if problem != nil {
			return problem
		}
		lock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
		if problem != nil {
			return problem
		}
		facts, problem := m.releaseFacts(ctx, origin, connection, pkg, release)
		if problem != nil {
			return problem
		}
		callees, problem := install.PublishedDependencies(pkg, lock.Bytes, facts.LockedRequirements)
		if problem != nil {
			return problem
		}
		for _, callee := range callees {
			edges = append(edges, &pb.ReleaseEdge{Caller: key, Callee: callee.Package + "@" + callee.Version})
			if problem := walk(callee.Package, callee.Version); problem != nil {
				return problem
			}
		}
		return nil
	}
	if problem := walk(pkg, release); problem != nil {
		return nil, problem
	}
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].Caller+"\x00"+edges[i].Callee < edges[j].Caller+"\x00"+edges[j].Callee
	})
	return edges, nil
}

// releaseFacts are one release's install facts for one machine: immutable (the release and
// the machine's image never change), so they are read from the hub once and kept.
func (m *machineRuns) releaseFacts(ctx context.Context, origin string, connection *machineConnection, pkg, release string) (orchestrator.PrepareFacts, *exit.Error) {
	name := sha256.Sum256([]byte(origin + "\x00" + connection.Name + "\x00" + connection.RentalID() + "\x00" + pkg + "@" + release))
	path := filepath.Join(m.layout.Root, "releases", "facts", hex.EncodeToString(name[:16])+".json")
	var facts orchestrator.PrepareFacts
	if raw, err := os.ReadFile(path); err == nil && json.Unmarshal(raw, &facts) == nil && facts.Application != "" {
		return facts, nil
	}
	facts, problem := connection.PrepareFacts(ctx, &pb.DownloadPackageRef{Package: pkg, Release: release})
	if problem != nil {
		return facts, problem
	}
	if raw, err := json.Marshal(facts); err == nil && os.MkdirAll(filepath.Dir(path), 0o700) == nil {
		if os.WriteFile(path+".tmp", raw, 0o600) == nil {
			_ = os.Rename(path+".tmp", path)
		}
	}
	return facts, nil
}

// submissionRefused names a failed release-root submission; a definitive refusal is recorded.
func (m *machineRuns) submissionRefused(requestID string, trailer metadata.MD, err error) *exit.Error {
	for _, code := range trailer.Get("cozy-error-code") {
		switch code {
		case "execution_workspace_changed", "execution_workspace_required":
			return exit.Named(exit.Conflict, "machine_execution.workspace_changed", "execution workspace no longer matches the frozen submission; prior acceptance remains unresolved")
		case "execution_submission_refused":
			return m.unaccepted(requestID, err)
		}
	}
	return machineTransport(err)
}

// compatibleModels readies a release root for a machine whose Runtime cannot take one: the
// client resolves each slot as it did before (the caller's choice, else the owner's binding
// or the authored default, as a ladder the machine's devices then pin) and the call goes the
// preparation path. It says so on the run, with how to reach the one-message path.
func (m *machineRuns) compatibleModels(ctx context.Context, request records.Request, connection *machineConnection) (records.Request, *exit.Error) {
	_ = m.store.AppendEvent(request.ID, "request.preparing", 0, map[string]any{"stage": "machine",
		"detail": "this machine's Runtime lacks release roots; used the compatibility path (" + machines.RuntimeUpdate(connection.Name) + " for the fast path)"})
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return request, problem
	}
	catalog := m.resolver.catalog(request.Hub)
	detail, problem := catalog.PackageRelease(ctx, ref, request.Release)
	if problem != nil {
		return request, problem
	}
	surface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return request, problem
	}
	entrypoint, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return request, problem
	}
	chosen := map[string]records.ModelRef{}
	for _, model := range request.Models {
		chosen[model.BindingSlot()] = model
	}
	models := make([]records.ModelRef, 0, len(entrypoint.Models))
	for _, slot := range entrypoint.Models {
		choice, explicit := chosen[slot.Path]
		if !explicit {
			selected, problem := m.resolver.childModelLadder(request.Hub, request.Package, request.Entrypoint, slot)
			if problem != nil {
				return request, problem
			}
			selected.Slot = slot.Path
			models = append(models, selected)
			continue
		}
		spec := choice.Model
		if choice.Manifest != "" {
			spec += "@" + choice.Manifest
		} else if choice.Release != "" {
			spec += "@" + choice.Release
		}
		lane := choice.Lane
		if choice.Manifest != "" {
			lane = ""
		}
		resolved, problem := catalog.ResolveModel(ctx, spec, lane)
		if problem != nil {
			return request, problem
		}
		model := records.ModelRef{Package: request.Package, Slot: slot.Path, BindingPath: slot.Path,
			Model: resolved.Model, CatalogRepository: resolved.Model, Release: resolved.Release, Lane: resolved.Lane,
			Manifest: resolved.ManifestID, ManifestLength: resolved.ManifestLength, Bytes: resolved.Bytes,
			ComponentBytes: resolved.ComponentBytes, ComponentUse: slot.ComponentUse}
		if choice.Manifest != "" && choice.Release == "" {
			// A checkpoint named by digest alone is downloaded as that checkpoint, whatever
			// release (if any) names it: without this the machine is never asked for it.
			model.Release, model.Lane, model.HubCheckpoint = "", "", true
		}
		models = append(models, model)
	}
	if problem := m.store.PinMachineModels(request.ID, models); problem != nil {
		return request, problem
	}
	request.Models = models
	return request, nil
}

// compatibleJobModels readies a job for a machine whose Runtime cannot take it by its
// release: its slots are resolved as the client resolved them before (the same selection
// and exact manifest reads), and the job goes the prepared path. It says so on the run.
func (m *machineRuns) compatibleJobModels(ctx context.Context, request records.Request, connection *machineConnection) (records.Request, *exit.Error) {
	_ = m.store.AppendEvent(request.ID, "request.preparing", 0, map[string]any{"stage": "machine",
		"detail": "this machine lacks release_root_jobs; used the prepared job path (" + machines.RuntimeUpdate(connection.Name) + " for the fast path)"})
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return request, problem
	}
	detail, problem := m.resolver.catalog(request.Hub).PackageRelease(ctx, ref, request.Release)
	if problem != nil {
		return request, problem
	}
	surface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return request, problem
	}
	job, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return request, problem
	}
	overrides := make(map[string]string, len(request.Models))
	for _, model := range request.Models {
		overrides[model.BindingSlot()] = choiceSpec(model)
	}
	resolving := &Context{Cfg: m.resolver.cfg.ForHub(request.Hub), Err: io.Discard}
	target := Target{Package: request.Package, Function: request.Entrypoint, Release: request.Release}
	selected, problem := invocationModelSpecs(resolving, target, job, overrides)
	if problem != nil {
		return request, problem
	}
	models, problem := resolveSelectedInvocationModels(resolving, target, job, selected)
	if problem == nil {
		models, problem = jobManifestInputs(resolving, job, models)
	}
	if problem != nil {
		return request, problem
	}
	if problem := m.store.PinMachineModels(request.ID, models); problem != nil {
		return request, problem
	}
	request.Models = models
	return request, nil
}

// choiceSpec spells a recorded model choice in the one model-ref grammar.
func choiceSpec(model records.ModelRef) string {
	spec := model.Model
	if model.Release != "" {
		spec += "@" + model.Release
		if model.Lane != "" {
			spec += "/" + model.Lane
		}
	} else if model.Lane != "" {
		spec += "#" + model.Lane
	}
	if model.Manifest != "" && (model.Release != "" || model.Lane == "") {
		spec += "#" + model.Manifest
	}
	return spec
}
