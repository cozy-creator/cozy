package orchestrator

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// LocalServingPreparation names package metadata retained by the install. It is
// launch configuration, not a guessed PlacementSet or an invocation binding.
type LocalServingPreparation struct {
	PythonVersion      string   `json:"python_version"`
	PythonRequires     string   `json:"python_requires"`
	Published          bool     `json:"published"`
	Application        string   `json:"application"`
	ModelSlotPaths     []string `json:"model_slot_paths"`
	PackageInterface   []byte   `json:"package_interface"`
	LockedRequirements string   `json:"locked_requirements"`
}

// prepareLocalServing uses the same Runtime preparation messages as the rental
// host, over this per-install worker's existing local preparation connection.
func (c *Orchestrator) prepareLocalServing(req records.Request, spec WorkerLaunchSpec) (WorkerLaunchSpec, string, *exit.Error) {
	instance, _, problem := c.EnsureWorker(spec)
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if problem := c.ensureWorkerClaimed(instance); problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	w, s, problem := c.localControl(instance)
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if s.preparation == nil || w.spec.Connection != nil || spec.Preparation == nil {
		return WorkerLaunchSpec{}, "", exit.Internalf("local serving preparation has no local worker connection")
	}
	if problem := requireMixedModelInputs(s, mixedModelInputs(req.Models)); problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if req.ParentRequestID != "" && len(downloadModelRefs(req.Models)) > 0 {
		acquirer, ok := c.opt.Packages.(interface{ EnsureLocalModels([]ModelRef) *exit.Error })
		if !ok {
			return WorkerLaunchSpec{}, "", exit.Unavailablef("local model acquisition owner is unavailable")
		}
		if problem := acquirer.EnsureLocalModels(req.Models); problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
	}
	w.localMu.Lock()
	defer w.localMu.Unlock()
	logical := LogicalPackage{Package: req.Package, Release: spec.Placement.Release, Function: req.Entrypoint,
		PlanID: req.PlanID, Models: req.Models, Outputs: splitOutputs(req.Outputs)}
	c.mu.Lock()
	already := w.spec.Placement.PlacementSetDigest != "" && selectionServes(req.Models, w.spec.Placement.Models) &&
		(req.LocalInstallationID == "" || req.LocalInstallationID == w.spec.Placement.InstallationID) && w.desiredRefusal == nil
	preparedSpec := w.spec
	c.mu.Unlock()
	if already && req.ParentRequestID == "" {
		plan, problem := preparedSpec.Placement.EntrypointDigest(req.Entrypoint)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		return preparedSpec, plan, nil
	}
	prep := spec.Preparation
	var result *pb.PreparePackageSetResult
	var rpcError error
	var trailer metadata.MD
	c.ObservePhase(instance, PhaseSample{Name: PhasePreparing, Detail: req.Package})
	if prep.Published {
		locked, err := os.ReadFile(prep.LockedRequirements)
		if err != nil {
			return WorkerLaunchSpec{}, "", exit.Internalf("cannot read retained package requirements: %s", err)
		}
		selected, problem := localDownloadSelection(req.Models, []*pb.DownloadPackageRef{{Package: req.Package, Release: logical.Release}})
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		root := spec.InstallRoot
		if root == "" {
			return WorkerLaunchSpec{}, "", exit.New(exit.Structural, "published local preparation has no worker environment store")
		}
		if problem := requireAdapterDownloadPeer(s, selected); problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		result, rpcError = s.preparation.PreparePackageSet(s.ctx, &pb.PreparePackageSetRequest{
			PythonRequires: prep.PythonRequires, PythonVersion: prep.PythonVersion, InstallRoot: root, DownloadDelegation: selected, Application: prep.Application,
			ModelSlotPaths: prep.ModelSlotPaths, LockedRequirements: locked,
			PackageInterface: append([]byte(nil), prep.PackageInterface...),
		})
	} else {
		revision, problem := c.opt.Packages.LocalInstallation(req.InstallID, req.LocalInstallationID)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		if revision.Package != req.Package || revision.Release != logical.Release {
			return WorkerLaunchSpec{}, "", exit.New(exit.Conflict, "local installation differs from selected package")
		}
		operation := req.ID
		files, problem := stageLocalPreparationFiles(spec.InstallRoot, operation, revision)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		result, rpcError = s.preparation.PrepareLocalPackage(s.ctx, &pb.PrepareLocalPackageRequest{
			PythonRequires: revision.PythonRequires, PythonVersion: revision.PythonVersion, InstallRoot: spec.InstallRoot, OperationId: operation,
			Package: &pb.DevelopmentPackage{Package: revision.Package, Release: revision.Release, InstallationId: revision.ID},
			Files:   files, SourceArchive: revision.SourceArchive, DependencyRequirements: append([]byte(nil), revision.DependencyRequirements...),
		}, grpc.Trailer(&trailer))
		if rpcError == nil && len(req.Models) > 0 {
			call := &pb.PreparePrivatePlacementRequest{OperationId: operation, InstallationId: revision.ID, Claim: s.claim}
			if req.ParentRequestID != "" {
				call.NativeModels, problem = c.nativeServingModels(req)
			}
			if problem == nil {
				downloads := make([]ModelRef, 0, len(req.Models))
				for _, model := range req.Models {
					if model.Downloadable() {
						downloads = append(downloads, model)
					}
				}
				if len(downloads) > 0 {
					call.DownloadDelegation, problem = localDownloadSelection(downloads, nil)
				}
			}
			if problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
			if problem := requireAdapterDownloadPeer(s, call.DownloadDelegation); problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
			if problem := requireAdapterPeer(s, hasModelAdapters(req.Models)); problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
			result, rpcError = s.preparation.PreparePrivatePlacement(s.ctx, call, grpc.Trailer(&trailer))
		}
	}
	if rpcError != nil {
		ended := classifyPrepareEnd(rpcError)
		if ended.err != nil {
			return WorkerLaunchSpec{}, "", exit.Unavailablef("local worker preparation interrupted: %s", ended.err)
		}
		return WorkerLaunchSpec{}, "", exit.Named(exit.Structural, "worker.prepare_refused", "local worker preparation refused: %s", ended.refusal).
			WithCause(runtimeErrorCode(trailer))
	}
	if result == nil || result.PlacementSet == nil {
		return WorkerLaunchSpec{}, "", exit.Internalf("local worker preparation returned no placement")
	}
	set := result.PlacementSet
	if !bytes.Equal(canonical.Digest(set.PlacementSetCanonicalBytes), set.PlacementSetDigest) {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Structural, "worker.prepare_identity_mismatch", "local worker placement changed its digest")
	}
	doc, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
	if err != nil || len(doc.List("placements")) != 1 {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Structural, "worker.prepare_document_invalid", "local worker returned an invalid placement")
	}
	desired, problem := PlacementFromExact(req.Package, req.InstallID, spellOf(set.PlacementSetDigest), set.PlacementSetCanonicalBytes, map[string][]string{req.Entrypoint: logical.Outputs})
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if desired.Release != logical.Release || (!prep.Published && desired.InstallationID != req.LocalInstallationID) ||
		(len(req.Models) > 0 && len(desired.Models) == 0) || !selectionServes(req.Models, desired.Models) {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "local_preparation_selection_changed", "worker placement differs from accepted code or model selections")
	}
	plan, problem := desired.EntrypointDigest(req.Entrypoint)
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if problem := c.opt.Store.BindRequestPlan(req.ID, plan); problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if err := os.MkdirAll(spec.ArtifactCache, 0o700); err != nil {
		return WorkerLaunchSpec{}, "", exit.Internalf("cannot create prepared placement cache: %s", err)
	}
	if err := os.WriteFile(filepath.Join(spec.ArtifactCache, desired.PlacementSetDigest[7:]), set.PlacementSetCanonicalBytes, 0o600); err != nil {
		return WorkerLaunchSpec{}, "", exit.Internalf("cannot retain worker-prepared placement: %s", err)
	}
	current, problem := c.opt.Store.RequestRow(req.ID)
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if current == nil || (current.State != "queued" && current.State != "submitted") {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "request.execution_stopped", "request stopped during local model preparation")
	}
	revision := c.nextRevision()
	c.mu.Lock()
	if c.sessions[s.bootID] != s || c.workers[instance] != w || w.stopping || w.exited {
		c.mu.Unlock()
		return WorkerLaunchSpec{}, "", exit.Unavailablef("local worker session changed during preparation")
	}
	w.hostPrepareSeq++
	seq := w.hostPrepareSeq
	w.revision, w.desiredRefusal = revision, nil
	w.spec.Placement = desired
	w.placementID = desired.PlacementID()
	w.planIDs = nil
	for _, entry := range desired.Entrypoints {
		w.planIDs = append(w.planIDs, entry.Digest)
	}
	sort.Strings(w.planIDs)
	preparedSpec = w.spec
	c.mu.Unlock()
	c.convergePrepared(s, w, seq, revision, "local serving", set)
	// The caller persists this exact binding and uses EnsurePlacementReady, the
	// ordinary local readiness path. Remote observations are not populated for a
	// locally spawned worker and cannot be used to wait for its activation.
	return preparedSpec, plan, nil
}

func splitOutputs(raw string) []string {
	if raw == "" {
		return nil
	}
	return strings.Split(raw, ",")
}

func localDownloadSelection(models []ModelRef, packages []*pb.DownloadPackageRef) ([]byte, *exit.Error) {
	selected := make([]*pb.DownloadModelRef, 0, len(models))
	for _, model := range models {
		if !model.Pinned() {
			return nil, exit.Named(exit.Validation, "local_model_selection_incomplete", "local preparation needs exact model manifests")
		}
		for _, slot := range append([]string{model.BindingSlot()}, model.SharedSlots...) {
			selected = append(selected, &pb.DownloadModelRef{Package: model.Package, Slot: slot, Model: model.Model, Release: model.Release, Lane: model.Lane, Manifest: model.Manifest, Adapters: downloadAdapters(model.Adapters)})
		}
	}
	sort.Slice(selected, func(i, j int) bool {
		return selected[i].Package+"/"+selected[i].Slot < selected[j].Package+"/"+selected[j].Slot
	})
	raw, _, err := canonical.Identity(&pb.DownloadDelegation{Models: selected, Packages: packages})
	if err != nil {
		return nil, exit.Internalf("cannot encode local model selection: %s", err)
	}
	return raw, nil
}

func stageLocalPreparationFiles(root, operation string, revision localpackage.Installation) ([]*pb.LocalPackageFile, *exit.Error) {
	directory := filepath.Join(root, ".stage", operation, "files")
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return nil, exit.Internalf("cannot stage local package files: %s", err)
	}
	rows := make([]*pb.LocalPackageFile, 0, len(revision.Files))
	for _, file := range revision.Files {
		destination := filepath.Join(directory, file.Filename)
		if filepath.Base(file.Filename) != file.Filename {
			return nil, exit.New(exit.Validation, "invalid staged file name")
		}
		if problem := copyLocalPreparationFile(file.Path, destination); problem != nil {
			return nil, problem
		}
		var digest []byte
		if file.Digest != "" {
			var err error
			digest, err = canonical.Raw(file.Digest)
			if err != nil {
				return nil, exit.New(exit.Validation, "invalid staged wheel integrity")
			}
		} else if file.Filename != revision.SourceArchive {
			return nil, exit.New(exit.Validation, "only private source may omit a wheel checksum")
		}
		rows = append(rows, &pb.LocalPackageFile{Digest: digest, Filename: file.Filename, Length: uint64(file.Length), Path: destination})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Filename < rows[j].Filename })
	return rows, nil
}

func copyLocalPreparationFile(source, destination string) *exit.Error {
	input, err := os.Open(source)
	if err != nil {
		return exit.Internalf("cannot read local installation file: %s", err)
	}
	defer input.Close()
	output, err := os.CreateTemp(filepath.Dir(destination), ".file-")
	if err != nil {
		return exit.Internalf("cannot stage local installation file: %s", err)
	}
	temporary := output.Name()
	defer os.Remove(temporary)
	_, err = io.Copy(output, input)
	if err == nil {
		err = output.Sync()
	}
	closed := output.Close()
	if err == nil {
		err = closed
	}
	if err != nil {
		return exit.Internalf("cannot retain local installation file: %s", err)
	}
	if err := os.Rename(temporary, destination); err != nil {
		return exit.Internalf("cannot commit staged local wheel: %s", err)
	}
	return nil
}
