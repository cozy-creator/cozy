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
	"google.golang.org/protobuf/proto"
)

type localPreparedCode struct {
	operation, revision, base string
	result                    *pb.PreparePackageSetResult
}

// LocalServingPreparation names package metadata retained by the install. It is
// launch configuration, not a guessed PlacementSet or an invocation binding.
type LocalServingPreparation struct {
	Published              bool     `json:"published"`
	Application            string   `json:"application"`
	ModelSlotPaths         []string `json:"model_slot_paths"`
	PackageInterfaceDigest string   `json:"package_interface_digest"`
	LockedRequirements     string   `json:"locked_requirements"`
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
		(req.LocalPackageDigest == "" || req.LocalPackageDigest == w.spec.Placement.LocalRevisionDigest) && w.desiredRefusal == nil
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
	var preparedCode *localPreparedCode
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
			root = filepath.Dir(filepath.Dir(spec.EnvironmentPython))
		}
		result, rpcError = s.preparation.PreparePackageSet(s.ctx, &pb.PreparePackageSetRequest{
			InstallRoot: root, DownloadDelegation: selected, Application: prep.Application,
			ModelSlotPaths: prep.ModelSlotPaths, LockedRequirements: locked,
		})
	} else {
		revision, problem := c.opt.Packages.LocalRevision(req.InstallID, req.LocalPackageDigest)
		if problem != nil {
			return WorkerLaunchSpec{}, "", problem
		}
		if revision.Package != req.Package || revision.Release != logical.Release || revision.PackageInterfaceDigest != prep.PackageInterfaceDigest {
			return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "local_package_revision_changed", "local serving revision differs from its install")
		}
		digest, _ := canonical.Raw(revision.Digest)
		operation := req.ID
		base, baseProblem := numericalEnvironment(s)
		if prior := w.localCode; baseProblem == nil && prior != nil && prior.base == base && prior.revision == revision.Digest {
			operation = prior.operation
			result = proto.Clone(prior.result).(*pb.PreparePackageSetResult)
			c.logf("%s reuses prepared code operation %s", req.ID, operation)
		} else {
			files, problem := stageLocalPreparationWheels(spec.InstallRoot, operation, revision)
			if problem != nil {
				return WorkerLaunchSpec{}, "", problem
			}
			source, _ := canonical.Raw(revision.SourceDigest)
			result, rpcError = s.preparation.PrepareLocalPackage(s.ctx, &pb.PrepareLocalPackageRequest{
				InstallRoot: spec.InstallRoot, OperationId: operation,
				Package: &pb.DevelopmentPackage{Package: revision.Package, Release: revision.Release, SourceDigest: source, LocalRevisionDigest: digest},
				Wheels:  files, DependencyRequirements: append([]byte(nil), revision.DependencyRequirements...),
			})
			if rpcError == nil && result != nil && baseProblem == nil {
				preparedCode = &localPreparedCode{operation: operation, revision: revision.Digest, base: base, result: proto.Clone(result).(*pb.PreparePackageSetResult)}
			}
		}
		if rpcError == nil && len(req.Models) > 0 {
			call := &pb.PreparePrivatePlacementRequest{OperationId: operation, LocalRevisionDigest: digest, Claim: s.claim}
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
			result, rpcError = s.preparation.PreparePrivatePlacement(s.ctx, call)
		}
	}
	if rpcError != nil {
		ended := classifyPrepareEnd(rpcError)
		if ended.err != nil {
			return WorkerLaunchSpec{}, "", exit.Unavailablef("local worker preparation interrupted: %s", ended.err)
		}
		return WorkerLaunchSpec{}, "", exit.Named(exit.Structural, "worker.prepare_refused", "local worker preparation refused: %s", ended.refusal)
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
	row := doc.List("placements")[0]
	if row.Sub("package_interface").Str("digest") != prep.PackageInterfaceDigest {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "package_interface_mismatch", "worker package interface differs from retained install")
	}
	desired, problem := PlacementFromExact(req.Package, req.InstallID, spellOf(set.PlacementSetDigest), set.PlacementSetCanonicalBytes, map[string][]string{req.Entrypoint: logical.Outputs})
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if desired.Release != logical.Release || (!prep.Published && desired.LocalRevisionDigest != req.LocalPackageDigest) ||
		(len(req.Models) > 0 && len(desired.Models) == 0) || !selectionServes(req.Models, desired.Models) {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "local_preparation_selection_changed", "worker placement differs from accepted code or model selections")
	}
	plan, problem := desired.EntrypointDigest(req.Entrypoint)
	if problem != nil {
		return WorkerLaunchSpec{}, "", problem
	}
	if req.PlanID != "" && req.PlanID != plan {
		return WorkerLaunchSpec{}, "", exit.Named(exit.Conflict, "plan_mismatch", "worker constructed a different binding than the requested plan")
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
	if preparedCode != nil {
		if base, problem := numericalEnvironment(s); problem == nil && base == preparedCode.base {
			w.localCode = preparedCode
		}
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
		path := model.Slot
		if model.BindingPath != "" {
			path = model.BindingPath
		}
		for _, slot := range append([]string{path}, model.SharedSlots...) {
			selected = append(selected, &pb.DownloadModelRef{Package: model.Package, Slot: slot, Model: model.Model, Release: model.Release, Lane: model.Lane, Manifest: model.Manifest})
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

func stageLocalPreparationWheels(root, operation string, revision localpackage.Revision) ([]*pb.LocalPackageWheel, *exit.Error) {
	directory := filepath.Join(root, ".stage", operation, "wheels")
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return nil, exit.Internalf("cannot stage local package wheels: %s", err)
	}
	rows := make([]*pb.LocalPackageWheel, 0, len(revision.Files))
	for _, file := range revision.Files {
		destination := filepath.Join(directory, file.Filename)
		if filepath.Base(file.Filename) != file.Filename {
			return nil, exit.New(exit.Validation, "invalid staged wheel name")
		}
		if problem := copyLocalPreparationWheel(file.Path, destination); problem != nil {
			return nil, problem
		}
		digest, err := canonical.Raw(file.Digest)
		if err != nil {
			return nil, exit.New(exit.Validation, "invalid staged wheel identity")
		}
		rows = append(rows, &pb.LocalPackageWheel{Digest: digest, Filename: file.Filename, Length: uint64(file.Length), Path: destination})
	}
	sort.Slice(rows, func(i, j int) bool { return bytes.Compare(rows[i].Digest, rows[j].Digest) < 0 })
	return rows, nil
}

func copyLocalPreparationWheel(source, destination string) *exit.Error {
	input, err := os.Open(source)
	if err != nil {
		return exit.Internalf("cannot read sealed local wheel: %s", err)
	}
	defer input.Close()
	output, err := os.CreateTemp(filepath.Dir(destination), ".wheel-")
	if err != nil {
		return exit.Internalf("cannot stage sealed local wheel: %s", err)
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
		return exit.Internalf("cannot retain sealed local wheel: %s", err)
	}
	if err := os.Rename(temporary, destination); err != nil {
		return exit.Internalf("cannot commit staged local wheel: %s", err)
	}
	return nil
}
