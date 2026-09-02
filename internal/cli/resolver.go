package cli

import (
	"context"
	"sort"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/records"
)

// The LOCAL module's package resolver: `org/name` -> the spec that makes its worker
// resident. ONE source, and it is the only one a user's machine will ever use — the
// INSTALL and its pin (cl-009's rows), resolved through internal/launch.
//
// cl-006 shipped a second source, `--dev-package <file>`: a hand-written PackageSpec
// document, because the install could not yet carry the launch facts a supervisor
// needs and guessing an interpreter would have been worse than refusing. cl-010 DELETES
// it — an install carries a venv, a proven descriptor and a binding table, which is
// every fact that document supplied. Nothing coexists "temporarily": the flag, the
// loader, and the driver's writer are all gone, and the live driver installs a package
// exactly as a user does.

// Resolver is the Cozy daemon's package resolver.
type Resolver struct {
	mu          sync.Mutex
	refreshMu   sync.Mutex
	selectionMu sync.Mutex
	store       *records.Store
	cfg         config.Config
	// cache holds the specs already derived this launch. Deriving one reads a descriptor
	// and asks the runtime for its artifact index; an install is IMMUTABLE, so doing it
	// twice would answer the same thing twice.
	cache    map[string]orchestrator.WorkerLaunchSpec
	selected map[string]orchestrator.WorkerLaunchSpec
	// placements contain only control-plane facts. Keeping this cache distinct is the
	// seam cl-020's verified control manifest will populate without a local venv.
	placements map[string]orchestrator.DesiredPlacement
	catalog    *hub.Client
	// Devices is the device envelope the daemon GRANTS a worker this host launches: what
	// it may SEE, and the space the worker's reported lanes index into (proto-024).
	Devices []string
}

// PreparePrivate freezes one editable install into exact wheels before rental attachment.
// Tensorhub selects the base; the worker validates the package against that exact base.
func (r *Resolver) PreparePrivate(ctx context.Context, installID string) (
	privatepackage.Revision, *exit.Error,
) {
	install, problem := r.store.Install(installID)
	if problem != nil {
		return privatepackage.Revision{}, problem
	}
	if install == nil {
		return privatepackage.Revision{}, exit.New(exit.NotFound, "install %s does not exist", installID)
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return privatepackage.Revision{}, problem
	}
	if problem := privatepackage.Sweep(layout, r.store); problem != nil {
		return privatepackage.Revision{}, problem
	}
	revision, problem := privatepackage.Stage(ctx, layout, *install)
	if problem != nil {
		return privatepackage.Revision{}, problem
	}
	return revision, nil
}

// PrivateRevision reopens the exact staged wheel set a durable request already names.
func (r *Resolver) PrivateRevision(installID, digest string) (privatepackage.Revision, *exit.Error) {
	install, problem := r.store.Install(installID)
	if problem != nil {
		return privatepackage.Revision{}, problem
	}
	if install == nil {
		return privatepackage.Revision{}, exit.New(exit.NotFound, "install %s does not exist", installID)
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return privatepackage.Revision{}, problem
	}
	return privatepackage.Open(layout, *install, digest)
}

// RefreshEditable is the daemon-owned pre-invocation fence for live source trees.
// It snapshots the current tree, builds a complete replacement install when it
// moved, and swaps the active pin only after Runtime accepted the replacement.
// A failure returns a typed refusal and leaves the last good install active.
func (r *Resolver) RefreshEditable(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	r.refreshMu.Lock()
	defer r.refreshMu.Unlock()
	pkg = strings.TrimSpace(pkg)
	current, problem := r.activeInstall(pkg)
	if problem != nil {
		return "", false, false, problem
	}
	if current.SourceKind != "local" {
		return current.ID, false, false, nil
	}
	refreshFailure := func(cause *exit.Error) (string, bool, bool, *exit.Error) {
		return current.ID, true, false, exit.Named(cause.Code, "editable_refresh_failed",
			"editable package %s could not refresh; install %s remains active",
			pkg, short12(current.ID)).
			WithRemedy("%s: %s", cause.ErrName(), cause.Message)
	}
	pack, problem := packagepublish.PrepareLocalFrom(current.SourceRef)
	if problem != nil {
		return refreshFailure(problem)
	}
	defer pack.Close()
	if "local/"+pack.Name != current.Package || pack.Release != current.Version {
		return refreshFailure(exit.Named(exit.Conflict, "editable_identity_changed",
			"editable metadata now names %s/%s@%s, not installed %s@%s",
			"local", pack.Name, pack.Release, current.Package, current.Version).
			WithRemedy("install the renamed package directory explicitly"))
	}
	digest, files, bytes, problem := pack.SourceIdentity()
	if problem != nil {
		return refreshFailure(problem)
	}
	if digest == current.SourceDigest {
		return current.ID, true, false, nil
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return refreshFailure(problem)
	}
	writer, problem := install.Lock(layout)
	if problem != nil {
		return refreshFailure(problem)
	}
	defer writer.Unlock()
	latest, problem := r.activeInstall(pkg)
	if problem != nil {
		return refreshFailure(problem)
	}
	if latest.ID != current.ID {
		current = latest
		if current.SourceKind != "local" {
			return current.ID, false, false, nil
		}
		if current.SourceRef != pack.Tree || current.Package != "local/"+pack.Name ||
			current.Version != pack.Release {
			return refreshFailure(exit.Named(exit.Conflict, "editable_refresh_raced",
				"the active package changed while its editable source was being checked").
				WithRemedy("retry against the current install"))
		}
		if current.SourceDigest == digest {
			return current.ID, true, false, nil
		}
	}
	result, problem := install.Run(layout, r.store, install.Request{
		Ref: install.Ref{Package: current.Package}, Force: true,
		Local: &install.LocalSource{SourceDigest: digest, Bytes: bytes, Files: files,
			Package: current.Package, Release: current.Version, Tree: current.SourceRef},
	})
	if problem != nil {
		return refreshFailure(problem)
	}
	r.mu.Lock()
	delete(r.cache, pkg)
	delete(r.placements, pkg)
	r.mu.Unlock()
	if result.Install.ID == "" {
		return refreshFailure(exit.Internalf("editable refresh returned no active install"))
	}
	if result.Superseded != "" {
		// Activation already committed the replacement. Reclaim the now-unpinned immutable
		// tree just like an explicit install does; an exceptional cleanup failure must
		// not misreport the newly active environment as a failed refresh.
		_, _ = install.Reclaim(layout, r.store, result.Superseded)
	}
	return result.Install.ID, true, !result.Idempotent, nil
}

func short12(value string) string {
	if len(value) > 12 {
		return value[:12]
	}
	return value
}

// NewResolver builds the resolver over the lifecycle authority. `devices` is the envelope
// the daemon grants every local worker it launches (orchestrator.LocalDeviceEnvelope).
func NewResolver(store *records.Store, cfg config.Config, devices []string) *Resolver {
	return &Resolver{
		store: store, cfg: cfg,
		cache:      map[string]orchestrator.WorkerLaunchSpec{},
		selected:   map[string]orchestrator.WorkerLaunchSpec{},
		placements: map[string]orchestrator.DesiredPlacement{},
		catalog:    hub.New(cfg, "cozy-daemon"),
		Devices:    append([]string(nil), devices...),
	}
}

// ResolvePlacement answers only WHAT a package target should host. The current local
// install derives that fact from its proven environment; a future control-manifest
// install can supply it directly without changing the API or orchestrator boundary.
func (r *Resolver) ResolvePlacement(pkg string) (orchestrator.DesiredPlacement, *exit.Error) {
	pkg = strings.TrimSpace(pkg)
	r.mu.Lock()
	placement, ok := r.placements[pkg]
	r.mu.Unlock()
	if ok {
		return placement, nil
	}
	inst, e := r.activeInstall(pkg)
	if e != nil {
		return orchestrator.DesiredPlacement{}, e
	}
	facts, e := launch.Read(*inst, r.cfg.Home, r.cfg.Tool())
	if e != nil {
		return orchestrator.DesiredPlacement{}, e
	}
	placement, e = facts.Placement()
	if e != nil {
		return orchestrator.DesiredPlacement{}, e
	}
	r.mu.Lock()
	r.placements[pkg] = placement
	r.mu.Unlock()
	return placement, nil
}

// Resolve answers with the spec for one package ref.
func (r *Resolver) Resolve(pkg string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	pkg = strings.TrimSpace(pkg)
	r.mu.Lock()
	spec, ok := r.cache[pkg]
	r.mu.Unlock()
	if ok {
		return spec, nil
	}
	inst, e := r.activeInstall(pkg)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	facts, e := launch.Read(*inst, r.cfg.Home, r.cfg.Tool())
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	spec, e = facts.Spec(r.Devices)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	r.mu.Lock()
	r.cache[pkg] = spec
	r.mu.Unlock()
	return spec, nil
}

// ResolveInstall answers from the exact immutable row a durable request retained.
func (r *Resolver) ResolveInstall(installID string, models []orchestrator.ModelRef) (
	orchestrator.WorkerLaunchSpec, *exit.Error,
) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	if len(models) == 0 {
		if facts.Install.PlacementSetDigest == "" {
			return orchestrator.WorkerLaunchSpec{}, exit.Named(exit.Validation,
				"package_model_binding_required",
				"%s is installed as code but this modeled invocation supplied no exact model binding",
				facts.Install.Package).
				WithRemedy("invoke through `cozy run` so package defaults resolve, or pass --model [slot=]org/model@release")
		}
		return facts.Spec(r.Devices)
	}
	key := selectedInstallKey(installID, models)
	r.mu.Lock()
	cached, ok := r.selected[key]
	r.mu.Unlock()
	if ok {
		return cached, nil
	}
	r.selectionMu.Lock()
	defer r.selectionMu.Unlock()
	r.mu.Lock()
	cached, ok = r.selected[key]
	r.mu.Unlock()
	if ok {
		return cached, nil
	}
	selected := make([]install.PublishedModel, 0, len(models))
	for _, model := range models {
		if model.Package != facts.Install.Package || model.Slot == "" || model.Model == "" ||
			model.Release == "" || model.Lane == "" || model.ManifestLength <= 0 {
			return orchestrator.WorkerLaunchSpec{}, exit.Named(exit.Validation,
				"local_model_selection_incomplete",
				"model selection for %s does not carry exact slot/release/lane/Manifest facts",
				facts.Install.Package)
		}
		selected = append(selected, install.PublishedModel{Package: model.Package, Slot: model.Slot,
			Model: model.Model, Release: model.Release, Lane: model.Lane,
			Manifest: model.Manifest, ManifestLength: model.ManifestLength})
	}
	layout, e := home.Open(r.cfg.Home)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	placement, e := install.PreparePublishedSelection(layout, facts.Install, selected)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	chosen := facts.Install
	chosen.PlacementSetDigest = placement.Digest
	selectedFacts, e := launch.Read(chosen, r.cfg.Home, r.cfg.Tool())
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	spec, e := selectedFacts.Spec(r.Devices)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	r.mu.Lock()
	r.selected[key] = spec
	r.mu.Unlock()
	return spec, nil
}

func selectedInstallKey(installID string, models []orchestrator.ModelRef) string {
	rows := append([]orchestrator.ModelRef(nil), models...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Slot < rows[j].Slot })
	var key strings.Builder
	key.WriteString(installID)
	for _, row := range rows {
		key.WriteByte(0)
		key.WriteString(row.Slot)
		key.WriteByte(0)
		key.WriteString(row.Model)
		key.WriteByte(0)
		key.WriteString(row.Release)
		key.WriteByte(0)
		key.WriteString(row.Lane)
		key.WriteByte(0)
		key.WriteString(row.Manifest)
	}
	return key.String()
}

func (r *Resolver) ResolveRemoteRelease(pkg, release, releaseDigest, function string,
	models []orchestrator.ModelRef,
) (
	orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error,
) {
	var empty orchestrator.LogicalPackage
	if _, err := canonical.Raw(releaseDigest); err != nil || release == "" {
		return empty, nil, exit.Named(exit.Structural, "rental.package_release_digest_invalid",
			"remote package release identity is incomplete")
	}
	ref, problem := hub.ParseRef(pkg)
	if problem != nil {
		return empty, nil, problem
	}
	ctx, cancel := hub.Context()
	defer cancel()
	detail, problem := r.catalog.PackageRelease(ctx, ref, release)
	if problem != nil {
		return empty, nil, problem
	}
	if detail.Release.Release != release || detail.Release.ReleaseDigest != releaseDigest ||
		detail.Release.PackageDescriptorLength != int64(len(detail.PackageDescriptor)) {
		return empty, nil, exit.Named(exit.Conflict, "rental.package_release_changed",
			"Tensorhub release %s@%s does not match the queued immutable release", pkg, release)
	}
	requirements, problem := detail.Requirements()
	if problem != nil {
		return empty, nil, problem
	}
	descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
	if problem != nil {
		return empty, nil, problem
	}
	if descriptor.Digest != detail.Release.PackageDescriptorDigest {
		return empty, nil, exit.Named(exit.Conflict, "rental.package_descriptor_digest_mismatch",
			"Tensorhub descriptor bytes do not match their release fact")
	}
	entrypoint, problem := descriptor.Function(function)
	if problem != nil {
		return empty, nil, problem
	}
	if len(entrypoint.Models) > 0 && len(descriptor.Entrypoints) != 1 {
		return empty, nil, exit.Named(exit.Unavailable, "rental.modeled_package_surface_unsupported",
			"the first modeled rental lane requires one serving entrypoint so its worker-derived binding is unambiguous")
	}
	models = append([]orchestrator.ModelRef(nil), models...)
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	if len(models) != len(entrypoint.Models) {
		return empty, nil, exit.Named(exit.Validation, "rental.model_selection_incomplete",
			"%s requires exactly one model for each of its %d slots", function, len(entrypoint.Models))
	}
	bySlot := make(map[string]orchestrator.ModelRef, len(models))
	for _, model := range models {
		if _, exists := bySlot[model.Slot]; exists {
			return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
				"model slot %s was selected more than once", model.Slot)
		}
		bySlot[model.Slot] = model
	}
	for _, slot := range entrypoint.Models {
		model, selected := bySlot[slot.Path]
		if model.Package != pkg || model.Slot != slot.Path || model.Release == "" {
			return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
				"model selection does not bind exact slot %s", slot.Path)
		}
		if !selected {
			return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
				"model selection does not bind exact slot %s", slot.Path)
		}
		if len(slot.Stamps) != 0 {
			return empty, nil, exit.Named(exit.Unavailable, "rental.model_stamps_unsupported",
				"model slot %s uses unsupported stamps", slot.Path)
		}
		if _, problem := hub.ParseRef(model.Model); problem != nil {
			return empty, nil, problem
		}
		if _, err := canonical.Raw(model.Manifest); err != nil {
			return empty, nil, exit.Named(exit.Validation, "rental.model_manifest_invalid",
				"model selection for %s has no exact manifest", slot.Path)
		}
	}
	planID := ""
	if len(models) == 0 {
		body, err := canonical.Write(map[string]canonical.Value{
			"name": function, "slots": []canonical.Value{},
		})
		if err != nil {
			return empty, nil, exit.Internalf("cannot derive remote binding identity: %s", err)
		}
		planID, err = canonical.Spell(canonical.Digest(body))
		if err != nil {
			return empty, nil, exit.Internalf("cannot spell remote binding identity: %s", err)
		}
	}
	return orchestrator.LogicalPackage{
		Package: pkg, Release: release, ReleaseDigest: releaseDigest,
		Function: function, Outputs: launch.AssetPaths(entrypoint.Result), PlanID: planID,
		Models: models, NeedsAccelerator: launch.AcceleratorRequired(requirements),
	}, entrypoint, nil
}

func (r *Resolver) ResolveRemoteJob(pkg, release, releaseDigest, function string,
	models []orchestrator.ModelRef, deferredModels bool,
) (
	orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error,
) {
	var empty orchestrator.LogicalJob
	if _, err := canonical.Raw(releaseDigest); err != nil || release == "" {
		return empty, nil, exit.Named(exit.Structural, "rental.package_release_digest_invalid",
			"remote package release identity is incomplete")
	}
	ref, problem := hub.ParseRef(pkg)
	if problem != nil {
		return empty, nil, problem
	}
	ctx, cancel := hub.Context()
	defer cancel()
	detail, problem := r.catalog.PackageRelease(ctx, ref, release)
	if problem != nil {
		return empty, nil, problem
	}
	if detail.Release.Release != release || detail.Release.ReleaseDigest != releaseDigest ||
		detail.Release.PackageDescriptorLength != int64(len(detail.PackageDescriptor)) {
		return empty, nil, exit.Named(exit.Conflict, "rental.package_release_changed",
			"Tensorhub release %s@%s does not match the queued immutable release", pkg, release)
	}
	requirements, problem := detail.Requirements()
	if problem != nil {
		return empty, nil, problem
	}
	descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
	if problem != nil || descriptor.Digest != detail.Release.PackageDescriptorDigest {
		return empty, nil, exit.Named(exit.Conflict, "rental.package_descriptor_digest_mismatch",
			"Tensorhub descriptor bytes do not match their release fact")
	}
	job, problem := descriptor.Function(function)
	if problem != nil {
		return empty, nil, problem
	}
	if job.Kind != "job" || job.DescriptorID == "" {
		return empty, nil, exit.Named(exit.Validation, "rental.job_descriptor_invalid",
			"%s is not a published job callable", function)
	}
	models = append([]orchestrator.ModelRef(nil), models...)
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	if !deferredModels && len(models) != len(job.Models) {
		return empty, nil, exit.Named(exit.Validation, "rental.job_model_selection_incomplete",
			"remote job %s requires exactly %d model Manifest binding(s)", function, len(job.Models))
	}
	byParam := make(map[string]orchestrator.ModelRef, len(models))
	for _, model := range models {
		if model.Package != pkg || model.Slot == "" || model.Model == "" ||
			model.ManifestLength <= 0 || model.ManifestLength > (int64(1)<<53)-1 {
			return empty, nil, exit.Named(exit.Validation, "rental.job_model_selection_mismatch",
				"remote job %s carries an incomplete model Manifest binding", function)
		}
		if _, exists := byParam[model.Slot]; exists {
			return empty, nil, exit.Named(exit.Validation, "rental.job_model_selection_mismatch",
				"remote job %s repeats model parameter %s", function, model.Slot)
		}
		if _, err := canonical.Raw(model.Manifest); err != nil {
			return empty, nil, exit.Named(exit.Validation, "rental.job_model_manifest_invalid",
				"remote job %s model parameter %s has no exact Manifest digest", function, model.Slot)
		}
		byParam[model.Slot] = model
	}
	for _, slot := range job.Models {
		if deferredModels {
			continue
		}
		if _, ok := byParam[slot.Param]; !ok {
			return empty, nil, exit.Named(exit.Validation, "rental.job_model_selection_mismatch",
				"remote job %s does not bind model parameter %s", function, slot.Param)
		}
		if len(slot.Stamps) != 0 {
			return empty, nil, exit.Named(exit.Unavailable, "rental.job_model_stamps_unsupported",
				"remote job %s model parameter %s uses unsupported stamps", function, slot.Param)
		}
	}
	weights := make([]orchestrator.WeightsOutput, 0, len(job.WeightsOutputs))
	params := make([]string, 0, len(job.Models))
	for _, model := range job.Models {
		params = append(params, model.Param)
	}
	outputs := launch.AssetPaths(job.Result)
	for _, output := range job.WeightsOutputs {
		var contract *records.ModelTransferContract
		if output.RequiredContract != nil {
			contract = &records.ModelTransferContract{TopologyDigest: output.RequiredContract.TopologyDigest,
				Encodings: append([]string(nil), output.RequiredContract.Encodings...)}
		}
		weights = append(weights, orchestrator.WeightsOutput{
			OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes,
			RequiredContract: contract,
		})
		outputs = append(outputs, output.OutputID)
	}
	return orchestrator.LogicalJob{
		Package: pkg, Release: release, ReleaseDigest: releaseDigest,
		Function: function, DescriptorID: job.DescriptorID, Outputs: outputs,
		WeightsOutputs: weights, NeedsAccelerator: launch.AcceleratorRequired(requirements), Models: models,
		ProducerParams: params,
	}, job, nil
}

func (r *Resolver) Entrypoint(installID, name string) (*launch.Entrypoint, bool, *exit.Error) {
	install, descriptor, problem := r.installDescriptor(installID)
	if problem != nil {
		return nil, false, problem
	}
	entrypoint, problem := descriptor.Function(name)
	return entrypoint, launch.AcceleratorRequired(strings.Split(install.Closure, "\n")), problem
}

func (r *Resolver) installRecord(installID string) (*records.PackageInstall, *exit.Error) {
	install, e := r.store.Install(strings.TrimSpace(installID))
	if e != nil {
		return nil, e
	}
	if install == nil {
		return nil, exit.New(exit.NotFound,
			"request install %s is no longer present", installID).
			WithRemedy("the durable request retains its immutable install until terminal")
	}
	return install, nil
}

func (r *Resolver) installDescriptor(installID string) (*records.PackageInstall,
	*launch.PackageDescriptor, *exit.Error) {
	install, e := r.installRecord(installID)
	if e != nil {
		return nil, nil, e
	}
	descriptor, e := launch.ReadDescriptor(launch.DescriptorPath(install.Dir), install.PackageDescriptor)
	if e != nil {
		return nil, nil, e
	}
	return install, descriptor, nil
}

func (r *Resolver) installFacts(installID string) (*launch.Facts, *exit.Error) {
	install, e := r.installRecord(installID)
	if e != nil {
		return nil, e
	}
	return launch.Read(*install, r.cfg.Home, r.cfg.Tool())
}

// ResolveJob answers with the spec that makes ONE job function's worker resident. It is
// the same install, the same venv and the same device envelope as `Resolve` — what
// differs is the plan record staged for it and the Directive mode it boots into.
//
// It is NOT cached: a job spec is per-function, and caching by package alone was exactly
// the shape that would hand a serving spec to a job.
func (r *Resolver) ResolveJob(pkg, function string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	inst, e := r.activeInstall(strings.TrimSpace(pkg))
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	facts, e := launch.Read(*inst, r.cfg.Home, r.cfg.Tool())
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	spec, _, e := facts.JobSpec(function, r.Devices)
	return spec, e
}

// ResolveJobInstall starts a job from the exact immutable install selected before the
// request entered the queue.
func (r *Resolver) ResolveJobInstall(installID, function string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	spec, _, e := facts.JobSpec(function, r.Devices)
	return spec, e
}

// Jobs names the `@job` functions one installed package registers, with the descriptor
// id each resolves to. `cozy run list` and the API's job listing read it.
func (r *Resolver) Jobs(pkg string) ([]launch.JobFacts, *exit.Error) {
	inst, e := r.activeInstall(strings.TrimSpace(pkg))
	if e != nil {
		return nil, e
	}
	facts, e := launch.Read(*inst, r.cfg.Home, r.cfg.Tool())
	if e != nil {
		return nil, e
	}
	return jobsOf(facts)
}

// JobsInstall lists jobs from one exact immutable install.
func (r *Resolver) JobsInstall(installID string) ([]launch.JobFacts, *exit.Error) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return nil, e
	}
	return jobsOf(facts)
}

func jobsOf(facts *launch.Facts) ([]launch.JobFacts, *exit.Error) {
	out := []launch.JobFacts{}
	for _, job := range facts.PackageDescriptor.Jobs {
		one, e := facts.Job(job.Name)
		if e != nil {
			return nil, e
		}
		out = append(out, *one)
	}
	return out, nil
}

// activeInstall resolves an active package pin. Internal callers may name a major as
// `org/name@v2`; otherwise the newest installed major wins.
func (r *Resolver) activeInstall(ref string) (*records.PackageInstall, *exit.Error) {
	pkg, major, hasMajor := splitMajor(ref)
	if r.store == nil {
		return nil, exit.Unavailablef("this Cozy daemon has no install records")
	}
	pins, e := r.store.Pins(pkg)
	if e != nil {
		return nil, e
	}
	if len(pins) == 0 {
		return nil, exit.New(exit.NotFound, "%s is not installed on this host", pkg).
			WithRemedy("`cozy package list` shows installed packages").
			WithNext("cozy package search " + pkg)
	}
	chosen := pins[len(pins)-1]
	if hasMajor {
		found := false
		for _, p := range pins {
			if p.Major == major {
				chosen, found = p, true
			}
		}
		if !found {
			return nil, exit.New(exit.NotFound, "%s is installed, but not at v%d", pkg, major).
				WithRemedy("installed majors: %s", majorsOf(pins)).
				WithNext("cozy package list")
		}
	}
	inst, e := r.store.Install(chosen.InstallID)
	if e != nil {
		return nil, e
	}
	if inst == nil {
		return nil, exit.Internalf("%s is pinned to install %s and that row is gone",
			pkg, chosen.InstallID)
	}
	return inst, nil
}

// splitMajor cuts `org/name@vN` into its parts.
func splitMajor(ref string) (pkg string, major int, ok bool) {
	name, suffix, cut := strings.Cut(ref, "@v")
	if !cut || suffix == "" || len(suffix) > 1 && suffix[0] == '0' {
		return ref, 0, false
	}
	n := 0
	for _, c := range suffix {
		if c < '0' || c > '9' {
			return ref, 0, false
		}
		n = n*10 + int(c-'0')
	}
	return name, n, true
}

func majorsOf(pins []records.Pin) string {
	out := make([]string, 0, len(pins))
	for _, p := range pins {
		out = append(out, "v"+itoa(p.Major))
	}
	sort.Strings(out)
	return strings.Join(out, ", ")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// List names every package this host can resolve.
func (r *Resolver) List() []string {
	out := []string{}
	if r.store != nil {
		if installed, e := r.store.Installed(); e == nil {
			for _, g := range installed {
				out = append(out, g.Package)
			}
		}
	}
	sort.Strings(out)
	return dedupe(out)
}

func dedupe(in []string) []string {
	out := in[:0]
	var last string
	for i, v := range in {
		if i > 0 && v == last {
			continue
		}
		out = append(out, v)
		last = v
	}
	return out
}
