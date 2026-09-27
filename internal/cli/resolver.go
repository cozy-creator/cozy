package cli

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// The LOCAL module's package resolver: `org/name` -> the spec that makes its worker
// resident. ONE source, and it is the only one a user's machine will ever use — the
// INSTALL and its pin (cl-009's rows), resolved through internal/launch.
//
// cl-006 shipped a second source, `--dev-package <file>`: a hand-written PackageSpec
// document, because the install could not yet carry the launch facts a supervisor
// needs and guessing an interpreter would have been worse than refusing. cl-010 DELETES
// it — an install carries a venv, a proven PackageInterface and a binding table, which is
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
	// cache holds the specs already derived this launch. Deriving one reads a PackageInterface
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

// PrepareLocal freezes one editable install into exact wheels before rental attachment.
// Tensorhub selects the base; the worker validates the package against that exact base.
func (r *Resolver) PrepareLocal(ctx context.Context, installID string) (
	localpackage.Installation, *exit.Error,
) {
	install, problem := r.store.Install(installID)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	if install == nil {
		return localpackage.Installation{}, exit.New(exit.NotFound, "install %s does not exist", installID)
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	revision, problem := localpackage.Stage(ctx, layout, *install)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	return revision, nil
}

// LocalRevision reopens the exact staged wheel set a durable request already names.
func (r *Resolver) LocalInstallation(installID, digest string) (localpackage.Installation, *exit.Error) {
	install, problem := r.store.Install(installID)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	if install == nil {
		return localpackage.Installation{}, exit.New(exit.NotFound, "install %s does not exist", installID)
	}
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	revision, problem := localpackage.Open(layout, *install, digest)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	return revision, nil
}

// EditableSnapshot is one reading of an editable install's live source tree against the
// install that was active when it was read. Close releases the prepared tree.
type EditableSnapshot struct {
	Package  string
	Current  *records.PackageInstall
	Editable bool
	Changed  bool
	Files    int
	Bytes    int64
	pack     *packagepublish.Package
}

func (s *EditableSnapshot) Close() {
	if s != nil && s.pack != nil {
		s.pack.Close()
		s.pack = nil
	}
}

// RefreshEditable is the daemon-owned pre-invocation fence for live source trees.
// It snapshots the current tree, builds a complete replacement install when it
// moved, and swaps the active pin only after Runtime accepted the replacement.
// A failure returns a typed refusal and leaves the last good install active.
func (r *Resolver) RefreshEditable(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	snapshot, problem := r.SnapshotEditable(pkg)
	if snapshot == nil {
		return "", false, false, problem
	}
	defer snapshot.Close()
	if problem != nil || !snapshot.Editable {
		return snapshot.Current.ID, snapshot.Editable, false, problem
	}
	installID, changed, problem = r.RefreshSnapshot(snapshot)
	return installID, true, changed, problem
}

// SnapshotEditable reads the tree once: the walk plus the hash of every source file. It
// takes no lock, so the source watcher can read a tree that is still moving and read it
// again; only RefreshSnapshot serializes. A tree that cannot be read answers the typed
// refusal beside the snapshot that names the install still active.
func (r *Resolver) SnapshotEditable(pkg string) (*EditableSnapshot, *exit.Error) {
	pkg = strings.TrimSpace(pkg)
	current, problem := r.activeInstall(pkg)
	if problem != nil {
		return nil, problem
	}
	snapshot := &EditableSnapshot{Package: pkg, Current: current}
	if current.SourceKind != "local" {
		return snapshot, nil
	}
	snapshot.Editable = true
	pack, problem := packagepublish.PrepareLocalFrom(current.SourceRef)
	if problem != nil {
		return snapshot, refreshFailure(pkg, current, problem)
	}
	if "local/"+pack.Name != current.Package {
		pack.Close()
		return snapshot, refreshFailure(pkg, current, exit.Named(exit.Conflict, "editable_identity_changed",
			"editable metadata now names local/%s, not installed %s", pack.Name, current.Package).
			WithRemedy("install the renamed package directory explicitly"))
	}
	stats, bytes, problem := pack.SourceStats()
	if problem != nil {
		pack.Close()
		return snapshot, refreshFailure(pkg, current, problem)
	}
	snapshot.pack, snapshot.Files, snapshot.Bytes = pack, len(stats), bytes
	snapshot.Changed = pack.Release != current.Version || !packagepublish.SourceStatsUnchanged(current.Dir, stats)
	return snapshot, nil
}

func refreshFailure(pkg string, current *records.PackageInstall, cause *exit.Error) *exit.Error {
	detail := cause.ErrName() + ": " + cause.Message
	if cause.Remedy != "" {
		detail += " — " + cause.Remedy
	}
	return exit.Named(cause.Code, "editable_refresh_failed",
		"editable package %s could not refresh; install %s remains active",
		pkg, short12(current.ID)).
		WithRemedy("%s", detail)
}

// RefreshSnapshot builds the replacement install a moved snapshot calls for and swaps the
// pin. It answers the active install afterwards and whether that is a new one; a snapshot
// that matches the active install, or one the pin already moved past, changes nothing.
func (r *Resolver) RefreshSnapshot(snapshot *EditableSnapshot) (installID string, changed bool, problem *exit.Error) {
	r.refreshMu.Lock()
	defer r.refreshMu.Unlock()
	if !snapshot.Editable || !snapshot.Changed {
		return snapshot.Current.ID, false, nil
	}
	pkg, current, pack := snapshot.Package, snapshot.Current, snapshot.pack
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return current.ID, false, refreshFailure(pkg, current, problem)
	}
	writer, problem := install.Lock(layout)
	if problem != nil {
		return current.ID, false, refreshFailure(pkg, current, problem)
	}
	defer writer.Unlock()
	latest, problem := r.activeInstall(pkg)
	if problem != nil {
		return current.ID, false, refreshFailure(pkg, current, problem)
	}
	if latest.ID != current.ID {
		current = latest
		if current.SourceKind != "local" {
			return current.ID, false, nil
		}
		if current.SourceRef != pack.Tree || current.Package != "local/"+pack.Name {
			return current.ID, false, refreshFailure(pkg, current, exit.Named(exit.Conflict, "editable_refresh_raced",
				"the active package changed while its editable source was being checked").
				WithRemedy("retry against the current install"))
		}
	}
	result, problem := install.Run(layout, r.store, install.Request{
		Ref: install.Ref{Package: current.Package}, Force: true,
		Local: &install.LocalSource{Bytes: snapshot.Bytes, Files: snapshot.Files,
			Package: current.Package, Release: pack.Release, Tree: current.SourceRef,
			Namespace: r.namespace},
	})
	if problem != nil {
		return current.ID, false, refreshFailure(pkg, current, problem)
	}
	r.mu.Lock()
	delete(r.cache, pkg)
	delete(r.placements, pkg)
	r.mu.Unlock()
	if result.Install.ID == "" {
		return current.ID, false, refreshFailure(pkg, current, exit.Internalf("editable refresh returned no active install"))
	}
	if result.Superseded != "" {
		// Activation already committed the replacement. Reclaim the now-unpinned immutable
		// tree just like an explicit install does; an exceptional cleanup failure must
		// not misreport the newly active environment as a failed refresh.
		_, _ = install.Reclaim(layout, r.store, result.Superseded)
	}
	return result.Install.ID, !result.Idempotent, nil
}

func short12(value string) string {
	if len(value) > 12 {
		return value[:12]
	}
	return value
}

// namespace answers the signed-in caller on the daemon's Tensorhub: the account an
// editable source's account index and org-relative model defaults resolve against.
func (r *Resolver) namespace() (packagepublish.Namespace, *exit.Error) {
	c := hub.New(r.cfg, "cozy-daemon").WithTokenSource(accountauth.New(r.cfg))
	ctx, cancel := hub.Context()
	defer cancel()
	account, problem := c.CurrentAccount(ctx)
	if problem != nil {
		return packagepublish.Namespace{}, problem
	}
	return packagepublish.Namespace{Hub: c.Base(), Account: account.Name}, nil
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
	if facts.Install.PlacementSetDigest == "" || len(models) > 0 {
		spec, problem := facts.PreparationSpec(r.Devices)
		if problem != nil {
			return spec, problem
		}
		spec.Placement.Models = append([]orchestrator.ModelRef(nil), models...)
		return spec, nil
	}
	return facts.Spec(r.Devices)
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
		key.WriteByte(0)
		key.WriteString(strconv.Itoa(len(row.Adapters)))
		for _, adapter := range row.Adapters {
			for _, part := range []string{adapter.Component, adapter.Model, adapter.Release, adapter.Lane, adapter.Manifest, adapter.SourceComponent, adapter.Scale} {
				key.WriteByte(0)
				key.WriteString(part)
			}
		}
	}
	return key.String()
}

func (r *Resolver) ResolveRemoteRelease(pkg, release, function string,
	models []orchestrator.ModelRef,
) (
	orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error,
) {
	var empty orchestrator.LogicalPackage
	if release == "" {
		return empty, nil, exit.Named(exit.Structural, "rental.package_release_invalid",
			"remote package release is absent")
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
	if detail.Release.Release != release {
		return empty, nil, exit.Named(exit.Conflict, "rental.package_release_changed",
			"Tensorhub answered release %s@%s for the queued release %s", pkg, detail.Release.Release, release)
	}
	requirements, problem := detail.Requirements()
	if problem != nil {
		return empty, nil, problem
	}
	packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return empty, nil, problem
	}
	entrypoint, problem := packageInterface.Function(function)
	if problem != nil {
		return empty, nil, problem
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
		// A retained Hub checkpoint has no release/lane yet. It is the same
		// exact downloadable input already accepted by jobs; an unbound name or
		// operation-local manifest is not a replacement for that resolved fact.
		if model.Package != pkg || model.Slot != slot.Path ||
			(model.Release == "" && (!model.Downloadable() || model.Lane != "" || model.ManifestLength <= 0)) {
			return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
				"model selection does not bind exact slot %s", slot.Path)
		}
		if !selected {
			return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
				"model selection does not bind exact slot %s", slot.Path)
		}
		if _, problem := hub.ParseRef(model.Model); problem != nil {
			return empty, nil, problem
		}
		// Exact, or a ladder every rung of which is exact: the machine decision pins one
		// rung once the machine exists (cl-166).
		exact := []string{model.Manifest}
		if !model.Pinned() {
			exact = exact[:0]
			for _, rung := range model.Ladder {
				exact = append(exact, rung.Manifest)
			}
		}
		if len(exact) == 0 {
			return empty, nil, exit.Named(exit.Validation, "rental.model_manifest_invalid",
				"model selection for %s has neither an exact manifest nor a ladder", slot.Path)
		}
		for _, manifest := range exact {
			if _, err := canonical.Raw(manifest); err != nil {
				return empty, nil, exit.Named(exit.Validation, "rental.model_manifest_invalid",
					"model selection for %s has no exact manifest", slot.Path)
			}
		}
	}
	if len(models) > 0 && len(packageInterface.Entrypoints) > 1 {
		// ONE PLACEMENT PER CONSTRUCTION (h3a-018). The selection also binds every
		// sibling slot the owner bound to the same model release under the same ladder,
		// so the pod prepares both entrypoints once and a switch between them is a
		// dispatch. Owner overrides and authored defaults use the same selection here.
		// Sharing only saves a later preparation, so an unreadable override set skips it.
		if rows, problem := r.catalog.PackageBindings(ctx, ref); problem == nil {
			defaults := effectiveModelBindings(declaredModelSlots(packageInterface.Entrypoints), rows, ref.Org)
			shareModelSlots(models, entrypoint, packageInterface.Entrypoints, defaults)
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
		Package: pkg, Release: release,
		Function: function, Outputs: launch.AssetPaths(entrypoint.Result), PlanID: planID,
		Models: models, NeedsAccelerator: launch.AcceleratorRequired(requirements),
	}, entrypoint, nil
}

func (r *Resolver) ResolveRemoteJob(pkg, release, function string,
	models []orchestrator.ModelRef, deferredModels bool,
) (
	orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error,
) {
	var empty orchestrator.LogicalJob
	if release == "" {
		return empty, nil, exit.Named(exit.Structural, "rental.package_release_invalid",
			"remote package release is absent")
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
	if detail.Release.Release != release {
		return empty, nil, exit.Named(exit.Conflict, "rental.package_release_changed",
			"Tensorhub answered release %s@%s for the queued release %s", pkg, detail.Release.Release, release)
	}
	requirements, problem := detail.Requirements()
	if problem != nil {
		return empty, nil, problem
	}
	packageInterface, problem := launch.DecodePackageInterface(detail.PackageInterface)
	if problem != nil {
		return empty, nil, problem
	}
	job, problem := packageInterface.Function(function)
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
	byParam := make(map[string]int, len(models))
	for index, model := range models {
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
		byParam[model.Slot] = index
	}
	for _, slot := range job.Models {
		if deferredModels {
			continue
		}
		index, ok := byParam[slot.Param]
		if !ok {
			return empty, nil, exit.Named(exit.Validation, "rental.job_model_selection_mismatch",
				"remote job %s does not bind model parameter %s", function, slot.Param)
		}
		if models[index].BindingPath != "" && models[index].BindingPath != slot.Path {
			return empty, nil, exit.Named(exit.Validation, "rental.job_model_selection_mismatch",
				"remote job %s model parameter %s changed its interface binding path", function, slot.Param)
		}
		models[index].BindingPath = slot.Path
	}
	weights := make([]orchestrator.WeightsOutput, 0, len(job.WeightsOutputs))
	params := make([]string, 0, len(job.Models))
	for _, model := range job.Models {
		params = append(params, model.Param)
	}
	outputs := launch.AssetPaths(job.Result)
	for _, output := range job.WeightsOutputs {
		weights = append(weights, orchestrator.WeightsOutput{
			OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes,
		})
		outputs = append(outputs, output.OutputID)
	}
	return orchestrator.LogicalJob{
		Package: pkg, Release: release,
		Function: function, DescriptorID: job.DescriptorID, Outputs: outputs,
		WeightsOutputs: weights, NeedsAccelerator: launch.AcceleratorRequired(requirements), Models: models,
		ProducerParams: params,
	}, job, nil
}

func (r *Resolver) Entrypoint(installID, name string) (*launch.Entrypoint, bool, *exit.Error) {
	install, packageInterface, problem := r.installPackageInterface(installID)
	if problem != nil {
		return nil, false, problem
	}
	entrypoint, problem := packageInterface.Function(name)
	if problem != nil {
		return nil, false, problem
	}
	accelerator := launch.AcceleratorRequired(strings.Split(install.Closure, "\n"))
	if entrypoint.Kind == "job" && len(entrypoint.Models) == 0 && len(entrypoint.WeightsOutputs) == 0 {
		parent, problem := r.store.CompositionParent(installID, name)
		if problem != nil {
			return nil, false, problem
		}
		accelerator = accelerator && !parent
	}
	return entrypoint, accelerator, nil
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

func (r *Resolver) installPackageInterface(installID string) (*records.PackageInstall,
	*launch.PackageInterface, *exit.Error) {
	install, e := r.installRecord(installID)
	if e != nil {
		return nil, nil, e
	}
	packageInterface, e := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir))
	if e != nil {
		return nil, nil, e
	}
	return install, packageInterface, nil
}

func (r *Resolver) installFacts(installID string) (*launch.Facts, *exit.Error) {
	installed, e := r.installRecord(installID)
	if e != nil {
		return nil, e
	}
	facts, problem := launch.Read(*installed, r.cfg.Home, r.cfg.Tool())
	if problem != nil {
		return nil, problem
	}
	facts.CPUOrchestration, problem = r.store.HasChildBindings(installID)
	if problem != nil {
		return nil, problem
	}
	facts.CPUOrchestration = facts.CPUOrchestration || facts.PackageInterface.Application == packagepublish.ScriptApplication
	facts.SelfCallable, problem = r.store.SelfCallableEntrypoints(installID)
	return facts, problem
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

// Jobs names the `@job` functions one installed package registers, with the PackageInterface
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
	for _, job := range facts.PackageInterface.Jobs {
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

// shareModelSlots binds only complete sibling constructions (h3a-018). Every slot
// must have a default satisfied by an already-selected model; a partial sibling
// would leave stranded selections that Runtime correctly refuses at preparation.
func shareModelSlots(models []orchestrator.ModelRef, own *launch.Entrypoint, all []launch.Entrypoint,
	defaults map[string]hub.PackageBindingRow) {
	classes := make(map[string]string, len(own.Models))
	for _, slot := range own.Models {
		classes[slot.Path] = slot.Class
	}
	for i := range models {
		models[i].SharedSlots = nil
	}
siblingLoop:
	for i := range all {
		sibling := &all[i]
		if sibling.Name == own.Name {
			continue
		}
		selected := make(map[int][]string)
		for _, slot := range sibling.Models {
			row, bound := defaults[slot.Path]
			if !bound {
				continue siblingLoop
			}
			matched := false
			for j, model := range models {
				if slot.Class == classes[model.Slot] && row.Model == model.Model &&
					row.Release == model.Release && ladderOffers(row.Ladder, model) {
					selected[j] = append(selected[j], slot.Path)
					matched = true
					break
				}
			}
			if !matched {
				continue siblingLoop
			}
		}
		for j, slots := range selected {
			models[j].SharedSlots = append(models[j].SharedSlots, slots...)
		}
	}
	for i := range models {
		sort.Strings(models[i].SharedSlots)
	}
}

// ladderOffers answers whether a sibling's ladder serves the selection: rung for rung the
// same ladder when the selection still carries one, or a rung on the pinned lane when the
// caller pinned it explicitly.
func ladderOffers(ladder []hub.BindingRung, model orchestrator.ModelRef) bool {
	if model.Pinned() {
		for _, rung := range ladder {
			if rung.Lane == model.Lane {
				return true
			}
		}
		return false
	}
	if len(ladder) != len(model.Ladder) {
		return false
	}
	for i, rung := range ladder {
		if rung.GPU != model.Ladder[i].GPU || rung.Lane != model.Ladder[i].Lane {
			return false
		}
	}
	return true
}

// ValidateExecutionCapture checks that the accepted installation is available.
// Runtime dependency constraints are resolved by uv during package installation.
func (r *Resolver) ValidateExecutionCapture(request records.Request) *exit.Error {
	_, problem := r.LocalInstallation(request.InstallID, request.LocalInstallationID)
	return problem
}
