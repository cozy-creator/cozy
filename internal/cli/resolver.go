package cli

import (
	"io"
	"slices"
	"sort"
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

// Resolver is Calcifer's package resolver.
type Resolver struct {
	refreshMu sync.Mutex
	store     *records.Store
	cfg       config.Config
	// catalogs is one public catalog client per Tensorhub origin; each request reads
	// its own hub's catalog.
	catalogMu sync.Mutex
	catalogs  map[string]*hub.Client
}

// LocalInstallation is the code of the unpublished install a request names, as its machine
// takes it: read from the install's own source root, never staged.
func (r *Resolver) LocalInstallation(installID, targetHub string) (localpackage.Installation, *exit.Error) {
	install, problem := r.store.Install(installID)
	if problem != nil {
		return localpackage.Installation{}, problem
	}
	if install == nil {
		return localpackage.Installation{}, exit.New(exit.NotFound, "install %s does not exist", installID)
	}
	origin := targetHub
	if origin == "" {
		origin = install.Hub
	}
	return localpackage.Open(*install, home.Paths(r.cfg.Home).HubWheels(), func() (packagepublish.Namespace, *exit.Error) {
		return r.namespaceAt(origin)
	}, targetHub)
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

// RefreshEditable is Calcifer-owned pre-invocation fence for live source trees.
// It snapshots the current tree, builds a complete replacement install when it
// moved, and swaps the active pin only after Runtime accepted the replacement.
// A failure returns a typed refusal and leaves the last good install active. Only a local/
// capture has a source tree; a published package passes untouched.
func (r *Resolver) RefreshEditable(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	if !strings.HasPrefix(strings.TrimSpace(pkg), "local/") {
		return "", false, false, nil
	}
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
	current, problem := r.activeInstall("", pkg)
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
	// Reading a tree takes a moment: another refresh, or an install, is waited for.
	writer, problem := home.WaitWriter(layout, false, io.Discard)
	if problem != nil {
		return current.ID, false, refreshFailure(pkg, current, problem)
	}
	defer writer.Unlock()
	latest, problem := r.activeInstall("", pkg)
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
			Package: current.Package, Release: pack.Release, Tree: current.SourceRef},
	})
	if problem != nil {
		return current.ID, false, refreshFailure(pkg, current, problem)
	}
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

// namespace answers the signed-in caller on Calcifer's Tensorhub: the account an
// editable source's account index and org-relative model defaults resolve against.
func (r *Resolver) namespace() (packagepublish.Namespace, *exit.Error) {
	return r.namespaceAt("")
}

// namespaceAt is the caller on one Tensorhub, "" Calcifer's. Its account is read once per
// credential and kept, so a run asks no Hub.
func (r *Resolver) namespaceAt(origin string) (packagepublish.Namespace, *exit.Error) {
	cfg := r.cfg.ForHub(origin)
	c := hub.New(cfg, "calcifer").WithTokenSource(accountauth.New(cfg))
	ctx, cancel := hub.Context()
	defer cancel()
	account, problem := c.CurrentAccount(ctx)
	if problem != nil {
		return packagepublish.Namespace{}, problem
	}
	return packagepublish.Namespace{Hub: c.Base(), Account: account.Name}, nil
}

// NewResolver builds the resolver over the lifecycle authority.
func NewResolver(store *records.Store, cfg config.Config) *Resolver {
	return &Resolver{
		store: store, cfg: cfg,
		catalogs: map[string]*hub.Client{},
	}
}

// catalog is one Tensorhub origin's catalog; "" is Calcifer's default hub. The
// static token reaches only the origin it was issued for.
func (r *Resolver) catalog(origin string) *hub.Client {
	cfg := r.cfg.ForHub(origin)
	r.catalogMu.Lock()
	defer r.catalogMu.Unlock()
	if r.catalogs == nil {
		r.catalogs = map[string]*hub.Client{}
	}
	if c := r.catalogs[cfg.HubURL]; c != nil {
		return c
	}
	c := hub.New(cfg, "calcifer")
	r.catalogs[cfg.HubURL] = c
	return c
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
		spec := facts.PreparationSpec()
		spec.Placement.Models = append([]orchestrator.ModelRef(nil), models...)
		return spec, nil
	}
	return facts.Spec()
}

func (r *Resolver) ResolveRemoteRelease(origin, pkg, release, function string,
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
	packageInterface, problem := r.releaseInterface(origin, ref, release)
	if problem != nil {
		return empty, nil, problem
	}
	entrypoint, problem := packageInterface.Function(function)
	if problem != nil {
		return empty, nil, problem
	}
	models = append([]orchestrator.ModelRef(nil), models...)
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	// The caller's own choices, and the defaults a rental for the run is sized by, each bind a
	// slot of the function, or a captured callee's; the machine resolves every other slot.
	for _, model := range models {
		if model.Choice && model.Callable != "" {
			_, path, valid := launch.CapturedModelSlot(model.Package + "/" + model.BindingSlot())
			callee, _, _ := strings.Cut(path, ".models.")
			if !valid || model.Callable != model.Package+"/"+callee {
				return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
					"captured model selection needs an exact callable slot")
			}
			continue // Runtime checks membership in the captured graph before acceptance
		}
		if model.Package != pkg || !slices.ContainsFunc(entrypoint.Models, func(slot launch.Slot) bool { return slot.Path == model.Slot }) {
			return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
				"model selection does not bind a slot of %s", function)
		}
		// A checkpoint named by hash alone is one exact retained checkpoint at the Hub.
		if model.HubCheckpoint || model.Release == "" && model.Manifest != "" {
			if _, err := canonical.Raw(model.Manifest); model.Manifest != "" && err != nil {
				return empty, nil, exit.Named(exit.Validation, "rental.model_manifest_invalid",
					"model selection for %s has no exact manifest", model.Slot)
			}
			if !model.Downloadable() || model.Release != "" || model.Lane != "" || model.ManifestLength <= 0 {
				return empty, nil, exit.Named(exit.Validation, "rental.model_selection_mismatch",
					"model selection for %s names no exact retained checkpoint", model.Slot)
			}
		}
	}
	planID := ""
	if len(models) == 0 && len(entrypoint.Models) == 0 {
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
		Function: function, Outputs: launch.OutputSlots(entrypoint.Result), PlanID: planID,
		Models: models, NeedsAccelerator: entrypoint.NeedsAccelerator(),
	}, entrypoint, nil
}

// releaseInterface is a published release's interface: this computer's verified install of it
// when there is one, else the interface its machine described here (a run on a known machine
// reads no Hub), else Tensorhub's release card (choosing a machine).
func (r *Resolver) releaseInterface(origin string, ref hub.Ref, release string) (*launch.PackageInterface, *exit.Error) {
	if _, installed, problem := r.store.ActivePackage(r.cfg.ForHub(origin).HubURL, ref.String()); problem == nil && installed != nil &&
		installed.Version == release {
		if _, packageInterface, problem := r.installPackageInterface(installed.ID); problem == nil {
			return packageInterface, nil
		}
	}
	if kept := readKeptRelease(home.Paths(r.cfg.Home).Root, r.cfg.ForHub(origin).HubURL, ref.String(), release); kept.Interface != nil {
		return launch.DecodePackageInterface(kept.Interface)
	}
	ctx, cancel := hub.Context()
	defer cancel()
	detail, problem := r.catalog(origin).PackageRelease(ctx, ref, release)
	if problem != nil {
		return nil, problem
	}
	if detail.Release.Release != release {
		return nil, exit.Named(exit.Conflict, "rental.package_release_changed",
			"Tensorhub answered release %s@%s for the queued release %s", ref, detail.Release.Release, release)
	}
	return launch.DecodePackageInterface(detail.PackageInterface)
}

func (r *Resolver) Entrypoint(installID, name string) (*launch.Entrypoint, bool, *exit.Error) {
	_, packageInterface, problem := r.installPackageInterface(installID)
	if problem != nil {
		return nil, false, problem
	}
	entrypoint, problem := packageInterface.Function(name)
	if problem != nil {
		return nil, false, problem
	}
	return entrypoint, entrypoint.NeedsAccelerator(), nil
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
	return facts, nil
}

func (r *Resolver) activeInstall(hub, ref string) (*records.PackageInstall, *exit.Error) {
	if r.store == nil {
		return nil, exit.Unavailablef("this Calcifer has no install records")
	}
	return activeInstall(r.store, hub, ref)
}

// activeInstall is the install a package reference resolves to at hub (local/ at none).
// Internal callers may name a major as `org/name@v2`.
func activeInstall(store *records.Store, hub, ref string) (*records.PackageInstall, *exit.Error) {
	pkg, major, hasMajor := splitMajor(ref)
	pin, inst, e := store.ActivePackage(hub, pkg)
	switch {
	case e != nil:
		return nil, e
	case pin == nil || hasMajor && pin.Major != major:
		where := "on this host"
		if hub := records.ReferenceHub(hub, pkg); hub != "" {
			where = "from " + hub
		}
		return nil, exit.New(exit.NotFound, "%s is not installed %s", ref, where).
			WithRemedy("`cozy package list` shows installed packages").
			WithNext("cozy package list")
	case inst == nil:
		return nil, exit.Internalf("%s is pinned to install %s and that row is gone", pkg, pin.InstallID)
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
