package cli

import (
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
	"github.com/cozy-creator/cozy/internal/records"
)

// The LOCAL module's package resolver: `org/name` -> the spec that makes its worker
// resident. ONE source, and it is the only one a user's machine will ever use — the
// INSTALL GENERATION and its pin (cl-009's rows), resolved through internal/launch.
//
// cl-006 shipped a second source, `--dev-package <file>`: a hand-written PackageSpec
// document, because the generation could not yet carry the launch facts a supervisor
// needs and guessing an interpreter would have been worse than refusing. cl-010 DELETES
// it — a generation carries a venv, a proven descriptor and a binding table, which is
// every fact that document supplied. Nothing coexists "temporarily": the flag, the
// loader, and the driver's writer are all gone, and the live driver installs a package
// exactly as a user does.

// Resolver is the Cozy daemon's package resolver.
type Resolver struct {
	mu        sync.Mutex
	refreshMu sync.Mutex
	store     *records.Store
	cfg       config.Config
	// cache holds the specs already derived this launch. Deriving one reads a descriptor
	// and asks the runtime for its artifact index; a generation is IMMUTABLE, so doing it
	// twice would answer the same thing twice.
	cache map[string]orchestrator.WorkerLaunchSpec
	// placements contain only control-plane facts. Keeping this cache distinct is the
	// seam cl-020's verified control manifest will populate without a local venv.
	placements map[string]orchestrator.DesiredPlacement
	catalog    *hub.Client
	published  map[string]*launch.PackageDescriptor
	// Devices is the device envelope a worker this host launches may SEE.
	Devices []string
}

// RefreshEditable is the daemon-owned pre-invocation fence for live source trees.
// It snapshots the current tree, builds a complete replacement generation when it
// moved, and swaps the active pin only after Runtime accepted the replacement.
// A failure returns a typed refusal and leaves the last good generation active.
func (r *Resolver) RefreshEditable(pkg string) (installID string, editable, changed bool, problem *exit.Error) {
	r.refreshMu.Lock()
	defer r.refreshMu.Unlock()
	pkg = strings.TrimSpace(pkg)
	current, problem := r.generation(pkg)
	if problem != nil {
		return "", false, false, problem
	}
	if current.SourceKind != "local" {
		return current.ID, false, false, nil
	}
	refreshFailure := func(cause *exit.Error) (string, bool, bool, *exit.Error) {
		return current.ID, true, false, exit.Named(cause.Code, "editable_refresh_failed",
			"editable package %s could not refresh; generation %s remains active",
			pkg, short12(current.ID)).
			WithRemedy("%s: %s", cause.ErrName(), cause.Message)
	}
	pack, problem := packagepublish.PrepareFrom(current.SourceRef)
	if problem != nil {
		return refreshFailure(problem)
	}
	defer pack.Close()
	if pack.Organization+"/"+pack.Name != current.Package || pack.Release != current.Version {
		return refreshFailure(exit.Named(exit.Conflict, "editable_identity_changed",
			"editable metadata now names %s/%s@%s, not installed %s@%s",
			pack.Organization, pack.Name, pack.Release, current.Package, current.Version).
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
	latest, problem := r.generation(pkg)
	if problem != nil {
		return refreshFailure(problem)
	}
	if latest.ID != current.ID {
		current = latest
		if current.SourceKind != "local" {
			return current.ID, false, false, nil
		}
		if current.SourceRef != pack.Tree || current.Package != pack.Organization+"/"+pack.Name ||
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
	if result.Gen.ID == "" {
		return refreshFailure(exit.Internalf("editable refresh returned no active generation"))
	}
	return result.Gen.ID, true, !result.Idempotent, nil
}

func short12(value string) string {
	if len(value) > 12 {
		return value[:12]
	}
	return value
}

// NewResolver builds the resolver over the lifecycle authority.
func NewResolver(store *records.Store, cfg config.Config) *Resolver {
	return &Resolver{
		store: store, cfg: cfg,
		cache:      map[string]orchestrator.WorkerLaunchSpec{},
		placements: map[string]orchestrator.DesiredPlacement{},
		catalog:    hub.New(cfg, "cozy-daemon"),
		published:  map[string]*launch.PackageDescriptor{},
		Devices:    []string{"0"},
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
	gen, e := r.generation(pkg)
	if e != nil {
		return orchestrator.DesiredPlacement{}, e
	}
	facts, e := launch.Read(*gen, r.cfg.Home, r.cfg.Tool())
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
	gen, e := r.generation(pkg)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	facts, e := launch.Read(*gen, r.cfg.Home, r.cfg.Tool())
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
func (r *Resolver) ResolveInstall(installID string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	return facts.Spec(r.Devices)
}

func (r *Resolver) ResolveLogicalInstall(installID, function string) (orchestrator.LogicalPackage, *exit.Error) {
	install, e := r.installRecord(installID)
	if e != nil {
		return orchestrator.LogicalPackage{}, e
	}
	descriptor, e := r.publishedDescriptor(install)
	if e != nil {
		return orchestrator.LogicalPackage{}, e
	}
	entrypoint, e := descriptor.Function(function)
	if e != nil {
		return orchestrator.LogicalPackage{}, e
	}
	if len(descriptor.Entrypoints) != 1 || len(entrypoint.Models) != 0 {
		return orchestrator.LogicalPackage{}, exit.Named(exit.Unavailable,
			"rental.logical_package_unsupported",
			"private package_set currently admits one weightless entrypoint and no model slots")
	}
	return orchestrator.LogicalPackage{
		Package: install.Package, Release: install.Version, ReleaseDigest: install.SourceDigest,
		InstallID: install.ID, Function: function,
		Outputs: launch.AssetPaths(entrypoint.Result),
	}, nil
}

// Entrypoint returns one exact install's verified request/result schema.
func (r *Resolver) Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error) {
	_, descriptor, e := r.installDescriptor(installID)
	if e != nil {
		return nil, e
	}
	return descriptor.Function(name)
}

func (r *Resolver) RemoteEntrypoint(installID, name string) (*launch.Entrypoint, *exit.Error) {
	install, e := r.installRecord(installID)
	if e != nil {
		return nil, e
	}
	descriptor, e := r.publishedDescriptor(install)
	if e != nil {
		return nil, e
	}
	return descriptor.Function(name)
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

func (r *Resolver) publishedDescriptor(install *records.PackageInstall) (*launch.PackageDescriptor,
	*exit.Error) {
	r.mu.Lock()
	descriptor := r.published[install.SourceDigest]
	r.mu.Unlock()
	if descriptor != nil {
		return descriptor, nil
	}
	descriptor, problem := publishedPackageDescriptor(r.catalog, install)
	if problem != nil {
		return nil, problem
	}
	r.mu.Lock()
	r.published[install.SourceDigest] = descriptor
	r.mu.Unlock()
	return descriptor, nil
}

func publishedPackageDescriptor(c *hub.Client,
	install *records.PackageInstall) (*launch.PackageDescriptor, *exit.Error) {
	if install.SourceKind != "tensorhub" || !install.Verified {
		return nil, exit.Named(exit.Unavailable, "rental.package_release_unpublished",
			"private package_set requires an immutable Tensorhub package release")
	}
	if _, err := canonical.Raw(install.SourceDigest); err != nil {
		return nil, exit.Named(exit.Structural, "rental.package_release_digest_invalid",
			"installed package %s has no valid immutable release digest", install.Package)
	}
	ref, problem := hub.ParseRef(install.Package)
	if problem != nil {
		return nil, problem
	}
	ctx, cancel := hub.Context()
	defer cancel()
	detail, problem := c.PackageRelease(ctx, ref, install.Version)
	if problem != nil {
		return nil, problem
	}
	if detail.Release.Release != install.Version || detail.Release.ReleaseDigest != install.SourceDigest {
		return nil, exit.Named(exit.Conflict, "rental.package_release_changed",
			"Tensorhub release %s@%s does not match the installed immutable release pin",
			install.Package, install.Version)
	}
	if detail.Release.PackageDescriptorLength != int64(len(detail.PackageDescriptor)) {
		return nil, exit.Named(exit.Conflict, "rental.package_descriptor_length_mismatch",
			"Tensorhub descriptor length does not match its release fact")
	}
	descriptor, problem := launch.DecodeDescriptor(detail.PackageDescriptor)
	if problem != nil {
		return nil, problem
	}
	if descriptor.Digest != detail.Release.PackageDescriptorDigest {
		return nil, exit.Named(exit.Conflict, "rental.package_descriptor_digest_mismatch",
			"Tensorhub descriptor bytes do not match their release fact")
	}
	return descriptor, nil
}

func (r *Resolver) installFacts(installID string) (*launch.Facts, *exit.Error) {
	install, e := r.installRecord(installID)
	if e != nil {
		return nil, e
	}
	return launch.Read(*install, r.cfg.Home, r.cfg.Tool())
}

// ResolveJob answers with the spec that makes ONE job function's worker resident. It is
// the same generation, the same venv and the same device envelope as `Resolve` — what
// differs is the plan record staged for it and the Directive mode it boots into.
//
// It is NOT cached: a job spec is per-function, and caching by package alone was exactly
// the shape that would hand a serving spec to a job.
func (r *Resolver) ResolveJob(pkg, function string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	gen, e := r.generation(strings.TrimSpace(pkg))
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	facts, e := launch.Read(*gen, r.cfg.Home, r.cfg.Tool())
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
	gen, e := r.generation(strings.TrimSpace(pkg))
	if e != nil {
		return nil, e
	}
	facts, e := launch.Read(*gen, r.cfg.Home, r.cfg.Tool())
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

// generation resolves an active package pin. Internal callers may name a major as
// `org/name@v2`; otherwise the newest installed major wins.
func (r *Resolver) generation(ref string) (*records.PackageInstall, *exit.Error) {
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
	gen, e := r.store.Install(chosen.InstallID)
	if e != nil {
		return nil, e
	}
	if gen == nil {
		return nil, exit.Internalf("%s is pinned to generation %s and that row is gone",
			pkg, chosen.InstallID)
	}
	return gen, nil
}

// splitMajor cuts `org/name@vN` into its parts.
func splitMajor(ref string) (pkg string, major int, ok bool) {
	name, suffix, cut := strings.Cut(ref, "@v")
	if !cut {
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
