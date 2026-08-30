package cli

import (
	"sort"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
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
	mu    sync.Mutex
	store *records.Store
	cfg   config.Config
	// cache holds the specs already derived this launch. Deriving one reads a descriptor
	// and asks the runtime for its artifact index; a generation is IMMUTABLE, so doing it
	// twice would answer the same thing twice.
	cache map[string]orchestrator.WorkerLaunchSpec
	// placements contain only control-plane facts. Keeping this cache distinct is the
	// seam cl-020's verified control manifest will populate without a local venv.
	placements map[string]orchestrator.DesiredPlacement
	// Devices is the device envelope a worker this host launches may SEE.
	Devices []string
}

// NewResolver builds the resolver over the lifecycle authority.
func NewResolver(store *records.Store, cfg config.Config) *Resolver {
	return &Resolver{
		store: store, cfg: cfg,
		cache:      map[string]orchestrator.WorkerLaunchSpec{},
		placements: map[string]orchestrator.DesiredPlacement{},
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

// Entrypoint returns one exact install's verified request/result schema.
func (r *Resolver) Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return nil, e
	}
	return facts.PackageDescriptor.Function(name)
}

func (r *Resolver) installFacts(installID string) (*launch.Facts, *exit.Error) {
	install, e := r.store.Install(strings.TrimSpace(installID))
	if e != nil {
		return nil, e
	}
	if install == nil {
		return nil, exit.New(exit.NotFound,
			"request install %s is no longer present", installID).
			WithRemedy("the durable request retains its immutable install until terminal")
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
