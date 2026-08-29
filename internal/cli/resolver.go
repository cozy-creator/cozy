package cli

import (
	"sort"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy-creator/internal/api"
	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/launch"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// The LOCAL module's endpoint resolver: `org/name` -> the spec that makes its worker
// resident. ONE source, and it is the only one a user's machine will ever use — the
// INSTALL GENERATION and its pin (cl-009's rows), resolved through internal/launch.
//
// cl-006 shipped a second source, `--dev-endpoint <file>`: a hand-written EndpointSpec
// document, because the generation could not yet carry the launch facts a supervisor
// needs and guessing an interpreter would have been worse than refusing. cl-010 DELETES
// it — a generation carries a venv, a proven descriptor and a binding table, which is
// every fact that document supplied. Nothing coexists "temporarily": the flag, the
// loader, and the driver's writer are all gone, and the live driver installs an endpoint
// exactly as a user does.

// Resolver is the LocalService's endpoint resolver.
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

// ResolvePlacement answers only WHAT an endpoint target should host. The current local
// install derives that fact from its proven environment; a future control-manifest
// install can supply it directly without changing the API or orchestrator boundary.
func (r *Resolver) ResolvePlacement(endpoint string) (orchestrator.DesiredPlacement, *exit.Error) {
	endpoint = strings.TrimSpace(endpoint)
	r.mu.Lock()
	placement, ok := r.placements[endpoint]
	r.mu.Unlock()
	if ok {
		return placement, nil
	}
	gen, e := r.generation(endpoint)
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
	r.placements[endpoint] = placement
	r.mu.Unlock()
	return placement, nil
}

// Resolve answers with the spec for one endpoint ref.
func (r *Resolver) Resolve(endpoint string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	endpoint = strings.TrimSpace(endpoint)
	r.mu.Lock()
	spec, ok := r.cache[endpoint]
	r.mu.Unlock()
	if ok {
		return spec, nil
	}
	gen, e := r.generation(endpoint)
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
	r.cache[endpoint] = spec
	r.mu.Unlock()
	return spec, nil
}

// ResolveInstall answers from one exact immutable install row rather than the active pin.
// It is the workflow recovery path: resolution can be retained without becoming identity.
func (r *Resolver) ResolveInstall(installID string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	return facts.Spec(r.Devices)
}

// Entrypoint returns one exact install's verified request/result schema for workflow
// validation before any child request exists.
func (r *Resolver) Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error) {
	facts, e := r.installFacts(installID)
	if e != nil {
		return nil, e
	}
	return facts.Descriptor.Function(name)
}

func (r *Resolver) installFacts(installID string) (*launch.Facts, *exit.Error) {
	install, e := r.store.Install(strings.TrimSpace(installID))
	if e != nil {
		return nil, e
	}
	if install == nil {
		return nil, exit.New(exit.NotFound,
			"workflow install %s is no longer present", installID).
			WithRemedy("a live workflow pins its immutable install; restore the records/directory before retrying")
	}
	return launch.Read(*install, r.cfg.Home, r.cfg.Tool())
}

// ResolveJob answers with the spec that makes ONE job function's worker resident. It is
// the same generation, the same venv and the same device envelope as `Resolve` — what
// differs is the plan record staged for it and the Directive mode it boots into.
//
// It is NOT cached: a job spec is per-function, and caching by endpoint alone was exactly
// the shape that would hand a serving spec to a job.
func (r *Resolver) ResolveJob(endpoint, function string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	gen, e := r.generation(strings.TrimSpace(endpoint))
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

// Jobs names the `@job` functions one installed endpoint registers, with the descriptor
// id each resolves to. `cozy job ls` and the API's job listing read it.
func (r *Resolver) Jobs(endpoint string) ([]launch.JobFacts, *exit.Error) {
	gen, e := r.generation(strings.TrimSpace(endpoint))
	if e != nil {
		return nil, e
	}
	facts, e := launch.Read(*gen, r.cfg.Home, r.cfg.Tool())
	if e != nil {
		return nil, e
	}
	out := []launch.JobFacts{}
	for _, job := range facts.Descriptor.Jobs {
		one, e := facts.Job(job.Name)
		if e != nil {
			return nil, e
		}
		out = append(out, *one)
	}
	return out, nil
}

// generation resolves the ACTIVE pin for an endpoint. A ref may name its major
// (`org/name@v2`); without one, a single pinned major answers and several refuse rather
// than picking.
func (r *Resolver) generation(ref string) (*records.EndpointInstall, *exit.Error) {
	endpoint, major, hasMajor := splitMajor(ref)
	if r.store == nil {
		return nil, exit.Unavailablef("this LocalService has no install records")
	}
	pins, e := r.store.Pins(endpoint)
	if e != nil {
		return nil, e
	}
	if len(pins) == 0 {
		return nil, api.UnknownEndpoint(endpoint)
	}
	chosen := pins[0]
	if hasMajor {
		found := false
		for _, p := range pins {
			if p.Major == major {
				chosen, found = p, true
			}
		}
		if !found {
			return nil, exit.New(exit.NotFound, "%s is installed, but not at v%d", endpoint, major).
				WithRemedy("installed majors: %s", majorsOf(pins)).
				WithNext("cozy ls")
		}
	} else if len(pins) > 1 {
		return nil, exit.Usagef("%s is installed at several majors and a ref must name one", endpoint).
			WithRemedy("majors: %s — a major is a required path segment, never a default", majorsOf(pins)).
			WithNext("cozy ls")
	}
	gen, e := r.store.Install(chosen.InstallID)
	if e != nil {
		return nil, e
	}
	if gen == nil {
		return nil, exit.Internalf("%s is pinned to generation %s and that row is gone",
			endpoint, chosen.InstallID)
	}
	return gen, nil
}

// splitMajor cuts `org/name@vN` into its parts.
func splitMajor(ref string) (endpoint string, major int, ok bool) {
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

// List names every endpoint this host can resolve.
func (r *Resolver) List() []string {
	out := []string{}
	if r.store != nil {
		if installed, e := r.store.Installed(); e == nil {
			for _, g := range installed {
				out = append(out, g.Endpoint)
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
