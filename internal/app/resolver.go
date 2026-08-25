package app

import (
	"encoding/json"
	"os"
	"sort"
	"sync"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

// The LOCAL module's endpoint resolver: `org/name` -> the spec that makes its worker
// resident. Two sources, in this order:
//
//  1. The INSTALL GENERATION and its pin (cl-009's rows). This is the real one, and it is
//     the only one a user's machine will ever use.
//  2. A DEV SPEC named at launch with `--dev-endpoint <file>`. It exists for exactly one
//     reason and it is a named consumer, not a convenience: cl-001's coordinator-kill
//     crash arm cannot be armed without a SEPARATE `cozy up` process to kill, and driving
//     a separate process means the driver must be able to ask the running service to make
//     a worker resident. Until cl-009's generation carries the launch facts a supervisor
//     needs (interpreter, argv, imposed env, binding records), the driver hands them over
//     as a document.
//
// cl-010 replaces (2)'s reason, not its shape: `cozy start` is a client of the same
// route, and the day the generation resolves fully, the dev door has no caller and goes.

// Resolver is the LocalService's endpoint resolver.
type Resolver struct {
	mu    sync.Mutex
	store *records.Store
	dev   map[string]coord.EndpointSpec
}

// NewResolver builds the resolver over the lifecycle authority.
func NewResolver(store *records.Store) *Resolver {
	return &Resolver{store: store, dev: map[string]coord.EndpointSpec{}}
}

// LoadDev reads one dev spec document. The file is a `coord.EndpointSpec` as JSON; a
// malformed one refuses at launch rather than at the first submission.
func (r *Resolver) LoadDev(path string) *exit.Error {
	data, err := os.ReadFile(path)
	if err != nil {
		return exit.New(exit.NotFound, "cannot read the dev endpoint spec %s: %s", path, err).
			WithRemedy("--dev-endpoint takes a JSON EndpointSpec; it is a development door")
	}
	var spec coord.EndpointSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		return exit.New(exit.Validation, "%s is not an EndpointSpec: %s", path, err)
	}
	if spec.Endpoint == "" || len(spec.Bindings) == 0 {
		return exit.New(exit.Validation, "%s names no endpoint or no binding", path)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.dev[spec.Endpoint] = spec
	return nil
}

// Resolve answers with the spec for one endpoint ref.
func (r *Resolver) Resolve(endpoint string) (coord.EndpointSpec, *exit.Error) {
	r.mu.Lock()
	spec, ok := r.dev[endpoint]
	r.mu.Unlock()
	if ok {
		return spec, nil
	}
	// The generation half: the pin exists and names a built venv, but the launch facts a
	// supervisor needs are cr-016's stable runtime verb, which this host does not call
	// yet (cl-001's seam names it as a one-line change to EndpointSpec.Args). Rather than
	// guess an interpreter and an argv, this refuses BY NAME.
	if r.store != nil {
		pins, e := r.store.Pins(endpoint)
		if e == nil && len(pins) > 0 {
			return coord.EndpointSpec{}, exit.New(exit.Unavailable,
				"%s is installed (generation %s) but this build cannot launch an installed generation",
				endpoint, pins[0].Generation).
				WithRemedy("the launch facts come from cr-016's runtime verb; cl-010 wires them")
		}
	}
	return coord.EndpointSpec{}, api.UnknownEndpoint(endpoint)
}

// List names every endpoint this host can resolve.
func (r *Resolver) List() []string {
	out := []string{}
	r.mu.Lock()
	for name := range r.dev {
		out = append(out, name)
	}
	r.mu.Unlock()
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
