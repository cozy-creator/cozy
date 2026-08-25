package launch

import (
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

const (
	mib = 1 << 20
	gib = 1 << 30
)

// The budgets this LOCAL coordinator declares for one attempt. They are a COORDINATOR
// policy, not an endpoint fact — the runtime prices the real ladder against the card's
// measured free bytes and confesses what it did (cr-008a/cr-008b). The same three numbers
// the runtime's own local-coordinator adapter uses for a bare-venv run, so the two doors
// price identically.
const (
	vramBudget   = 7 * gib
	hostBudget   = 2 * gib
	pinnedBudget = 256 * mib
	vramFloor    = 1 * gib
)

// Facts is everything one installed generation needs to be served, gathered once.
type Facts struct {
	Generation records.Generation
	Source     string
	Descriptor *Descriptor
	Runtime    Runtime
}

// Read gathers a generation's facts: where its source is, the surface it proved at
// install, and the runtime that proved it.
func Read(gen records.Generation, cozyHome string, env []string) (*Facts, *exit.Error) {
	source := SourceDir(gen)
	d, e := ReadDescriptor(source, gen.Descriptor)
	if e != nil {
		return nil, e
	}
	return &Facts{
		Generation: gen,
		Source:     source,
		Descriptor: d,
		Runtime:    Runtime{Bin: Binary(gen.Dir), Dir: source, Home: cozyHome, Env: env},
	}, nil
}

// SourceDir is where a generation's endpoint tree lives. An archive install stages it
// under the generation; a `--dir` install builds a venv against the live tree and records
// its absolute path (cl-009's editable development door).
func SourceDir(gen records.Generation) string {
	if gen.SourceKind == "dir" && gen.SourceRef != "" {
		return gen.SourceRef
	}
	return filepath.Join(gen.Dir, "source")
}

// ReleaseID is the endpoint release identity this host serves the generation under. It is
// what the coordinator pins and what a registering worker must match: an install
// generation of one endpoint version is one provisioned instance lifetime.
func ReleaseID(gen records.Generation) string {
	version := gen.Version
	if version == "" {
		version = gen.ID
	}
	return gen.Endpoint + "@" + version
}

// Spec builds the EndpointSpec the coordinator launches — the object cl-006's
// `--dev-endpoint` document used to supply by hand.
//
// The interpreter is the GENERATION'S OWN: the venv `uv sync --locked` built from the
// release's own lock, so the cozy-runtime that serves an endpoint is the one the release
// pinned and never this host's. That is the whole reason a generation is a venv.
func (f *Facts) Spec(devices []string) (coord.EndpointSpec, *exit.Error) {
	// THE RESOLVED SELECTION, from the one resolver that owns the grammar
	// (`cozy-runtime bindings`). cl-010 read `endpoint.toml`'s `[bindings]` table here
	// because no runtime verb emitted the resolved record; cr-016 added the verb, and this
	// host's second reader of that closed grammar is deleted rather than kept in step.
	resolved, e := f.Runtime.Bindings()
	if e != nil {
		return coord.EndpointSpec{}, e
	}
	table := map[string]Binding{}
	for _, b := range resolved {
		table[b.Path] = b
	}
	spec := coord.EndpointSpec{
		Endpoint:   f.Generation.Endpoint,
		ReleaseID:  ReleaseID(f.Generation),
		Generation: f.Generation.ID,
		// THE SUPERVISOR ENTRY, and it is now the VERB. cl-004 recorded the private
		// `internal.worker.session:main` import as owed: `serve` was dead on arrival
		// because the CLI read the config at its entrypoint and `session.main` read it
		// again, which the runtime's own one-authority rule refuses. cozy-runtime
		// `db4ab8a` passes the config it already read into `session.supervise`, so the
		// import — a second process entry into one loop, across a private module path a
		// release is free to move — deletes.
		Python:   Binary(f.Generation.Dir),
		Args:     []string{"serve"},
		Dir:      f.Source,
		Devices:  devices,
		GraceSec: 3,
	}
	for i := range f.Descriptor.Entrypoints {
		ep := &f.Descriptor.Entrypoints[i]
		binding, e := f.binding(ep, table)
		if e != nil {
			return coord.EndpointSpec{}, e
		}
		spec.Bindings = append(spec.Bindings, binding)
	}
	if len(spec.Bindings) == 0 {
		// Every entrypoint mints a record now — weightless included — so this is the one
		// remaining case: a release whose descriptor registers no entrypoint at all. A job
		// function is not one; it is served through `cozy job`, on a spec of its own.
		return coord.EndpointSpec{}, exit.Named(exit.Structural, "no_servable_function",
			"%s registers no entrypoint, and a worker with no plan to advertise has nothing to serve",
			f.Generation.Endpoint).
			WithRemedy("its functions: %s — an `@app.entrypoint` is what a request dispatches to",
				strings.Join(f.Descriptor.Names(), ", "))
	}
	return spec, nil
}

// binding mints ONE entrypoint's local pinned-binding record. A weightless entrypoint
// (no declared model slot) still gets a record: it names the project and the entrypoint,
// which is all the runtime needs to construct nothing.
func (f *Facts) binding(ep *Entrypoint, table map[string]Binding) (*coord.Binding, *exit.Error) {
	record := map[string]any{
		"project":                   f.Source,
		"entrypoint":                ep.Name,
		"model_construction_digest": "",
	}
	if len(ep.Models) > 1 {
		// The record shape carries ONE model slot. Two slots is cr-008b's
		// construction-identity coalescing and th-004's multi-binding plan document;
		// refusing by name beats minting a record that silently binds one of them.
		return nil, exit.Named(exit.Structural, "multi_slot_unsupported",
			"%s declares %d model binding paths and this host mints ONE record per attempt",
			ep.Name, len(ep.Models)).
			WithRemedy("multi-slot local serving lands with th-004's EntrypointBindingPlan document")
	}
	if len(ep.Models) == 0 {
		// A WEIGHTLESS entrypoint declares no model, so its record declares none either —
		// and that IS the record, not an absence of one. cozy-runtime `a3c3d72` reads a
		// record carrying none of the seven model keys as weightless, keys its executor by
		// release and project, and returns from `prepare` before the torch import. So the
		// coordinator gets a plan digest to dispatch by, and a machine with no card and no
		// bench store can serve something. That was cl-010's named seam and it is closed.
		//
		// `release` is the endpoint release rather than an artifact ref because there is no
		// artifact: every weightless entrypoint of one install generation shares one
		// executor, which is exactly what the runtime's construction key spells.
		record["release"] = ReleaseID(f.Generation)
		return &coord.Binding{
			Entrypoint: ep.Name, Record: record, Outputs: AssetPaths(ep.Result),
		}, nil
	}

	slot := ep.Models[0]
	selected, ok := table[slot.Path]
	if !ok {
		known := make([]string, 0, len(table))
		for path := range table {
			known = append(known, path)
		}
		sort.Strings(known)
		return nil, exit.Named(exit.Structural, "unbound_slot",
			"%s binds %s and the release's own resolver reports no selection for it",
			ep.Name, slot.Path).
			WithRemedy("it resolves: %s — code states capability, bindings state selection",
				strings.Join(known, ", ")).
			WithNext("cozy describe " + f.Generation.Endpoint)
	}
	artifact, e := f.Runtime.Find(selected.Ref)
	if e != nil {
		return nil, e
	}
	// The DECLARED component set for this function narrows what the record stages; an
	// endpoint that declares none takes the artifact whole.
	components := artifact.Components()
	if declared := slot.ComponentUse[ep.Name]; len(declared) > 0 {
		narrowed := []string{}
		for _, name := range components {
			for _, want := range declared {
				if name == want {
					narrowed = append(narrowed, name)
				}
			}
		}
		if len(narrowed) > 0 {
			components = narrowed
		}
	}
	pairs := make([]string, 0, len(artifact.Snapshots))
	for _, name := range artifact.Components() {
		pairs = append(pairs, name+"="+artifact.Snapshots[name])
	}
	floor := artifact.VRAMFloorBytes
	if floor == 0 {
		floor = vramFloor
	}
	record["model_class"] = slot.Class
	record["binding_path"] = slot.Path
	record["param"] = slot.Param
	record["component"] = components[0]
	record["components"] = strings.Join(components, ",")
	record["store"] = artifact.Store
	record["config"] = artifact.Config
	record["snapshot"] = artifact.Snapshots[components[0]]
	record["snapshots"] = strings.Join(pairs, ",")
	record["release"] = artifact.Ref
	record["variant"] = artifact.Variant
	record["vram_bytes"] = int64(vramBudget)
	record["host_bytes"] = int64(hostBudget)
	record["pinned_bytes"] = int64(pinnedBudget)
	record["vram_floor_bytes"] = floor
	record["resident_budget_bytes"] = int64(0)
	record["tenancy"] = "local"
	record["custody"] = artifact.Custody
	record["strict_keys"] = false
	return &coord.Binding{
		Entrypoint: ep.Name, Record: record, Outputs: AssetPaths(ep.Result),
	}, nil
}
