package launch

import (
	"path/filepath"
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
	table, e := ReadBindings(f.Source)
	if e != nil {
		return coord.EndpointSpec{}, e
	}
	spec := coord.EndpointSpec{
		Endpoint:   f.Generation.Endpoint,
		ReleaseID:  ReleaseID(f.Generation),
		Generation: f.Generation.ID,
		Python:     filepath.Join(f.Generation.Dir, "venv", "bin", "python"),
		// THE SUPERVISOR ENTRY. `cozy-runtime serve` is cr-016's documented process entry
		// but takes only --socket/--cas/--out, and the coordinator must name the instance,
		// the release, the device envelope and the grace clock — so this launches the
		// supervisor's own main with the full flag set the protocol needs. Recorded as
		// cl-010's seam for cr-016: when `serve` carries those flags it becomes the entry
		// and this argv collapses to `[serve, --socket, …]`.
		Args: []string{"-c",
			"import sys; from cozy_runtime.internal.worker.session import main; " +
				"raise SystemExit(main(sys.argv[1:]))"},
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
		if binding != nil {
			spec.Bindings = append(spec.Bindings, binding)
		}
	}
	if len(spec.Bindings) == 0 {
		return coord.EndpointSpec{}, exit.Named(exit.Structural, "no_servable_function",
			"%s registers no function with a model binding, and a weightless function has no plan to dispatch by",
			f.Generation.Endpoint).
			WithRemedy("its functions: %s — serve one with `cozy-runtime run` in its own venv",
				strings.Join(f.Descriptor.Names(), ", "))
	}
	return spec, nil
}

// binding mints ONE entrypoint's local pinned-binding record. A weightless entrypoint
// (no declared model slot) still gets a record: it names the project and the entrypoint,
// which is all the runtime needs to construct nothing.
func (f *Facts) binding(ep *Entrypoint, table map[string]Selection) (*coord.Binding, *exit.Error) {
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
		// A WEIGHTLESS entrypoint has no binding-plan record to stage, and a worker
		// advertises exactly the plans it loaded — so this coordinator has no digest to
		// dispatch it by. It is not served here, and it is not silently missing either:
		// `cozy describe` lists it and says so. The runtime's own one-shot `run` serves
		// it today (its attempt carries zero binding plans), and serving one THROUGH the
		// protocol wants a modelless plan in the plan vocabulary — cl-010's named seam,
		// not a record with empty paths in it.
		return nil, nil
	}

	slot := ep.Models[0]
	selection, e := Select(table, slot)
	if e != nil {
		return nil, e
	}
	artifact, e := f.Runtime.Find(selection.Ref())
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
