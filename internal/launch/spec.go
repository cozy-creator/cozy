package launch

import (
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

const (
	mib = 1 << 20
	gib = 1 << 30
)

// The budgets this LOCAL orchestrator declares for one attempt. They are ORCHESTRATOR
// policy, not an endpoint fact — the runtime prices the real ladder against the card's
// measured free bytes and confesses what it did (cr-008a/cr-008b). The same numbers the
// runtime's own local-orchestrator adapter uses for a bare-venv run, so the two doors
// price identically.
//
// `vramBudget` IS GONE (#569b). It was 7 GiB, and it was not policy: it was a guess at how
// much a model weighs, made by a host that may never have held the model. Bound against a
// 92.42 GiB artifact on a card with 149 GB free, the fill refused `budget_overrun` — the
// ninth field of the same mistake #567e corrected in the other eight. What a model weighs
// is the artifact's own byte total, which the border counted and the index row carries, so
// the machine that holds the bytes states it. Host and pinned staging buffers stay here,
// because those really are this orchestrator's policy and not facts about anybody's weights.
const (
	hostBudget   = 2 * gib
	pinnedBudget = 256 * mib
)

// Facts is everything one endpoint install needs to be served, gathered once.
type Facts struct {
	Install    records.EndpointInstall
	Source     string
	Descriptor *Descriptor
	RuntimeCLI RuntimeCLI
}

// Read gathers a generation's facts: where its source is, the surface it proved at
// install, and the runtime that proved it.
func Read(gen records.EndpointInstall, cozyHome string, env []string) (*Facts, *exit.Error) {
	source := SourceDir(gen)
	d, e := ReadDescriptor(source, gen.Descriptor)
	if e != nil {
		return nil, e
	}
	return &Facts{
		Install:    gen,
		Source:     source,
		Descriptor: d,
		RuntimeCLI: RuntimeCLI{Bin: Binary(gen.Dir), Dir: source, Home: cozyHome, Env: env},
	}, nil
}

// SourceDir is where a generation's endpoint tree lives. An archive install stages it
// under the generation; a `--dir` install builds a venv against the live tree and records
// its absolute path (cl-009's editable development door).
func SourceDir(gen records.EndpointInstall) string {
	if gen.SourceKind == "dir" && gen.SourceRef != "" {
		return gen.SourceRef
	}
	return filepath.Join(gen.Dir, "source")
}

// ReleaseID is the endpoint release identity this host serves the generation under. It is
// what the orchestrator pins and what a registering worker must match: an install
// generation of one endpoint version is one provisioned instance lifetime.
func ReleaseID(gen records.EndpointInstall) string {
	version := gen.Version
	if version == "" {
		version = gen.ID
	}
	return gen.Endpoint + "@" + version
}

// Placement builds the platform-neutral control facts a worker is asked to host. It says
// WHAT release and binding plans should serve, never HOW a process is launched. Today the
// local install derives the records through its own runtime; cl-020's signed control
// manifest will supply the same result without constructing a target environment here.
func (f *Facts) Placement() (orchestrator.DesiredPlacement, *exit.Error) {
	// THE RESOLVED SELECTION, from the one resolver that owns the grammar
	// (`cozy-runtime bindings`). cl-010 read `endpoint.toml`'s `[bindings]` table here
	// because no runtime verb emitted the resolved record; cr-016 added the verb, and this
	// host's second reader of that closed grammar is deleted rather than kept in step.
	resolved, weightlessPlans, e := f.RuntimeCLI.Bindings()
	if e != nil {
		return orchestrator.DesiredPlacement{}, e
	}
	table := map[string]Binding{}
	for _, b := range resolved {
		table[b.Path] = b
	}
	plans := map[string]WeightlessPlan{}
	for _, p := range weightlessPlans {
		if _, exists := plans[p.Entrypoint]; exists {
			return orchestrator.DesiredPlacement{}, exit.Internalf(
				"runtime reported weightless plan %s more than once", p.Entrypoint)
		}
		plans[p.Entrypoint] = p
	}
	if len(plans) > 0 && len(table) > 0 {
		return orchestrator.DesiredPlacement{}, exit.Internalf(
			"runtime reported modeled bindings and weightless plans for one placement")
	}
	placement := orchestrator.DesiredPlacement{
		Endpoint:  f.Install.Endpoint,
		ReleaseID: ReleaseID(f.Install),
		InstallID: f.Install.ID,
		// The descriptor this install already VERIFIED in the generation's own venv
		// (cr-003's `describe --check`). It names the placement's surface by digest.
		DescriptorDigest: f.Install.Descriptor,
	}
	hidden := []string{}
	for i := range f.Descriptor.Entrypoints {
		ep := &f.Descriptor.Entrypoints[i]
		// THE SERVING SET EXCLUDES HIDDEN SURFACES (#572d). Not a filter on rendering — a
		// filter on what gets a binding staged at all, which is the only place the hide can
		// be structural rather than documentary.
		if ep.Hidden {
			hidden = append(hidden, ep.Name)
			continue
		}
		var binding *orchestrator.Binding
		if len(plans) > 0 {
			p, ok := plans[ep.Name]
			if !ok {
				return orchestrator.DesiredPlacement{}, exit.Internalf(
					"runtime reported no canonical weightless plan for visible entrypoint %s", ep.Name)
			}
			binding = &orchestrator.Binding{
				Entrypoint: ep.Name,
				Outputs:    AssetPaths(ep.Result),
				RuntimePlan: &orchestrator.BindingPlanSubject{
					SubjectID: p.SubjectID,
					Kind:      p.Kind,
					Digest:    p.Digest,
					Length:    p.Length,
				},
			}
		} else {
			binding, e = f.binding(ep, table)
			if e != nil {
				return orchestrator.DesiredPlacement{}, e
			}
		}
		placement.Bindings = append(placement.Bindings, binding)
	}
	if len(hidden) > 0 {
		placement.Hidden = hidden
	}
	if len(placement.Bindings) == 0 {
		// A release whose descriptor registers no entrypoint has no plan to advertise. A job
		// function is not one; it is served through `cozy job`, on a spec of its own.
		return orchestrator.DesiredPlacement{}, exit.Named(exit.Structural, "no_servable_function",
			"%s registers no entrypoint, and a worker with no plan to advertise has nothing to serve",
			f.Install.Endpoint).
			WithRemedy("its functions: %s — an `@app.entrypoint` is what a request dispatches to",
				strings.Join(f.Descriptor.Names(), ", "))
	}
	if len(plans) != len(placement.Bindings) && len(plans) > 0 {
		return orchestrator.DesiredPlacement{}, exit.Internalf(
			"runtime reported %d weightless plans for %d visible entrypoints",
			len(plans), len(placement.Bindings))
	}
	return placement, nil
}

// Spec adds this host's target-environment materialization to a placement. The
// interpreter is the GENERATION'S OWN: the venv `uv sync --locked` built from the
// release's own lock, so the cozy-runtime that serves an endpoint is the one the release
// pinned and never this host's. A connected worker never calls this method.
func (f *Facts) Spec(devices []string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	placement, e := f.Placement()
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	args := []string{"serve"}
	if placement.RuntimeStagesBindings() {
		args = append(args, "--weightless-endpoint", f.Source)
	}
	return orchestrator.WorkerLaunchSpec{
		Placement: placement,
		// THE WORKER ENTRY is the public verb. Creator never imports Runtime internals or
		// gives the weightless constructor a second source tree: the exact same f.Source
		// is both the child working directory and --weightless-endpoint.
		Python:   Binary(f.Install.Dir),
		Args:     args,
		Dir:      f.Source,
		Devices:  devices,
		GraceSec: 3,
	}, nil
}

// binding mints ONE modeled entrypoint's local pinned-binding record. Weightless plans
// are the installed runtime's canonical documents and arrive through weightless_plans;
// this writer must never approximate them or keep the retired Record/2 path as a fallback.
func (f *Facts) binding(ep *Entrypoint, table map[string]Binding) (*orchestrator.Binding, *exit.Error) {
	if len(ep.Models) == 0 {
		return nil, exit.Named(exit.Structural, "weightless_plan_unavailable",
			"the installed runtime reported no canonical weightless plan for %s", ep.Name).
			WithRemedy("install a runtime whose bindings document carries weightless_plans; " +
				"Creator will not recreate the retired flat binding record")
	}
	record := map[string]any{
		// RESOLUTION, not identity (#506a): `project` is where THIS machine staged the
		// endpoint tree, and a pod that installed the byte-identical archive staged it
		// somewhere else. What names the endpoint inside the identity is
		// `endpoint_release` below — the same string on both machines by construction.
		"project":                   f.Source,
		"endpoint_release":          ReleaseID(f.Install),
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
			WithNext("cozy describe " + f.Install.Endpoint)
	}
	// THE RECORD NAMES THE ARTIFACT, IT DOES NOT RESOLVE IT (#567e).
	//
	// This used to call `RuntimeCLI.Find(selected.Ref)` and copy the answer — store root,
	// per-component snapshot map, config path, variant, custody, VRAM floor — into the
	// document. Every one of those is a fact about a DISK, and the disk it was read from is
	// this box's. A rented pod holding the artifact perfectly was dispatched a record
	// describing a store it has never seen; worse, the owner could not mint the record at
	// all, because a host that has not pulled 92 GiB it never intends to serve exits 4 on
	// its own empty index before a request leaves it. That is the whole of hard stop 2.
	//
	// #565b's rule already covered `project` and this is its exact twin: RESOLUTION IS THE
	// RESOLVER'S OWN. So the record states the REF — identity, the same string on both
	// machines — and the worker resolves it against the index of the machine that is going
	// to serve. A local run is unchanged by construction: there, both machines are this one.
	//
	// `components` stays, and is a different KIND of fact: it is the CLASS's declared
	// component set, read off the descriptor of the release BOTH machines installed. The
	// worker intersects it with what the artifact carries.
	record["model_class"] = slot.Class
	record["model_binding_path"] = slot.Path
	record["model_parameter_name"] = slot.Param
	record["artifact_ref"] = selected.Ref
	record["components"] = declaredComponents(slot)
	record["release"] = selected.Ref
	record["host_bytes"] = int64(hostBudget)
	record["pinned_bytes"] = int64(pinnedBudget)
	// `resident_budget_bytes` is DELIBERATELY ABSENT (cl-027, se-002 defect 4's ruling
	// finished). Runtime defines zero as "every component stays resident", so the
	// unconditional `0` this record used to carry was an unsafe all-resident declaration
	// authored by a host that never saw the card — the same ninth-field mistake as the
	// deleted vramBudget (#569b). The residence ceiling derives from card + artifact on
	// the machine that holds both; zero is the EXPLICIT operator override meaning
	// all-resident, never a default this side spells for it.
	record["tenancy"] = "local"
	record["strict_keys"] = false
	return &orchestrator.Binding{
		Entrypoint: ep.Name, Record: record, Outputs: AssetPaths(ep.Result),
	}, nil
}

// declaredComponents is the UNION of the component names a slot's class declares across all
// its methods — the set that class's construction binds, and the scope its read plan is
// checked against (#570b).
//
// It used to read `slot.ComponentUse[ep.Name]`, which is always EMPTY: `component_use` is
// keyed by METHOD (`predict_data_velocity`, `condition_text`, …) and never by entrypoint.
// The bug was invisible while an empty set fell back to the artifact's whole component
// list — every binding simply took all of them, which is what a single-component artifact
// wants anyway. On H3's N-ary artifact it stopped being invisible: Ref2VAModel was handed
// all five components including the sibling `transformer` its contract forbids it to touch,
// and construction refused `undeclared_component`.
//
// An empty union still means "take the artifact whole", which is the honest reading for a
// class that declares no component use at all.
func declaredComponents(slot Slot) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, names := range slot.ComponentUse {
		for _, n := range names {
			if !seen[n] {
				seen[n] = true
				out = append(out, n)
			}
		}
	}
	sort.Strings(out)
	return out
}
