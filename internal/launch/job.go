package launch

import (
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
)

// THE JOB HALF of an installed generation (cl-004). A job is an attempt class on the one
// machinery (cr-009), so this file mints exactly what the serving half mints — a local
// plan record and the digest that names it — over the descriptor's `jobs` list instead of
// its `entrypoints` list.
//
// The two records are deliberately NOT the same document, because they resolve different
// things: a binding plan resolves an artifact against a card, and a job plan resolves a
// CALLABLE plus its caps and its publication declaration. There is no residency, no
// component set and no construction digest on a job record — a job has no model
// residency at all (cr-009 §2).

// JobResourceCap is the orchestrator's declared bound for one local job attempt. Jobs are
// CPU-class here: the census-shaped work this host runs reads canonical headers and
// derives projections, and a job that needs a card declares `gpu_count` on its own
// surface, which the record below carries as a FLOOR.
const jobRSSBudget = int64(8) * gib

// JobFacts is one resolved `@job` on an installed generation.
type JobFacts struct {
	Name string
	// DescriptorID is `job_descriptor_id`: sha256 over the canonical bytes of
	// `{"format":"cozy.runtime.JobDescriptor/1", …the job's own descriptor entry}`.
	// DERIVED, never stored (cr-009's seam) — every environment of one release computes
	// the same one, and the worker resolves its local record by exactly this string.
	DescriptorID string
	// Outputs are the job's declared asset result field paths. They ARE the output ids
	// the publication grant names, one destination each.
	Outputs []string
	// Publishes is the job's own `publishes=` declaration. A grant mints off the
	// DECLARATION, never off the kind (cr-009).
	Publishes bool
	GPUCount  int64
}

// JobSpec builds the WorkerLaunchSpec that makes ONE job function's worker resident. It
// is the job lane's `Facts.Spec`: same worker, same venv, same device envelope, a
// JobDirective instead of a placement set.
//
// A JOB WORKER HOSTS NO PLACEMENT. Its DesiredPlacement carries `Jobs` and no bindings,
// which is what puts the DesiredWorkerState's `mode` oneof on the job branch — a worker is
// in exactly ONE mode until the next revision.
//
// ONE WORKER PER (endpoint, install, job function), and it is RECLAIMED at its outcome
// like every other job worker: one immutable build, one bounded attempt, outcome, reclaim
// (worker-protocol, cr-009). Deep queueing is the ORCHESTRATOR's — the dispatch queue holds
// the work and select-or-start makes the next worker resident — and it does not require
// keeping a job worker warm after it finishes. Warm persistence is a serving concern and
// stays one.
func (f *Facts) JobSpec(function string, devices []string) (orchestrator.WorkerLaunchSpec, *JobFacts, *exit.Error) {
	facts, e := f.Job(function)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, nil, e
	}
	spec := orchestrator.WorkerLaunchSpec{
		Placement: orchestrator.DesiredPlacement{
			Endpoint:         f.Install.Endpoint,
			ReleaseID:        ReleaseID(f.Install),
			InstallID:        f.Install.ID,
			DescriptorDigest: f.Install.Descriptor,
			Jobs: []*orchestrator.JobPlan{{
				Function:     facts.Name,
				DescriptorID: facts.DescriptorID,
				Outputs:      facts.Outputs,
				// The record's key set is CLOSED at both ends: `plan.py::JobBinding.read`
				// refuses an unknown key, exactly as the binding record's reader does.
				Record: map[string]any{
					"job_descriptor_id":           facts.DescriptorID,
					"build_id":                    ReleaseID(f.Install),
					"project":                     f.Source,
					"job":                         facts.Name,
					"gpu_count":                   facts.GPUCount,
					"publishes":                   facts.Publishes,
					"emits_media":                 false,
					"gpu_rate_micro_usd_per_hour": int64(0),
					"cap_micro_usd":               int64(0),
					"reclaim_on_terminal":         true,
				},
				RSSCap: jobRSSBudget,
			}},
		},
		// The same entry the serving lane uses: the runtime's own public verb (spec.go).
		Python:   Binary(f.Install.Dir),
		Args:     []string{"serve"},
		Dir:      f.Source,
		Devices:  devices,
		GraceSec: 3,
	}
	return spec, facts, nil
}

// Job resolves one declared `@job` and READS its descriptor id from the runtime that owns
// the derivation. cl-004 reproduced `internal/descriptor.py::job_descriptor_id` here in Go
// because no verb would say it; cr-016's `describe <job> --json` now carries it beside the
// entry, so the second implementation of the canonical form — and the whole class of
// refusals it owed for values the protocol profile cannot spell (a float bound, a null
// default) — deletes with it.
func (f *Facts) Job(function string) (*JobFacts, *exit.Error) {
	entry, e := f.jobEntry(function)
	if e != nil {
		return nil, e
	}
	var declared Entrypoint
	if err := json.Unmarshal(entry, &declared); err != nil {
		return nil, exit.Internalf("the descriptor's job entry for %q is unreadable: %s", function, err)
	}
	var said struct {
		DescriptorID string `json:"job_descriptor_id"`
	}
	if e := f.RuntimeCLI.call(&said, "describe", function); e != nil {
		return nil, e
	}
	if said.DescriptorID == "" {
		return nil, exit.Named(exit.Structural, "job_descriptor_id_absent",
			"`cozy-runtime describe %s` named no job_descriptor_id", function).
			WithRemedy("the id is the runtime's own derivation (cr-016); a release pinning an older runtime cannot be dispatched by it")
	}
	facts := &JobFacts{
		Name: function, DescriptorID: said.DescriptorID, Outputs: AssetPaths(declared.Result),
	}
	var shape struct {
		Publishes bool `json:"publishes"`
		Resources struct {
			GPUCount int64 `json:"gpu_count"`
		} `json:"resources"`
	}
	_ = json.Unmarshal(entry, &shape)
	facts.Publishes, facts.GPUCount = shape.Publishes, shape.Resources.GPUCount
	return facts, nil
}

// jobEntry finds one job's EXACT descriptor entry, as bytes. The typed `Descriptor.Jobs`
// view is a projection of this document and would not reproduce it — the digest is over
// what the release committed, so it is taken over the release's own bytes.
func (f *Facts) jobEntry(function string) (json.RawMessage, *exit.Error) {
	var doc struct {
		Jobs []json.RawMessage `json:"jobs"`
	}
	if err := json.Unmarshal(f.Descriptor.Raw, &doc); err != nil {
		return nil, exit.Internalf("this generation's descriptor is unreadable: %s", err)
	}
	names := []string{}
	for _, raw := range doc.Jobs {
		var named struct {
			Name string `json:"name"`
		}
		if json.Unmarshal(raw, &named) != nil {
			continue
		}
		names = append(names, named.Name)
		if named.Name == function {
			return raw, nil
		}
	}
	known := strings.Join(names, ", ")
	if known == "" {
		known = "no jobs"
	}
	return nil, exit.Named(exit.NotFound, "unknown_job",
		"%s registers no job named %q", f.Install.Endpoint, function).
		WithRemedy("it registers: %s", known).
		WithNext("cozy describe " + f.Install.Endpoint)
}
