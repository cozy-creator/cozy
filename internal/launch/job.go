package launch

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
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

// JobResourceCap is the coordinator's declared bound for one local job attempt. Jobs are
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

// JobSpec builds the EndpointSpec that makes ONE job function's worker resident. It is
// the job lane's `Facts.Spec`: same supervisor, same venv, same device envelope, a
// JobDirective instead of a ServingDirective.
//
// ONE WORKER PER (endpoint, generation, job function), and it is RECLAIMED at its
// terminal like every other job worker: one immutable build, one bounded attempt,
// terminal, reclaim (worker-protocol, cr-009). Deep queueing is the COORDINATOR's — the
// dispatch queue holds the work and select-or-start makes the next worker resident — and
// it does not require keeping a job worker warm after it finishes. Warm persistence is a
// serving concern and stays one.
func (f *Facts) JobSpec(function string, devices []string) (coord.EndpointSpec, *JobFacts, *exit.Error) {
	facts, e := f.Job(function)
	if e != nil {
		return coord.EndpointSpec{}, nil, e
	}
	spec := coord.EndpointSpec{
		Endpoint:   f.Generation.Endpoint,
		ReleaseID:  ReleaseID(f.Generation),
		Generation: f.Generation.ID,
		Python:     f.Generation.Dir + "/venv/bin/python",
		Args: []string{"-c",
			"import sys; from cozy_runtime.internal.worker.session import main; " +
				"raise SystemExit(main(sys.argv[1:]))"},
		Dir:      f.Source,
		Devices:  devices,
		GraceSec: 3,
		Jobs: []*coord.JobPlan{{
			Function:     facts.Name,
			DescriptorID: facts.DescriptorID,
			Outputs:      facts.Outputs,
			// The record's key set is CLOSED at both ends: `plan.py::JobBinding.read`
			// refuses an unknown key, exactly as the binding record's reader does.
			Record: map[string]any{
				"job_descriptor_id":           facts.DescriptorID,
				"build_id":                    ReleaseID(f.Generation),
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
	}
	return spec, facts, nil
}

// Job resolves one declared `@job` and derives its descriptor id.
func (f *Facts) Job(function string) (*JobFacts, *exit.Error) {
	entry, e := f.jobEntry(function)
	if e != nil {
		return nil, e
	}
	id, e := jobDescriptorID(entry)
	if e != nil {
		return nil, e
	}
	var declared Entrypoint
	if err := json.Unmarshal(entry, &declared); err != nil {
		return nil, exit.Internalf("the descriptor's job entry for %q is unreadable: %s", function, err)
	}
	facts := &JobFacts{
		Name: function, DescriptorID: id, Outputs: AssetPaths(declared.Result),
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
		"%s registers no job named %q", f.Generation.Endpoint, function).
		WithRemedy("it registers: %s", known).
		WithNext("cozy describe " + f.Generation.Endpoint)
}

// jobDescriptorID reproduces `internal/descriptor.py::job_descriptor_id` — sha256 over
// the canonical bytes of the job's own entry under its format tag.
//
// It is DERIVED here rather than read, because the id is deliberately absent from the
// document (`_fixed_point` refuses a digest anywhere in a source-stable surface). The
// derivation is checked the only way it can be: the worker resolves its local record BY
// this string, so a wrong one is `unresolved_job_descriptor` on the very first Report and
// never a silently mismatched attempt.
//
// SEAM (cr-016): this host's canonical writer is the PROTOCOL profile — integer-only,
// printable ASCII, no null — and a descriptor may legitimately carry a float bound or a
// None default (cozy-runtime's own writer implements full JCS for those three). Such a
// job refuses HERE, by name, rather than being handed a plausible wrong digest. The fix
// is `cozy-runtime describe` printing the id it already knows how to derive.
func jobDescriptorID(entry json.RawMessage) (string, *exit.Error) {
	var decoded map[string]any
	dec := json.NewDecoder(strings.NewReader(string(entry)))
	if err := dec.Decode(&decoded); err != nil {
		return "", exit.Internalf("a descriptor job entry is not an object: %s", err)
	}
	doc := map[string]canonical.Value{}
	for k, v := range decoded {
		value, err := canonicalOf(k, v)
		if err != nil {
			return "", exit.Named(exit.Structural, "job_descriptor_id_underivable",
				"this job's descriptor entry carries %s", err).
				WithRemedy("this host derives the id under the protocol's integer-only ASCII profile; " +
					"the runtime that owns the derivation should print it (cr-016 seam)")
		}
		doc[k] = value
	}
	doc["format"] = "cozy.runtime.JobDescriptor/1"
	data, err := canonical.Write(doc)
	if err != nil {
		return "", exit.Named(exit.Structural, "job_descriptor_id_underivable",
			"the job's descriptor entry does not canonicalize: %s", err)
	}
	spelled, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot spell the job descriptor id: %s", err)
	}
	return spelled, nil
}

// canonicalOf converts one decoded JSON value into a canonical Value, or names why it
// cannot. `encoding/json` decodes every number as a float64, so an integral one is the
// integer it spells and a fractional one refuses — the same rule `coord.Binding.PlanID`
// applies to a record that crossed JSON.
func canonicalOf(path string, v any) (canonical.Value, error) {
	switch t := v.(type) {
	case nil:
		return nil, fmt.Errorf("a null at %s, which these documents cannot spell", path)
	case bool:
		return t, nil
	case string:
		return t, nil
	case float64:
		if t != float64(int64(t)) {
			return nil, fmt.Errorf("the fractional number %v at %s", t, path)
		}
		return int64(t), nil
	case []any:
		out := make([]canonical.Value, 0, len(t))
		for i, e := range t {
			value, err := canonicalOf(fmt.Sprintf("%s[%d]", path, i), e)
			if err != nil {
				return nil, err
			}
			out = append(out, value)
		}
		return out, nil
	case map[string]any:
		out := map[string]canonical.Value{}
		for k, e := range t {
			value, err := canonicalOf(path+"."+k, e)
			if err != nil {
				return nil, err
			}
			out[k] = value
		}
		return out, nil
	}
	return nil, fmt.Errorf("a %T at %s", v, path)
}
