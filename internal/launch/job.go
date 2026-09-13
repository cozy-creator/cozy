package launch

import (
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

// THE JOB HALF of an installed package (cl-004). A job is an attempt class on the one
// machinery (cr-009), so this file mints exactly what the serving half mints — a local
// plan record and the digest that names it — over the PackageInterface's `jobs` list instead of
// its `entrypoints` list.
//
// The two records are deliberately NOT the same document, because they resolve different
// things: a binding plan resolves an artifact against a card, and a job plan resolves a
// CALLABLE plus its caps and its publication declaration. There is no residency, no
// component set and no construction digest on a job record — a job has no model
// residency at all (cr-009 §2).

// JobResourceCap is the orchestrator's declared host-memory bound for one local job attempt.
const jobRSSBudget = orchestrator.DefaultJobRSSCap

// JobFacts is one resolved `@job` on an installed package.
type JobFacts struct {
	Name string
	// Request is the exact callable schema used by the admission authority before
	// the request enters the ordinary queue.
	Request          Struct
	Result           Struct
	Assets           *AssetsSlot
	RetainsArtifacts bool
	// DescriptorID is `job_descriptor_id`: sha256 over the canonical bytes of
	// `{"format":"cozy.runtime.JobDescriptor/1", …the job's own descriptor entry}`.
	// DERIVED, never stored (cr-009's seam) — every environment of one release computes
	// the same one, and the worker resolves its local record by exactly this string.
	DescriptorID string
	// Outputs are the job's declared asset result field paths. They ARE the output ids
	// the publication grant names, one destination each.
	Outputs        []string
	WeightsOutputs []orchestrator.WeightsOutput
	ModelParams    []string
	// Publishes is the job's own `publishes=` declaration. A grant mints off the
	// DECLARATION, never off the kind (cr-009).
	Publishes        bool
	NeedsAccelerator bool
}

// JobSpec builds the WorkerLaunchSpec that makes ONE job function's worker resident. It
// is the job lane's `Facts.Spec`: same worker, same venv, same device envelope, a
// JobDirective instead of a placement set.
//
// A JOB WORKER HOSTS NO PLACEMENT. Its DesiredPlacement carries `Jobs` and no bindings,
// which is what puts the DesiredWorkerState's `mode` oneof on the job branch — a worker is
// in exactly ONE mode until the next revision.
//
// ONE WORKER PER (package, install, job function), and it is RECLAIMED at its outcome
// like every other job worker: one immutable build, one bounded attempt, outcome, reclaim
// (worker-protocol, cr-009). Deep queueing is the ORCHESTRATOR's — the dispatch queue holds
// the work and select-or-start makes the next worker resident — and it does not require
// keeping a job worker warm after it finishes. Warm persistence is a serving concern and
// stays one.
func (f *Facts) JobSpec(function string, devices []string) (orchestrator.WorkerLaunchSpec, *JobFacts, *exit.Error) {
	if f.Install.SourceKind == "local" && (f.Install.ProjectDir == "" ||
		filepath.Clean(f.Install.SourceRef) != filepath.Join(f.Install.Dir, "source")) {
		return orchestrator.WorkerLaunchSpec{}, nil, exit.Named(exit.Structural,
			"job_snapshot_required", "local jobs require a captured source revision").
			WithRemedy("run the local job through `cozy run` to capture its source and dependencies")
	}
	facts, e := f.Job(function)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, nil, e
	}
	if !facts.NeedsAccelerator {
		devices = nil
	}
	runtimeBin, e := hostruntime.Path(f.RuntimeCLI.Env)
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, nil, e
	}
	placement, buildID, e := f.JobCodeIdentity()
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, nil, e
	}
	cache := filepath.Join(f.Install.Dir, "artifact-cache")
	environmentPython := f.environmentPython()
	environmentContent := placement.EnvironmentDigest
	if environmentContent == "" {
		environmentContent = f.Install.LockDigest
	}
	placement.Jobs = []*orchestrator.JobPlan{{
		Function:       facts.Name,
		DescriptorID:   facts.DescriptorID,
		BuildID:        buildID,
		Outputs:        facts.Outputs,
		WeightsOutputs: facts.WeightsOutputs,
		// The record's key set is CLOSED at both ends: `plan.py::JobBinding.read`
		// refuses an unknown key, exactly as the binding record's reader does.
		Record: map[string]any{
			"job_descriptor_id":           facts.DescriptorID,
			"build_id":                    buildID,
			"application":                 f.PackageInterface.Application,
			"package_interface":           PackageInterfacePath(f.Install.Dir),
			"python":                      environmentPython,
			"environment_content_digest":  environmentContent,
			"job":                         facts.Name,
			"publishes":                   facts.Publishes,
			"emits_media":                 false,
			"gpu_rate_micro_usd_per_hour": int64(0),
			"cap_micro_usd":               int64(0),
			"reclaim_on_terminal":         true,
		},
		RSSCap: jobRSSBudget, NeedsAccelerator: facts.NeedsAccelerator,
	}}
	spec := orchestrator.WorkerLaunchSpec{
		Placement: placement,
		// The same entry the serving lane uses: the runtime's own public verb (spec.go).
		Python:            runtimeBin,
		Args:              []string{"serve"},
		Dir:               f.Source,
		Devices:           devices,
		GraceSec:          3,
		ArtifactCache:     cache,
		EnvironmentPython: environmentPython,
		TensorFSRoot:      config.Frozen().TensorFSRoot,
	}
	return spec, facts, nil
}

// Job resolves one declared `@job` and READS its descriptor id from the runtime that owns
// the derivation. cl-004 reproduced `internal/package_interface.py::job_descriptor_id` here in Go
// because no verb would say it; cr-016's `describe <job> --json` now carries it beside the
// entry, so the second implementation of the canonical form — and the whole class of
// refusals it owed for values the protocol profile cannot spell (a float bound, a null
// default) — deletes with it. The reader is THIS host's Runtime (hostruntime.Path, cl-175): a
// published install is read from its exact retained interface, an editable one from its
// source, and neither imports the package.
func (f *Facts) Job(function string) (*JobFacts, *exit.Error) {
	var declared *Entrypoint
	for i := range f.PackageInterface.Jobs {
		if f.PackageInterface.Jobs[i].Name == function {
			declared = &f.PackageInterface.Jobs[i]
			break
		}
	}
	if declared == nil {
		return nil, exit.Named(exit.NotFound, "unknown_job",
			"%s registers no job named %q", f.Install.Package, function).
			WithRemedy("it registers: %s", strings.Join(f.PackageInterface.Names(), ", ")).
			WithNext("cozy package list --full")
	}
	runtimeBin, problem := hostruntime.Path(f.RuntimeCLI.Env)
	if problem != nil {
		return nil, problem
	}
	var said struct {
		DescriptorID string `json:"job_descriptor_id"`
	}
	runtime := f.RuntimeCLI
	runtime.Bin = runtimeBin
	if e := runtime.call(&said, "describe", function); e != nil {
		return nil, e
	}
	if said.DescriptorID == "" {
		return nil, exit.Named(exit.Structural, "job_descriptor_id_absent",
			"`cozy-runtime describe %s` named no job_descriptor_id", function).
			WithRemedy("the id is the runtime's own derivation (cr-016); a release pinning an older runtime cannot be dispatched by it")
	}
	assets := AssetPaths(declared.Result)
	if len(assets) > 0 && len(declared.WeightsOutputs) > 0 {
		return nil, exit.Named(exit.Structural, "mixed_job_output_kinds",
			"job %s mixes %d result asset output(s) with %d weights output(s); rev5 OutputBinding cannot distinguish them",
			function, len(assets), len(declared.WeightsOutputs))
	}
	weightsOutputs := make([]orchestrator.WeightsOutput, 0, len(declared.WeightsOutputs))
	outputs := append([]string(nil), assets...)
	for _, output := range declared.WeightsOutputs {
		weightsOutputs = append(weightsOutputs, orchestrator.WeightsOutput{
			OutputID: output.OutputID, MimeType: output.MimeType, MaxBytes: output.MaxBytes,
		})
		outputs = append(outputs, output.OutputID)
	}
	facts := &JobFacts{
		Name: function, Request: declared.Request, Result: declared.Result, Assets: declared.Assets, DescriptorID: said.DescriptorID, Outputs: outputs,
		WeightsOutputs:   weightsOutputs,
		Publishes:        declared.Publishes,
		NeedsAccelerator: AcceleratorRequired(strings.Split(f.Install.Closure, "\n")) && !(f.CPUOrchestration && !f.SelfCallable[function] && len(declared.Models) == 0 && len(declared.WeightsOutputs) == 0),
	}
	if f.Install.Package == "local/"+runtimeoperation.Name && f.PackageInterface.Application == runtimeoperation.Application {
		// The fixed builtin encodes through native TensorFS/NumPy. Optional GPU
		// packages in the Runtime base do not turn this CPU operation into inference.
		facts.NeedsAccelerator = false
	}
	facts.RetainsArtifacts = len(ModelArtifactPaths(declared.Result)) > 0 || len(RetainedAssetPaths(declared)) > 0
	for _, model := range declared.Models {
		facts.ModelParams = append(facts.ModelParams, model.Param)
	}
	return facts, nil
}
