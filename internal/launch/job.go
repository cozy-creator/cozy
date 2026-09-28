package launch

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
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

// JobFacts is one resolved `@job` on an installed package.
type JobFacts struct {
	Name     string
	Internal bool
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
			WithRemedy("it registers: %s", strings.Join(f.PackageInterface.PublicNames(), ", ")).
			WithNext("cozy package list --full")
	}
	if declared.DescriptorID == "" {
		return nil, exit.Named(exit.Structural, "job_descriptor_id_absent",
			"installed job %s has no descriptor identity", function)
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
		Name: function, Request: declared.Request, Result: declared.Result, Assets: declared.Assets, DescriptorID: declared.DescriptorID, Outputs: outputs,
		Internal:         declared.Internal,
		WeightsOutputs:   weightsOutputs,
		Publishes:        declared.Publishes,
		NeedsAccelerator: declared.NeedsAccelerator(strings.Split(f.Install.Closure, "\n")),
	}
	if declared.Accelerator == nil && f.CPUOrchestration && !f.SelfCallable[function] && len(declared.Models) == 0 && len(declared.WeightsOutputs) == 0 {
		facts.NeedsAccelerator = false
	}
	facts.RetainsArtifacts = len(ModelArtifactPaths(declared.Result)) > 0 || len(RetainedAssetPaths(declared)) > 0
	for _, model := range declared.Models {
		facts.ModelParams = append(facts.ModelParams, model.Param)
	}
	return facts, nil
}
