package api

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// resolveDescribedCall admits an unversioned call using the contract the CLI read
// from its machine (or selected Hub while the machine was unavailable). This is
// input/export metadata, not a release selection: Runtime resolves the package and
// validates the actual callable, models and payload when the machine accepts it.
func (s *Server) resolveDescribedCall(raw json.RawMessage, kind string, out orchestrator.Submission) (orchestrator.Submission, *launch.Entrypoint, *exit.Error) {
	if _, problem := hub.ParseRef(out.Package); problem != nil {
		return out, nil, problem
	}
	surface, problem := launch.DecodePackageInterface(raw)
	if problem != nil {
		return out, nil, problem
	}
	entry, problem := surface.Function(out.Entrypoint)
	if problem != nil {
		return out, nil, problem
	}
	if entry.Kind != kind {
		return out, nil, exit.New(exit.Validation, "%s/%s is a %s, not a %s", out.Package, out.Entrypoint, entry.Kind, kind)
	}
	if problem := validateInputs(entry, &out); problem != nil {
		return out, nil, problem
	}
	// Model choice syntax remains validated here; the selected package on the machine
	// owns whether each choice names an actual model parameter of its callable.
	if _, problem := orchestrator.ModelChoices(records.Request{Package: out.Package, Entrypoint: out.Entrypoint}, out.Models); problem != nil {
		return out, nil, problem
	}
	out.PlanID, out.Release, out.InstallID = "", "", ""
	out.Outputs = launch.OutputSlots(entry.Result)
	out.NeedsAccelerator = len(entry.Models) > 0
	if entry.Accelerator != nil {
		out.NeedsAccelerator = *entry.Accelerator
	}
	if kind == "job" {
		facts := launch.Facts{Install: records.PackageInstall{Package: out.Package}, PackageInterface: surface}
		job, problem := facts.Job(out.Entrypoint)
		if problem != nil {
			return out, nil, problem
		}
		out.PlanID, out.Outputs = job.DescriptorID, job.Outputs
		out.WeightsOutputs, out.ProducerParams = job.WeightsOutputs, job.ModelParams
		out.ChildArtifacts = job.RetainsArtifacts
	}
	if problem := s.deriveOutputExport(entry, &out); problem != nil {
		return out, nil, problem
	}
	return out, entry, nil
}
