package modeltransfer

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// ValidateSubmission binds a model-transfer intent to the already-resolved job
// PackageInterface by model-input and output-slot names only.
func ValidateSubmission(spec orchestrator.Submission) *exit.Error {
	intent := spec.ModelTransfer
	if intent == nil {
		return nil
	}
	if intent.Destination == "" && (!spec.RetainWork || spec.Kind != "job" || spec.Package == "cozy/platform") {
		return exit.New(exit.Validation, "unpublished source results require a retained ordinary job")
	}
	if !intent.HasAcquisition() && (spec.Rental || spec.RentalRequired) && spec.Worker == "" {
		// The existing rental workload declaration carries these exact model
		// identities to the Hub's closure union measurement, for jobs as well
		// as serving. Private operation-local manifests have no such authority.
		for _, model := range spec.Models {
			// A choice on a named rental is resolved to a Hub checkpoint before it runs: by
			// the machine (release roots) or by the daemon at dispatch (prepared path).
			if model.Choice && spec.RequestedRental != "" {
				continue
			}
			if !model.Downloadable() {
				return exit.Named(exit.Validation, "model_transfer.prepared_rental_required",
					"operation-local model inputs require an explicitly selected prepared rental").
					WithRemedy("use Hub-resolved checkpoints so Tensorhub can measure their closures before renting")
			}
		}
	}
	if spec.Rental || spec.RentalRequired {
		for _, file := range intent.SourceFiles {
			if len(file.Header) == 0 {
				return exit.Named(exit.Validation, "model_transfer.source_header_missing",
					"source header for %s must be inspected before renting", file.Member)
			}
		}
	}
	if len(intent.Outputs) != len(spec.WeightsOutputs) ||
		(intent.HasAcquisition() && (len(intent.SourceProfiles) == 0 ||
			!profilesCoverParams(intent.SourceProfiles, spec.ProducerParams))) {
		return exit.New(exit.Validation, "producer transfer inputs/outputs do not match the job PackageInterface")
	}
	declared := make(map[string]bool, len(spec.WeightsOutputs))
	for index := range spec.WeightsOutputs {
		output := &spec.WeightsOutputs[index]
		declared[output.OutputID] = true
	}
	for _, output := range intent.Outputs {
		if !declared[output.Name] {
			return exit.New(exit.Validation,
				"producer transfer output %s is absent from its PackageInterface", output.Name)
		}
	}
	return nil
}

// profilesCoverParams requires the caller's intent (--source-profile) to bind
// exactly the job's model inputs: the PackageInterface declares no source selection
// (cr-077), so dispatch is the one place a producer's sources are named.
func profilesCoverParams(intent map[string]string, params []string) bool {
	if len(intent) != len(params) {
		return false
	}
	for _, param := range params {
		if intent[param] == "" {
			return false
		}
	}
	return true
}
