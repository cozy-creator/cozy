package modeltransfer

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// ValidateProducer checks the ordinary modeled job shape used by source preparation.
// Payload requirements are checked by the existing typed job payload validator.
// Output tensor schemas depend on the selected source and remain owned by the produced
// CozyTensors header; publication declares only bounded output slots here.
// The PackageInterface declares no source selection (cr-077): every model input is bound
// by the caller's --source-profile at dispatch.
func ValidateProducer(name string, job *launch.Entrypoint, supplied map[string]string) *exit.Error {
	if len(job.Models) == 0 {
		return exit.Named(exit.Validation, "model_producer.source_inputs_absent",
			"producer job %s has no typed model input", name)
	}
	declared := map[string]bool{}
	for _, slot := range job.Models {
		declared[slot.Param] = true
		if supplied[slot.Param] == "" {
			return exit.Named(exit.Validation, "model_producer.source_profile_absent",
				"producer job %s model input %s is unbound; pass --source-profile %s=<reviewed profile>", name, slot.Param, slot.Param)
		}
	}
	for param := range supplied {
		if !declared[param] {
			return exit.Named(exit.Validation, "model_producer.source_profile_unknown_slot",
				"--source-profile names %s, which is not a model input of producer job %s", param, name)
		}
	}
	if len(job.WeightsOutputs) == 0 {
		return exit.Named(exit.Validation, "model_producer.outputs_absent",
			"producer job %s has no WeightsSink model output", name)
	}
	return nil
}

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
			if !model.Published() {
				return exit.Named(exit.Validation, "model_transfer.prepared_rental_required",
					"unpublished model inputs require an explicitly selected prepared rental").
					WithRemedy("use published model references so Tensorhub can measure their closures before renting")
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
	platformPassThrough := spec.Package == "cozy/platform" && spec.Entrypoint == "model-pass-through"
	if spec.Package == "cozy/platform" || spec.Entrypoint == "model-pass-through" {
		if !platformPassThrough {
			return exit.New(exit.Validation, "platform pass-through requires exact package and function")
		}
		if !intent.HasAcquisition() {
			return exit.New(exit.Validation, "platform pass-through requires a source acquisition")
		}
		if len(intent.Outputs) != 1 || intent.Outputs[0].Name != "model" {
			return exit.New(exit.Validation, "platform pass-through requires exactly output model")
		}
		return nil
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
