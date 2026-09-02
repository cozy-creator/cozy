package modeltransfer

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// ValidateProducer checks only the callable shape model download/upload needs.
// A producer may leave its output contract open when the exact source determines
// the topology; TensorFS still derives and verifies the produced checkpoint facts.
func ValidateProducer(name string, job *launch.Entrypoint, supplied map[string]string) *exit.Error {
	if len(job.Models) == 0 {
		return exit.Named(exit.Validation, "model_producer.source_inputs_absent",
			"producer job %s has no typed model input", name)
	}
	declared := map[string]bool{}
	for _, slot := range job.Models {
		declared[slot.Param] = true
		if slot.SourceProfile != "" && supplied[slot.Param] != "" {
			return exit.Named(exit.Validation, "model_producer.source_profile_conflict",
				"producer job %s model input %s declares a TensorFS source profile; --source-profile cannot override it", name, slot.Param)
		}
		if slot.SourceProfile == "" && supplied[slot.Param] == "" {
			return exit.Named(exit.Validation, "model_producer.source_profile_absent",
				"producer job %s model input %s has no TensorFS source profile; declare one or pass --source-profile %s=<reviewed profile>", name, slot.Param, slot.Param)
		}
	}
	for param := range supplied {
		if !declared[param] {
			return exit.Named(exit.Validation, "model_producer.source_profile_unknown_slot",
				"--source-profile names %s, which is not a model input of producer job %s", param, name)
		}
	}
	for _, field := range job.Request.Fields {
		if field.Wire == "required" {
			return exit.Named(exit.Validation, "model_producer.argument_missing",
				"producer job %s requires argument %s", name, field.Name)
		}
	}
	if len(job.WeightsOutputs) == 0 {
		return exit.Named(exit.Validation, "model_producer.outputs_absent",
			"producer job %s has no WeightsSink model output", name)
	}
	return nil
}

// ValidateSubmission binds a model transfer intent to the already-resolved job
// descriptor. An omitted contract matches only another omission; when a producer
// declares one, exact topology and encoding equality remains mandatory.
func ValidateSubmission(spec orchestrator.Submission) *exit.Error {
	intent := spec.ModelTransfer
	if intent == nil {
		return nil
	}
	platformPassThrough := spec.Package == "cozy/platform" && spec.Entrypoint == "model-pass-through"
	if spec.Package == "cozy/platform" || spec.Entrypoint == "model-pass-through" {
		if !platformPassThrough {
			return exit.New(exit.Validation, "platform pass-through requires exact package and function")
		}
		if len(intent.Outputs) != 1 || intent.Outputs[0].Name != "model" ||
			intent.Outputs[0].RequiredContract != nil {
			return exit.New(exit.Validation, "platform pass-through requires exactly output model")
		}
		return nil
	}
	if len(intent.SourceProfiles) == 0 || len(intent.Outputs) != len(spec.WeightsOutputs) ||
		!sameProfileMap(intent.SourceProfiles, spec.ProducerProfiles) {
		return exit.New(exit.Validation, "producer transfer inputs/outputs do not match the job descriptor")
	}
	declared := make(map[string]*orchestrator.WeightsOutput, len(spec.WeightsOutputs))
	for index := range spec.WeightsOutputs {
		output := &spec.WeightsOutputs[index]
		declared[output.OutputID] = output
	}
	for _, output := range intent.Outputs {
		declaredOutput := declared[output.Name]
		if declaredOutput == nil || !sameContract(output.RequiredContract, declaredOutput.RequiredContract) {
			return exit.New(exit.Validation,
				"producer transfer output %s differs from its descriptor contract", output.Name)
		}
	}
	return nil
}

func sameProfileMap(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if right[key] != value {
			return false
		}
	}
	return true
}

func sameContract(left, right *records.ModelTransferContract) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.TopologyDigest == right.TopologyDigest &&
		strings.Join(left.Encodings, "\x00") == strings.Join(right.Encodings, "\x00")
}
