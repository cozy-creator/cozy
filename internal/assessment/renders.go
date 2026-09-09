package assessment

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// RenderInspection is the installed cozy-eval reader's @3 association projection.
// It contains no metric parser or implementation of the evaluator's gate rules.
type RenderInspection struct {
	Schema string `json:"schema"`
	Report struct {
		Digest string `json:"digest"`
		Length int64  `json:"length"`
		Schema string `json:"schema"`
	} `json:"report"`
	Subject struct {
		Candidate string `json:"candidate_checkpoint"`
		Reference string `json:"reference_checkpoint"`
		Workloads []struct {
			Entrypoint string `json:"entrypoint"`
			Digest     string `json:"digest"`
		} `json:"workloads"`
		Arms map[string]RenderArm `json:"arms"`
	} `json:"subject"`
	Environment     map[string]Environment `json:"environment"`
	Verdict         string                 `json:"publisher_reported_verdict"`
	Required        []string               `json:"required"`
	SamePod         bool                   `json:"same_pod"`
	SamePodRequired bool                   `json:"same_pod_required"`
}

type RenderArm struct {
	Checkpoint string   `json:"checkpoint"`
	Requests   []string `json:"requests"`
	Media      []string `json:"media"`
	Audio      []string `json:"audio"`
	Captures   []string `json:"captures"`
}

// BoundRender identifies only the retained execution/input/media binding checked
// here. It does NOT authorize attachment: cr-113 must additionally bind the reported
// capture content identity and observed execution environment to this same outcome.
type BoundRender struct {
	Arm              string
	RequestID        string
	Ordinal          int64
	InvocationDigest string
	OutcomeDigest    string
}

func renderRefusal() *exit.Error {
	return exit.Named(exit.Conflict, "assessment.render_binding_mismatch", "the report or workload does not match retained render execution")
}

// ReadRenderInspection accepts facts only from the caller's trusted installed report
// inspector. The original artifact bytes must match its report identity exactly.
func ReadRenderInspection(report, inspected []byte) (RenderInspection, *exit.Error) {
	var out RenderInspection
	if len(report) == 0 || len(report) > MaxBytes || len(inspected) > MaxBytes || json.Unmarshal(inspected, &out) != nil ||
		out.Schema != "cozy-eval/report-inspection@4" || out.Report.Schema != "cozy-eval/checkpoint-validation@4" ||
		out.Report.Digest != spell(report) || out.Report.Length != int64(len(report)) ||
		len(out.Subject.Arms) != 3 || len(out.Environment) != 3 {
		return out, renderRefusal()
	}

	var envelope struct {
		Body struct {
			Environment map[string]Environment `json:"environment"`
		} `json:"body"`
	}
	if json.Unmarshal(report, &envelope) != nil || len(envelope.Body.Environment) != 3 {
		return out, renderRefusal()
	}
	for arm, projected := range out.Environment {
		full, ok := envelope.Body.Environment[arm]
		if !ok || !sameProjection(projected, full) {
			return out, renderRefusal()
		}
	}
	out.Environment = envelope.Body.Environment
	switch out.Verdict {
	case "pass", "fail", "indeterminate":
	default:
		return out, renderRefusal()
	}
	return out, nil
}

// VerifyRenderBindings validates canonical workload preimages, the captured child
// target, ordered payloads, distinct successful executions, checkpoint inputs, boot
// identity and terminal media/audio outputs. Workload.entrypoint is the captured
// invocable's module.export, not a mutable package handle. Model argument slots are
// excluded from the payload comparison because the candidate/reference intentionally
// select different checkpoints; their input binding is checked separately.
//
// parentID is the retained render-composition owner, resolved from report custody;
// it need not be the parent invoking attachment after a nested composition returns.
// This is deliberately narrower than full assessment verification. A caller must
// still verify cr-113 capture and execution observations before any Hub mutation.
func VerifyRenderBindings(st *records.Store, parentID string, info RenderInspection, workloadBytes []byte) ([]BoundRender, *exit.Error) {
	if len(workloadBytes) == 0 || len(workloadBytes) > MaxBytes {
		return nil, renderRefusal()
	}
	normalized, err := canonical.NormalizeJCS(workloadBytes)
	if err != nil || !bytes.Equal(normalized, workloadBytes) {
		return nil, renderRefusal()
	}
	var preimages []json.RawMessage
	if json.Unmarshal(workloadBytes, &preimages) != nil || len(preimages) == 0 || len(preimages) != len(info.Subject.Workloads) {
		return nil, renderRefusal()
	}
	parent, problem := st.RequestRow(parentID)
	if problem != nil {
		return nil, problem
	}
	if parent == nil || !parent.IsJob() || !parent.RetainWork || parent.InstallID == "" {
		return nil, renderRefusal()
	}
	bindings, problem := st.ChildBindings(parent.InstallID)
	if problem != nil {
		return nil, problem
	}
	type invocation struct {
		binding records.ChildBinding
		payload []byte
	}
	var inputs []invocation
	for i, raw := range preimages {
		var workload struct {
			Entrypoint string            `json:"entrypoint"`
			Payloads   []json.RawMessage `json:"payloads"`
		}
		if spell(raw) != info.Subject.Workloads[i].Digest || json.Unmarshal(raw, &workload) != nil || workload.Entrypoint != info.Subject.Workloads[i].Entrypoint || len(workload.Payloads) == 0 {
			return nil, renderRefusal()
		}
		var matches []records.ChildBinding
		for _, binding := range bindings {
			if binding.Module+"."+binding.Export == workload.Entrypoint {
				matches = append(matches, binding)
			}
		}
		if len(matches) != 1 {
			return nil, renderRefusal()
		}
		for _, payload := range workload.Payloads {
			if len(payload) == 0 || payload[0] != '{' {
				return nil, renderRefusal()
			}
			inputs = append(inputs, invocation{matches[0], payload})
		}
	}
	if len(inputs) == 0 {
		return nil, renderRefusal()
	}
	seen := map[string]bool{}
	out := make([]BoundRender, 0, len(inputs)*3)
	for _, name := range []string{"reference", "repeat", "candidate"} {
		arm, ok := info.Subject.Arms[name]
		checkpoint := info.Subject.Reference
		if name == "candidate" {
			checkpoint = info.Subject.Candidate
		}
		if !ok || arm.Checkpoint != checkpoint || len(arm.Requests) != len(inputs) || len(arm.Media) != len(inputs) ||
			(len(arm.Audio) != 0 && len(arm.Audio) != len(inputs)) || (len(arm.Captures) != 0 && len(arm.Captures) != len(inputs)) {
			return nil, renderRefusal()
		}
		for i, id := range arm.Requests {
			if seen[id] {
				return nil, renderRefusal()
			}
			seen[id] = true
			request, problem := st.RequestRow(id)
			if problem != nil {
				return nil, problem
			}
			target := inputs[i]
			if request == nil || request.ParentRequestID != parentID || request.InstallID != target.binding.ChildInstallID || request.Entrypoint != target.binding.Entrypoint ||
				request.State != "succeeded" || request.ChildReusable || request.ReusedFrom != "" || len(request.Models) != 1 || request.Models[0].Manifest != checkpoint {
				return nil, renderRefusal()
			}
			arguments, problem := st.ChildArguments(*request)
			if problem != nil {
				return nil, problem
			}
			intent := map[string]any{"interface_digest": target.binding.InterfaceDigest, "module": target.binding.Module, "export": target.binding.Export, "request": json.RawMessage(arguments)}
			if request.Capture != "" {
				intent["capture"] = json.RawMessage(request.Capture)
			}
			identity, _ := json.Marshal(intent)
			identity, err = canonical.NormalizeJCS(identity)
			if err != nil || spell(identity) != request.ChildIntentDigest {
				return nil, renderRefusal()
			}
			var payload map[string]json.RawMessage
			if json.Unmarshal(request.Payload, &payload) != nil {
				return nil, renderRefusal()
			}
			if request.IsJob() {
				for _, model := range request.Models {
					delete(payload, model.Slot)
				}
			}
			projected, _ := json.Marshal(payload)
			projected, err = canonical.NormalizeJCS(projected)
			if err != nil || !bytes.Equal(projected, target.payload) {
				return nil, renderRefusal()
			}
			attempt, problem := st.AttemptRow(id, request.Ordinal)
			if problem != nil {
				return nil, problem
			}
			if attempt == nil || (attempt.State != "terminal" && attempt.State != "closed") || attempt.TerminalStatus != "SUCCEEDED" ||
				attempt.SessionID == "" || attempt.SessionID != info.Environment[name].WorkerBootID ||
				spell(attempt.InvocationCanonical) != attempt.InvocationDigest || spell(attempt.TerminalBody) != attempt.TerminalDigest {
				return nil, renderRefusal()
			}
			spec, err := canonical.Read(attempt.InvocationCanonical, &pb.InvocationSpec{})
			if err != nil || spec.Str("payload_digest") != spell(request.Payload) {
				return nil, renderRefusal()
			}
			if request.IsJob() {
				if spec.Sub("job").Str("build_id") != target.binding.LocalRevisionDigest {
					return nil, renderRefusal()
				}
				modelInput := false
				for _, input := range spec.List("inputs") {
					if input.Str("input_id") == "model:"+request.Models[0].Slot && input.Str("digest") == checkpoint {
						modelInput = true
					}
				}
				if !modelInput {
					return nil, renderRefusal()
				}
			} else if !servingModelMatches(*request, *attempt, target.binding, checkpoint) {
				return nil, renderRefusal()
			}
			terminal, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
			if err != nil || terminal.Str("request_id") != id || terminal.Int("attempt_ordinal") != request.Ordinal || terminal.Str("invocation_spec_digest") != attempt.InvocationDigest || terminal.Int("status") != int64(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED) || terminal["execution_started"] != true {
				return nil, renderRefusal()
			}
			outputs := terminal.Sub("output_manifest").List("outputs")
			hasOutput := func(digest, mediaPrefix string) bool {
				for _, output := range outputs {
					if output.Str("digest") == digest && strings.HasPrefix(output.Str("mime_type"), mediaPrefix) && output.Int("length") > 0 {
						return true
					}
				}
				return false
			}
			if !hasOutput(arm.Media[i], "image/") && !hasOutput(arm.Media[i], "video/") {
				return nil, renderRefusal()
			}
			if len(arm.Audio) != 0 && !hasOutput(arm.Audio[i], "audio/") {
				return nil, renderRefusal()
			}
			out = append(out, BoundRender{name, id, request.Ordinal, attempt.InvocationDigest, attempt.TerminalDigest})
		}
	}
	return out, nil
}

func servingModelMatches(request records.Request, attempt records.Attempt, binding records.ChildBinding, checkpoint string) bool {
	placement, problem := records.BoundServingPlacement(request, attempt)
	if problem != nil || placement == nil || placement.Sub("development").Str("local_revision_digest") != binding.LocalRevisionDigest || placement.Sub("package_interface").Str("digest") != binding.InterfaceDigest {
		return false
	}
	models := map[string]canonical.Doc{}
	for _, model := range placement.List("models") {
		id := model.Str("id")
		if id == "" || models[id] != nil {
			return false
		}
		models[id] = model
	}
	for _, entrypoint := range placement.List("entrypoints") {
		if entrypoint.Str("name") != request.Entrypoint {
			continue
		}
		slots := entrypoint.List("slots")
		if len(slots) != 1 || slots[0].Str("slot") != request.Models[0].Slot {
			return false
		}
		model := models[slots[0].Str("reference_model_id")]
		return model != nil && model.Sub("manifest").Str("digest") == checkpoint && model.Sub("manifest").Int("length") == request.Models[0].ManifestLength
	}
	return false
}

func spell(raw []byte) string { out, _ := canonical.Spell(canonical.Digest(raw)); return out }
