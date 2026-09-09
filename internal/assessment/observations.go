package assessment

import (
	"encoding/json"
	"reflect"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Environment contains only association facts from the validated report. Numerical
// interpretation and verdict derivation remain in cozy-eval.
type Environment struct {
	RuntimeVersion    string `json:"runtime_version"`
	WorkerImage       string `json:"worker_image"`
	Accelerator       string `json:"accelerator"`
	Driver            string `json:"driver"`
	CUDA              string `json:"cuda"`
	WorkerBootID      string `json:"worker_boot_id"`
	ExecutionLane     string `json:"execution_lane"`
	ExecutionContract string `json:"execution_contract"`
	KernelSymbol      string `json:"kernel_symbol"`
}

func sameProjection(projected, full Environment) bool {
	projected.Driver = full.Driver
	projected.CUDA = full.CUDA
	return projected == full
}
func observedEnvironment(doc canonical.Doc) Environment {
	return Environment{
		RuntimeVersion: doc.Str("runtime_version"), WorkerImage: doc.Str("worker_image_digest"), Accelerator: doc.Str("accelerator"), Driver: doc.Str("driver"), CUDA: doc.Str("cuda"), WorkerBootID: doc.Str("worker_boot_id"), ExecutionLane: doc.Str("execution_lane"), ExecutionContract: doc.Str("execution_contract_digest"), KernelSymbol: doc.Str("kernel_symbol"),
	}
}

// VerifyObservations joins the report to actual outcome-owned observations and
// independently retained native outputs. Empty image/driver/CUDA are legitimate
// local CPU facts; they must match exactly, and no eager/kernel fact is inferred.
func VerifyObservations(st *records.Store, owner string, info RenderInspection, workloads []byte, bound []BoundRender) ([]records.NativeArtifactRetention, *exit.Error) {
	refuse := func() ([]records.NativeArtifactRetention, *exit.Error) {
		return nil, exit.Named(exit.Conflict, "assessment.observation_mismatch", "assessment environment, capture or byte custody differs from retained execution")
	}
	var plans []struct {
		Payloads []json.RawMessage `json:"payloads"`
		Capture  []string          `json:"capture"`
		Steps    []uint32          `json:"capture_steps"`
	}
	if json.Unmarshal(workloads, &plans) != nil {
		return refuse()
	}
	type capture struct {
		Components []string `json:"components"`
		Steps      []uint32 `json:"steps"`
	}
	var expected []capture
	for _, plan := range plans {
		for range plan.Payloads {
			expected = append(expected, capture{plan.Capture, plan.Steps})
		}
	}
	counters := map[string]int{}
	var holds []records.NativeArtifactRetention
	for _, render := range bound {
		index := counters[render.Arm]
		counters[render.Arm]++
		arm, ok := info.Subject.Arms[render.Arm]
		if !ok || index >= len(expected) {
			return refuse()
		}
		request, problem := st.RequestRow(render.RequestID)
		if problem != nil {
			return nil, problem
		}
		if request == nil || request.ParentRequestID != owner || !request.RetainWork {
			return refuse()
		}
		attempt, problem := st.AttemptRow(render.RequestID, render.Ordinal)
		if problem != nil {
			return nil, problem
		}
		if attempt == nil || attempt.InvocationDigest != render.InvocationDigest || attempt.TerminalDigest != render.OutcomeDigest {
			return refuse()
		}
		terminal, err := canonical.Read(attempt.TerminalBody, &pb.AttemptOutcomeBody{})
		if err != nil {
			return refuse()
		}
		observation := terminal.Sub("observation")
		environment := observation.Sub("environment")
		if len(environment) == 0 || observedEnvironment(environment) != info.Environment[render.Arm] {
			return refuse()
		}
		if environment.Str("worker_boot_id") != attempt.SessionID || environment.Str("runtime_version") == "" || environment.Str("accelerator") == "" {
			return refuse()
		}
		var wanted []string
		wanted = append(wanted, arm.Media[index])
		if len(arm.Audio) > 0 {
			wanted = append(wanted, arm.Audio[index])
		}
		captureID := ""
		if len(arm.Captures) > 0 {
			if index >= len(arm.Captures) || arm.Captures[index] == "" || request.Capture == "" {
				return refuse()
			}
			var actual capture
			if json.Unmarshal([]byte(request.Capture), &actual) != nil || !reflect.DeepEqual(actual, expected[index]) {
				return refuse()
			}
			spec, err := canonical.Read(attempt.InvocationCanonical, &pb.InvocationSpec{})
			if err != nil {
				return refuse()
			}
			raw, _ := json.Marshal(spec.Sub("capture"))
			var accepted capture
			if json.Unmarshal(raw, &accepted) != nil || !reflect.DeepEqual(actual, accepted) {
				return refuse()
			}
			observed := observation.Sub("capture")
			if observed.Str("content_digest") != arm.Captures[index] || observed.Str("output_id") != "runtime.capture" {
				return refuse()
			}
			captureID = "runtime.capture"
		} else if request.Capture != "" || len(observation.Sub("capture")) != 0 {
			return refuse()
		}
		outputs, problem := st.ByteOutputs(render.RequestID, render.Ordinal)
		if problem != nil {
			return nil, problem
		}
		retained, problem := st.NativeArtifactRetentions(render.RequestID)
		if problem != nil {
			return nil, problem
		}
		for _, digest := range wanted {
			matched := false
			for _, output := range outputs {
				if output.Digest != digest {
					continue
				}
				h, ok := heldOutput(retained, owner, output)
				if ok {
					holds = append(holds, h)
					matched = true
					break
				}
			}
			if !matched {
				return refuse()
			}
		}
		if captureID != "" {
			matched := false
			for _, output := range outputs {
				if output.OutputID != captureID || output.MimeType != "application/vnd.cozy.tree-manifest" {
					continue
				}
				h, ok := heldOutput(retained, owner, output)
				if ok {
					holds = append(holds, h)
					matched = true
					break
				}
			}
			if !matched {
				return refuse()
			}
		}
	}
	for _, arm := range []string{"reference", "repeat", "candidate"} {
		if counters[arm] != len(expected) {
			return refuse()
		}
	}
	return holds, nil
}
func heldOutput(holds []records.NativeArtifactRetention, owner string, b records.ByteOutput) (records.NativeArtifactRetention, bool) {
	for _, h := range holds {
		if h.ArtifactKind == "tree" && h.Kind == "result" && h.ParentRequestID == owner && h.ProducerID == b.RequestID && h.ProducerAttempt == b.Attempt && h.ProducerOutputID == b.OutputID && h.State == "held" && h.TransactionID == b.ProducerRootID && h.ReceiptDigest == b.ReceiptDigest && h.ManifestID == b.ManifestID && h.ManifestLength == b.ManifestLength && h.ContentBytes == b.ContentBytes {
			return h, true
		}
	}
	return records.NativeArtifactRetention{}, false
}
