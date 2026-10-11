package records

import (
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
)

// OperationKey names computation, never its run history or custody recipients.
// ChildTargetDigest binds the exact captured implementation and interface. The
// selected numerical environment is bound separately by QualifiedOperationKey.
// Only declared injected Models hide receipt provenance
// from the implementation; ordinary value arguments remain intact.
func OperationKey(request Request) (string, *exit.Error) {
	if _, err := canonical.Raw(request.ChildTargetDigest); err != nil {
		return "", exit.New(exit.Validation, "operation implementation identity is malformed")
	}
	var inputs map[string]json.RawMessage
	if json.Unmarshal(request.Payload, &inputs) != nil || inputs == nil {
		return "", exit.New(exit.Validation, "operation inputs must be one canonical object")
	}
	seen := map[string]bool{}
	for _, model := range request.Models {
		if model.Slot == "" || seen[model.Slot] || model.ManifestLength <= 0 {
			return "", exit.New(exit.Validation, "operation model binding is incomplete or repeated")
		}
		seen[model.Slot] = true
		if _, err := canonical.Raw(model.Manifest); err != nil {
			return "", exit.New(exit.Validation, "operation model manifest identity is malformed")
		}
		value := inputs[model.Slot]
		if len(value) == 0 {
			value = json.RawMessage("null")
		}
		artifact, problem := DecodeModelArtifact(value)
		if problem != nil {
			return "", problem
		}
		if artifact != nil && (artifact.Manifest.Digest != model.Manifest || artifact.Manifest.Length != model.ManifestLength) {
			return "", exit.Named(exit.Conflict, "operation.model_changed", "operation model input differs from its resolved manifest")
		}
		inputs[model.Slot], _ = json.Marshal(map[string]any{"manifest": ArtifactObjectRef{Digest: model.Manifest, Length: model.ManifestLength}})
	}
	identity := map[string]any{"target_digest": request.ChildTargetDigest, "inputs": inputs}
	if request.Capture != "" {
		identity["capture"] = json.RawMessage(request.Capture)
	}
	if request.AttentionKernel != "" {
		// A KERNEL NAMES COMPUTATION (cr-125). An 8-bit arm returns different bytes from the
		// same inputs, so a memo keyed without the pin would hand back another kernel's
		// result — which is the same lie a silent fallback would be, one layer up.
		identity["attention_kernel"] = request.AttentionKernel
	}
	if request.GPUs > 0 {
		identity["gpus"] = request.GPUs
	}
	raw, err := json.Marshal(identity)
	if err != nil {
		return "", exit.Internalf("cannot encode operation identity: %s", err)
	}
	raw, err = canonical.NormalizeJCS(raw)
	if err != nil {
		return "", exit.New(exit.Validation, "operation inputs are not canonical values")
	}
	digest, _ := canonical.Spell(canonical.Digest(raw))
	return digest, nil
}
