// Package workflow owns the one durable ordered-workflow form. Endpoints remain ordinary
// request producers: only Creator parses this plan, materializes backward output bindings,
// and submits ordinary child requests.
package workflow

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/inputasset"
)

const (
	PlanFormat        = "cozy.workflow.Plan/1"
	MaxSteps          = 16
	MaxPayloadBytes   = 8 << 20
	MaxFieldPathDepth = 16
	MaxFieldPathIndex = 1024
)

// Plan is execution meaning only. Local install ids and staged paths are resolution facts
// and therefore do not have fields here.
type Plan struct {
	Format             string `json:"format"`
	CreativePlanDigest string `json:"creative_plan_digest"`
	Steps              []Step `json:"steps"`
}

type Step struct {
	Endpoint                string          `json:"endpoint"`
	EndpointReleaseID       string          `json:"endpoint_release_id"`
	Entrypoint              string          `json:"entrypoint"`
	EntrypointBindingPlanID string          `json:"entrypoint_binding_plan_id"`
	PayloadBase64           string          `json:"payload_b64"`
	Outputs                 []string        `json:"outputs"`
	Assets                  []AssetClaim    `json:"assets,omitempty"`
	Bindings                []OutputBinding `json:"bindings,omitempty"`
}

// AssetClaim is content identity. Its service-owned staged path is supplied separately
// and never enters the plan or workflow digest.
type AssetClaim struct {
	FieldPath string `json:"field_path"`
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type"`
	Kind      string `json:"kind"`
	Order     uint32 `json:"order"`
}

// OutputBinding can only name an earlier ordinal. Array position is the one-based step
// ordinal, which makes a cycle or forward edge unrepresentable after validation.
type OutputBinding struct {
	FieldPath         string `json:"field_path"`
	PriorStep         int    `json:"prior_step"`
	OutputName        string `json:"output_name"`
	ExpectedMediaKind string `json:"expected_media_kind"`
}

func DecodePlan(data []byte) (*Plan, []byte, string, *exit.Error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var plan Plan
	if err := decoder.Decode(&plan); err != nil {
		return nil, nil, "", exit.Named(exit.Validation, "workflow_plan_malformed",
			"the workflow plan is not a strict %s document: %s", PlanFormat, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return nil, nil, "", exit.Named(exit.Validation, "workflow_plan_malformed",
			"the workflow plan carries trailing JSON after its document")
	}
	if e := plan.validate(); e != nil {
		return nil, nil, "", e
	}
	canonicalBytes, err := canonical.Write(plan.document())
	if err != nil {
		return nil, nil, "", exit.Named(exit.Validation, "workflow_plan_noncanonical",
			"the workflow plan cannot enter the canonical identity profile: %s", err)
	}
	digest, err := canonical.Spell(canonical.Digest(canonicalBytes))
	if err != nil {
		return nil, nil, "", exit.Internalf("cannot spell the workflow plan digest: %s", err)
	}
	return &plan, canonicalBytes, digest, nil
}

func (p Plan) validate() *exit.Error {
	if p.Format != PlanFormat {
		return exit.Named(exit.Validation, "workflow_plan_format",
			"workflow format %q is not %q", p.Format, PlanFormat)
	}
	if _, err := canonical.Raw(p.CreativePlanDigest); err != nil {
		return exit.Named(exit.Validation, "workflow_creative_digest",
			"creative_plan_digest is not a lowercase sha256 digest: %s", err)
	}
	if len(p.Steps) == 0 || len(p.Steps) > MaxSteps {
		return exit.Named(exit.Validation, "workflow_step_count",
			"a workflow has 1-%d steps; this plan has %d", MaxSteps, len(p.Steps))
	}
	for index, step := range p.Steps {
		ordinal := index + 1
		if e := step.validate(ordinal, p.Steps); e != nil {
			return e
		}
	}
	return nil
}

func (s Step) validate(ordinal int, steps []Step) *exit.Error {
	if strings.TrimSpace(s.Endpoint) == "" || strings.TrimSpace(s.Entrypoint) == "" {
		return invalidStep(ordinal, "endpoint and entrypoint are required")
	}
	if strings.TrimSpace(s.EndpointReleaseID) == "" || !strings.Contains(s.EndpointReleaseID, "@") {
		return invalidStep(ordinal, "endpoint_release_id must be an exact release identity")
	}
	lowerRelease := strings.ToLower(s.EndpointReleaseID)
	for _, mutable := range []string{"@latest", "@main", "@master"} {
		if strings.HasSuffix(lowerRelease, mutable) {
			return invalidStep(ordinal, "endpoint_release_id may not be a mutable ref")
		}
	}
	if _, err := canonical.Raw(s.EntrypointBindingPlanID); err != nil {
		return invalidStep(ordinal, "entrypoint_binding_plan_id is not a lowercase sha256 digest")
	}
	payload, err := base64.StdEncoding.Strict().DecodeString(s.PayloadBase64)
	if err != nil || len(payload) > MaxPayloadBytes || !jsonObject(payload) {
		return invalidStep(ordinal,
			fmt.Sprintf("payload_b64 must encode one JSON object of at most %d bytes", MaxPayloadBytes))
	}
	outputs := map[string]bool{}
	for _, output := range s.Outputs {
		if output == "" || outputs[output] {
			return invalidStep(ordinal, "outputs must be non-empty and unique")
		}
		outputs[output] = true
	}
	if len(outputs) == 0 {
		return invalidStep(ordinal, "at least one declared output is required")
	}
	targets := map[string]bool{}
	for _, asset := range s.Assets {
		if validateFieldPath(asset.FieldPath) != nil || inputasset.ValidateID(asset.FieldPath) != nil ||
			targets[asset.FieldPath] {
			return invalidStep(ordinal, "each asset field_path must be non-empty and unique")
		}
		targets[asset.FieldPath] = true
		if _, err := canonical.Raw(asset.Digest); err != nil || asset.Length <= 0 {
			return invalidStep(ordinal, "every asset needs an exact digest and positive length")
		}
		if !validKind(asset.Kind) || kindOf(asset.MediaType) != asset.Kind {
			return invalidStep(ordinal, "asset kind and media_type disagree")
		}
	}
	for _, binding := range s.Bindings {
		if validateFieldPath(binding.FieldPath) != nil || inputasset.ValidateID(binding.FieldPath) != nil ||
			targets[binding.FieldPath] {
			return invalidStep(ordinal,
				"each authored asset or prior-output binding must own one unique target field")
		}
		targets[binding.FieldPath] = true
		if binding.PriorStep < 1 || binding.PriorStep >= ordinal {
			return invalidStep(ordinal, "a binding may name only an earlier step ordinal")
		}
		if !validKind(binding.ExpectedMediaKind) {
			return invalidStep(ordinal, "expected_media_kind is not image, video, audio, or file")
		}
		priorOutputs := map[string]bool{}
		for _, name := range steps[binding.PriorStep-1].Outputs {
			priorOutputs[name] = true
		}
		if binding.OutputName == "" || !priorOutputs[binding.OutputName] {
			return invalidStep(ordinal, "a binding names an output the prior step does not declare")
		}
	}
	return nil
}

func validateFieldPath(path string) *exit.Error {
	if path == "" || len(path) > 512 {
		return exit.Named(exit.Validation, "workflow_asset_path",
			"workflow asset field path is empty or longer than 512 bytes")
	}
	parts := strings.Split(path, ".")
	if len(parts) > MaxFieldPathDepth {
		return exit.Named(exit.Validation, "workflow_asset_path",
			"workflow asset field path %q is deeper than %d", path, MaxFieldPathDepth)
	}
	for _, part := range parts {
		if part == "" || len(part) > 128 || strings.ContainsAny(part, `/\`) {
			return exit.Named(exit.Validation, "workflow_asset_path",
				"workflow asset field path %q has a malformed component", path)
		}
		if index, err := strconv.ParseUint(part, 10, 31); err == nil && index > MaxFieldPathIndex {
			return exit.Named(exit.Validation, "workflow_asset_path",
				"workflow asset field path %q index %d exceeds %d",
				path, index, MaxFieldPathIndex)
		}
	}
	return nil
}

func invalidStep(ordinal int, detail string) *exit.Error {
	return exit.Named(exit.Validation, "workflow_step_invalid", "workflow step %d: %s", ordinal, detail)
}

func jsonObject(data []byte) bool {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value map[string]any
	return decoder.Decode(&value) == nil && value != nil && decoder.Decode(&struct{}{}) == io.EOF
}

func validKind(kind string) bool {
	switch kind {
	case "image", "video", "audio", "file":
		return true
	}
	return false
}

func kindOf(mediaType string) string {
	prefix, _, _ := strings.Cut(strings.ToLower(strings.TrimSpace(mediaType)), "/")
	switch prefix {
	case "image", "video", "audio":
		return prefix
	}
	if mediaType != "" {
		return "file"
	}
	return ""
}

func (p Plan) document() map[string]canonical.Value {
	steps := make([]canonical.Value, 0, len(p.Steps))
	for _, step := range p.Steps {
		steps = append(steps, step.document())
	}
	return map[string]canonical.Value{
		"format":               PlanFormat,
		"creative_plan_digest": p.CreativePlanDigest,
		"steps":                steps,
	}
}

func (s Step) document() map[string]canonical.Value {
	outputs := make([]canonical.Value, 0, len(s.Outputs))
	for _, output := range s.Outputs {
		outputs = append(outputs, output)
	}
	assets := make([]canonical.Value, 0, len(s.Assets))
	for _, asset := range s.Assets {
		assets = append(assets, map[string]canonical.Value{
			"field_path": asset.FieldPath, "digest": asset.Digest, "length": asset.Length,
			"media_type": asset.MediaType, "kind": asset.Kind, "order": int64(asset.Order),
		})
	}
	bindings := make([]canonical.Value, 0, len(s.Bindings))
	for _, binding := range s.Bindings {
		bindings = append(bindings, map[string]canonical.Value{
			"field_path": binding.FieldPath, "prior_step": int64(binding.PriorStep),
			"output_name": binding.OutputName, "expected_media_kind": binding.ExpectedMediaKind,
		})
	}
	return map[string]canonical.Value{
		"endpoint": s.Endpoint, "endpoint_release_id": s.EndpointReleaseID,
		"entrypoint": s.Entrypoint, "entrypoint_binding_plan_id": s.EntrypointBindingPlanID,
		"payload_b64": s.PayloadBase64, "outputs": outputs, "assets": assets, "bindings": bindings,
	}
}
