package workflow

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

const materializedFormat = "cozy.workflow.MaterializedSubmission/1"

type ResolvedOutput struct {
	Binding   OutputBinding
	RequestID string
	Attempt   int64
	Output    records.Output
}

type MaterializedAsset struct {
	FieldPath string `json:"field_path"`
	Digest    string `json:"digest"`
	Length    int64  `json:"length"`
	MediaType string `json:"media_type"`
	Order     uint32 `json:"order"`
}

type MaterializedSubmission struct {
	Format                  string              `json:"format"`
	Endpoint                string              `json:"endpoint"`
	EndpointReleaseID       string              `json:"endpoint_release_id"`
	Entrypoint              string              `json:"entrypoint"`
	EntrypointBindingPlanID string              `json:"entrypoint_binding_plan_id"`
	PayloadBase64           string              `json:"payload_b64"`
	Outputs                 []string            `json:"outputs"`
	Assets                  []MaterializedAsset `json:"assets"`
}

func MaterializeStep(plan *Plan, ordinal int, authored []records.AssetBinding,
	prior []ResolvedOutput) (MaterializedSubmission, []byte, string,
	[]records.ResolvedBinding, *exit.Error) {
	if ordinal < 1 || ordinal > len(plan.Steps) {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Internalf("workflow step ordinal %d is outside its plan", ordinal)
	}
	step := plan.Steps[ordinal-1]
	payload, _ := base64.StdEncoding.Strict().DecodeString(step.PayloadBase64)
	var document map[string]any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Internalf("validated workflow step %d payload no longer decodes: %s", ordinal, err)
	}

	authoredByField := map[string]records.AssetBinding{}
	for _, asset := range authored {
		authoredByField[asset.FieldPath] = asset
	}
	assets := make([]MaterializedAsset, 0, len(step.Assets)+len(step.Bindings))
	paths := map[string]bool{}
	for _, claim := range step.Assets {
		asset, ok := authoredByField[claim.FieldPath]
		if !ok || asset.Digest != claim.Digest || asset.Length != claim.Length ||
			asset.MediaType != claim.MediaType || asset.Order != claim.Order {
			return MaterializedSubmission{}, nil, "", nil,
				exit.Named(exit.Conflict, "workflow_asset_resolution",
					"workflow step %d asset %s does not match its staged content identity",
					ordinal, claim.FieldPath)
		}
		if e := setPayloadRef(document, claim.FieldPath, claim.Digest); e != nil {
			return MaterializedSubmission{}, nil, "", nil, e
		}
		paths[claim.FieldPath] = true
		assets = append(assets, MaterializedAsset{
			FieldPath: claim.FieldPath, Digest: claim.Digest, Length: claim.Length,
			MediaType: claim.MediaType, Order: claim.Order,
		})
	}
	if len(authoredByField) != len(step.Assets) {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Named(exit.Conflict, "workflow_asset_resolution",
				"workflow step %d carries staged assets its plan does not declare", ordinal)
	}

	priorByField := map[string]ResolvedOutput{}
	for _, output := range prior {
		priorByField[output.Binding.FieldPath] = output
	}
	resolved := make([]records.ResolvedBinding, 0, len(step.Bindings))
	for _, binding := range step.Bindings {
		one, ok := priorByField[binding.FieldPath]
		if !ok || one.Binding != binding || one.Output.OutputID != binding.OutputName {
			return MaterializedSubmission{}, nil, "", nil,
				exit.Named(exit.Conflict, "workflow_output_resolution",
					"workflow step %d binding %s has no exact prior output", ordinal, binding.FieldPath)
		}
		if kindOf(one.Output.MimeType) != binding.ExpectedMediaKind {
			return MaterializedSubmission{}, nil, "", nil,
				exit.Named(exit.Validation, "workflow_output_kind",
					"workflow step %d binding %s expected %s and prior output %s is %s",
					ordinal, binding.FieldPath, binding.ExpectedMediaKind,
					binding.OutputName, one.Output.MimeType)
		}
		if e := setPayloadRef(document, binding.FieldPath, one.Output.Digest); e != nil {
			return MaterializedSubmission{}, nil, "", nil, e
		}
		paths[binding.FieldPath] = true
		assets = append(assets, MaterializedAsset{
			FieldPath: binding.FieldPath, Digest: one.Output.Digest,
			Length: one.Output.Length, MediaType: one.Output.MimeType,
			Order: fieldOrder(binding.FieldPath),
		})
		resolved = append(resolved, records.ResolvedBinding{
			FieldPath: binding.FieldPath, PriorStep: binding.PriorStep,
			PriorRequestID: one.RequestID, PriorAttempt: one.Attempt,
			OutputName: binding.OutputName, Digest: one.Output.Digest,
			Length: one.Output.Length, MediaType: one.Output.MimeType,
			ExpectedMediaKind: binding.ExpectedMediaKind,
		})
	}
	if len(priorByField) != len(step.Bindings) || len(paths) != len(step.Assets)+len(step.Bindings) {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Internalf("workflow step %d materialization target set disagrees", ordinal)
	}
	renderedPayload, err := json.Marshal(document)
	if err != nil {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Internalf("cannot render workflow step %d payload: %s", ordinal, err)
	}
	materialized := MaterializedSubmission{
		Format: materializedFormat, Endpoint: step.Endpoint,
		EndpointReleaseID: step.EndpointReleaseID, Entrypoint: step.Entrypoint,
		EntrypointBindingPlanID: step.EntrypointBindingPlanID,
		PayloadBase64:           base64.StdEncoding.EncodeToString(renderedPayload),
		Outputs:                 append([]string(nil), step.Outputs...), Assets: assets,
	}
	canonicalBytes, err := canonical.Write(materialized.document())
	if err != nil {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Internalf("cannot canonicalize workflow step %d: %s", ordinal, err)
	}
	digest, err := canonical.Spell(canonical.Digest(canonicalBytes))
	if err != nil {
		return MaterializedSubmission{}, nil, "", nil,
			exit.Internalf("cannot spell workflow step %d digest: %s", ordinal, err)
	}
	return materialized, canonicalBytes, digest, resolved, nil
}

func ChildKey(workflowDigest string, ordinal int, materialized MaterializedSubmission,
	materializedDigest string) (string, *exit.Error) {
	data, err := canonical.Write(map[string]canonical.Value{
		"format": "cozy.workflow.ChildIdentity/1", "workflow_execution_digest": workflowDigest,
		"step_ordinal": int64(ordinal), "endpoint_release_id": materialized.EndpointReleaseID,
		"entrypoint":                     materialized.Entrypoint,
		"entrypoint_binding_plan_id":     materialized.EntrypointBindingPlanID,
		"materialized_submission_digest": materializedDigest,
	})
	if err != nil {
		return "", exit.Internalf("cannot canonicalize workflow child identity: %s", err)
	}
	key, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot spell workflow child identity: %s", err)
	}
	return key, nil
}

func DecodeMaterialized(data []byte) (MaterializedSubmission, *exit.Error) {
	var out MaterializedSubmission
	if _, err := canonical.ReadObject(data); err != nil {
		return out, exit.Internalf("stored workflow materialization is not canonical: %s", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&out); err != nil || out.Format != materializedFormat {
		return out, exit.Internalf("stored workflow materialization is unreadable: %v", err)
	}
	return out, nil
}

func (m MaterializedSubmission) Submission(idemKey, digest, installID, worker string,
	paths map[string]string, limits map[string]int64) (orchestrator.Submission, *exit.Error) {
	payload, err := base64.StdEncoding.Strict().DecodeString(m.PayloadBase64)
	if err != nil {
		return orchestrator.Submission{}, exit.Internalf("stored workflow payload is unreadable: %s", err)
	}
	assets := make([]records.AssetBinding, 0, len(m.Assets))
	for _, asset := range m.Assets {
		path := paths[asset.FieldPath]
		if path == "" {
			return orchestrator.Submission{}, exit.Internalf(
				"stored workflow asset %s has no private resolution", asset.FieldPath)
		}
		limit := limits[asset.FieldPath]
		if limit <= 0 || asset.Length > limit {
			return orchestrator.Submission{}, exit.Named(exit.Validation,
				"workflow_asset_over_bound",
				"workflow asset %s is %d B and its pinned field admits %d B",
				asset.FieldPath, asset.Length, limit)
		}
		assets = append(assets, records.AssetBinding{
			FieldPath: asset.FieldPath, LocalPath: path, Digest: asset.Digest,
			Length: asset.Length, MediaType: asset.MediaType, Order: asset.Order, MaxBytes: limit,
		})
	}
	return orchestrator.Submission{
		IdemKey: idemKey, BodyDigest: digest, Endpoint: m.Endpoint,
		Entrypoint: m.Entrypoint, PlanID: m.EntrypointBindingPlanID,
		Payload: payload, Outputs: append([]string(nil), m.Outputs...),
		Assets: assets, InstallID: installID, Worker: worker,
	}, nil
}

func (m MaterializedSubmission) document() map[string]canonical.Value {
	outputs := make([]canonical.Value, 0, len(m.Outputs))
	for _, output := range m.Outputs {
		outputs = append(outputs, output)
	}
	assets := make([]canonical.Value, 0, len(m.Assets))
	for _, asset := range m.Assets {
		assets = append(assets, map[string]canonical.Value{
			"field_path": asset.FieldPath, "digest": asset.Digest, "length": asset.Length,
			"media_type": asset.MediaType, "order": int64(asset.Order),
		})
	}
	return map[string]canonical.Value{
		"format": materializedFormat, "endpoint": m.Endpoint,
		"endpoint_release_id": m.EndpointReleaseID, "entrypoint": m.Entrypoint,
		"entrypoint_binding_plan_id": m.EntrypointBindingPlanID,
		"payload_b64":                m.PayloadBase64, "outputs": outputs, "assets": assets,
	}
}

func setPayloadRef(document map[string]any, path, ref string) *exit.Error {
	if problem := validateFieldPath(path); problem != nil {
		return problem
	}
	parts := strings.Split(path, ".")
	_, e := setValue(document, parts, ref, path)
	return e
}

func setValue(value any, parts []string, ref, whole string) (any, *exit.Error) {
	if len(parts) == 0 {
		if value != nil && value != "" && value != ref {
			return value, exit.New(exit.Conflict,
				"workflow asset field %q already has a different payload value", whole)
		}
		return ref, nil
	}
	part, rest := parts[0], parts[1:]
	switch node := value.(type) {
	case map[string]any:
		child := node[part]
		if child == nil && len(rest) > 0 {
			child = containerFor(rest[0])
		}
		updated, e := setValue(child, rest, ref, whole)
		if e == nil {
			node[part] = updated
		}
		return node, e
	case []any:
		position, err := strconv.Atoi(part)
		if err != nil || position < 0 || position > MaxFieldPathIndex {
			return value, exit.New(exit.Validation,
				"%q is not a list position in workflow asset field %s", part, whole)
		}
		for len(node) <= position {
			node = append(node, nil)
		}
		child := node[position]
		if child == nil && len(rest) > 0 {
			child = containerFor(rest[0])
		}
		updated, e := setValue(child, rest, ref, whole)
		if e == nil {
			node[position] = updated
		}
		return node, e
	default:
		return value, exit.New(exit.Conflict,
			"workflow asset field %q collides with a scalar payload value", whole)
	}
}

func containerFor(next string) any {
	if _, err := strconv.ParseUint(next, 10, 31); err == nil {
		return []any{}
	}
	return map[string]any{}
}

func fieldOrder(path string) uint32 {
	var order uint64
	for _, part := range strings.Split(path, ".") {
		if index, err := strconv.ParseUint(part, 10, 32); err == nil {
			order = index
		}
	}
	return uint32(order)
}
