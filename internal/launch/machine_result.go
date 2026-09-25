package launch

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ValidateMachineResult verifies the captured schema against final result
// metadata. Asset results carry their complete immutable facts, while the
// ordinary payload validator accepts only an input reference at those positions.
func ValidateMachineResult(schema json.RawMessage, envelope *pb.ResultEnvelope) *exit.Error {
	if envelope == nil || envelope.ResultBlob != nil || len(envelope.InlineResult) == 0 || len(envelope.InlineResult) > pb.MaxInlineControlBytes {
		return exit.New(exit.Conflict, "machine result requires its bounded inline metadata")
	}
	normalized, err := canonical.NormalizeJCS(schema)
	if err != nil || !bytes.Equal(canonical.Digest(normalized), envelope.ResultSchemaDigest) {
		return exit.New(exit.Conflict, "machine result differs from its captured schema")
	}
	inline, err := canonical.NormalizeJCS(envelope.InlineResult)
	if err != nil || !bytes.Equal(inline, envelope.InlineResult) {
		return exit.New(exit.Conflict, "machine result is not canonical JSON")
	}
	var declared Struct
	if json.Unmarshal(schema, &declared) != nil {
		return exit.New(exit.Conflict, "machine result schema is unreadable")
	}
	var value any
	decoder := json.NewDecoder(bytes.NewReader(inline))
	decoder.UseNumber()
	if decoder.Decode(&value) != nil {
		return exit.New(exit.Conflict, "machine result is unreadable")
	}
	entrypoint := &Entrypoint{Result: declared}
	for _, path := range AssetPaths(declared) {
		parts := strings.Split(path, ".")
		parent := value
		for i, part := range parts {
			var child any
			switch node := parent.(type) {
			case map[string]any:
				child = node[part]
			case []any:
				index, err := strconv.Atoi(part)
				if err != nil || index < 0 || index >= len(node) {
					return exit.New(exit.Conflict, "machine asset result position is absent")
				}
				child = node[index]
			default:
				return exit.New(exit.Conflict, "machine asset result position is absent")
			}
			if i < len(parts)-1 {
				parent = child
				continue
			}
			asset, ok := child.(map[string]any)
			spec, declared := ResultAssetSpec(entrypoint, path)
			if !ok || !declared || asset["kind"] != spec.Kind || spec.Kind == "tree" && len(asset) != 4 || spec.Kind != "tree" && len(asset) != 5 {
				return exit.New(exit.Conflict, "machine asset result has no closed typed metadata")
			}
			ref, ok := asset["asset_ref"].(string)
			if !ok || asset["digest"] != ref {
				return exit.New(exit.Conflict, "machine asset result changed its content identity")
			}
			if _, err := canonical.Raw(ref); err != nil {
				return exit.New(exit.Conflict, "machine asset result digest is invalid")
			}
			size, ok := asset["size_bytes"].(json.Number)
			length, err := size.Int64()
			media, mediaOK := asset["media_type"].(string)
			if !ok || err != nil || length < 0 || spec.MaxBytes > 0 && length > spec.MaxBytes || spec.Kind != "tree" && (!mediaOK || !spec.AcceptsMediaType(media)) {
				return exit.New(exit.Conflict, "machine asset result exceeds its declared media contract")
			}
			switch node := parent.(type) {
			case map[string]any:
				node[part] = ref
			case []any:
				index, _ := strconv.Atoi(part)
				node[index] = ref
			}
		}
	}
	wrapped, err := json.Marshal(map[string]any{"result": value})
	if err != nil {
		return exit.New(exit.Conflict, "machine result cannot be validated")
	}
	return ValidatePayload("machine result", &Entrypoint{Name: "result", Request: Struct{Fields: []Field{{Name: "result", Wire: "required", Type: schema}}}}, wrapped)
}

// Runtime023PNGResultSchema recognizes the released writer's exact omission for
// a bare PNG result. The caller must verify the captured decoded bound against
// copied image bytes before releasing custody or acknowledging the outcome.
func Runtime023PNGResultSchema(schema json.RawMessage, envelope *pb.ResultEnvelope, runtimeVersion string) (json.RawMessage, int64, bool) {
	if runtimeVersion != "0.18.23" || envelope == nil {
		return nil, 0, false
	}
	var document map[string]json.RawMessage
	if json.Unmarshal(schema, &document) != nil || len(document) != 1 {
		return nil, 0, false
	}
	var fields []map[string]json.RawMessage
	if json.Unmarshal(document["fields"], &fields) != nil || len(fields) != 1 {
		return nil, 0, false
	}
	field := fields[0]
	var name string
	var kind map[string]string
	var bound map[string]json.RawMessage
	if json.Unmarshal(field["name"], &name) != nil || name != "value" ||
		json.Unmarshal(field["type"], &kind) != nil || len(kind) != 1 || kind["asset"] != "image" ||
		json.Unmarshal(field["asset_bound"], &bound) != nil {
		return nil, 0, false
	}
	var decoded int64
	var media []string
	if json.Unmarshal(bound["max_decoded_bytes"], &decoded) != nil || decoded <= 0 ||
		json.Unmarshal(bound["media_types"], &media) != nil || len(media) != 1 || media[0] != "image/png" {
		return nil, 0, false
	}
	delete(bound, "max_decoded_bytes")
	field["asset_bound"], _ = json.Marshal(bound)
	document["fields"], _ = json.Marshal(fields)
	projected, err := json.Marshal(document)
	if err != nil {
		return nil, 0, false
	}
	projected, err = canonical.NormalizeJCS(projected)
	if err != nil || !bytes.Equal(canonical.Digest(projected), envelope.ResultSchemaDigest) {
		return nil, 0, false
	}
	return projected, decoded, true
}
