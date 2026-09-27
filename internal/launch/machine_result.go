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

// ValidateMachineResult checks final result metadata against the captured schema. The
// worker's own schema digest and byte spelling are another Runtime version's facts, so
// the result is judged by the values this host consumes. Asset results carry their
// immutable facts, while the ordinary payload validator accepts only an input reference
// at those positions.
func ValidateMachineResult(schema json.RawMessage, envelope *pb.ResultEnvelope) *exit.Error {
	if envelope == nil || envelope.ResultBlob != nil || len(envelope.InlineResult) == 0 || len(envelope.InlineResult) > pb.MaxInlineControlBytes {
		return exit.New(exit.Conflict, "machine result requires its bounded inline metadata")
	}
	inline := envelope.InlineResult
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
			if !ok || !declared || asset["kind"] != spec.Kind {
				return exit.New(exit.Conflict, "machine asset result has no typed metadata")
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
