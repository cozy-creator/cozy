package launch

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// mapAssetFilenames walks declared media leaves, never arbitrary strings or tree inputs.
// The schema decides which JSON strings grant files. Existing content references retain
// their meaning, and ambiguous string-or-asset unions remain ordinary string inputs.
func mapAssetFilenames(ep *Entrypoint, payload json.RawMessage, transform func(string, string) (any, *exit.Error)) (json.RawMessage, *exit.Error) {
	var document map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil || document == nil {
		return nil, exit.New(exit.Validation, "input payload must be a JSON object")
	}
	var visit func(any, any, string) (any, *exit.Error)
	visit = func(schema, value any, path string) (any, *exit.Error) {
		object, ok := schema.(map[string]any)
		if !ok || value == nil {
			return value, nil
		}
		if _, asset := object["asset"]; asset {
			if source, ok := value.(string); ok && source != "" && !strings.HasPrefix(source, "sha256:") {
				return transform(path, source)
			}
			return value, nil
		}
		if union, ok := object["union"].([]any); ok {
			// A plain string branch makes a filename interpretation ambiguous.
			if _, text := value.(string); text {
				for _, branch := range union {
					if branch == "str" {
						return value, nil
					}
					if literal, ok := branch.(map[string]any); ok && literal["literal"] != nil {
						raw, _ := json.Marshal(branch)
						if validateRenderedInto(raw, value, path, nil) == nil {
							return value, nil
						}
					}
				}
			}
			tag, tagged := object["tag_field"].(string)
			for _, branch := range union {
				if tagged {
					variant, variantOK := branch.(map[string]any)
					fields, fieldsOK := value.(map[string]any)
					if !variantOK || !fieldsOK || variant["tag"] != fields[tag] {
						continue
					}
				}
				var problem *exit.Error
				value, problem = visit(branch, value, path)
				if problem != nil {
					return nil, problem
				}
			}
			return value, nil
		}
		if item, list := object["list"]; list {
			if values, ok := value.([]any); ok {
				for index, child := range values {
					updated, problem := visit(item, child, fmt.Sprintf("%s.%d", path, index))
					if problem != nil {
						return nil, problem
					}
					values[index] = updated
				}
			}
			return value, nil
		}
		fields, fieldsOK := object["fields"].([]any)
		values, valuesOK := value.(map[string]any)
		if fieldsOK && valuesOK {
			for _, row := range fields {
				field, ok := row.(map[string]any)
				if !ok {
					continue
				}
				name, _ := field["name"].(string)
				if child, exists := values[name]; exists {
					updated, problem := visit(field["type"], child, path+"."+name)
					if problem != nil {
						return nil, problem
					}
					values[name] = updated
				}
			}
		}
		return value, nil
	}
	for _, field := range ep.Request.Fields {
		value, exists := document[field.Name]
		if !exists {
			continue
		}
		var schema any
		decoder := json.NewDecoder(strings.NewReader(string(field.Type)))
		decoder.UseNumber()
		if err := decoder.Decode(&schema); err != nil {
			return nil, exit.Internalf("cannot read asset field schema: %s", err)
		}
		updated, problem := visit(schema, value, field.Name)
		if problem != nil {
			return nil, problem
		}
		document[field.Name] = updated
	}
	raw, err := json.Marshal(document)
	if err != nil {
		return nil, exit.Internalf("cannot render asset input payload: %s", err)
	}
	return raw, nil
}
