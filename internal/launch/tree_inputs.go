package launch

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// TreeInputRefs resolves the existing scalar/nested Tree fields against actual
// argument values. A directory alias never authorizes an undeclared position.
func TreeInputRefs(entry *Entrypoint, payload []byte) (map[string]string, *exit.Error) {
	var value map[string]any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if decoder.Decode(&value) != nil {
		return nil, exit.New(exit.Validation, "tree arguments are unreadable")
	}
	refs := map[string]string{}
	for _, name := range AssetPaths(entry.Request) {
		spec, ok := AssetSpec(entry, name)
		if !ok || spec.Kind != "tree" {
			continue
		}
		current := value
		parts := strings.Split(name, ".")
		for index, part := range parts {
			item, exists := current[part]
			if !exists || item == nil {
				break
			}
			if index == len(parts)-1 {
				ref, ok := item.(string)
				if !ok || ref == "" {
					return nil, exit.New(exit.Validation, "Tree field %s needs one input reference", name)
				}
				refs[name] = ref
			} else {
				current, ok = item.(map[string]any)
				if !ok {
					return nil, exit.New(exit.Validation, "Tree field %s has an invalid parent", name)
				}
			}
		}
	}
	return refs, nil
}

func ReplaceTreeInputRefs(payload []byte, replacements map[string]string) ([]byte, *exit.Error) {
	var value map[string]any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if decoder.Decode(&value) != nil {
		return nil, exit.New(exit.Validation, "tree arguments are unreadable")
	}
	for name, digest := range replacements {
		current := value
		parts := strings.Split(name, ".")
		for _, part := range parts[:len(parts)-1] {
			next, ok := current[part].(map[string]any)
			if !ok {
				return nil, exit.New(exit.Validation, "Tree field %s has an invalid parent", name)
			}
			current = next
		}
		current[parts[len(parts)-1]] = digest
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, exit.Internalf("cannot encode captured Tree arguments: %s", err)
	}
	return raw, nil
}
