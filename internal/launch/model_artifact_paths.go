package launch

import (
	"encoding/json"
	"sort"
)

// ModelArtifactPaths enumerates only schema-declared native model references.
// A user field that happens to spell a receipt field is ordinary result data.
// A '*' segment means each element of a schema-declared list.
func ModelArtifactPaths(result Struct) [][]string {
	raw, _ := json.Marshal(result)
	var schema any
	_ = json.Unmarshal(raw, &schema)
	var paths [][]string
	var visit func(any, []string)
	visit = func(value any, path []string) {
		node, ok := value.(map[string]any)
		if !ok {
			return
		}
		if node["input"] == "model" {
			paths = append(paths, append([]string(nil), path...))
			return
		}
		if list, ok := node["list"]; ok {
			visit(list, append(append([]string(nil), path...), "*"))
		}
		if branches, ok := node["union"].([]any); ok {
			for _, branch := range branches {
				visit(branch, path)
			}
		}
		if fields, ok := node["fields"].([]any); ok {
			for _, field := range fields {
				if row, ok := field.(map[string]any); ok {
					if name, ok := row["name"].(string); ok {
						visit(row["type"], append(append([]string(nil), path...), name))
					}
				}
			}
		}
	}
	visit(schema, nil)
	sort.Slice(paths, func(i, j int) bool {
		a, _ := json.Marshal(paths[i])
		b, _ := json.Marshal(paths[j])
		return string(a) < string(b)
	})
	return paths
}
