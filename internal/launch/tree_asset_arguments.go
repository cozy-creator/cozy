package launch

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
)

// ParseTreeAssets routes named Tree directories through the same native snapshot
// intake as explicit job input trees. File and media arguments remain unchanged.
func ParseTreeAssets(entry *Entrypoint, payload json.RawMessage, specs []string) (json.RawMessage, []string, []string, *exit.Error) {
	var document map[string]any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if decoder.Decode(&document) != nil || document == nil {
		return nil, nil, nil, exit.New(exit.Validation, "job asset arguments are unreadable")
	}
	var files, trees []string
	seen := map[string]bool{}
	for _, value := range specs {
		field, source, _, named := splitAssetArgument(entry, value)
		if !named {
			files = append(files, value)
			continue
		}
		parts, problem := assetPath(strings.TrimSpace(field))
		if problem != nil {
			return nil, nil, nil, problem
		}
		parts[0], problem = canonicalFieldKey(entry, parts[0])
		if problem != nil {
			return nil, nil, nil, problem
		}
		field = strings.Join(parts, ".")
		policy, ok := AssetSpec(entry, field)
		if !ok || policy.Kind != "tree" {
			files = append(files, value)
			continue
		}
		if seen[field] {
			return nil, nil, nil, exit.New(exit.Validation, "input Tree field %q was supplied more than once", field)
		}
		seen[field] = true
		if problem := inputasset.ValidateID(field); problem != nil {
			return nil, nil, nil, problem
		}
		if strings.TrimSpace(source) == "" {
			return nil, nil, nil, exit.Usagef("--asset %q needs a directory", value)
		}
		alias := "asset:" + field
		if problem := setAssetRef(document, parts, alias); problem != nil {
			return nil, nil, nil, problem
		}
		trees = append(trees, alias+"="+strings.TrimSpace(source))
	}
	if len(trees) == 0 {
		return payload, files, nil, nil
	}
	raw, err := json.Marshal(document)
	if err != nil {
		return nil, nil, nil, exit.Internalf("cannot encode typed Tree arguments: %s", err)
	}
	return raw, files, trees, nil
}
