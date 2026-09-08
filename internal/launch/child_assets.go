package launch

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/records"
)

// InheritChildAssets binds schema-known media references to the parent's already
// owned immutable input bytes. It never interprets an author path or fetch URL.
func InheritChildAssets(ep *Entrypoint, payload []byte, parent []records.AssetBinding) ([]records.AssetBinding, *exit.Error) {
	var document map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.UseNumber()
	if decoder.Decode(&document) != nil {
		return nil, exit.New(exit.Validation, "child media payload is not an object")
	}
	var paths []string
	for _, field := range ep.Request.Fields {
		value, exists := document[field.Name]
		if !exists {
			continue
		}
		if problem := validateFieldInto(field, value, field.Name, &paths); problem != nil {
			return nil, problem
		}
	}
	if len(paths) > 32 {
		return nil, exit.New(exit.Validation, "child media exceeds 32 input references")
	}
	result := make([]records.AssetBinding, 0, len(paths))
	for _, path := range paths {
		var value any = document
		parts := strings.Split(path, ".")
		for _, part := range parts {
			switch held := value.(type) {
			case map[string]any:
				value = held[part]
			case []any:
				index, err := strconv.Atoi(part)
				if err != nil || index < 0 || index >= len(held) {
					return nil, exit.New(exit.Validation, "child media path is invalid")
				}
				value = held[index]
			default:
				return nil, exit.New(exit.Validation, "child media path is invalid")
			}
		}
		digest, ok := value.(string)
		if _, err := canonical.Raw(digest); !ok || err != nil {
			return nil, exit.Named(exit.Conflict, "child.asset_ungranted", "child media needs a verified immutable input reference")
		}
		var owned *records.AssetBinding
		for i := range parent {
			if parent[i].Digest == digest {
				owned = &parent[i]
				break
			}
		}
		if owned == nil {
			return nil, exit.Named(exit.Conflict, "child.asset_ungranted", "child media was not granted to the parent")
		}
		spec, ok := AssetSpecForMedia(ep, path, owned.MediaType)
		if !ok {
			return nil, exit.New(exit.Validation, "child media field has no matching media contract")
		}
		maximum := spec.MaxBytes
		if maximum <= 0 {
			maximum = inputasset.MaxBytes
		}
		if owned.Length > maximum || !spec.AcceptsMediaType(owned.MediaType) {
			return nil, exit.Named(exit.Validation, "child.asset_bound", "inherited media exceeds the child field's byte or MIME bound")
		}
		if problem := inputasset.Verify(*owned, maximum); problem != nil {
			return nil, problem
		}
		bound := *owned
		bound.FieldPath, bound.Order, bound.MaxBytes = path, pathOrder(parts), maximum
		result = append(result, bound)
	}
	return result, nil
}
