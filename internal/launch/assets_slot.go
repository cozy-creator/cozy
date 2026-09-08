package launch

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/records"
)

// AssetsSlot is the explicit callable input contract authored by Runtime.
// Occurrence labels and references live in its ordinary request payload.
type AssetsSlot struct {
	Parameter string       `json:"parameter"`
	Kinds     []AssetsKind `json:"kinds"`
}

type AssetsKind struct {
	Kind            string   `json:"kind"`
	MediaTypes      []string `json:"media_types"`
	MaxBytes        int64    `json:"max_bytes,omitempty"`
	MaxDecodedBytes int64    `json:"max_decoded_bytes,omitempty"`
}

func validateAssetsSlot(raw, request json.RawMessage) error {
	value, err := exactKeys(raw, []string{"parameter", "kinds"}, nil)
	if err != nil {
		return err
	}
	var slot AssetsSlot
	if json.Unmarshal(raw, &slot) != nil || slot.Parameter == "" || strings.ContainsAny(slot.Parameter, ". /\\") {
		return fmt.Errorf("assets parameter must be one field identifier")
	}
	var schema Struct
	if json.Unmarshal(request, &schema) != nil {
		return fmt.Errorf("assets request is invalid")
	}
	matches := 0
	for _, field := range schema.Fields {
		if field.Name != slot.Parameter {
			continue
		}
		var kind struct {
			List Struct `json:"list"`
		}
		if json.Unmarshal(field.Type, &kind) != nil || len(kind.List.Fields) != 2 {
			return fmt.Errorf("assets parameter must name a list of asset/label records")
		}
		asset, label := false, false
		for _, item := range kind.List.Fields {
			var typ any
			_ = json.Unmarshal(item.Type, &typ)
			if object, ok := typ.(map[string]any); ok && item.Name == "asset" && len(object) == 1 && object["asset"] == "file" {
				asset = true
			}
			if item.Name == "label" && typ == "str" && item.Wire == "optional" {
				label = true
			}
		}
		if !asset || !label {
			return fmt.Errorf("assets parameter must name a list of asset/label records")
		}
		matches++
	}
	if matches != 1 || len(slot.Kinds) < 1 || len(slot.Kinds) > 4 {
		return fmt.Errorf("assets needs one request field and 1..4 kinds")
	}
	var rows []json.RawMessage
	if json.Unmarshal(value["kinds"], &rows) != nil {
		return fmt.Errorf("assets kinds are invalid")
	}
	seen, media := map[string]bool{}, map[string]bool{}
	for i, row := range rows {
		fields, err := exactKeys(row, []string{"kind", "media_types"}, []string{"max_bytes", "max_decoded_bytes"})
		if err != nil {
			return err
		}
		kind := slot.Kinds[i]
		if seen[kind.Kind] || (kind.Kind != "image" && kind.Kind != "video" && kind.Kind != "audio" && kind.Kind != "file") || (len(kind.MediaTypes) == 0 && kind.Kind != "file") {
			return fmt.Errorf("assets kinds must be unique media contracts")
		}
		seen[kind.Kind] = true
		for _, mime := range kind.MediaTypes {
			if mime != strings.TrimSpace(strings.ToLower(mime)) || !strings.Contains(mime, "/") || media[mime] {
				return fmt.Errorf("assets MIME types must be canonical and disjoint")
			}
			media[mime] = true
		}
		if fields["max_bytes"] != nil && kind.MaxBytes <= 0 || fields["max_decoded_bytes"] != nil && kind.MaxDecodedBytes <= 0 {
			return fmt.Errorf("assets byte limits must be positive")
		}
	}
	return nil
}

func (s *AssetsSlot) contains(parts []string) bool {
	if s == nil || len(parts) != 3 || parts[0] != s.Parameter || parts[2] != "asset" {
		return false
	}
	i, err := strconv.ParseUint(parts[1], 10, 31)
	return err == nil && strconv.FormatUint(i, 10) == parts[1]
}

func (s *AssetsSlot) maxBytes() int64 {
	var limit int64
	for _, kind := range s.Kinds {
		limit = max(limit, effectiveAssetMax(kind.MaxBytes))
	}
	return limit
}
func effectiveAssetMax(limit int64) int64 {
	if limit <= 0 {
		return inputasset.MaxBytes
	}
	return limit
}

// AssetSpecForMedia chooses the descriptor's exact encoded-byte policy after the
// existing file fingerprint has observed its MIME. Decoding remains Runtime-owned.
func AssetSpecForMedia(ep *Entrypoint, path, mediaType string) (AssetField, bool) {
	parts, problem := assetPath(path)
	if problem != nil {
		return AssetField{}, false
	}
	if !ep.Assets.contains(parts) {
		return AssetSpec(ep, path)
	}
	var fallback *AssetsKind
	for i := range ep.Assets.Kinds {
		kind := &ep.Assets.Kinds[i]
		if kind.Kind == "file" && len(kind.MediaTypes) == 0 {
			fallback = kind
			continue
		}
		for _, mime := range kind.MediaTypes {
			if mime == mediaType {
				return AssetField{Kind: kind.Kind, MaxBytes: effectiveAssetMax(kind.MaxBytes), MediaTypes: kind.MediaTypes}, true
			}
		}
	}
	if fallback != nil {
		return AssetField{Kind: "file", MaxBytes: effectiveAssetMax(fallback.MaxBytes)}, true
	}
	return AssetField{}, false
}

// BindChildAssets forwards only explicit occurrences whose content the parent
// already owns. A cache hit or ambient file path never grants a child authority.
func BindChildAssets(ep *Entrypoint, payload json.RawMessage, parent []records.AssetBinding) ([]records.AssetBinding, *exit.Error) {
	if ep.Assets == nil {
		return nil, nil
	}
	var doc map[string]json.RawMessage
	if json.Unmarshal(payload, &doc) != nil {
		return nil, exit.New(exit.Validation, "child Assets input is not an object")
	}
	if _, present := doc[ep.Assets.Parameter]; !present {
		return nil, nil
	}
	var values []struct {
		Asset string `json:"asset"`
		Label string `json:"label"`
	}
	if json.Unmarshal(doc[ep.Assets.Parameter], &values) != nil {
		return nil, exit.New(exit.Validation, "child Assets input is not an occurrence list")
	}
	out := make([]records.AssetBinding, 0, len(values))
	labels := map[string]bool{}
	for i, value := range values {
		if value.Label != "" && labels[value.Label] {
			return nil, exit.New(exit.Validation, "asset label %q was supplied more than once", value.Label)
		}
		labels[value.Label] = true
		var held *records.AssetBinding
		for j := range parent {
			if parent[j].Digest == value.Asset {
				held = &parent[j]
				break
			}
		}
		if held == nil {
			return nil, exit.Named(exit.Conflict, "child.asset_scope", "child asset %d is not authorized by the parent request", i)
		}
		bound := *held
		bound.FieldPath = fmt.Sprintf("%s.%d.asset", ep.Assets.Parameter, i)
		bound.Order = uint32(i)
		spec, ok := AssetSpecForMedia(ep, bound.FieldPath, bound.MediaType)
		if !ok || bound.Length > spec.MaxBytes {
			return nil, exit.Named(exit.Validation, "input_asset_bound", "child asset %d is not admitted by the callable's media policy", i)
		}
		bound.MaxBytes = spec.MaxBytes
		out = append(out, bound)
	}
	return out, nil
}

// Count/label checks precede file reads. Explicit named payload bindings never
// consume an occurrence in the separate collection.
func preflightAssetCount(ep *Entrypoint, document map[string]any, specs []string) *exit.Error {
	if ep.Assets == nil {
		return nil
	}
	var values []any
	if value, exists := document[ep.Assets.Parameter]; exists {
		var ok bool
		values, ok = value.([]any)
		if !ok {
			return exit.New(exit.Validation, "Assets input %s must be a list", ep.Assets.Parameter)
		}
	}
	count := int64(len(values))
	labels := map[string]bool{}
	addLabel := func(label string) *exit.Error {
		if label == "" {
			return nil
		}
		if labels[label] {
			return exit.New(exit.Validation, "asset label %q was supplied more than once", label)
		}
		labels[label] = true
		return nil
	}
	for _, value := range values {
		if entry, ok := value.(map[string]any); ok {
			if label, ok := entry["label"].(string); ok {
				if problem := addLabel(label); problem != nil {
					return problem
				}
			}
		}
	}
	for _, spec := range specs {
		name, _, named := strings.Cut(spec, "=")
		if named {
			if namedAssetSpec(ep, name) {
				continue
			}
			if problem := addLabel(name); problem != nil {
				return problem
			}
		}
		count++
	}
	for _, field := range ep.Request.Fields {
		if field.Name == ep.Assets.Parameter && field.Constraints.MaxLength != nil && count > *field.Constraints.MaxLength {
			return exit.New(exit.Validation, "Assets input %s has %d items; maximum is %d", field.Name, count, *field.Constraints.MaxLength)
		}
	}
	return nil
}

func namedAssetSpec(ep *Entrypoint, path string) bool {
	parts, problem := assetPath(strings.TrimSpace(path))
	if problem != nil {
		return false
	}
	folded, problem := canonicalFieldKey(ep, parts[0])
	if problem != nil {
		return false
	}
	parts[0] = folded
	_, ok := AssetSpec(ep, strings.Join(parts, "."))
	return ok
}
