package launch

import (
	"encoding/json"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/inputasset"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

// ParseAssets turns repeated `--asset <field-path>=<file>` flags into one payload plus
// exact byte bindings. The payload carries only an opaque content reference; paths and
// bytes travel out-of-band through DeliveryGrant. A nested path is resolved against the
// endpoint's recorded schema, so `references.0.image` cannot accidentally grant a file
// to a scalar or to a misspelled field.
func ParseAssets(ep *Entrypoint, payload json.RawMessage, specs []string) (json.RawMessage, []records.AssetBinding, *exit.Error) {
	if len(specs) == 0 {
		return payload, nil, nil
	}
	var document map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil {
		return nil, nil, exit.Internalf("cannot add input assets to the payload: %s", err)
	}

	seen := map[string]bool{}
	assets := make([]records.AssetBinding, 0, len(specs))
	for _, spec := range specs {
		fieldPath, source, ok := strings.Cut(spec, "=")
		fieldPath, source = strings.TrimSpace(fieldPath), strings.TrimSpace(source)
		if !ok || fieldPath == "" || source == "" {
			return nil, nil, exit.Usagef("--asset %q is not <field-path>=<file>", spec).
				WithRemedy("examples: `--asset first_frame=frame.png` or `--asset references.0.image=ref.jpg`")
		}
		if seen[fieldPath] {
			return nil, nil, exit.New(exit.Validation, "input asset field %q was supplied more than once", fieldPath)
		}
		if e := inputasset.ValidateID(fieldPath); e != nil {
			return nil, nil, e
		}
		seen[fieldPath] = true

		parts, e := assetPath(fieldPath)
		if e != nil {
			return nil, nil, e
		}
		if !assetAt(ep.Request, parts) {
			return nil, nil, exit.New(exit.Validation,
				"%s.%s is not an asset field in this release's request schema", ep.Name, fieldPath).
				WithRemedy("`cozy describe <org/endpoint>/%s` prints the recorded request schema", ep.Name)
		}
		assetSpec, _ := AssetSpec(ep, fieldPath)
		maxBytes := assetSpec.MaxBytes
		if maxBytes <= 0 {
			maxBytes = inputasset.MaxBytes
		}

		absolute, err := filepath.Abs(source)
		if err != nil {
			return nil, nil, exit.New(exit.NotFound, "cannot resolve input asset %s: %s", source, err)
		}
		length, digest, mediaType, e := inputasset.Fingerprint(absolute, maxBytes)
		if e != nil {
			return nil, nil, e
		}
		if !assetSpec.AcceptsMediaType(mediaType) {
			return nil, nil, exit.New(exit.Validation,
				"%s.%s accepts media types [%s], not %q",
				ep.Name, fieldPath, strings.Join(assetSpec.MediaTypes, ", "), mediaType)
		}
		if e := setAssetRef(document, parts, digest); e != nil {
			return nil, nil, e
		}
		assets = append(assets, records.AssetBinding{
			FieldPath: fieldPath, LocalPath: absolute, Digest: digest,
			Length: length, MediaType: mediaType, Order: pathOrder(parts), MaxBytes: maxBytes,
		})
	}
	sort.Slice(assets, func(i, j int) bool { return assets[i].FieldPath < assets[j].FieldPath })
	rendered, err := json.Marshal(document)
	if err != nil {
		return nil, nil, exit.Internalf("cannot render the payload with input assets: %s", err)
	}
	return rendered, assets, nil
}

// LegacyFileTerm returns the first `key=@file` payload term. That spelling embeds bytes
// into a JSON string and has no field-path binding, digest, or grant identity; a remote
// run must refuse it rather than sending plausible-looking garbage to another machine.
func LegacyFileTerm(terms []string) string {
	for _, term := range terms {
		key, value, ok := strings.Cut(term, "=")
		if ok && !strings.HasSuffix(key, ":") && strings.HasPrefix(value, "@") {
			return term
		}
	}
	return ""
}

func assetPath(path string) ([]string, *exit.Error) {
	parts := strings.Split(path, ".")
	for _, part := range parts {
		if part == "" || strings.ContainsAny(part, `/\\`) {
			return nil, exit.Usagef("--asset field path %q is malformed", path).
				WithRemedy("use descriptor field names separated by dots; list positions are decimal indexes")
		}
	}
	return parts, nil
}

func assetAt(root Struct, parts []string) bool {
	_, ok := assetSpecAt(root, parts, AssetField{})
	return ok
}

type AssetField struct {
	Kind       string
	MaxBytes   int64
	MediaTypes []string
}

func (f AssetField) AcceptsMediaType(actual string) bool {
	if len(f.MediaTypes) == 0 {
		return true
	}
	for _, mediaType := range f.MediaTypes {
		if strings.EqualFold(strings.TrimSpace(mediaType), strings.TrimSpace(actual)) {
			return true
		}
	}
	return false
}

// AssetSpec returns the exact asset kind, compressed-byte bound, and accepted media types
// at one request path.
func AssetSpec(ep *Entrypoint, path string) (AssetField, bool) {
	parts, problem := assetPath(path)
	if problem != nil {
		return AssetField{}, false
	}
	return assetSpecAt(ep.Request, parts, AssetField{})
}

func assetSpecAt(root Struct, parts []string, inherited AssetField) (AssetField, bool) {
	if len(parts) == 0 {
		return AssetField{}, false
	}
	for _, field := range root.Fields {
		if field.Name != parts[0] {
			continue
		}
		spec := inherited
		if field.AssetBound.MaxBytes > 0 {
			spec.MaxBytes = field.AssetBound.MaxBytes
		}
		if len(field.AssetBound.MediaTypes) > 0 {
			spec.MediaTypes = append([]string(nil), field.AssetBound.MediaTypes...)
		}
		var schema any
		if json.Unmarshal(field.Type, &schema) != nil {
			return AssetField{}, false
		}
		return assetSpecAtValue(schema, parts[1:], spec)
	}
	return AssetField{}, false
}

func assetSpecAtValue(schema any, parts []string, inherited AssetField) (AssetField, bool) {
	object, ok := schema.(map[string]any)
	if !ok {
		return AssetField{}, false
	}
	if kind, ok := object["asset"].(string); ok {
		inherited.Kind = kind
		return inherited, len(parts) == 0
	}
	if union, ok := object["union"].([]any); ok {
		for _, branch := range union {
			if spec, ok := assetSpecAtValue(branch, parts, inherited); ok {
				return spec, true
			}
		}
		return AssetField{}, false
	}
	if item, ok := object["list"]; ok {
		if len(parts) == 0 {
			return AssetField{}, false
		}
		if _, err := strconv.ParseUint(parts[0], 10, 31); err != nil {
			return AssetField{}, false
		}
		return assetSpecAtValue(item, parts[1:], inherited)
	}
	fields, ok := object["fields"].([]any)
	if !ok || len(parts) == 0 {
		return AssetField{}, false
	}
	for _, row := range fields {
		encoded, err := json.Marshal(row)
		if err != nil {
			continue
		}
		var field Field
		if json.Unmarshal(encoded, &field) != nil || field.Name != parts[0] {
			continue
		}
		spec := inherited
		if field.AssetBound.MaxBytes > 0 {
			spec.MaxBytes = field.AssetBound.MaxBytes
		}
		if len(field.AssetBound.MediaTypes) > 0 {
			spec.MediaTypes = append([]string(nil), field.AssetBound.MediaTypes...)
		}
		var nested any
		if json.Unmarshal(field.Type, &nested) != nil {
			return AssetField{}, false
		}
		return assetSpecAtValue(nested, parts[1:], spec)
	}
	return AssetField{}, false
}

func setAssetRef(document map[string]any, parts []string, ref string) *exit.Error {
	_, e := setPath(document, parts, ref, strings.Join(parts, "."))
	return e
}

func setPath(value any, parts []string, ref, whole string) (any, *exit.Error) {
	if len(parts) == 0 {
		if occupied(value, ref) {
			return value, assetCollision(whole)
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
		updated, e := setPath(child, rest, ref, whole)
		if e == nil {
			node[part] = updated
		}
		return node, e
	case []any:
		position, err := strconv.Atoi(part)
		if err != nil || position < 0 {
			return value, exit.New(exit.Validation, "%q is not a list position in asset field %s", part, whole)
		}
		for len(node) <= position {
			node = append(node, nil)
		}
		child := node[position]
		if child == nil && len(rest) > 0 {
			child = containerFor(rest[0])
		}
		updated, e := setPath(child, rest, ref, whole)
		if e == nil {
			node[position] = updated
		}
		return node, e
	default:
		return value, assetCollision(whole)
	}
}

func containerFor(next string) any {
	if _, err := strconv.ParseUint(next, 10, 31); err == nil {
		return []any{}
	}
	return map[string]any{}
}

func occupied(value any, ref string) bool {
	return value != nil && value != "" && value != ref
}

func assetCollision(path string) *exit.Error {
	return exit.New(exit.Conflict, "input asset field %q already has a payload value", path).
		WithRemedy("supply an asset field once, with `--asset %s=<file>`", path)
}

func pathOrder(parts []string) uint32 {
	var order uint64
	for _, part := range parts {
		if index, err := strconv.ParseUint(part, 10, 32); err == nil {
			order = index
		}
	}
	return uint32(order)
}
