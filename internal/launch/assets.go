package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/records"
)

// ParseAssets resolves files and label=file occurrences in a declared Assets slot,
// or explicit field-path=file bindings, into one payload plus exact byte bindings.
// The payload carries only an opaque content reference; paths and
// bytes travel out-of-band through DeliveryGrant. A nested path is resolved against the
// package's recorded schema, so `references.0.image` cannot accidentally grant a file
// to a scalar or to a misspelled field.
type ImagePreparer func(source string, kind AssetsKind) (string, *exit.Error)

func ParseAssets(ep *Entrypoint, payload json.RawMessage, specs, fidelities []string, prepare ImagePreparer) (json.RawMessage, []records.AssetBinding, *exit.Error) {
	// JSON filenames join the ordinary --asset path before any fingerprinting or grants.
	// Clear only these local spellings so setAssetRef can insert their verified identity.
	var embedded []string
	var problem *exit.Error
	payload, problem = mapAssetFilenames(ep, payload, func(path, source string) (any, *exit.Error) {
		if strings.Contains(source, "://") {
			return nil, exit.New(exit.Validation, "%s needs a local asset filename, not a URL", path)
		}
		embedded = append(embedded, path+"="+source)
		return nil, nil
	})
	if problem != nil {
		return nil, nil, problem
	}
	specs = append(embedded, specs...)
	if len(specs) == 0 && len(fidelities) == 0 && ep.Assets == nil {
		return payload, nil, nil
	}
	var document map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil || document == nil {
		return nil, nil, exit.Internalf("cannot add input assets to the payload: %s", err)
	}
	if ep.Assets != nil {
		if _, present := document[ep.Assets.Parameter]; !present {
			document[ep.Assets.Parameter] = []any{}
		}
	}

	if problem := preflightAssetCount(ep, document, specs); problem != nil {
		return nil, nil, problem
	}
	seen := map[string]bool{}
	type pendingAsset struct {
		fieldPath, source string
		parts             []string
	}
	pending := make([]pendingAsset, 0, len(specs))
	for _, spec := range specs {
		fieldPath, source, label, explicit := splitAssetArgument(ep, spec)
		if !explicit {
			if ep.Assets == nil {
				return nil, nil, exit.Usagef("%s declares no Assets input for --asset %q", ep.Name, spec).
					WithRemedy("use --asset <field-path>=<file> for a named payload asset")
			}
			values, _ := document[ep.Assets.Parameter].([]any)
			fieldPath = fmt.Sprintf("%s.%d.asset", ep.Assets.Parameter, len(values))
			entry := map[string]any{}
			if label != "" {
				entry["label"] = label
			}
			document[ep.Assets.Parameter] = append(values, entry)
		}
		fieldPath, source = strings.TrimSpace(fieldPath), strings.TrimSpace(source)
		if fieldPath == "" || source == "" {
			return nil, nil, exit.Usagef("--asset %q needs a file", spec)
		}
		parts, e := assetPath(fieldPath)
		if e != nil {
			return nil, nil, e
		}
		// The same case-insensitive NAME fold ParsePayload applies (canonicalFieldKey):
		// the top-level segment folds onto the PackageInterface's spelling, so the binding,
		// the payload ref, and the worker-protocol input id all carry the canonical name.
		folded, e := canonicalFieldKey(ep, parts[0])
		if e != nil {
			return nil, nil, e
		}
		if folded != parts[0] {
			parts[0] = folded
			fieldPath = strings.Join(parts, ".")
		}
		if seen[fieldPath] {
			return nil, nil, exit.New(exit.Validation, "input asset field %q was supplied more than once", fieldPath)
		}
		if e := inputasset.ValidateID(fieldPath); e != nil {
			return nil, nil, e
		}
		seen[fieldPath] = true
		if !assetAt(ep.Request, parts) {
			return nil, nil, exit.New(exit.Validation,
				"%s.%s is not an asset field in this release's request schema", ep.Name, fieldPath).
				WithRemedy("the installed package.package-interface.json declares %s's request schema", ep.Name)
		}
		if ep.Assets.contains(parts) {
			values := document[ep.Assets.Parameter].([]any)
			index, _ := strconv.Atoi(parts[1])
			for len(values) <= index {
				values = append(values, nil)
			}
			if values[index] == nil {
				values[index] = map[string]any{}
			}
			document[ep.Assets.Parameter] = values
		}
		pending = append(pending, pendingAsset{fieldPath, source, parts})
	}
	if problem := applyAssetFidelity(ep, document, fidelities); problem != nil {
		return nil, nil, problem
	}
	assets := make([]records.AssetBinding, 0, len(pending))
	for _, item := range pending {
		fieldPath, source, parts := item.fieldPath, item.source, item.parts
		assetSpec, _ := AssetSpec(ep, fieldPath)
		maxBytes := assetSpec.MaxBytes
		if maxBytes <= 0 {
			maxBytes = inputasset.MaxBytes
		}

		if source == "~" || strings.HasPrefix(source, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				return nil, nil, exit.New(exit.NotFound, "cannot resolve input asset home: %s", err)
			}
			if source == "~" {
				source = home
			} else {
				source = filepath.Join(home, strings.TrimPrefix(source, "~/"))
			}
		}
		absolute, err := filepath.Abs(source)
		if err != nil {
			return nil, nil, exit.New(exit.NotFound, "cannot resolve input asset %s: %s", source, err)
		}
		if kind := ep.Assets.preparedImageKind(); prepare != nil && kind != nil && ep.Assets.contains(parts) {
			_, mediaType, problem := inputasset.Probe(absolute)
			if problem != nil {
				return nil, nil, problem
			}
			if (AssetField{MediaTypes: kind.MediaTypes}).AcceptsMediaType(mediaType) {
				absolute, problem = prepare(absolute, *kind)
				if problem != nil {
					return nil, nil, problem
				}
			}
		}
		facts, e := inputasset.Fingerprint(absolute, maxBytes)
		if e != nil {
			return nil, nil, e
		}
		length, digest, mediaType := facts.Length, facts.Digest, facts.MediaType
		selected, admitted := AssetSpecForMedia(ep, fieldPath, mediaType)
		if !admitted || !selected.AcceptsMediaType(mediaType) {
			return nil, nil, exit.New(exit.Validation,
				"%s.%s accepts media types [%s], not %q",
				ep.Name, fieldPath, strings.Join(assetSpec.MediaTypes, ", "), mediaType)
		}
		maxBytes = effectiveAssetMax(selected.MaxBytes)
		if length > maxBytes {
			return nil, nil, exit.Named(exit.Validation, "input_asset_bound", "input asset %s is %d B; its media policy permits %d B", fieldPath, length, maxBytes)
		}
		if e := setAssetRef(document, parts, digest); e != nil {
			return nil, nil, e
		}
		assets = append(assets, records.AssetBinding{
			FieldPath: fieldPath, LocalPath: absolute, Digest: digest,
			Length: length, MediaType: mediaType, Order: pathOrder(parts), MaxBytes: maxBytes,
			ModTime: facts.ModTime,
		})
	}
	if problem := ValidateAssetCounts(ep, assets); problem != nil {
		return nil, nil, problem
	}
	sort.Slice(assets, func(i, j int) bool { return assets[i].FieldPath < assets[j].FieldPath })
	rendered, err := json.Marshal(document)
	if err != nil {
		return nil, nil, exit.Internalf("cannot render the payload with input assets: %s", err)
	}
	return rendered, assets, nil
}

// Explicit path syntax wins over label/field assignment, so a=b.png stays a
// filename in /refs/a=b.png, ~/refs/a=b.png or ./refs/a=b.png.
func splitAssetArgument(ep *Entrypoint, spec string) (field, source, label string, named bool) {
	if filepath.IsAbs(spec) || spec == "~" || strings.HasPrefix(spec, "~/") ||
		strings.HasPrefix(spec, "./") || strings.HasPrefix(spec, "../") {
		return "", spec, "", false
	}
	field, source, named = strings.Cut(spec, "=")
	if !named {
		return "", spec, "", false
	}
	if ep.Assets == nil || namedAssetSpec(ep, field) {
		return field, source, "", true
	}
	return "", source, field, false
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
				WithRemedy("use package-interface field names separated by dots; list positions are decimal indexes")
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
	TagField   string
	TagValue   any
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
	if ep.Assets.contains(parts) {
		return AssetField{Kind: "file", MaxBytes: ep.Assets.maxBytes()}, true
	}
	return assetSpecAt(ep.Request, parts, AssetField{})
}

// ResultAssetSpec returns the exact asset kind, byte bound, and media types declared for
// one result path. Unlike AssetSpec it reads the result schema; output naming uses it to
// announce an exact destination before execution when the package declares one MIME type.
func ResultAssetSpec(ep *Entrypoint, path string) (AssetField, bool) {
	parts, problem := assetPath(path)
	if problem != nil {
		return AssetField{}, false
	}
	return assetSpecAt(ep.Result, parts, AssetField{})
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
	if object["input"] == "tree" {
		inherited.Kind = "tree"
		return inherited, len(parts) == 0
	}
	if kind, ok := object["asset"].(string); ok {
		inherited.Kind = kind
		return inherited, len(parts) == 0
	}
	if field, ok := object["tag_field"].(string); ok && field != "" {
		inherited.TagField, inherited.TagValue = field, object["tag"]
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

// RetainedAssetPaths names native file/tree results; ordinary rendered media
// keeps its publication path even when returned beside a retained artifact.
func RetainedAssetPaths(ep *Entrypoint) map[string]bool {
	result := map[string]bool{}
	for _, path := range AssetPaths(ep.Result) {
		if spec, ok := ResultAssetSpec(ep, path); ok && (spec.Kind == "file" || spec.Kind == "tree") {
			result[path] = true
		}
	}
	return result
}
