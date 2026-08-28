package launch

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

// The `run` payload grammar (cozy-runtime-cli.md), the same one cr-016 implements for a
// bare-venv run: a positional PRIMARY filling the first declared field, `key=value`
// scalars, `key=@file` local file contents encoded as one JSON string,
// `key:=<json>` raw JSON for nested values, and
// `--in <file>` supplying the whole payload with `key=value` merged on top.
//
// TYPING IS THE DESCRIPTOR'S, not this module's. `steps=20` is an int because the field's
// rendered schema says int, never because the string looked numeric — the difference shows
// up the first time an endpoint declares a string field whose value is digits. The schema
// is the surface the release's own runtime vouched for at install, read back from the
// generation, so this costs microseconds and no subprocess: cozy-creator.md's
// "payload validation is client-side from the recorded schema, near-instant".
//
// An undeclared key refuses HERE, before a request is recorded and long before a model
// loads, because a typo should cost a millisecond.

// ParsePayload builds one request document from the argv terms.
func ParsePayload(ep *Entrypoint, terms []string, infile string) (json.RawMessage, *exit.Error) {
	document := map[string]json.RawMessage{}

	if infile != "" {
		data, err := os.ReadFile(infile)
		if err != nil {
			return nil, exit.New(exit.NotFound, "--in %s: %s", infile, err).
				WithRemedy("--in takes one JSON file holding the whole payload")
		}
		var loaded map[string]json.RawMessage
		if err := json.Unmarshal(data, &loaded); err != nil {
			return nil, exit.New(exit.Validation, "--in %s does not hold one JSON object: %s", infile, err)
		}
		for k, v := range loaded {
			document[k] = v
		}
	}

	var positional []string
	for _, term := range terms {
		colon := strings.Index(term, ":=")
		// `key:=json` only when the FIRST `=` is the one in `:=` — `a=b:=c` is a scalar.
		if colon >= 0 && strings.Index(term, "=") == colon+1 {
			key, raw := term[:colon], term[colon+2:]
			if !json.Valid([]byte(raw)) {
				return nil, exit.Usagef("%s:=… is not JSON: %s", key, raw).
					WithRemedy("`key:=<json>` carries a nested value verbatim; `key=value` is the scalar form")
			}
			if e := declared(ep, key); e != nil {
				return nil, e
			}
			document[key] = json.RawMessage(raw)
			continue
		}
		if key, raw, ok := strings.Cut(term, "="); ok {
			if e := declared(ep, key); e != nil {
				return nil, e
			}
			if after, isFile := strings.CutPrefix(raw, "@"); isFile {
				rendered, _ := ep.TypeOfField(key)
				kind, _ := typeOf(rendered)
				if kind == "asset" {
					return nil, exit.New(exit.Validation,
						"%s.%s is an input asset and key=@file has no grant identity", ep.Name, key).
						WithRemedy("use `--asset %s=%s`; the schema field path becomes the input identity", key, after)
				}
				data, err := os.ReadFile(after)
				if err != nil {
					return nil, exit.New(exit.NotFound, "%s=@%s: %s", key, after, err)
				}
				encoded, err := json.Marshal(string(data))
				if err != nil {
					return nil, exit.Internalf("cannot carry %s: %s", after, err)
				}
				document[key] = encoded
				continue
			}
			value, e := typed(ep, key, raw)
			if e != nil {
				return nil, e
			}
			document[key] = value
			continue
		}
		positional = append(positional, term)
	}

	if len(positional) > 0 {
		primary := ""
		if len(ep.Request.Fields) > 0 {
			// The FIRST declared field, in the author's own declaration order (msgspec
			// preserves it), never a name this module blesses.
			primary = ep.Request.Fields[0].Name
		}
		if primary == "" {
			return nil, exit.Usagef("%s takes no positional value: it declares no request field", ep.Name)
		}
		if len(positional) > 1 {
			return nil, exit.Usagef("%s takes ONE positional value (%s); got %d",
				ep.Name, primary, len(positional)).
				WithRemedy("every other field is `key=value`, `key=@file` or `key:=<json>`")
		}
		value, e := typed(ep, primary, positional[0])
		if e != nil {
			return nil, e
		}
		document[primary] = value
	}

	data, err := json.Marshal(document)
	if err != nil {
		return nil, exit.Internalf("cannot render the payload: %s", err)
	}
	return data, nil
}

// ValidatePayload checks one already-rendered request object against the exact descriptor
// schema. It is the workflow/composer side of the same recorded-schema gate ParsePayload
// applies while building a CLI payload.
func ValidatePayload(ep *Entrypoint, payload json.RawMessage) *exit.Error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var document map[string]any
	if err := decoder.Decode(&document); err != nil || document == nil {
		return exit.New(exit.Validation, "%s payload is not one JSON object", ep.Name)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return exit.New(exit.Validation, "%s payload carries trailing JSON", ep.Name)
	}
	fieldsByName := map[string]Field{}
	for _, field := range ep.Request.Fields {
		fieldsByName[field.Name] = field
		if field.Wire == "required" {
			if _, ok := document[field.Name]; !ok {
				return exit.New(exit.Validation, "%s payload omits required field %q", ep.Name, field.Name)
			}
		}
	}
	for name, value := range document {
		field, ok := fieldsByName[name]
		if !ok {
			return declared(ep, name)
		}
		if problem := validateField(field, value, name); problem != nil {
			return problem
		}
	}
	return nil
}

// PopulatedAssetPaths returns every request-schema asset field carrying a non-empty
// reference in one already-validated payload. It lets a workflow prove that each opaque
// reference has an out-of-band grant instead of accepting a string the worker cannot open.
func PopulatedAssetPaths(ep *Entrypoint, payload json.RawMessage) ([]string, *exit.Error) {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var document map[string]any
	if err := decoder.Decode(&document); err != nil || document == nil {
		return nil, exit.New(exit.Validation, "%s payload is not one JSON object", ep.Name)
	}
	var out []string
	for _, field := range ep.Request.Fields {
		value, present := document[field.Name]
		if !present {
			continue
		}
		if problem := validateRenderedInto(field.Type, value, field.Name, &out); problem != nil {
			return nil, problem
		}
	}
	sort.Strings(out)
	return out, nil
}

// ValidatePayloadAssets validates one authored payload after substituting exact-shaped
// opaque references at the declared out-of-band grant paths. It is the pre-record gate
// shared by higher-level composers that know asset identity before the digest exists in
// a backward workflow binding.
func ValidatePayloadAssets(ep *Entrypoint, payload json.RawMessage, assetPaths []string) *exit.Error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var document map[string]any
	if err := decoder.Decode(&document); err != nil || document == nil {
		return exit.New(exit.Validation, "%s payload is not one JSON object", ep.Name)
	}
	expected := append([]string(nil), assetPaths...)
	sort.Strings(expected)
	for index, path := range expected {
		if index > 0 && expected[index-1] == path {
			return exit.New(exit.Validation, "request asset path %s appears more than once", path)
		}
		parts, problem := assetPath(path)
		if problem != nil {
			return problem
		}
		if problem := setAssetRef(document, parts,
			"sha256:0000000000000000000000000000000000000000000000000000000000000000"); problem != nil {
			return problem
		}
	}
	rendered, err := json.Marshal(document)
	if err != nil {
		return exit.Internalf("cannot render %s validation payload: %s", ep.Name, err)
	}
	if problem := ValidatePayload(ep, rendered); problem != nil {
		return problem
	}
	populated, problem := PopulatedAssetPaths(ep, rendered)
	if problem != nil {
		return problem
	}
	if len(populated) != len(expected) {
		return exit.Named(exit.Validation, "request_asset_resolution",
			"%s payload asset references do not exactly match its declared grants", ep.Name)
	}
	for index := range populated {
		if populated[index] != expected[index] {
			return exit.Named(exit.Validation, "request_asset_resolution",
				"%s payload asset references do not exactly match its declared grants", ep.Name)
		}
	}
	return nil
}

func validateField(field Field, value any, path string) *exit.Error {
	return validateFieldInto(field, value, path, nil)
}

func validateFieldInto(field Field, value any, path string, assets *[]string) *exit.Error {
	if len(field.Constraints.Unknown) > 0 {
		return exit.Named(exit.Structural, "descriptor_constraint_unknown",
			"request field %s uses unsupported constraints: %s",
			path, strings.Join(field.Constraints.Unknown, ", "))
	}
	if problem := validateRenderedInto(field.Type, value, path, assets); problem != nil {
		return problem
	}
	if field.Constraints.MinLength != nil || field.Constraints.MaxLength != nil {
		var length int64
		switch typed := value.(type) {
		case string:
			length = int64(utf8.RuneCountInString(typed))
		case []any:
			length = int64(len(typed))
		default:
			return exit.Named(exit.Structural, "descriptor_constraint_unknown",
				"request field %s has length constraints on a non-string, non-list type", path)
		}
		if minimum := field.Constraints.MinLength; minimum != nil && length < *minimum {
			return exit.New(exit.Validation,
				"request field %s has length %d; its minimum is %d", path, length, *minimum)
		}
		if maximum := field.Constraints.MaxLength; maximum != nil && length > *maximum {
			return exit.New(exit.Validation,
				"request field %s has length %d; its maximum is %d", path, length, *maximum)
		}
	}
	if field.Constraints.GT != nil || field.Constraints.GE != nil || field.Constraints.LE != nil {
		number, ok := value.(json.Number)
		if !ok {
			return exit.Named(exit.Structural, "descriptor_constraint_unknown",
				"request field %s has numeric constraints on a non-numeric type", path)
		}
		parsed, err := strconv.ParseFloat(number.String(), 64)
		if err != nil {
			return exit.New(exit.Validation, "request field %s is not a finite number", path)
		}
		if minimum := field.Constraints.GE; minimum != nil && parsed < *minimum {
			return exit.New(exit.Validation,
				"request field %s is %s; its minimum is %v", path, number.String(), *minimum)
		}
		if minimum := field.Constraints.GT; minimum != nil && parsed <= *minimum {
			return exit.New(exit.Validation,
				"request field %s is %s; it must be greater than %v", path, number.String(), *minimum)
		}
		if maximum := field.Constraints.LE; maximum != nil && parsed > *maximum {
			return exit.New(exit.Validation,
				"request field %s is %s; its maximum is %v", path, number.String(), *maximum)
		}
	}
	return nil
}

func validateRenderedInto(raw json.RawMessage, value any, path string, assets *[]string) *exit.Error {
	var scalar string
	if json.Unmarshal(raw, &scalar) == nil {
		switch scalar {
		case "str":
			if _, ok := value.(string); ok {
				return nil
			}
		case "int":
			if number, ok := value.(json.Number); ok {
				if _, err := strconv.ParseInt(number.String(), 10, 64); err == nil {
					return nil
				}
			}
		case "float":
			if number, ok := value.(json.Number); ok {
				if _, err := strconv.ParseFloat(number.String(), 64); err == nil {
					return nil
				}
			}
		case "bool":
			if _, ok := value.(bool); ok {
				return nil
			}
		case "null":
			if value == nil {
				return nil
			}
		default:
			return exit.Named(exit.Structural, "descriptor_type_unknown",
				"request field %s has unsupported descriptor scalar %q", path, scalar)
		}
		return exit.New(exit.Validation, "request field %s does not match declared %s", path, scalar)
	}
	var schema map[string]json.RawMessage
	if json.Unmarshal(raw, &schema) != nil {
		return exit.Named(exit.Structural, "descriptor_type_unknown",
			"request field %s has an unreadable descriptor type", path)
	}
	if _, ok := schema["asset"]; ok {
		if ref, ok := value.(string); ok && ref != "" {
			if assets != nil {
				*assets = append(*assets, path)
			}
			return nil
		}
		return exit.New(exit.Validation, "request asset field %s is not a non-empty reference", path)
	}
	if input, ok := schema["input"]; ok && string(input) == `"tree"` {
		if _, ok := value.(string); ok {
			return nil
		}
		return exit.New(exit.Validation, "request tree field %s is not a reference", path)
	}
	if literal, ok := schema["literal"]; ok {
		var members []json.RawMessage
		if json.Unmarshal(literal, &members) != nil || len(members) == 0 {
			return exit.Named(exit.Structural, "descriptor_type_unknown",
				"request field %s has an unreadable literal", path)
		}
		actual, err := json.Marshal(value)
		if err == nil {
			for _, member := range members {
				var compact bytes.Buffer
				if json.Compact(&compact, member) == nil && bytes.Equal(actual, compact.Bytes()) {
					return nil
				}
			}
		}
		return exit.New(exit.Validation, "request field %s is not one of its declared literals", path)
	}
	if union, ok := schema["union"]; ok {
		var branches []json.RawMessage
		if json.Unmarshal(union, &branches) != nil {
			return exit.Named(exit.Structural, "descriptor_type_unknown",
				"request field %s has an unreadable union", path)
		}
		for _, branch := range branches {
			var found []string
			if validateRenderedInto(branch, value, path, &found) == nil {
				if assets != nil {
					*assets = append(*assets, found...)
				}
				return nil
			}
		}
		return exit.New(exit.Validation, "request field %s matches no declared union branch", path)
	}
	if item, ok := schema["list"]; ok {
		values, ok := value.([]any)
		if !ok {
			return exit.New(exit.Validation, "request field %s is not a list", path)
		}
		for index, element := range values {
			if problem := validateRenderedInto(item, element,
				path+"."+strconv.Itoa(index), assets); problem != nil {
				return problem
			}
		}
		return nil
	}
	if fields, ok := schema["fields"]; ok {
		var nested Struct
		if json.Unmarshal(raw, &nested) != nil {
			return exit.Named(exit.Structural, "descriptor_type_unknown",
				"request field %s has unreadable nested fields", path)
		}
		object, ok := value.(map[string]any)
		if !ok {
			return exit.New(exit.Validation, "request field %s is not an object", path)
		}
		declared := map[string]Field{}
		for _, field := range nested.Fields {
			declared[field.Name] = field
			if field.Wire == "required" {
				if _, ok := object[field.Name]; !ok {
					return exit.New(exit.Validation, "request field %s omits required %s", path, field.Name)
				}
			}
		}
		for name, element := range object {
			field, ok := declared[name]
			if !ok {
				return exit.New(exit.Validation, "request field %s declares no nested field %q", path, name)
			}
			if problem := validateFieldInto(field, element, path+"."+name, assets); problem != nil {
				return problem
			}
		}
		_ = fields
		return nil
	}
	return exit.Named(exit.Structural, "descriptor_type_unknown",
		"request field %s has an unsupported descriptor type", path)
}

func declared(ep *Entrypoint, key string) *exit.Error {
	if _, ok := ep.TypeOfField(key); ok {
		return nil
	}
	return exit.New(exit.Validation, "%s declares no request field %q", ep.Name, key).
		WithRemedy("it declares: %s", strings.Join(ep.RequestFields(), ", ")).
		WithNext("cozy describe <org/endpoint>")
}

// typed spells one scalar the way the field's rendered schema declares it.
func typed(ep *Entrypoint, key, raw string) (json.RawMessage, *exit.Error) {
	rendered, ok := ep.TypeOfField(key)
	if !ok {
		return nil, declared(ep, key)
	}
	kind, _ := typeOf(rendered)
	switch kind {
	case "scalar:int":
		if _, err := strconv.ParseInt(raw, 10, 64); err != nil {
			return nil, wrongType(ep, key, raw, "int")
		}
		return json.RawMessage(raw), nil
	case "scalar:float":
		if _, err := strconv.ParseFloat(raw, 64); err != nil {
			return nil, wrongType(ep, key, raw, "float")
		}
		return json.RawMessage(raw), nil
	case "scalar:bool":
		switch raw {
		case "true", "false":
			return json.RawMessage(raw), nil
		}
		return nil, wrongType(ep, key, raw, "bool")
	case "scalar:str", "tree":
		// A TREE's wire value IS its ref (cr-009): the path rides the field VALUE at the
		// far end, hydrated from the grant, and a ref the grant does not cover never
		// reaches a filesystem. So the scalar spelling is the ref, and `--input
		// <ref>=<dir>` is what grants the read.
		encoded, err := json.Marshal(raw)
		if err != nil {
			return nil, exit.Internalf("cannot carry %s: %s", key, err)
		}
		return encoded, nil
	case "asset":
		return nil, exit.New(exit.Validation,
			"%s.%s is an input asset and `key=value` cannot grant its bytes", ep.Name, key).
			WithRemedy("use `--asset %s=<file>`; the schema field path becomes the input identity", key)
	}
	// Anything structured — an asset, a list, a nested struct — has no scalar spelling.
	// `key:=<json>` is the form that carries it, and saying so beats guessing.
	return nil, exit.New(exit.Validation,
		"%s.%s is not a scalar and `key=value` cannot spell one", ep.Name, key).
		WithRemedy("carry it as `%s:=<json>`; asset fields use `--asset <field-path>=<file>`", key)
}

func wrongType(ep *Entrypoint, key, raw, want string) *exit.Error {
	return exit.New(exit.Validation,
		"%s.%s is declared %s and %q is not one", ep.Name, key, want, raw).
		WithRemedy("the schema is the release's own — `cozy describe <org/endpoint>/%s` prints it", ep.Name)
}
