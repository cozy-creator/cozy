package launch

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// The `run` payload grammar (cozy-runtime-cli.md), the same one cr-016 implements for a
// bare-venv run: a positional PRIMARY filling the first declared field, `key=value`
// scalars, `key=@file` local file contents encoded as one JSON string,
// `key:=<json>` raw JSON for nested values, and
// `--in <file>` supplying the whole payload with `key=value` merged on top.
//
// TYPING IS THE DESCRIPTOR'S, not this module's. `steps=20` is an int because the field's
// rendered schema says int, never because the string looked numeric — the difference shows
// up the first time a package declares a string field whose value is digits. The schema
// is the surface the release's own runtime vouched for at install, read back from the
// install, so this costs microseconds and no subprocess: cozy-creator.md's
// "payload validation is client-side from the recorded schema, near-instant".
//
// An undeclared key refuses HERE, before a request is recorded and long before a model
// loads, because a typo should cost a millisecond.

// KernelAxes is the execution-path override's vocabulary of AXES (cr-125), and it is one:
// attention. A GEMM or fusion pin is not here because no measurement has asked for one. The
// KERNEL names are the runtime's own (`attention.BY_NAME`) and are deliberately NOT restated
// here — a second copy of that vocabulary would drift from the image that ships the kernels,
// so an unknown NAME refuses at the worker, which knows what it can actually run.
var KernelAxes = []string{"attention"}

// RunKeys are the RESERVED dotted namespaces a run term may claim before the payload grammar
// sees it. `model.<param>=<ref>` picks the WEIGHTS off the owner's ladder (cl-109);
// `kernel.<axis>=<name>` picks the EXECUTION PATH off the runtime's own selection (cr-125).
// Neither can reach a payload field, because a wire name carries no dot.
type RunKeys struct {
	// Models is the declared slot path -> ref the caller pinned.
	Models map[string]string
	// AttentionKernel rides the InvocationSpec to the worker; empty leaves the runtime's own
	// selection alone, which is what every ordinary run does.
	AttentionKernel string
}

// ParsePayload builds one request document from the argv terms, and collects the reserved
// run keys beside it. The exact-case dotted prefixes are matched BEFORE field case-folding
// and can never reach a payload field — wire names carry no dot — so they route to the
// returned RunKeys. A bare `model=` or `kernel=` term stays an ordinary payload field.
func ParsePayload(ep *Entrypoint, terms []string, infile string) (
	json.RawMessage, RunKeys, *exit.Error,
) {
	document := map[string]json.RawMessage{}

	if infile != "" {
		data, err := os.ReadFile(infile)
		if err != nil {
			return nil, RunKeys{}, exit.New(exit.NotFound, "--in %s: %s", infile, err).
				WithRemedy("--in takes one JSON file holding the whole payload")
		}
		var loaded map[string]json.RawMessage
		if err := json.Unmarshal(data, &loaded); err != nil {
			return nil, RunKeys{}, exit.New(exit.Validation, "--in %s does not hold one JSON object: %s", infile, err)
		}
		for k, v := range loaded {
			document[k] = v
		}
	}

	keys := RunKeys{Models: map[string]string{}}
	var positional []string
	for _, term := range terms {
		if strings.HasPrefix(term, "model.") && strings.Contains(term, "=") {
			slotPath, ref, e := modelOverrideTerm(ep, term)
			if e != nil {
				return nil, RunKeys{}, e
			}
			if _, dup := keys.Models[slotPath]; dup {
				return nil, RunKeys{}, exit.Usagef("model slot %s was bound more than once", slotPath)
			}
			keys.Models[slotPath] = ref
			continue
		}
		if strings.HasPrefix(term, "kernel.") {
			axis, name, e := kernelOverrideTerm(term)
			if e != nil {
				return nil, RunKeys{}, e
			}
			switch axis {
			case "attention":
				if keys.AttentionKernel != "" {
					return nil, RunKeys{}, exit.Usagef("kernel.attention was pinned more than once")
				}
				keys.AttentionKernel = name
			default:
				// A NAMED AXIS WITH NOWHERE TO PUT IT. Reachable only by adding a name to
				// KernelAxes without a field beside it, and silently dropping the pin there
				// would be the exact failure the whole namespace exists to prevent.
				return nil, RunKeys{}, exit.Internalf("kernel.%s is declared but carries nowhere", axis)
			}
			continue
		}
		colon := strings.Index(term, ":=")
		// `key:=json` only when the FIRST `=` is the one in `:=` — `a=b:=c` is a scalar.
		if colon >= 0 && strings.Index(term, "=") == colon+1 {
			key, raw := term[:colon], term[colon+2:]
			if !json.Valid([]byte(raw)) {
				return nil, RunKeys{}, exit.Usagef("%s:=… is not JSON: %s", key, raw).
					WithRemedy("`key:=<json>` carries a nested value verbatim; `key=value` is the scalar form")
			}
			key, e := canonicalFieldKey(ep, key)
			if e != nil {
				return nil, RunKeys{}, e
			}
			if e := declared(ep, key); e != nil {
				return nil, RunKeys{}, e
			}
			document[key] = json.RawMessage(raw)
			continue
		}
		if key, raw, ok := strings.Cut(term, "="); ok {
			key, e := canonicalFieldKey(ep, key)
			if e != nil {
				return nil, RunKeys{}, e
			}
			if e := declared(ep, key); e != nil {
				return nil, RunKeys{}, e
			}
			if after, isFile := strings.CutPrefix(raw, "@"); isFile {
				rendered, _ := ep.TypeOfField(key)
				kind, _ := typeOf(rendered)
				if kind == "asset" {
					return nil, RunKeys{}, exit.New(exit.Validation,
						"%s.%s is an input asset and key=@file has no grant identity", ep.Name, key).
						WithRemedy("use `--asset %s=%s`; the schema field path becomes the input identity", key, after)
				}
				data, err := os.ReadFile(after)
				if err != nil {
					return nil, RunKeys{}, exit.New(exit.NotFound, "%s=@%s: %s", key, after, err)
				}
				encoded, err := json.Marshal(string(data))
				if err != nil {
					return nil, RunKeys{}, exit.Internalf("cannot carry %s: %s", after, err)
				}
				document[key] = encoded
				continue
			}
			value, e := typed(ep, key, raw)
			if e != nil {
				return nil, RunKeys{}, e
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
			return nil, RunKeys{}, exit.Usagef("%s takes no positional value: it declares no request field", ep.Name)
		}
		if len(positional) > 1 {
			return nil, RunKeys{}, exit.Usagef("%s takes ONE positional value (%s); got %d",
				ep.Name, primary, len(positional)).
				WithRemedy("every other field is `key=value`, `key=@file` or `key:=<json>`")
		}
		value, e := typed(ep, primary, positional[0])
		if e != nil {
			return nil, RunKeys{}, e
		}
		document[primary] = value
	}

	data, err := json.Marshal(document)
	if err != nil {
		return nil, RunKeys{}, exit.Internalf("cannot render the payload: %s", err)
	}
	return data, keys, nil
}

// kernelOverrideTerm claims one `kernel.`-prefixed argv term for the execution-path override
// (cr-125). The namespace is claimed WHOLE: a term that is not a legal override refuses here
// rather than falling through to become a payload field named `kernel.gemm`, which is exactly
// the mistyped-pin failure the reservation exists to prevent. The `:=` form is refused with
// the rest — a kernel name is one word, so a structured spelling would be a second grammar
// for the same fact.
func kernelOverrideTerm(term string) (axis, name string, problem *exit.Error) {
	refuse := func() *exit.Error {
		spellings := make([]string, 0, len(KernelAxes))
		for _, a := range KernelAxes {
			spellings = append(spellings, "kernel."+a+"=<name>")
		}
		return exit.Named(exit.Usage, "kernel_override_unknown",
			"%s is not an execution-path override", term).
			WithRemedy("the reserved namespace is %s", strings.Join(spellings, ", "))
	}
	key, raw, ok := strings.Cut(term, "=")
	if !ok || raw == "" || strings.HasSuffix(key, ":") {
		return "", "", refuse()
	}
	asked := strings.TrimPrefix(key, "kernel.")
	if !slices.Contains(KernelAxes, asked) {
		return "", "", refuse()
	}
	return asked, strings.TrimSpace(raw), nil
}

// modelOverrideTerm claims one `model.`-prefixed argv term for the reserved run-key
// grammar and resolves its `<param>` suffix onto a declared model slot. The structured
// `model.<param>:={…}` form is refused: the ref suffix grammar already spells every
// fact the resolver accepts.
func modelOverrideTerm(ep *Entrypoint, term string) (slotPath, ref string, problem *exit.Error) {
	colon := strings.Index(term, ":=")
	if colon >= 0 && strings.Index(term, "=") == colon+1 {
		return "", "", exit.Usagef("%s has no structured spelling", term[:colon]).
			WithRemedy("the ref grammar carries every fact: %s=org/model[@release[/lane]][#sha256:<hex>]", term[:colon])
	}
	key, raw, _ := strings.Cut(term, "=")
	slot, e := modelOverrideSlot(ep, strings.TrimPrefix(key, "model."))
	if e != nil {
		return "", "", e
	}
	return slot.Path, strings.TrimSpace(raw), nil
}

// modelOverrideSlot resolves a run key's `<param>` onto one declared slot. Params and
// full slot paths are disjoint spellings (a param carries no dot, a path always does),
// but the ambiguity guard stays: guessing between two slots is never right.
func modelOverrideSlot(ep *Entrypoint, asked string) (*Slot, *exit.Error) {
	var match *Slot
	for i := range ep.Models {
		slot := &ep.Models[i]
		if asked != slot.Path && asked != slot.Param {
			continue
		}
		if match != nil {
			return nil, exit.Usagef("model slot %q is ambiguous; use its full descriptor path", asked)
		}
		match = slot
	}
	if match != nil {
		return match, nil
	}
	if len(ep.Models) == 0 {
		return nil, exit.Named(exit.Usage, "model_slot_unknown",
			"no such model slot %q: %s declares no model slots", asked, ep.Name)
	}
	params := make([]string, 0, len(ep.Models))
	for _, slot := range ep.Models {
		params = append(params, slot.Param)
	}
	return nil, exit.Named(exit.Usage, "model_slot_unknown",
		"no such model slot %q for %s", asked, ep.Name).
		WithRemedy("%s declares: %s", ep.Name, strings.Join(params, ", "))
}

// ValidatePayload checks one already-rendered request object against the exact PackageInterface
// schema — the daemon-submit half of the same recorded-schema gate ParsePayload applies
// while building a CLI payload. It fires BEFORE a request row exists or an idempotency
// key is recorded (cl-105): every offending field is named in ONE typed
// `request_payload_invalid` refusal whose remedy is the callable's usage line. A
// PackageInterface the validator itself cannot read stays a structural refusal: that is a host
// fault, not a payload fault.
func ValidatePayload(pkg string, ep *Entrypoint, payload json.RawMessage) *exit.Error {
	target := ep.Name
	if pkg != "" {
		target = pkg + "/" + ep.Name
	}
	refuse := func(problems []string) *exit.Error {
		return exit.Named(exit.Validation, "request_payload_invalid",
			"%s request payload is invalid: %s", target, strings.Join(problems, "; ")).
			WithRemedy("%s", UsageLine(target, ep))
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var document map[string]any
	if err := decoder.Decode(&document); err != nil || document == nil {
		return refuse([]string{"the payload is not one JSON object"})
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return refuse([]string{"the payload carries trailing JSON"})
	}
	var problems []string
	known := map[string]bool{}
	var missing []string
	for _, field := range ep.Request.Fields {
		known[field.Name] = true
		if field.Wire == "required" {
			if _, ok := document[field.Name]; !ok {
				missing = append(missing, strconv.Quote(field.Name))
			}
		}
	}
	if len(missing) > 0 {
		problems = append(problems, "missing required "+fieldWord(len(missing))+" "+
			strings.Join(missing, ", "))
	}
	var unknown []string
	for name := range document {
		if !known[name] {
			unknown = append(unknown, strconv.Quote(name))
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		problems = append(problems, "unknown "+fieldWord(len(unknown))+" "+
			strings.Join(unknown, ", ")+" (it declares: "+
			strings.Join(ep.RequestFields(), ", ")+")")
	}
	for _, field := range ep.Request.Fields {
		value, ok := document[field.Name]
		if !ok {
			continue
		}
		if problem := validateField(field, value, field.Name); problem != nil {
			if problem.Code != exit.Validation {
				return problem
			}
			problems = append(problems, problem.Message)
		}
	}
	if len(problems) > 0 {
		if len(missing) > 0 {
			message := "provide required arguments: [" + strings.Join(missing, ", ") + "]"
			if len(problems) > 1 {
				message += "; " + strings.Join(problems[1:], "; ")
			}
			return exit.Named(exit.Validation, "request_payload_invalid", "%s", message).
				WithRemedy("%s", UsageLine(target, ep))
		}
		return refuse(problems)
	}
	return nil
}

func fieldWord(n int) string {
	if n == 1 {
		return "field"
	}
	return "fields"
}

func validateField(field Field, value any, path string) *exit.Error {
	return validateFieldInto(field, value, path, nil)
}

func validateFieldInto(field Field, value any, path string, assets *[]string) *exit.Error {
	if len(field.Constraints.Unknown) > 0 {
		return exit.Named(exit.Structural, "package_interface_constraint_unknown",
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
			return exit.Named(exit.Structural, "package_interface_constraint_unknown",
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
			return exit.Named(exit.Structural, "package_interface_constraint_unknown",
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
			return exit.Named(exit.Structural, "package_interface_type_unknown",
				"request field %s has unsupported package-interface scalar %q", path, scalar)
		}
		return exit.New(exit.Validation, "request field %s does not match declared %s", path, scalar)
	}
	var schema map[string]json.RawMessage
	if json.Unmarshal(raw, &schema) != nil {
		return exit.Named(exit.Structural, "package_interface_type_unknown",
			"request field %s has an unreadable package-interface type", path)
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
			if assets != nil {
				*assets = append(*assets, path)
			}
			return nil
		}
		return exit.New(exit.Validation, "request tree field %s is not a reference", path)
	}
	if input, ok := schema["input"]; ok && string(input) == `"model"` {
		raw, _ := json.Marshal(value)
		artifact, problem := records.DecodeModelArtifact(raw)
		if problem != nil {
			return problem
		}
		if artifact == nil {
			return exit.New(exit.Validation, "request model field %s requires an artifact reference", path)
		}
		return nil
	}
	if literal, ok := schema["literal"]; ok {
		var members []json.RawMessage
		if json.Unmarshal(literal, &members) != nil || len(members) == 0 {
			return exit.Named(exit.Structural, "package_interface_type_unknown",
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
			return exit.Named(exit.Structural, "package_interface_type_unknown",
				"request field %s has an unreadable union", path)
		}
		for _, branch := range branches {
			// Tagged unions carry the discriminator name once on the union and
			// its value on each struct branch. Pass both to the existing struct
			// validator instead of treating the tag as an undeclared payload field.
			if tagField := schema["tag_field"]; tagField != nil {
				var tagged map[string]json.RawMessage
				if json.Unmarshal(branch, &tagged) != nil || tagged == nil {
					return exit.Named(exit.Structural, "package_interface_type_unknown",
						"request field %s has an unreadable tagged branch", path)
				}
				tagged["tag_field"] = tagField
				branch, _ = json.Marshal(tagged)
			}
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
			return exit.Named(exit.Structural, "package_interface_type_unknown",
				"request field %s has unreadable nested fields", path)
		}
		object, ok := value.(map[string]any)
		if !ok {
			return exit.New(exit.Validation, "request field %s is not an object", path)
		}
		if nested.TagField != "" {
			value, present := object[nested.TagField]
			// Reuse literal validation for string and integer discriminator values.
			tagType, _ := json.Marshal(map[string][]json.RawMessage{"literal": {nested.Tag}})
			if !present || validateRenderedInto(tagType, value, path+"."+nested.TagField, nil) != nil {
				return exit.New(exit.Validation, "request field %s has an absent or incorrect %s tag", path, nested.TagField)
			}
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
			if nested.TagField != "" && name == nested.TagField {
				continue
			}
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
	return exit.Named(exit.Structural, "package_interface_type_unknown",
		"request field %s has an unsupported package-interface type", path)
}

// canonicalFieldKey folds one typed argv key onto the PackageInterface's own field spelling
// (Paul, 2026-09-02): request-field NAMES are case-insensitive at the CLI composition
// seam, the PackageInterface's spelling is canonical, and the wire carries ONLY the canonical
// name — the daemon-side validator stays strict. An exact match always wins; a fold that
// could reach two fields differing only by case refuses as ambiguous rather than
// guessing. Values are untouched. An unmatched key returns unchanged so `declared`
// refuses it with the contract remedy.
func canonicalFieldKey(ep *Entrypoint, key string) (string, *exit.Error) {
	if _, ok := ep.TypeOfField(key); ok {
		return key, nil
	}
	match, count := "", 0
	for _, field := range ep.Request.Fields {
		if strings.EqualFold(field.Name, key) {
			match = field.Name
			count++
		}
	}
	if count > 1 {
		return "", exit.Named(exit.Validation, "request_field_case_ambiguous",
			"%s declares %d request fields differing only by case; %q cannot fold onto one",
			ep.Name, count, key).
			WithRemedy("spell the field exactly; it declares: %s", strings.Join(ep.RequestFields(), ", "))
	}
	if count == 1 {
		return match, nil
	}
	return key, nil
}

func declared(ep *Entrypoint, key string) *exit.Error {
	if _, ok := ep.TypeOfField(key); ok {
		return nil
	}
	return exit.New(exit.Validation, "%s declares no request field %q", ep.Name, key).
		WithRemedy("it declares: %s", strings.Join(ep.RequestFields(), ", ")).
		WithNext("cozy package list --full")
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
	case "literal":
		return typedLiteral(ep, key, rendered, raw)
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

func typedLiteral(ep *Entrypoint, key string, rendered json.RawMessage, raw string) (json.RawMessage, *exit.Error) {
	var schema struct {
		Literal []json.RawMessage `json:"literal"`
	}
	if json.Unmarshal(rendered, &schema) != nil || len(schema.Literal) == 0 {
		return nil, exit.Named(exit.Structural, "package_interface_type_unknown",
			"%s.%s has an unreadable literal type", ep.Name, key)
	}
	// A bare CLI token naturally spells a string literal. Check strings first so a
	// declaration containing both "1" and 1 resolves `field=1` to the string.
	for _, member := range schema.Literal {
		var text string
		if json.Unmarshal(member, &text) == nil && text == raw {
			encoded, _ := json.Marshal(raw)
			return encoded, nil
		}
	}
	// Numeric, boolean and null literals already have unambiguous JSON spellings.
	if json.Valid([]byte(raw)) {
		decoder := json.NewDecoder(strings.NewReader(raw))
		decoder.UseNumber()
		var value any
		if decoder.Decode(&value) == nil && validateRenderedInto(rendered, value, key, nil) == nil {
			return json.RawMessage(raw), nil
		}
	}
	allowed := make([]string, 0, len(schema.Literal))
	for _, member := range schema.Literal {
		allowed = append(allowed, string(member))
	}
	return nil, exit.New(exit.Validation,
		"%s.%s must be one of: %s", ep.Name, key, strings.Join(allowed, ", "))
}

func wrongType(ep *Entrypoint, key, raw, want string) *exit.Error {
	return exit.New(exit.Validation,
		"%s.%s is declared %s and %q is not one", ep.Name, key, want, raw).
		WithRemedy("the installed package.package-interface.json declares %s's request schema", ep.Name)
}
