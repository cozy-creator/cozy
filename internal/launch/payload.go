package launch

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// The `run` payload grammar (cozy-runtime-cli.md), the same one cr-016 implements for a
// bare-venv run: a positional PRIMARY filling the first declared field, `key=value`
// scalars, `key=@file` local file contents encoded as one JSON string,
// `key:=<json>` raw JSON for nested values, and
// `--input <file>` supplying the whole payload with `key=value` merged on top.
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
	// Overlays is the declared slot path -> ordered generic adapter overlays. The
	// Runtime/package compatibility declaration resolves components and bounds later;
	// the CLI only canonicalizes syntax and decimal weights here.
	Overlays map[string][]ModelOverlay
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
	keys := RunKeys{Models: map[string]string{}, Overlays: map[string][]ModelOverlay{}}

	if infile != "" {
		data, err := os.ReadFile(infile)
		if err != nil {
			return nil, RunKeys{}, exit.New(exit.NotFound, "--input %s: %s", infile, err).
				WithRemedy("--input takes one JSON file holding the whole payload")
		}
		var loaded map[string]json.RawMessage
		if err := json.Unmarshal(data, &loaded); err != nil {
			return nil, RunKeys{}, exit.New(exit.Validation, "--input %s does not hold one JSON object: %s", infile, err)
		}
		// Field names fold onto the declared spelling, as terms do; an exact spelling wins.
		folded := make(map[string]json.RawMessage, len(loaded))
		for k, v := range loaded {
			key, problem := canonicalFieldKey(ep, k)
			if problem != nil {
				return nil, RunKeys{}, problem
			}
			if _, exact := loaded[key]; !exact || key == k {
				folded[key] = v
			}
		}
		data, _ = json.Marshal(folded)
		// Only declared media fields are filenames; ordinary prompt strings are untouched.
		data, problem := mapAssetFilenames(ep, data, func(path, source string) (any, *exit.Error) {
			if strings.Contains(source, "://") {
				// Defer refusal until after inline overrides replace file values.
				return source, nil
			}
			if filepath.IsAbs(source) || source == "~" || strings.HasPrefix(source, "~/") {
				return source, nil
			}
			absolute, err := filepath.Abs(filepath.Join(filepath.Dir(infile), source))
			if err != nil {
				return nil, exit.New(exit.Validation, "cannot resolve %s: %s", path, err)
			}
			return absolute, nil
		})
		if problem != nil {
			return nil, RunKeys{}, problem
		}
		loaded = nil
		if err := json.Unmarshal(data, &loaded); err != nil {
			return nil, RunKeys{}, exit.Internalf("cannot read normalized JSON inputs: %s", err)
		}
		if raw, ok := loaded["models"]; ok {
			if problem := parseModelEnvelope(ep, raw, &keys); problem != nil {
				return nil, RunKeys{}, problem
			}
			delete(loaded, "models")
		}
		for k, v := range loaded {
			document[k] = v
		}
	}

	var positional []string
	for _, term := range terms {
		if strings.HasPrefix(term, "model.") && strings.Contains(term, "=") {
			if strings.Contains(term, ".lora:=") || strings.HasSuffix(strings.SplitN(term, "=", 2)[0], ".lora") {
				if problem := parseModelOverlayTerm(ep, term, &keys); problem != nil {
					return nil, RunKeys{}, problem
				}
				continue
			}
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
		// Model-binding overlays are reserved even without the historical `model.`
		// prefix. Dotted payload fields are not legal, so this spelling cannot shadow a
		// request field and keeps `base_model.lora:=…` useful in shell scripts.
		if strings.Contains(term, ".lora=") {
			if problem := parseModelOverlayTerm(ep, term, &keys); problem != nil {
				return nil, RunKeys{}, problem
			}
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

// ModelOverlay is the canonical ordered adapter selection carried beside one model
// slot. The ref is resolved to an exact manifest only after the slot's compatibility
// declaration has accepted it; the parser deliberately does not treat a digest as a
// compatibility claim. Weight is a canonical decimal string so request identity does
// not depend on JSON number formatting.
type ModelOverlay struct {
	Ref       string `json:"ref"`
	Weight    string `json:"weight"`
	Component string `json:"component,omitempty"`
}

func (o *ModelOverlay) UnmarshalJSON(data []byte) error {
	var raw struct {
		Ref       string          `json:"ref"`
		Weight    json.RawMessage `json:"weight"`
		Component string          `json:"component,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	weight := string(raw.Weight)
	if len(raw.Weight) > 0 && raw.Weight[0] == '"' {
		if err := json.Unmarshal(raw.Weight, &weight); err != nil {
			return err
		}
	}
	o.Ref, o.Weight, o.Component = raw.Ref, weight, raw.Component
	return nil
}

func parseModelEnvelope(ep *Entrypoint, raw json.RawMessage, keys *RunKeys) *exit.Error {
	var envelope map[string]struct {
		Ref  string         `json:"ref"`
		LoRA []ModelOverlay `json:"lora"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil || envelope == nil {
		return exit.Named(exit.Usage, "model_binding_invalid", "models must be an object keyed by model slot").
			WithRemedy(`use {"models":{"slot":{"ref":"org/model@release","lora":[{"ref":"org/adapter@release","weight":0.8}]}}}`)
	}
	for asked, value := range envelope {
		slot, problem := modelOverrideSlot(ep, asked)
		if problem != nil {
			return problem
		}
		if value.Ref != "" {
			if _, exists := keys.Models[slot.Path]; exists {
				return exit.Usagef("model slot %s was bound more than once", slot.Path)
			}
			keys.Models[slot.Path] = value.Ref
		}
		for _, overlay := range value.LoRA {
			if problem := appendModelOverlay(ep, slot.Path, overlay, keys); problem != nil {
				return problem
			}
		}
	}
	return nil
}

func parseModelOverlayTerm(ep *Entrypoint, term string, keys *RunKeys) *exit.Error {
	key, raw, ok := strings.Cut(term, ":=")
	if !ok {
		key, raw, ok = strings.Cut(term, "=")
	}
	if !ok || !strings.HasSuffix(key, ".lora") {
		return exit.Named(exit.Usage, "model_overlay_invalid", "%s is not a model overlay", term).
			WithRemedy("use model.<slot>.lora:=<json-list> or model.<slot>.lora=<ref>,weight=<number>")
	}
	asked := strings.TrimSuffix(strings.TrimPrefix(key, "model."), ".lora")
	slot, problem := modelOverrideSlot(ep, asked)
	if problem != nil {
		return problem
	}
	if strings.HasPrefix(raw, "[") {
		var overlays []ModelOverlay
		if err := json.Unmarshal([]byte(raw), &overlays); err != nil {
			return exit.Named(exit.Usage, "model_overlay_invalid", "%s is not a JSON overlay list: %s", key, err)
		}
		for _, overlay := range overlays {
			if problem := appendModelOverlay(ep, slot.Path, overlay, keys); problem != nil {
				return problem
			}
		}
		return nil
	}
	ref, weight := raw, "1"
	parts := strings.Split(raw, ",")
	if len(parts) > 0 {
		ref = parts[0]
	}
	for _, part := range parts[1:] {
		name, value, found := strings.Cut(part, "=")
		if !found || name != "weight" || value == "" {
			return exit.Usagef("%s shorthand expects <ref>,weight=<number>", key)
		}
		weight = value
	}
	return appendModelOverlay(ep, slot.Path, ModelOverlay{Ref: ref, Weight: weight}, keys)
}

func appendModelOverlay(ep *Entrypoint, slotPath string, overlay ModelOverlay, keys *RunKeys) *exit.Error {
	if strings.TrimSpace(overlay.Ref) == "" {
		return exit.Usagef("model overlay for %s has an empty ref", slotPath)
	}
	weight := overlay.Weight
	if weight == "" {
		return exit.Usagef("model overlay for %s requires numeric weight", slotPath)
	}
	canonical, problem := CanonicalLoRAScale(weight)
	if problem != nil {
		return problem
	}
	overlay.Ref, overlay.Weight = strings.TrimSpace(overlay.Ref), canonical
	keys.Overlays[slotPath] = append(keys.Overlays[slotPath], overlay)
	return nil
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
	if problem := ValidateAttentionOverride(raw); problem != nil {
		return "", "", problem
	}
	return asked, raw, nil
}

// ValidateAttentionOverride checks transport syntax only. Runtime owns backend names,
// component existence, hardware eligibility and compiled/context-parallel restrictions.
// Empty is the ordinary request with no override; nonempty values are carried verbatim.
func ValidateAttentionOverride(pin string) *exit.Error {
	if pin == "" {
		return nil
	}
	scope, backend, scoped := strings.Cut(pin, "=")
	if strings.IndexFunc(pin, unicode.IsSpace) >= 0 ||
		(scoped && (scope == "" || backend == "" || strings.Contains(backend, "="))) {
		return exit.Named(exit.Usage, "attention_override_invalid",
			"invalid attention override %q", pin).
			WithRemedy("use --attention-kernel=backend or --attention-kernel=[model/]component=backend; no whitespace")
	}
	return nil
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
				missing = append(missing, fieldSignature(&field))
			}
		}
	}
	if len(missing) > 0 {
		problems = append(problems, "missing required "+fieldWord(len(missing))+" "+
			strings.Join(missing, "; "))
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
			message := "provide required arguments: [" + strings.Join(missing, "; ") + "]"
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

// reading is what one validation collects beside its verdict. assets gathers the asset
// positions it passed; ignored, when set, turns an undeclared nested field into an ignored
// path instead of a refusal. That is how a peer's RESULT is read: a newer package or
// Runtime may add fields this host does not declare. A caller's request never sets it,
// because there an undeclared key is a typo.
type reading struct {
	assets  *[]string
	ignored *[]string
}

func collecting(assets *[]string) *reading {
	if assets == nil {
		return nil
	}
	return &reading{assets: assets}
}

func (r *reading) asset(path string) {
	if r != nil && r.assets != nil {
		*r.assets = append(*r.assets, path)
	}
}

// branch is a scratch reading for one union branch, merged only if the branch matches.
func (r *reading) branch() *reading {
	out := &reading{assets: new([]string)}
	if r != nil && r.ignored != nil {
		out.ignored = new([]string)
	}
	return out
}

func (r *reading) merge(branch *reading) {
	if r == nil {
		return
	}
	if r.assets != nil {
		*r.assets = append(*r.assets, *branch.assets...)
	}
	if r.ignored != nil {
		*r.ignored = append(*r.ignored, *branch.ignored...)
	}
}

// validateFieldInto is the early check before anything is rented. Runtime validates the
// payload against the package's own types at execution, so a type or constraint this host
// cannot read is left to it rather than refused here.
func validateFieldInto(field Field, value any, path string, assets *[]string) *exit.Error {
	return validateFieldReading(field, value, path, collecting(assets))
}

func validateFieldReading(field Field, value any, path string, r *reading) *exit.Error {
	if problem := validateRendered(field.Type, value, path, r); problem != nil {
		return problem
	}
	if field.Constraints.MinLength != nil || field.Constraints.MaxLength != nil {
		length := int64(-1)
		switch typed := value.(type) {
		case string:
			length = int64(utf8.RuneCountInString(typed))
		case []any:
			length = int64(len(typed))
		case map[string]any:
			length = int64(len(typed))
		}
		if length < 0 {
			return nil
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
	if field.Constraints.GT != nil || field.Constraints.GE != nil || field.Constraints.LE != nil || field.Constraints.MultipleOf != nil {
		number, ok := value.(json.Number)
		if !ok {
			return nil
		}
		parsed, err := strconv.ParseFloat(number.String(), 64)
		if err != nil {
			return exit.New(exit.Validation, "request field %s is not a finite number", path)
		}
		if multiple := field.Constraints.MultipleOf; multiple != nil {
			if matches, err := matchesMultipleOf(number, *multiple); err == nil && !matches {
				return exit.New(exit.Validation, "request field %s is %s; it must be a multiple of %s", path, number.String(), multiple.String())
			}
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
	return validateRendered(raw, value, path, collecting(assets))
}

func validateRendered(raw json.RawMessage, value any, path string, r *reading) *exit.Error {
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
			return nil
		}
		return exit.New(exit.Validation, "request field %s does not match declared %s", path, scalar)
	}
	var schema map[string]json.RawMessage
	if json.Unmarshal(raw, &schema) != nil {
		return nil
	}
	if _, ok := schema["asset"]; ok {
		if ref, ok := value.(string); ok && ref != "" {
			r.asset(path)
			return nil
		}
		return exit.New(exit.Validation, "request asset field %s is not a non-empty reference", path)
	}
	if input, ok := schema["input"]; ok && string(input) == `"tree"` {
		if _, ok := value.(string); ok {
			r.asset(path)
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
			return nil
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
		if json.Unmarshal(union, &branches) != nil || len(branches) == 0 {
			return nil
		}
		for _, branch := range branches {
			// Tagged unions carry the discriminator name once on the union and
			// its value on each struct branch. Pass both to the existing struct
			// validator instead of treating the tag as an undeclared payload field.
			if tagField := schema["tag_field"]; tagField != nil {
				var tagged map[string]json.RawMessage
				if json.Unmarshal(branch, &tagged) != nil || tagged == nil {
					return nil
				}
				tagged["tag_field"] = tagField
				branch, _ = json.Marshal(tagged)
			}
			found := r.branch()
			if validateRendered(branch, value, path, found) == nil {
				r.merge(found)
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
			if problem := validateRendered(item, element,
				path+"."+strconv.Itoa(index), r); problem != nil {
				return problem
			}
		}
		return nil
	}
	if fields, ok := schema["fields"]; ok {
		var nested Struct
		if json.Unmarshal(raw, &nested) != nil {
			return nil
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
			if !ok && r != nil && r.ignored != nil {
				*r.ignored = append(*r.ignored, path+"."+name)
				continue
			}
			if !ok {
				return exit.New(exit.Validation, "request field %s declares no nested field %q", path, name)
			}
			if problem := validateFieldReading(field, element, path+"."+name, r); problem != nil {
				return problem
			}
		}
		_ = fields
		return nil
	}
	return nil
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
	if value, handled, problem := typedUnion(ep, key, rendered, raw); handled {
		return value, problem
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

// typedUnion spells a bare token for a union such as `int | None`. A branch with an
// unambiguous JSON spelling (int, float, bool, null, a non-string literal) wins over a
// string branch, so `seed=7000` is 7000 and `seed=null` is null; a token no typed branch
// reads falls to a string branch when one is declared. `key:=<json>` stays the raw form.
func typedUnion(ep *Entrypoint, key string, rendered json.RawMessage, raw string) (json.RawMessage, bool, *exit.Error) {
	var schema struct {
		Union []json.RawMessage `json:"union"`
	}
	if json.Unmarshal(rendered, &schema) != nil || len(schema.Union) == 0 {
		return nil, false, nil
	}
	// Only scalar and literal branches have a bare spelling; an asset, tree or model
	// branch is granted by its own flag and never by a string that happens to fit.
	scalars := make([]json.RawMessage, 0, len(schema.Union))
	for _, branch := range schema.Union {
		if kind, _ := typeOf(branch); strings.HasPrefix(kind, "scalar:") || kind == "literal" {
			scalars = append(scalars, branch)
		}
	}
	matches := func(value any) bool {
		for _, branch := range scalars {
			if validateRenderedInto(branch, value, key, nil) == nil {
				return true
			}
		}
		return false
	}
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.UseNumber()
	var value any
	if json.Valid([]byte(raw)) && decoder.Decode(&value) == nil {
		if _, isString := value.(string); !isString && matches(value) {
			return json.RawMessage(raw), true, nil
		}
	}
	if matches(raw) {
		encoded, _ := json.Marshal(raw)
		return encoded, true, nil
	}
	return nil, true, exit.New(exit.Validation,
		"%s.%s is declared %s and %q is not one", ep.Name, key, spellUnion(schema.Union), raw).
		WithRemedy("structured values are `%s:=<json>`; asset fields use `--asset <field-path>=<file>`", key)
}

func spellUnion(branches []json.RawMessage) string {
	names := make([]string, 0, len(branches))
	for _, branch := range branches {
		var scalar string
		if json.Unmarshal(branch, &scalar) == nil {
			names = append(names, scalar)
			continue
		}
		names = append(names, string(branch))
	}
	return strings.Join(names, "|")
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
