package launch

import (
	"encoding/json"
	"fmt"
	"strings"
)

// The ONE contract printer (cl-105/cl-106). The PackageInterface already carries every fact a
// caller needs to spell a valid request; this file renders those facts three ways —
// the AXI-style usage line (install's "what now?", the submit refusal's remedy), the
// `cozy run <target> --describe` contract, and nothing else. No other module re-renders
// a request schema.

// UsageLine is one concrete invocation for a callable: required terms bare, optional
// terms bracketed, each spelled in the exact grammar that supplies it.
func UsageLine(target string, ep *Entrypoint) string {
	line := "cozy run " + target
	if slot := ConversionSlot(ep); slot != nil {
		line += " <" + slot.Param + "> [<org/model>]"
		for _, other := range ep.Models[1:] {
			line += " model." + other.Param + "=<ref>"
		}
	}
	// An invocable job also carries each model parameter as a request field for managed
	// calls; on the command line that input is the slot binding above, so it is not repeated.
	params := map[string]bool{}
	for _, slot := range ep.Models {
		params[slot.Param] = true
	}
	for i := range ep.Request.Fields {
		if ep.Invocable != nil && params[ep.Request.Fields[i].Name] {
			continue
		}
		line += " " + usageTerm(&ep.Request.Fields[i], ep.Assets)
	}
	return line
}

func usageTerm(f *Field, assets *AssetsSlot) string {
	if isAssetsField(f, assets) {
		term := "[--asset <file>]..."
		if !acceptsEmptyAssets(f) {
			term = "--asset <file> " + term
		}
		return term
	}
	kind, _ := typeOf(f.Type)
	var term string
	switch {
	case kind == "asset":
		term = "--asset " + f.Name + "=<file>"
	case kind == "tree":
		term = f.Name + "=<tree-ref>"
	case strings.HasPrefix(kind, "scalar:"), kind == "literal":
		term = f.Name + "=<" + typePhrase(f.Type) + ">"
	case kind == "unknown" && unionPhrase(f.Type) != "":
		term = f.Name + "=<" + unionPhrase(f.Type) + ">"
	default:
		term = f.Name + ":=<json>"
	}
	if f.Wire != "required" {
		term = "[" + term + "]"
	}
	return term
}

// typePhrase is a rendered type in one compact word: the PackageInterface's own scalar names,
// literal members spelled out, asset/tree/list/struct in the payload grammar's terms.
func typePhrase(raw json.RawMessage) string {
	kind, _ := typeOf(raw)
	switch {
	case strings.HasPrefix(kind, "scalar:"):
		return strings.TrimPrefix(kind, "scalar:")
	case kind == "literal":
		var schema struct {
			Literal []json.RawMessage `json:"literal"`
		}
		if json.Unmarshal(raw, &schema) != nil {
			return "literal"
		}
		members := make([]string, 0, len(schema.Literal))
		for _, member := range schema.Literal {
			var text string
			if json.Unmarshal(member, &text) == nil {
				members = append(members, text)
			} else {
				members = append(members, string(member))
			}
		}
		return strings.Join(members, "|")
	case kind == "asset":
		var schema struct {
			Asset string `json:"asset"`
		}
		_ = json.Unmarshal(raw, &schema)
		return schema.Asset + "-file"
	case kind == "tree":
		return "tree-ref"
	case kind == "list":
		var schema struct {
			List json.RawMessage `json:"list"`
		}
		if json.Unmarshal(raw, &schema) == nil {
			return "list[" + typePhrase(schema.List) + "]"
		}
		return "list"
	case kind == "struct":
		return "object"
	}
	if phrase := unionPhrase(raw); phrase != "" {
		return phrase
	}
	return "json"
}

// unionPhrase spells a union's branches joined by `|`, or "" when raw is not a union.
func unionPhrase(raw json.RawMessage) string {
	var schema struct {
		Union []json.RawMessage `json:"union"`
	}
	if json.Unmarshal(raw, &schema) != nil || len(schema.Union) == 0 {
		return ""
	}
	branches := make([]string, 0, len(schema.Union))
	for _, branch := range schema.Union {
		branches = append(branches, typePhrase(branch))
	}
	return strings.Join(branches, "|")
}

// DescribeContract is the callable's whole contract, rendered for a person: the request
// fields, the model slots, and the result shape. `bindings` maps a slot path to its
// current default binding line, "" meaning unresolved.
func DescribeContract(target string, ep *Entrypoint, bindings map[string]string) string {
	var b strings.Builder
	b.WriteString(target)
	if ep.Kind == "job" {
		b.WriteString(" (job)")
	}
	b.WriteString("\n")
	if len(ep.Request.Fields) == 0 {
		b.WriteString("  request: none\n")
	} else {
		b.WriteString("  request:\n")
		writeFields(&b, ep.Request.Fields, "    ", "request", ep.Invocable, ep.Assets)
	}
	if len(ep.Models) > 0 {
		b.WriteString("  models:\n")
		for _, slot := range ep.Models {
			b.WriteString("    " + slot.Param + ": " + slot.Class)
			if binding := bindings[slot.Path]; binding != "" {
				b.WriteString(" (default: " + binding + ")")
			}
			b.WriteString("\n")
		}
	}
	if len(ep.Result.Fields) > 0 {
		b.WriteString("  output:\n")
		writeFields(&b, ep.Result.Fields, "    ", "result", ep.Invocable, nil)
	}
	b.WriteString("\nusage: " + UsageLine(target, ep) + "\n")
	return b.String()
}

// DescribeArguments uses the same schema printer as --describe without resolving
// model defaults, so an invalid invocation can show its arguments immediately.
func DescribeArguments(ep *Entrypoint) string {
	var b strings.Builder
	b.WriteString("Arguments:\n")
	writeFields(&b, ep.Request.Fields, "  ", "request", ep.Invocable, ep.Assets)
	return b.String()
}

func writeFields(b *strings.Builder, fields []Field, indent, path string, invocable *Invocable, assets *AssetsSlot) {
	for i := range fields {
		field := &fields[i]
		fieldPath := path + "/" + field.Name
		b.WriteString(indent + field.Name + ": " + describePhrase(field))
		b.WriteString(fieldNotes(field, fieldPath, invocable, assets))
		b.WriteString("\n")
		if kind, nested := typeOf(field.Type); kind == "struct" && len(indent) < 12 {
			writeFields(b, nested.Fields, indent+"  ", fieldPath, invocable, nil)
		}
	}
}

// describePhrase is typePhrase with the asset spelled as a person reads it.
func describePhrase(f *Field) string {
	kind, _ := typeOf(f.Type)
	if kind == "asset" {
		var schema struct {
			Asset string `json:"asset"`
		}
		_ = json.Unmarshal(f.Type, &schema)
		phrase := schema.Asset + " asset"
		if len(f.AssetBound.MediaTypes) > 0 {
			phrase += " (" + strings.Join(f.AssetBound.MediaTypes, ", ") + ")"
		}
		return phrase
	}
	return typePhrase(f.Type)
}

func fieldNotes(f *Field, path string, invocable *Invocable, assets *AssetsSlot) string {
	notes := ""
	if c := constraintPhrase(f.Constraints); c != "" {
		notes += " " + c
	}
	if invocable != nil {
		if value, ok := invocable.Defaults[path]; ok {
			return notes + " (default = " + string(value) + ")"
		}
	}
	if isAssetsField(f, assets) && acceptsEmptyAssets(f) {
		// ParseAssets supplies an empty bundle when this framework parameter is
		// omitted. This is input syntax, not a default for arbitrary list fields.
		return notes + " (default = [])"
	}
	switch f.Wire {
	case "optional":
		notes += " (optional)"
	case "omissible":
		notes += " (optional; model default)"
	}
	return notes
}

func isAssetsField(field *Field, assets *AssetsSlot) bool {
	return assets != nil && field.Name == assets.Parameter
}

func acceptsEmptyAssets(field *Field) bool {
	return field.Constraints.MinLength == nil || *field.Constraints.MinLength == 0
}

func constraintPhrase(c FieldConstraints) string {
	parts := []string{}
	if c.GT != nil {
		parts = append(parts, fmt.Sprintf(">%v", *c.GT))
	}
	if c.GE != nil {
		parts = append(parts, fmt.Sprintf(">=%v", *c.GE))
	}
	if c.LE != nil {
		parts = append(parts, fmt.Sprintf("<=%v", *c.LE))
	}
	if c.MinLength != nil {
		parts = append(parts, fmt.Sprintf("min length %d", *c.MinLength))
	}
	if c.MaxLength != nil {
		parts = append(parts, fmt.Sprintf("max length %d", *c.MaxLength))
	}
	if len(parts) == 0 {
		return ""
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

// RawRequest is the callable's request struct verbatim from the canonical PackageInterface
// document — what `--describe --json` emits.
func (d *PackageInterface) RawRequest(name string) (json.RawMessage, bool) {
	var doc struct {
		Entrypoints []struct {
			Name    string          `json:"name"`
			Request json.RawMessage `json:"request"`
		} `json:"entrypoints"`
		Jobs []struct {
			Name    string          `json:"name"`
			Request json.RawMessage `json:"request"`
		} `json:"jobs"`
	}
	if json.Unmarshal(d.Raw, &doc) != nil {
		return nil, false
	}
	for _, row := range doc.Entrypoints {
		if row.Name == name {
			return row.Request, true
		}
	}
	for _, row := range doc.Jobs {
		if row.Name == name {
			return row.Request, true
		}
	}
	return nil, false
}
