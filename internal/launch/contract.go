package launch

import (
	"encoding/json"
	"fmt"
	"strings"
)

// The ONE contract printer (cl-105/cl-106). The descriptor already carries every fact a
// caller needs to spell a valid request; this file renders those facts three ways —
// the AXI-style usage line (install's "what now?", the submit refusal's remedy), the
// `cozy run <target> --describe` contract, and nothing else. No other module re-renders
// a request schema.

// UsageLine is one concrete invocation for a callable: required terms bare, optional
// terms bracketed, each spelled in the exact grammar that supplies it.
func UsageLine(target string, ep *Entrypoint) string {
	line := "cozy run " + target
	for i := range ep.Request.Fields {
		line += " " + usageTerm(&ep.Request.Fields[i])
	}
	return line
}

func usageTerm(f *Field) string {
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

// typePhrase is a rendered type in one compact word: the descriptor's own scalar names,
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
		writeFields(&b, ep.Request.Fields, "    ")
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
		b.WriteString("  result:\n")
		writeFields(&b, ep.Result.Fields, "    ")
	}
	b.WriteString("\nusage: " + UsageLine(target, ep) + "\n")
	return b.String()
}

func writeFields(b *strings.Builder, fields []Field, indent string) {
	for i := range fields {
		field := &fields[i]
		b.WriteString(indent + field.Name + ": " + describePhrase(field))
		b.WriteString(fieldNotes(field))
		b.WriteString("\n")
		if kind, nested := typeOf(field.Type); kind == "struct" && len(indent) < 12 {
			writeFields(b, nested.Fields, indent+"  ")
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

func fieldNotes(f *Field) string {
	notes := ""
	if c := constraintPhrase(f.Constraints); c != "" {
		notes += " " + c
	}
	switch f.Wire {
	case "optional":
		notes += " (optional)"
	case "omissible":
		notes += " (optional; model default)"
	}
	return notes
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

// RawRequest is the callable's request struct verbatim from the canonical descriptor
// document — what `--describe --json` emits.
func (d *PackageDescriptor) RawRequest(name string) (json.RawMessage, bool) {
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
