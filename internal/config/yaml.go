package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"regexp"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// yamlAliasBudget bounds the values anchors may expand to, so a few lines of aliases
// cannot unfold into gigabytes.
const yamlAliasBudget = 1 << 16

// The YAML 1.2 core schema: `yes`, `on`, `2026-09-28` and `1_000` stay strings.
var (
	coreNull   = regexp.MustCompile(`^(|~|null|Null|NULL)$`)
	coreBool   = regexp.MustCompile(`^(true|True|TRUE|false|False|FALSE)$`)
	coreInt    = regexp.MustCompile(`^([-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$`)
	coreFloat  = regexp.MustCompile(`^([-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?|[-+]?\.(inf|Inf|INF)|\.(nan|NaN|NAN))$`)
	jsonNumber = regexp.MustCompile(`^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?$`)
	yamlLine   = regexp.MustCompile(`^line ([0-9]+): (.*)$`)
	// The libyaml parser's (not scanner's) problems.
	yamlParserProblem = regexp.MustCompile(`^(did not find expected (<document start>|<stream-start>|'-' indicator|key|node content|',' or '[]}]')|found (duplicate %(YAML|TAG) directive|incompatible YAML document|undefined tag handle))$`)
)

// YAMLError is a refused YAML document. Kind is syntax, documents, aliases, key or value;
// Line is 1-based, 0 when the parser named none.
type YAMLError struct {
	Kind   string
	Line   int
	Reason string
}

// YAMLToJSON reads exactly one YAML document as the JSON text of the same value: keys in
// document order, integers exact at any size, and only core-schema scalars typed.
func YAMLToJSON(data []byte) ([]byte, *YAMLError) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var document, extra yaml.Node
	if err := decoder.Decode(&document); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, &YAMLError{"documents", 1, "the file holds no YAML document"}
		}
		return nil, yamlSyntax(err)
	}
	if err := decoder.Decode(&extra); err == nil {
		return nil, &YAMLError{"documents", extra.Line, "a second YAML document; the file must hold one"}
	} else if !errors.Is(err, io.EOF) {
		return nil, yamlSyntax(err)
	}
	var w yamlJSON
	if problem := w.write(&document); problem != nil {
		return nil, problem
	}
	return w.out.Bytes(), nil
}

func yamlSyntax(err error) *YAMLError {
	reason, line := strings.TrimPrefix(err.Error(), "yaml: "), 0
	if m := yamlLine.FindStringSubmatch(reason); m != nil {
		line, _ = strconv.Atoi(m[1])
		reason = m[2]
	}
	// yaml.v3 counts a parser error's line from 0, a scanner error's from 1, and omits 0.
	if yamlParserProblem.MatchString(reason) {
		line++
	} else if line == 0 && !strings.HasPrefix(reason, "unknown anchor") {
		line = 1
	}
	return &YAMLError{"syntax", line, reason}
}

type yamlJSON struct {
	out      bytes.Buffer
	alias    int // line of the outermost alias being expanded; 0 outside one
	expanded int
}

func (w *yamlJSON) write(n *yaml.Node) *YAMLError {
	switch n.Kind {
	case yaml.DocumentNode:
		return w.write(n.Content[0])
	case yaml.AliasNode:
		if w.alias != 0 {
			return w.write(n.Alias)
		}
		w.alias = n.Line
		defer func() { w.alias = 0 }()
		return w.write(n.Alias)
	}
	if w.alias != 0 {
		if w.expanded++; w.expanded > yamlAliasBudget {
			return &YAMLError{"aliases", w.alias, fmt.Sprintf("aliases expand past %d values", yamlAliasBudget)}
		}
	}
	switch n.Kind {
	case yaml.MappingNode:
		seen := map[string]bool{}
		w.out.WriteByte('{')
		for i := 0; i < len(n.Content); i += 2 {
			key := n.Content[i]
			if key.Kind != yaml.ScalarNode || coreTag(key) != "!!str" {
				return &YAMLError{"key", key.Line, fmt.Sprintf("map key %q is not a string", key.Value)}
			}
			if seen[key.Value] {
				return &YAMLError{"key", key.Line, fmt.Sprintf("map key %q appears twice", key.Value)}
			}
			seen[key.Value] = true
			if i > 0 {
				w.out.WriteByte(',')
			}
			text, _ := json.Marshal(key.Value)
			w.out.Write(text)
			w.out.WriteByte(':')
			if problem := w.write(n.Content[i+1]); problem != nil {
				return problem
			}
		}
		w.out.WriteByte('}')
	case yaml.SequenceNode:
		w.out.WriteByte('[')
		for i, item := range n.Content {
			if i > 0 {
				w.out.WriteByte(',')
			}
			if problem := w.write(item); problem != nil {
				return problem
			}
		}
		w.out.WriteByte(']')
	default:
		text, ok := scalarJSON(n)
		if !ok {
			return &YAMLError{"value", n.Line, fmt.Sprintf("%s %q has no JSON value", coreTag(n), n.Value)}
		}
		w.out.WriteString(text)
	}
	return nil
}

// coreTag is a scalar's explicit tag, or its core-schema resolution.
func coreTag(n *yaml.Node) string {
	switch v := n.Value; {
	case n.Style&yaml.TaggedStyle != 0:
		return n.ShortTag()
	case n.Style != 0:
		return "!!str"
	case coreNull.MatchString(v):
		return "!!null"
	case coreBool.MatchString(v):
		return "!!bool"
	case coreInt.MatchString(v):
		return "!!int"
	case coreFloat.MatchString(v):
		return "!!float"
	}
	return "!!str"
}

func scalarJSON(n *yaml.Node) (string, bool) {
	v := n.Value
	switch coreTag(n) {
	case "!!str":
		text, _ := json.Marshal(v)
		return string(text), true
	case "!!null":
		return "null", coreNull.MatchString(v)
	case "!!bool":
		return strings.ToLower(v), coreBool.MatchString(v)
	case "!!int":
		i, ok := new(big.Int).SetString(v, 10)
		if !ok {
			i, ok = new(big.Int).SetString(v, 0) // 0o and 0x
		}
		return i.String(), ok && coreInt.MatchString(v)
	case "!!float":
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || !coreFloat.MatchString(v) {
			return "", false // .inf and .nan too: JSON has neither
		}
		if jsonNumber.MatchString(v) && strings.ContainsAny(v, ".eE") {
			return v, true
		}
		text := strconv.FormatFloat(f, 'g', -1, 64)
		if !strings.ContainsAny(text, ".e") {
			text += ".0"
		}
		return text, true
	}
	return "", false
}
