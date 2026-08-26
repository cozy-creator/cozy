package canonical

import (
	"bytes"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"
)

// Read turns exact canonical bytes into a typed document. It refuses everything the
// writer could not have produced — that is the reading side of law 4, and the reason a
// planted `service_class` key on an ExecutionSpec is a REFUSAL and not an ignored field.
//
// The refusals, in the order they fire: size cap, JSON grammar (no null, no float,
// printable ASCII, only \" and \\), duplicate key, the RE-EMIT LAW (the bytes must be
// the canonical encoding of their own content), the document's `format` tag, and finally
// the message's closed key set.
func Read(data []byte, m proto.Message) (Doc, error) {
	if len(data) > DocMax {
		return nil, refuse("size_cap", "%d B over the %d B cap", len(data), DocMax)
	}
	p := &parser{src: data}
	v, err := p.value(0)
	if err != nil {
		return nil, err
	}
	p.space()
	if p.i != len(p.src) {
		return nil, refuse("noncanonical_encoding", "%d trailing byte(s) after the document", len(p.src)-p.i)
	}
	again, err := Write(v)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(again, data) {
		return nil, refuse("noncanonical_encoding", "bytes are not the canonical encoding of their own content")
	}
	obj, ok := v.(map[string]Value)
	if !ok {
		return nil, refuse("wrong_type", "a document is a JSON object")
	}
	d := m.ProtoReflect().Descriptor()
	want := Format(m)
	if obj["format"] != want {
		return nil, refuse("unknown_format", "%v is not %q", obj["format"], want)
	}
	known := map[string]bool{"format": true}
	for i := 0; i < d.Fields().Len(); i++ {
		known[string(d.Fields().Get(i).Name())] = true
	}
	for k := range obj {
		if !known[k] {
			return nil, refuse("unknown_field", "%s: unknown field %q", want, k)
		}
	}
	if err := semantics(string(d.FullName()), Doc(obj)); err != nil {
		return nil, err
	}
	return Doc(obj), nil
}

// semantics carries the few document rules the KEY SET cannot state. A closed key set says
// which keys may appear; it cannot say that one of them has exactly one legal spelling of
// "absent", and #485b makes that a wire law rather than a convention.
func semantics(name string, d Doc) error {
	switch name {
	case "cozy.worker.v1.EndpointEnvironmentSpec":
		// PLATFORMTARGET HAS ONE CANONICAL ENCODING (#485b). The absent libc is the single
		// value "none" — never "", never omitted. Two spellings of "no libc" would digest
		// to two environments and split every cache and receipt built on either, so the
		// unspelled form refuses at parse instead of quietly becoming a second identity.
		if pt, ok := d["platform_target"]; ok {
			target, _ := pt.(map[string]Value)
			if libc, _ := target["libc"].(string); libc == "" {
				return refuse("libc_unspelled",
					"platform_target.libc has one canonical encoding and the absent case is "+
						`"none"; an omitted or empty libc is a second spelling of the same environment`)
			}
		}
	}
	return nil
}

// Str reads one string field off a parsed document; a missing or wrong-typed field
// answers "" rather than a zero value pretending to be a fact.
func (d Doc) Str(key string) string {
	s, _ := d[key].(string)
	return s
}

// Int reads one integer field off a parsed document.
func (d Doc) Int(key string) int64 {
	n, _ := d[key].(int64)
	return n
}

// Sub reads one nested document off a parsed document.
func (d Doc) Sub(key string) Doc {
	m, _ := d[key].(map[string]Value)
	return Doc(m)
}

// List reads one repeated field off a parsed document as documents. An entry that is not
// an object is skipped rather than guessed at: the caller wanted rows.
func (d Doc) List(key string) []Doc {
	items, _ := d[key].([]Value)
	out := make([]Doc, 0, len(items))
	for _, item := range items {
		if m, ok := item.(map[string]Value); ok {
			out = append(out, Doc(m))
		}
	}
	return out
}

// --------------------------------------------------------------------------- parser

type parser struct {
	src []byte
	i   int
}

// space skips insignificant whitespace. The parser ACCEPTS it so the re-emit law is
// what refuses it — one rule refusing non-canonical bytes, never two.
func (p *parser) space() {
	for p.i < len(p.src) {
		switch p.src[p.i] {
		case ' ', '\t', '\n', '\r':
			p.i++
		default:
			return
		}
	}
}

func (p *parser) value(depth int) (Value, error) {
	if depth > depthMax {
		return nil, refuse("depth_cap", "nesting deeper than %d", depthMax)
	}
	p.space()
	if p.i >= len(p.src) {
		return nil, refuse("malformed_json", "document ends where a value was expected")
	}
	switch c := p.src[p.i]; {
	case c == '{':
		return p.object(depth)
	case c == '[':
		return p.array(depth)
	case c == '"':
		return p.str()
	case c == 't':
		return true, p.lit("true")
	case c == 'f':
		return false, p.lit("false")
	case c == 'n':
		return nil, refuse("wrong_type", "null has no canonical spelling in this profile")
	default:
		return p.number()
	}
}

func (p *parser) lit(word string) error {
	if !bytes.HasPrefix(p.src[p.i:], []byte(word)) {
		return refuse("malformed_json", "expected %q at offset %d", word, p.i)
	}
	p.i += len(word)
	return nil
}

func (p *parser) object(depth int) (Value, error) {
	p.i++ // '{'
	out := map[string]Value{}
	p.space()
	if p.i < len(p.src) && p.src[p.i] == '}' {
		p.i++
		return out, nil
	}
	for {
		p.space()
		if p.i >= len(p.src) || p.src[p.i] != '"' {
			return nil, refuse("key_grammar", "an object key must be a string, at offset %d", p.i)
		}
		key, err := p.str()
		if err != nil {
			return nil, err
		}
		name := key.(string)
		if _, seen := out[name]; seen {
			return nil, refuse("duplicate_key", "key %q appears twice", name)
		}
		p.space()
		if p.i >= len(p.src) || p.src[p.i] != ':' {
			return nil, refuse("malformed_json", "expected ':' at offset %d", p.i)
		}
		p.i++
		v, err := p.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out[name] = v
		p.space()
		if p.i >= len(p.src) {
			return nil, refuse("malformed_json", "unterminated object")
		}
		switch p.src[p.i] {
		case ',':
			p.i++
		case '}':
			p.i++
			return out, nil
		default:
			return nil, refuse("malformed_json", "expected ',' or '}' at offset %d", p.i)
		}
	}
}

func (p *parser) array(depth int) (Value, error) {
	p.i++ // '['
	out := []Value{}
	p.space()
	if p.i < len(p.src) && p.src[p.i] == ']' {
		p.i++
		return out, nil
	}
	for {
		v, err := p.value(depth + 1)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
		p.space()
		if p.i >= len(p.src) {
			return nil, refuse("malformed_json", "unterminated array")
		}
		switch p.src[p.i] {
		case ',':
			p.i++
		case ']':
			p.i++
			return out, nil
		default:
			return nil, refuse("malformed_json", "expected ',' or ']' at offset %d", p.i)
		}
	}
}

func (p *parser) str() (Value, error) {
	p.i++ // '"'
	var sb strings.Builder
	for p.i < len(p.src) {
		c := p.src[p.i]
		switch {
		case c == '"':
			p.i++
			return sb.String(), nil
		case c == '\\':
			if p.i+1 >= len(p.src) {
				return nil, refuse("malformed_json", "escape at end of document")
			}
			switch p.src[p.i+1] {
			case '"':
				sb.WriteByte('"')
			case '\\':
				sb.WriteByte('\\')
			default:
				return nil, refuse("noncanonical_encoding",
					"escape \\%c is outside the profile's two escapes", p.src[p.i+1])
			}
			p.i += 2
		case c < 0x20 || c > 0x7e:
			return nil, refuse("non_ascii_field", "byte 0x%02x in a string; fields are printable ASCII", c)
		default:
			sb.WriteByte(c)
			p.i++
		}
	}
	return nil, refuse("malformed_json", "unterminated string")
}

func (p *parser) number() (Value, error) {
	start := p.i
	for p.i < len(p.src) {
		c := p.src[p.i]
		if (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' {
			p.i++
			continue
		}
		break
	}
	tok := string(p.src[start:p.i])
	if tok == "" {
		return nil, refuse("malformed_json", "expected a value at offset %d", start)
	}
	if strings.ContainsAny(tok, ".eE") {
		return nil, refuse("non_integer_number", "%s is a float; documents are integer-only", tok)
	}
	n, err := strconv.ParseInt(tok, 10, 64)
	if err != nil || n < intMin || n > intMax {
		return nil, refuse("number_range", "%s is outside the interoperable integer range", tok)
	}
	return n, nil
}
