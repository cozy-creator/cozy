package canonical

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"math"
	"strconv"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Read turns a peer's document bytes into a typed document. The digest a caller checks is
// over these exact bytes, so the reader needs no second identity law: a document from an
// older or newer peer keeps loading. Members the message does not declare, and null
// members, are dropped; whitespace and any valid JSON spelling are accepted. What still
// refuses is what cannot be read unambiguously: malformed JSON, a duplicate key, the caps,
// and a `format` tag naming another document or another major version.
func Read(data []byte, m proto.Message) (Doc, error) {
	obj, err := ReadObject(data)
	if err != nil {
		return nil, err
	}
	d := m.ProtoReflect().Descriptor()
	want := Format(m)
	if obj["format"] != want {
		return nil, refuse("unknown_format", "%v is not %q", obj["format"], want)
	}
	prune(obj, d)
	obj["format"] = want
	if err := semantics(string(d.FullName()), obj); err != nil {
		return nil, err
	}
	return obj, nil
}

// ReadObject parses one JSON object under the size, depth and duplicate-key bounds
// without assigning a document schema.
func ReadObject(data []byte) (Doc, error) {
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
		return nil, refuse("malformed_json", "%d trailing byte(s) after the document", len(p.src)-p.i)
	}
	obj, ok := v.(map[string]Value)
	if !ok {
		return nil, refuse("wrong_type", "a document is a JSON object")
	}
	return Doc(obj), nil
}

// prune keeps the members a message declares, recursively. The dropped members are
// another version's additions this reader has no use for.
func prune(obj map[string]Value, d protoreflect.MessageDescriptor) {
	for key, value := range obj {
		field := d.Fields().ByName(protoreflect.Name(key))
		if field == nil {
			delete(obj, key)
			continue
		}
		if field.Kind() != protoreflect.MessageKind || field.IsMap() {
			continue
		}
		if nested, ok := value.(map[string]Value); ok {
			prune(nested, field.Message())
		}
		if items, ok := value.([]Value); ok {
			for _, item := range items {
				if nested, ok := item.(map[string]Value); ok {
					prune(nested, field.Message())
				}
			}
		}
	}
}

// semantics carries the few document rules the KEY SET cannot state. A closed key set says
// which keys may appear; it cannot say that one of them has exactly one legal spelling of
// "absent", and #485b makes that a wire law rather than a convention.
func semantics(name string, d Doc) error {
	switch name {
	case "cozy.worker.v1.AttemptOutcomeBody":
		return weightsReceiptList(d)
	case "cozy.worker.v1.WeightsReceipt":
		return weightsReceipt(d)
	}
	return nil
}

func weightsReceiptList(d Doc) error {
	raw, present := d["weights_receipts"]
	if !present {
		return nil
	}
	items, ok := raw.([]Value)
	if !ok {
		return refuse("weights_receipt_shape", "weights_receipts is a list")
	}
	if len(items) == 0 {
		delete(d, "weights_receipts")
		return nil
	}
	if len(items) > pb.MaxWeightsReceipts {
		return refuse("weights_receipt_count_cap", "%d receipts exceeds the %d-item cap",
			len(items), pb.MaxWeightsReceipts)
	}
	aggregate, slots := 0, map[string]bool{}
	for _, item := range items {
		fields, ok := item.(map[string]Value)
		if !ok {
			return refuse("weights_receipt_shape", "an weights_receipts item is not an object")
		}
		receipt, size, err := readWeightsReceiptRef(Doc(fields))
		if err != nil {
			return err
		}
		aggregate += size
		slot := receipt.Str("output_slot")
		if slots[slot] {
			return refuse("weights_receipt_duplicate", "output slot %q has two receipts", slot)
		}
		slots[slot] = true
	}
	if aggregate > pb.MaxWeightsReceiptAggregateBytes {
		return refuse("weights_receipt_aggregate_cap", "%d receipt bytes exceeds the %d-byte cap",
			aggregate, pb.MaxWeightsReceiptAggregateBytes)
	}
	return nil
}

func weightsReceipt(d Doc) error {
	for _, field := range []string{"owner_authority_scope", "request_id", "invocation_spec_digest",
		"output_slot", "weights_transaction_id", "tensorfs_receipt_digest",
		"tensorfs_receipt_canonical_bytes"} {
		if d.Str(field) == "" {
			return refuse("weights_receipt_incomplete", "%s is empty or absent", field)
		}
	}
	nested, err := decodeCanonicalBytes(d.Str("tensorfs_receipt_canonical_bytes"))
	if err != nil {
		return refuse("weights_receipt_nested_malformed", "%s", err)
	}
	if !sameDigest(nested, d.Str("tensorfs_receipt_digest")) {
		return refuse("weights_receipt_nested_digest_mismatch",
			"tensorfs_receipt_digest does not hash the exact carried bytes")
	}
	if _, err := Raw(d.Str("invocation_spec_digest")); err != nil {
		return refuse("weights_receipt_identity_malformed", "invocation_spec_digest: %s", err)
	}
	return nil
}

func readWeightsReceiptRef(ref Doc) (Doc, int, error) {
	if ref.Str("weights_receipt_digest") == "" || ref.Str("weights_receipt_canonical_bytes") == "" {
		return nil, 0, refuse("weights_receipt_ref_shape", "the reference carries a digest and bytes")
	}
	data, err := decodeCanonicalBytes(ref.Str("weights_receipt_canonical_bytes"))
	if err != nil {
		return nil, 0, refuse("weights_receipt_malformed", "%s", err)
	}
	if len(data) == 0 || len(data) > pb.MaxWeightsReceiptBytes {
		return nil, 0, refuse("weights_receipt_item_cap", "%d B is outside the 1..%d B range",
			len(data), pb.MaxWeightsReceiptBytes)
	}
	if !sameDigest(data, ref.Str("weights_receipt_digest")) {
		return nil, 0, refuse("weights_receipt_digest_mismatch",
			"weights_receipt_digest does not hash the exact carried bytes")
	}
	receipt, err := Read(data, &pb.WeightsReceipt{})
	return receipt, len(data), err
}

func decodeCanonicalBytes(value string) ([]byte, error) {
	return base64.StdEncoding.Strict().DecodeString(value)
}

func sameDigest(data []byte, spelled string) bool {
	raw, err := Raw(spelled)
	return err == nil && bytes.Equal(Digest(data), raw)
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

// Ints reads one repeated integer field off a parsed document. An entry that is not an
// integer is skipped rather than guessed at.
func (d Doc) Ints(key string) []int64 {
	items, _ := d[key].([]Value)
	out := make([]int64, 0, len(items))
	for _, item := range items {
		if n, ok := item.(int64); ok {
			out = append(out, n)
		}
	}
	return out
}

// Strs reads one repeated string field off a parsed document.
func (d Doc) Strs(key string) []string {
	items, _ := d[key].([]Value)
	out := make([]string, 0, len(items))
	for _, item := range items {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
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

// space skips insignificant whitespace.
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
		return null{}, p.lit("null")
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
		if _, absent := v.(null); !absent {
			out[name] = v
		}
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
		if _, absent := v.(null); !absent {
			out = append(out, v)
		}
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
	start := p.i
	p.i++ // '"'
	for p.i < len(p.src) {
		switch p.src[p.i] {
		case '\\':
			p.i += 2
		case '"':
			p.i++
			var out string
			if err := json.Unmarshal(p.src[start:p.i], &out); err != nil {
				return nil, refuse("malformed_json", "invalid string at offset %d", start)
			}
			return out, nil
		default:
			p.i++
		}
	}
	return nil, refuse("malformed_json", "unterminated string")
}

// number reads integers as int64 and any other JSON number as float64. A float can only
// sit in a member this reader does not consume; accessors read integers.
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
	if n, err := strconv.ParseInt(tok, 10, 64); err == nil {
		return n, nil
	}
	f, err := strconv.ParseFloat(tok, 64)
	if err != nil {
		return nil, refuse("malformed_json", "%s is not a number", tok)
	}
	if f == math.Trunc(f) && f >= float64(intMin) && f <= float64(intMax) {
		return int64(f), nil
	}
	return f, nil
}

// null marks a JSON null while parsing; the containing object or array drops it.
type null struct{}
