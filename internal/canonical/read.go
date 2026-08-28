package canonical

import (
	"bytes"
	"encoding/base64"
	"strconv"
	"strings"

	"google.golang.org/protobuf/proto"

	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
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
	obj, err := ReadObject(data)
	if err != nil {
		return nil, err
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
	if err := semantics(string(d.FullName()), obj); err != nil {
		return nil, err
	}
	return obj, nil
}

// ReadObject validates the shared canonical JSON profile without assigning a
// document schema. It is for exact control documents whose schemas live outside
// worker-protocol (for example EndpointBindingRelease). Callers must still
// enforce their closed key set and semantic joins.
func ReadObject(data []byte) (Doc, error) {
	obj, err := ParseObject(data)
	if err != nil {
		return nil, err
	}
	again, err := Write(map[string]Value(obj))
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(again, data) {
		return nil, refuse("noncanonical_encoding", "bytes are not the canonical encoding of their own content")
	}
	return obj, nil
}

// ParseObject applies the bounded JSON grammar and duplicate-key refusal but
// does not require compact canonical rendering. Descriptor bytes have their own
// exact stored-byte identity and may be pretty-printed; their schema owner, not
// this codec, decides that presentation.
func ParseObject(data []byte) (Doc, error) {
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
	obj, ok := v.(map[string]Value)
	if !ok {
		return nil, refuse("wrong_type", "a document is a JSON object")
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
	case "cozy.worker.v1.AttemptOutcomeBody":
		return artifactReceiptList(d)
	case "cozy.worker.v1.ArtifactReceipt":
		return artifactReceipt(d)
	case "cozy.worker.v1.ArtifactFinalizeDecision":
		return artifactFinalizeDecision(d)
	case "cozy.worker.v1.ArtifactFinalizeResult":
		return artifactFinalizeResult(d)
	}
	return nil
}

func artifactReceiptList(d Doc) error {
	raw, present := d["artifact_receipts"]
	if !present {
		return nil
	}
	items, ok := raw.([]Value)
	if !ok || len(items) == 0 {
		return refuse("artifact_receipt_shape", "artifact_receipts is a non-empty list when present")
	}
	if len(items) > pb.MaxArtifactReceipts {
		return refuse("artifact_receipt_count_cap", "%d receipts exceeds the %d-item cap",
			len(items), pb.MaxArtifactReceipts)
	}
	aggregate, prior := 0, ""
	for _, item := range items {
		fields, ok := item.(map[string]Value)
		if !ok {
			return refuse("artifact_receipt_shape", "an artifact_receipts item is not an object")
		}
		receipt, size, err := readArtifactReceiptRef(Doc(fields))
		if err != nil {
			return err
		}
		aggregate += size
		slot := receipt.Str("output_slot")
		if slot <= prior {
			return refuse("artifact_receipt_order", "output slot %q is not strictly after %q", slot, prior)
		}
		prior = slot
	}
	if aggregate > pb.MaxArtifactReceiptAggregateBytes {
		return refuse("artifact_receipt_aggregate_cap", "%d receipt bytes exceeds the %d-byte cap",
			aggregate, pb.MaxArtifactReceiptAggregateBytes)
	}
	return nil
}

func artifactReceipt(d Doc) error {
	for _, field := range []string{"owner_authority_scope", "request_id", "invocation_spec_digest",
		"output_slot", "artifact_transaction_id", "tensorfs_receipt_digest",
		"tensorfs_receipt_canonical_bytes"} {
		if d.Str(field) == "" {
			return refuse("artifact_receipt_incomplete", "%s is empty or absent", field)
		}
	}
	nested, err := decodeCanonicalBytes(d.Str("tensorfs_receipt_canonical_bytes"))
	if err != nil {
		return refuse("artifact_receipt_nested_malformed", "%s", err)
	}
	if !sameDigest(nested, d.Str("tensorfs_receipt_digest")) {
		return refuse("artifact_receipt_nested_digest_mismatch",
			"tensorfs_receipt_digest does not hash the exact carried bytes")
	}
	if _, err := Raw(d.Str("invocation_spec_digest")); err != nil {
		return refuse("artifact_receipt_identity_malformed", "invocation_spec_digest: %s", err)
	}
	return nil
}

func artifactFinalizeDecision(d Doc) error {
	for _, field := range []string{"owner_authority_scope", "request_id", "invocation_spec_digest", "output_slot"} {
		if d.Str(field) == "" {
			return refuse("artifact_finalize_decision_incomplete", "%s is empty or absent", field)
		}
	}
	disposition := d.Int("disposition")
	receipt, root := d.Str("artifact_receipt_digest"), d.Str("scratch_root_id")
	if receipt != "" {
		if _, err := Raw(receipt); err != nil {
			return refuse("artifact_finalize_decision_shape", "artifact_receipt_digest: %s", err)
		}
	}
	switch pb.ArtifactFinalizeDisposition(disposition) {
	case pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ADOPT:
		if receipt != "" && root != "" {
			return nil
		}
	case pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON:
		if receipt != "" && root == "" {
			return nil
		}
	case pb.ArtifactFinalizeDisposition_ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED:
		if receipt == "" && root == "" {
			return nil
		}
	}
	return refuse("artifact_finalize_decision_shape", "disposition %d has an invalid receipt/root shape", disposition)
}

func artifactFinalizeResult(d Doc) error {
	for _, field := range []string{"owner_authority_scope", "request_id", "invocation_spec_digest", "output_slot"} {
		if d.Str(field) == "" {
			return refuse("artifact_finalize_result_incomplete", "%s is empty or absent", field)
		}
	}
	outcome := pb.ArtifactFinalizeOutcome(d.Int("outcome"))
	rawReceipt, hasReceipt := d["artifact_receipt"]
	if outcome != pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ADOPTED &&
		outcome != pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ABANDONED {
		return refuse("artifact_finalize_result_shape", "outcome %d is not final", outcome)
	}
	if outcome == pb.ArtifactFinalizeOutcome_ARTIFACT_FINALIZE_OUTCOME_ADOPTED && !hasReceipt {
		return refuse("artifact_finalize_result_shape", "ADOPTED requires its exact receipt")
	}
	if !hasReceipt {
		return nil
	}
	fields, ok := rawReceipt.(map[string]Value)
	if !ok {
		return refuse("artifact_finalize_result_shape", "artifact_receipt is not an object")
	}
	_, _, err := readArtifactReceiptRef(Doc(fields))
	return err
}

func readArtifactReceiptRef(ref Doc) (Doc, int, error) {
	if len(ref) != 2 || ref.Str("artifact_receipt_digest") == "" ||
		ref.Str("artifact_receipt_canonical_bytes") == "" {
		return nil, 0, refuse("artifact_receipt_ref_shape", "the reference has exactly digest and bytes")
	}
	data, err := decodeCanonicalBytes(ref.Str("artifact_receipt_canonical_bytes"))
	if err != nil {
		return nil, 0, refuse("artifact_receipt_malformed", "%s", err)
	}
	if len(data) == 0 || len(data) > pb.MaxArtifactReceiptBytes {
		return nil, 0, refuse("artifact_receipt_item_cap", "%d B is outside the 1..%d B range",
			len(data), pb.MaxArtifactReceiptBytes)
	}
	if !sameDigest(data, ref.Str("artifact_receipt_digest")) {
		return nil, 0, refuse("artifact_receipt_digest_mismatch",
			"artifact_receipt_digest does not hash the exact carried bytes")
	}
	receipt, err := Read(data, &pb.ArtifactReceipt{})
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
