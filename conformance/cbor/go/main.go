package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/fxamacker/cbor/v2"
)

const (
	format               = "cozytensors/1"
	docMax               = 64 << 20
	maxComponents        = 256
	maxTensors           = 100_000
	maxParts             = 16
	maxRank              = 32
	maxKey               = 4096
	maxName              = 128
	maxSegments          = 65_536
	maxTotalRefs         = 65_536
	maxTotalBytes uint64 = 1 << 50
	inlineMax            = 256
)

var (
	dtypeNames = []string{"f64", "f32", "f16", "bf16", "f8_e4m3fn", "f8_e5m2", "i64", "i32", "i16", "i8", "u8", "bool"}
	widths     = []uint64{8, 4, 2, 2, 1, 1, 8, 4, 2, 1, 1, 1}
	dtypes     = func() map[string]uint64 {
		out := map[string]uint64{}
		for i, name := range dtypeNames {
			out[name] = uint64(i)
		}
		return out
	}()
	encMode       = mustEncMode()
	decMode       = mustDecMode()
	reservedStems = func() map[string]bool {
		out := map[string]bool{"con": true, "prn": true, "aux": true, "nul": true}
		for i := 1; i <= 9; i++ {
			out[fmt.Sprintf("com%d", i)] = true
			out[fmt.Sprintf("lpt%d", i)] = true
		}
		return out
	}()
)

type refusal struct{ code, detail string }

func (e refusal) Error() string                   { return e.detail }
func fail(code, format string, args ...any) error { return refusal{code, fmt.Sprintf(format, args...)} }

func mustEncMode() cbor.EncMode {
	m, err := cbor.CoreDetEncOptions().EncMode()
	if err != nil {
		panic(err)
	}
	return m
}

func mustDecMode() cbor.DecMode {
	m, err := (cbor.DecOptions{
		MaxNestedLevels:  16,
		MaxArrayElements: 131072,
		MaxMapPairs:      16,
		IndefLength:      cbor.IndefLengthForbidden,
		TagsMd:           cbor.TagsForbidden,
		IntDec:           cbor.IntDecConvertNone,
	}).DecMode()
	if err != nil {
		panic(err)
	}
	return m
}

func array(v any, arity int, where string) ([]any, error) {
	a, ok := v.([]any)
	if !ok {
		return nil, fail("WRONG_TYPE", "%s must be an array", where)
	}
	if arity >= 0 && len(a) != arity {
		return nil, fail("WRONG_ARITY", "%s has %d fields, expected %d", where, len(a), arity)
	}
	return a, nil
}

func object(v any, keys []string, where string) (map[string]any, error) {
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fail("WRONG_TYPE", "%s must be an object", where)
	}
	if len(m) != len(keys) {
		return nil, fail("WRONG_FIELDS", "%s has the wrong fields", where)
	}
	for _, key := range keys {
		if _, ok := m[key]; !ok {
			return nil, fail("WRONG_FIELDS", "%s is missing %q", where, key)
		}
	}
	return m, nil
}

func text(v any, where string, limit int, empty bool) (string, error) {
	s, ok := v.(string)
	if !ok {
		return "", fail("WRONG_TYPE", "%s must be text", where)
	}
	if (!empty && len(s) == 0) || len(s) > limit {
		return "", fail("TEXT_GRAMMAR", "%s is outside bounded printable ASCII", where)
	}
	for _, c := range []byte(s) {
		if c < 0x20 || c > 0x7e {
			return "", fail("TEXT_GRAMMAR", "%s is outside bounded printable ASCII", where)
		}
	}
	return s, nil
}

func closedName(v any, where string, limit int) (string, error) {
	s, err := text(v, where, limit, false)
	if err != nil {
		return "", err
	}
	for _, c := range []byte(s) {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.' || c == '/') {
			return "", fail("TEXT_GRAMMAR", "%s leaves the closed name grammar", where)
		}
	}
	return s, nil
}

func assetName(v any, where string) (string, error) {
	s, err := text(v, where, maxKey, false)
	if err != nil {
		return "", err
	}
	if strings.HasPrefix(s, "/") || strings.HasSuffix(s, "/") || strings.Contains(s, "//") {
		return "", fail("TEXT_GRAMMAR", "%s has an empty path component", where)
	}
	for _, component := range strings.Split(s, "/") {
		if component == "." || component == ".." {
			return "", fail("TEXT_GRAMMAR", "%s has a dot component", where)
		}
		if strings.HasPrefix(component, " ") || strings.HasSuffix(component, " ") || strings.HasSuffix(component, ".") {
			return "", fail("TEXT_GRAMMAR", "%s has a reserved edge", where)
		}
		if strings.ContainsAny(component, `\:*?"<>|`) {
			return "", fail("TEXT_GRAMMAR", "%s has a reserved character", where)
		}
		stem := strings.ToLower(strings.SplitN(component, ".", 2)[0])
		if reservedStems[stem] {
			return "", fail("TEXT_GRAMMAR", "%s has a Windows-reserved stem", where)
		}
	}
	return s, nil
}

func unsigned(v any, where string, positive bool) (uint64, error) {
	var n uint64
	switch x := v.(type) {
	case uint64:
		n = x
	case json.Number:
		var err error
		n, err = strconv.ParseUint(string(x), 10, 64)
		if err != nil {
			return 0, fail("WRONG_TYPE", "%s must be a uint64", where)
		}
	default:
		return 0, fail("WRONG_TYPE", "%s must be a uint64", where)
	}
	if positive && n == 0 {
		return 0, fail("ZERO_ELEMENT", "%s must be positive", where)
	}
	return n, nil
}

func rawDigest(v any, where string) ([]byte, error) {
	b, ok := v.([]byte)
	if !ok || len(b) != 32 {
		return nil, fail("MALFORMED_DIGEST", "%s must be exactly 32 bytes", where)
	}
	return append([]byte(nil), b...), nil
}

func diagnosticDigest(v any, where string) ([]byte, error) {
	s, err := text(v, where, 71, false)
	if err != nil || len(s) != 71 || s[:7] != "sha256:" {
		return nil, fail("MALFORMED_DIGEST", "%s must be sha256:<64 lowercase hex>", where)
	}
	b, e := hex.DecodeString(s[7:])
	if e != nil || s != "sha256:"+hex.EncodeToString(b) {
		return nil, fail("MALFORMED_DIGEST", "%s must be sha256:<64 lowercase hex>", where)
	}
	return b, nil
}

func dtype(v any, where string, diagnostic bool) (uint64, error) {
	if diagnostic {
		s, ok := v.(string)
		if !ok {
			return 0, fail("DTYPE_MISMATCH", "%s is not a closed dtype name", where)
		}
		n, ok := dtypes[s]
		if !ok {
			return 0, fail("DTYPE_MISMATCH", "%s is not a closed dtype name", where)
		}
		return n, nil
	}
	n, err := unsigned(v, where, false)
	if err != nil || n >= uint64(len(dtypeNames)) {
		return 0, fail("DTYPE_MISMATCH", "%s is outside the closed dtype table", where)
	}
	return n, nil
}

func shape(v any, where string) ([]any, error) {
	a, err := array(v, -1, where)
	if err != nil {
		return nil, err
	}
	if len(a) > maxRank {
		return nil, fail("RANK_CAP", "%s exceeds rank %d", where, maxRank)
	}
	out := make([]any, len(a))
	for i, value := range a {
		n, e := unsigned(value, fmt.Sprintf("%s[%d]", where, i), true)
		if e != nil {
			return nil, e
		}
		out[i] = n
	}
	return out, nil
}

func shapeBytes(dt uint64, dims []any, where string) (uint64, error) {
	total := widths[dt]
	for _, value := range dims {
		dim := value.(uint64)
		if dim != 0 && total > maxTotalBytes/dim {
			return 0, fail("TOTAL_BYTES_CAP", "%s exceeds the byte cap", where)
		}
		total *= dim
	}
	return total, nil
}

func addBytes(total, value uint64, where string) (uint64, error) {
	if value > maxTotalBytes || total > maxTotalBytes-value {
		return 0, fail("TOTAL_BYTES_CAP", "%s exceeds the byte cap", where)
	}
	return total + value, nil
}

func objectRef(v any, where string, diagnostic bool) ([]any, error) {
	a, err := array(v, 2, where)
	if err != nil {
		return nil, err
	}
	var dg []byte
	if diagnostic {
		dg, err = diagnosticDigest(a[0], where+".digest")
	} else {
		dg, err = rawDigest(a[0], where+".digest")
	}
	if err != nil {
		return nil, err
	}
	n, err := unsigned(a[1], where+".length", true)
	if err != nil {
		return nil, err
	}
	return []any{dg, n}, nil
}

func sortedStrings(rows []any, where string) error {
	last := ""
	for i, value := range rows {
		row := value.([]any)
		key := row[0].(string)
		if i > 0 && key <= last {
			return fail("SORT_ORDER", "%s must be sorted and unique", where)
		}
		last = key
	}
	return nil
}

func sortedDigests(rows []any, where string) error {
	var last []byte
	for i, value := range rows {
		key := value.([]any)[0].([]byte)
		if i > 0 && bytes.Compare(key, last) <= 0 {
			return fail("SORT_ORDER", "%s must be sorted and unique", where)
		}
		last = key
	}
	return nil
}

func normalizeDiagnostic(v any) ([]any, error) {
	m, err := object(v, []string{"format", "configs", "assets", "encodings", "components"}, "diagnostic")
	if err != nil {
		return nil, err
	}
	return normalizeHeader([]any{m["format"], m["configs"], m["assets"], m["encodings"], m["components"]}, true)
}

func normalizeHeader(v any, diagnostic bool) ([]any, error) {
	root, err := array(v, 5, "header")
	if err != nil {
		return nil, err
	}
	if root[0] != format {
		return nil, fail("UNKNOWN_FORMAT", "header format is not %s", format)
	}

	rawConfigs, err := array(root[1], -1, "configs")
	if err != nil {
		return nil, err
	}
	if len(rawConfigs) > maxComponents {
		return nil, fail("COUNT_CAP", "config count exceeds its cap")
	}
	configs := make([]any, 0, len(rawConfigs))
	for i, value := range rawConfigs {
		where := fmt.Sprintf("configs[%d]", i)
		row, e := array(value, 4, where)
		if e != nil {
			return nil, e
		}
		name, e := closedName(row[0], where+".name", maxName)
		if e != nil {
			return nil, e
		}
		var dg []byte
		if diagnostic {
			dg, e = diagnosticDigest(row[1], where+".digest")
		} else {
			dg, e = rawDigest(row[1], where+".digest")
		}
		if e != nil {
			return nil, e
		}
		length, e := unsigned(row[2], where+".length", true)
		if e != nil {
			return nil, e
		}
		media, e := text(row[3], where+".media_type", maxName, false)
		if e != nil {
			return nil, e
		}
		configs = append(configs, []any{name, dg, length, media})
	}
	if err := sortedStrings(configs, "configs"); err != nil {
		return nil, err
	}

	rawAssets, err := array(root[2], -1, "assets")
	if err != nil {
		return nil, err
	}
	if len(rawAssets) > maxSegments {
		return nil, fail("COUNT_CAP", "asset count exceeds its cap")
	}
	assets := make([]any, 0, len(rawAssets))
	for i, value := range rawAssets {
		where := fmt.Sprintf("assets[%d]", i)
		row, e := array(value, 5, where)
		if e != nil {
			return nil, e
		}
		name, e := assetName(row[0], where+".name")
		if e != nil {
			return nil, e
		}
		var dg []byte
		if diagnostic {
			dg, e = diagnosticDigest(row[1], where+".digest")
		} else {
			dg, e = rawDigest(row[1], where+".digest")
		}
		if e != nil {
			return nil, e
		}
		length, e := unsigned(row[2], where+".logical_length", true)
		if e != nil {
			return nil, e
		}
		media, e := text(row[3], where+".media_type", maxName, false)
		if e != nil {
			return nil, e
		}
		rawSegs, e := array(row[4], -1, where+".segments")
		if e != nil {
			return nil, e
		}
		if len(rawSegs) == 0 || len(rawSegs) > maxSegments {
			return nil, fail("COUNT_CAP", "%s.segments count is out of bounds", where)
		}
		segs := make([]any, 0, len(rawSegs))
		var sum uint64
		for j, value := range rawSegs {
			ref, e := objectRef(value, fmt.Sprintf("%s.segments[%d]", where, j), diagnostic)
			if e != nil {
				return nil, e
			}
			segs = append(segs, ref)
			sum, e = addBytes(sum, ref[1].(uint64), where)
			if e != nil {
				return nil, e
			}
		}
		if sum != length {
			return nil, fail("BYTE_LENGTH_MISMATCH", "%s segment lengths do not equal logical length", where)
		}
		assets = append(assets, []any{name, dg, length, media, segs})
	}
	if err := sortedStrings(assets, "assets"); err != nil {
		return nil, err
	}
	foldedAssets := map[string]bool{}
	for _, value := range assets {
		folded := strings.ToLower(value.([]any)[0].(string))
		if foldedAssets[folded] {
			return nil, fail("DUPLICATE_KEY", "asset names collide under ASCII case-folding")
		}
		foldedAssets[folded] = true
	}

	rawEncodings, err := array(root[3], -1, "encodings")
	if err != nil {
		return nil, err
	}
	if len(rawEncodings) == 0 || len(rawEncodings) > maxTotalRefs {
		return nil, fail("COUNT_CAP", "encoding count is out of bounds")
	}
	encodings := make([]any, 0, len(rawEncodings))
	for i, value := range rawEncodings {
		ref, e := objectRef(value, fmt.Sprintf("encodings[%d]", i), diagnostic)
		if e != nil {
			return nil, e
		}
		encodings = append(encodings, ref)
	}
	if err := sortedDigests(encodings, "encodings"); err != nil {
		return nil, err
	}

	rawComponents, err := array(root[4], -1, "components")
	if err != nil {
		return nil, err
	}
	if len(rawComponents) == 0 || len(rawComponents) > maxComponents {
		return nil, fail("EMPTY_COMPONENTS", "components count is out of bounds")
	}
	components := make([]any, 0, len(rawComponents))
	componentNames := map[string]bool{}
	cited := map[uint64]bool{}
	tensorCount := 0
	totalRefs := len(configs) + len(encodings)
	var totalBytes uint64
	for _, value := range configs {
		totalBytes, err = addBytes(totalBytes, value.([]any)[2].(uint64), "configs")
		if err != nil {
			return nil, err
		}
	}
	for _, value := range encodings {
		totalBytes, err = addBytes(totalBytes, value.([]any)[1].(uint64), "encodings")
		if err != nil {
			return nil, err
		}
	}
	for _, value := range assets {
		row := value.([]any)
		totalRefs += len(row[4].([]any))
		totalBytes, err = addBytes(totalBytes, row[2].(uint64), "assets")
		if err != nil {
			return nil, err
		}
	}
	if totalRefs > maxTotalRefs || totalBytes > maxTotalBytes {
		return nil, fail("TOTAL_REFS_CAP", "header aggregate work exceeds its cap")
	}
	for ci, value := range rawComponents {
		cw := fmt.Sprintf("components[%d]", ci)
		component, e := array(value, 2, cw)
		if e != nil {
			return nil, e
		}
		name, e := closedName(component[0], cw+".name", maxName)
		if e != nil {
			return nil, e
		}
		if componentNames[name] {
			return nil, fail("DUPLICATE_KEY", "component %q appears twice", name)
		}
		componentNames[name] = true
		rawTensors, e := array(component[1], -1, cw+".tensors")
		if e != nil {
			return nil, e
		}
		if len(rawTensors) == 0 {
			return nil, fail("EMPTY_COMPONENTS", "component %q has no tensors", name)
		}
		tensors := make([]any, 0, len(rawTensors))
		tensorNames := map[string]bool{}
		for ti, value := range rawTensors {
			tw := fmt.Sprintf("%s.tensors[%d]", cw, ti)
			tensor, e := array(value, 5, tw)
			if e != nil {
				return nil, e
			}
			key, e := closedName(tensor[0], tw+".key", maxKey)
			if e != nil {
				return nil, e
			}
			if tensorNames[key] {
				return nil, fail("DUPLICATE_KEY", "%s/%s appears twice", name, key)
			}
			tensorNames[key] = true
			logicalDT, e := dtype(tensor[1], tw+".logical_dtype", diagnostic)
			if e != nil {
				return nil, e
			}
			logicalShape, e := shape(tensor[2], tw+".logical_shape")
			if e != nil {
				return nil, e
			}
			encoding, e := unsigned(tensor[3], tw+".encoding_index", false)
			if e != nil || encoding >= uint64(len(encodings)) {
				return nil, fail("ENCODING_MISMATCH", "%s.encoding_index is out of range", tw)
			}
			cited[encoding] = true
			rawParts, e := array(tensor[4], -1, tw+".parts")
			if e != nil {
				return nil, e
			}
			if len(rawParts) == 0 || len(rawParts) > maxParts {
				return nil, fail("COUNT_CAP", "%s.parts count is out of bounds", tw)
			}
			parts := make([]any, 0, len(rawParts))
			partNames := map[string]bool{}
			for pi, value := range rawParts {
				pw := fmt.Sprintf("%s.parts[%d]", tw, pi)
				part, e := array(value, 4, pw)
				if e != nil {
					return nil, e
				}
				role, e := closedName(part[0], pw+".role", maxName)
				if e != nil {
					return nil, e
				}
				if partNames[role] {
					return nil, fail("DUPLICATE_KEY", "%s role %q appears twice", tw, role)
				}
				partNames[role] = true
				carrier, e := dtype(part[1], pw+".carrier_dtype", diagnostic)
				if e != nil {
					return nil, e
				}
				storedShape, e := shape(part[2], pw+".storage_shape")
				if e != nil {
					return nil, e
				}
				wanted, e := shapeBytes(carrier, storedShape, pw)
				if e != nil {
					return nil, e
				}
				var body any
				if diagnostic {
					bm, e := part[3].(map[string]any)
					if !e || len(bm) != 1 {
						return nil, fail("WRONG_TYPE", "%s.body must select inline_hex or segments", pw)
					}
					if value, ok := bm["inline_hex"]; ok {
						s, e := text(value, pw+".inline_hex", inlineMax*2, true)
						if e != nil {
							return nil, e
						}
						b, e := hex.DecodeString(s)
						if e != nil || s != hex.EncodeToString(b) {
							return nil, fail("WRONG_TYPE", "%s.inline_hex is not lowercase hex", pw)
						}
						body = b
					} else if value, ok := bm["segments"]; ok {
						rawSegs, e := array(value, -1, pw+".segments")
						if e != nil {
							return nil, e
						}
						segs := make([]any, 0, len(rawSegs))
						for j, value := range rawSegs {
							ref, e := objectRef(value, fmt.Sprintf("%s.segments[%d]", pw, j), true)
							if e != nil {
								return nil, e
							}
							segs = append(segs, ref)
						}
						body = segs
					} else {
						return nil, fail("WRONG_TYPE", "%s.body must select inline_hex or segments", pw)
					}
				} else {
					switch x := part[3].(type) {
					case []byte:
						body = append([]byte(nil), x...)
					case []any:
						segs := make([]any, 0, len(x))
						for j, value := range x {
							ref, e := objectRef(value, fmt.Sprintf("%s.segments[%d]", pw, j), false)
							if e != nil {
								return nil, e
							}
							segs = append(segs, ref)
						}
						body = segs
					default:
						return nil, fail("WRONG_TYPE", "%s.body must be bytes or a segment array", pw)
					}
				}
				switch x := body.(type) {
				case []byte:
					if uint64(len(x)) != wanted {
						return nil, fail("BYTE_LENGTH_MISMATCH", "%s inline length differs from shape", pw)
					}
					if len(x) > inlineMax {
						return nil, fail("INLINE_THRESHOLD", "%s must be segmented", pw)
					}
					if carrier == dtypes["bool"] {
						for _, b := range x {
							if b > 1 {
								return nil, fail("BOOL_BITS", "%s contains a non-boolean byte", pw)
							}
						}
					}
				case []any:
					if len(x) == 0 || len(x) > maxSegments {
						return nil, fail("COUNT_CAP", "%s segment count is out of bounds", pw)
					}
					var sum uint64
					for _, value := range x {
						sum, e = addBytes(sum, value.([]any)[1].(uint64), pw)
						if e != nil {
							return nil, e
						}
					}
					if sum != wanted {
						return nil, fail("BYTE_LENGTH_MISMATCH", "%s segment lengths differ from shape", pw)
					}
					if wanted <= inlineMax {
						return nil, fail("INLINE_THRESHOLD", "%s must be inline", pw)
					}
					totalRefs += len(x)
				}
				totalBytes, e = addBytes(totalBytes, wanted, pw)
				if e != nil {
					return nil, e
				}
				if totalRefs > maxTotalRefs || totalBytes > maxTotalBytes {
					return nil, fail("TOTAL_REFS_CAP", "header aggregate work exceeds its cap")
				}
				parts = append(parts, []any{role, carrier, storedShape, body})
			}
			tensors = append(tensors, []any{key, logicalDT, logicalShape, encoding, parts})
			tensorCount++
			if tensorCount > maxTensors {
				return nil, fail("COUNT_CAP", "tensor count exceeds its cap")
			}
		}
		components = append(components, []any{name, tensors})
	}
	if len(cited) != len(encodings) {
		return nil, fail("UNCITED_ENCODING", "every encoding must be cited")
	}
	return []any{format, configs, assets, encodings, components}, nil
}

func encodeDiagnostic(raw []byte) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var value any
	if err := dec.Decode(&value); err != nil {
		return nil, fail("MALFORMED_DIAGNOSTIC", "%v", err)
	}
	header, err := normalizeDiagnostic(value)
	if err != nil {
		return nil, err
	}
	return encMode.Marshal(header)
}

func decodeStrict(raw []byte) ([]any, error) {
	if len(raw) > docMax {
		return nil, fail("SIZE_CAP", "header is larger than %d bytes", docMax)
	}
	var value any
	if err := decMode.Unmarshal(raw, &value); err != nil {
		return nil, fail("MALFORMED_CBOR", "%v", err)
	}
	header, err := normalizeHeader(value, false)
	if err != nil {
		return nil, err
	}
	encoded, err := encMode.Marshal(header)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(encoded, raw) {
		return nil, fail("NONCANONICAL_ENCODING", "decode plus deterministic re-encode changed the bytes")
	}
	return header, nil
}

func ref(row []any) []any { return []any{"sha256:" + hex.EncodeToString(row[0].([]byte)), row[1]} }

func project(h []any) map[string]any {
	configs := []any{}
	for _, value := range h[1].([]any) {
		r := value.([]any)
		configs = append(configs, []any{r[0], "sha256:" + hex.EncodeToString(r[1].([]byte)), r[2], r[3]})
	}
	assets := []any{}
	for _, value := range h[2].([]any) {
		r := value.([]any)
		segs := []any{}
		for _, s := range r[4].([]any) {
			segs = append(segs, ref(s.([]any)))
		}
		assets = append(assets, []any{r[0], "sha256:" + hex.EncodeToString(r[1].([]byte)), r[2], r[3], segs})
	}
	encodings := []any{}
	for _, value := range h[3].([]any) {
		encodings = append(encodings, ref(value.([]any)))
	}
	components := []any{}
	for _, cv := range h[4].([]any) {
		c := cv.([]any)
		tensors := []any{}
		for _, tv := range c[1].([]any) {
			t := tv.([]any)
			parts := []any{}
			for _, pv := range t[4].([]any) {
				p := pv.([]any)
				var body map[string]any
				switch x := p[3].(type) {
				case []byte:
					body = map[string]any{"inline_hex": hex.EncodeToString(x)}
				case []any:
					segs := []any{}
					for _, s := range x {
						segs = append(segs, ref(s.([]any)))
					}
					body = map[string]any{"segments": segs}
				}
				parts = append(parts, []any{p[0], dtypeNames[p[1].(uint64)], p[2], body})
			}
			tensors = append(tensors, []any{t[0], dtypeNames[t[1].(uint64)], t[2], t[3], parts})
		}
		components = append(components, []any{c[0], tensors})
	}
	return map[string]any{"format": format, "configs": configs, "assets": assets, "encodings": encodings, "components": components}
}

func write(path string, raw []byte) error {
	if path == "" {
		_, err := os.Stdout.Write(raw)
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: codec encode|decode|check|id INPUT [--out PATH]")
		os.Exit(2)
	}
	cmd, input := os.Args[1], os.Args[2]
	out := ""
	if len(os.Args) == 5 && os.Args[3] == "--out" {
		out = os.Args[4]
	} else if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "bad arguments")
		os.Exit(2)
	}
	raw, err := os.ReadFile(input)
	if err == nil {
		switch cmd {
		case "encode":
			raw, err = encodeDiagnostic(raw)
			if err == nil {
				err = write(out, raw)
			}
		case "decode":
			var h []any
			h, err = decodeStrict(raw)
			if err == nil {
				raw, err = json.MarshalIndent(project(h), "", "  ")
				raw = append(raw, '\n')
				if err == nil {
					err = write(out, raw)
				}
			}
		case "check":
			_, err = decodeStrict(raw)
		case "id":
			_, err = decodeStrict(raw)
			if err == nil {
				sum := sha256.Sum256(raw)
				fmt.Printf("sha256:%x\n", sum)
			}
		default:
			err = errors.New("unknown command")
		}
	}
	if err != nil {
		var r refusal
		if errors.As(err, &r) {
			fmt.Fprintf(os.Stderr, "%s: %s\n", r.code, r.detail)
		} else {
			fmt.Fprintln(os.Stderr, err)
		}
		os.Exit(2)
	}
}
