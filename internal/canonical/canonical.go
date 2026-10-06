// Package canonical provides bounded data JSON, digests and JCS normalization.
// Stored document schemas are owned by their consumers, not protobuf descriptors.
package canonical

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const (
	intMax   = int64(1)<<53 - 1
	intMin   = -intMax
	depthMax = 12
	// DocMax is the document size cap; bytes past it are refused, never truncated.
	DocMax = 8 << 20
)

// Value is bool | int64 | string | []Value | map[string]Value. No float, no null:
// the protocol profile is integer-only printable ASCII.
type Value any

// Doc is a parsed canonical document — the JSON value, deliberately not a re-encoded
// message. Re-encoding would create a second representation free to disagree with the
// bytes the digest was taken over.
type Doc map[string]Value

// Error is a canonical-document refusal, named by code so a caller can classify it.
type Error struct {
	Code   string
	Detail string
}

func (e *Error) Error() string { return e.Code + ": " + e.Detail }

func refuse(code, format string, args ...any) *Error {
	return &Error{Code: code, Detail: fmt.Sprintf(format, args...)}
}

// Code returns err's refusal code, or "" if it is not a canonical refusal.
func Code(err error) string {
	if e, ok := err.(*Error); ok {
		return e.Code
	}
	return ""
}

// --------------------------------------------------------------------------- digests

// Spell turns the 32 raw bytes protobuf transports into `sha256:<hex>`.
func Spell(raw []byte) (string, error) {
	if len(raw) != 32 {
		return "", refuse("malformed_digest", "%d B is not a 32-byte SHA-256", len(raw))
	}
	return "sha256:" + hex.EncodeToString(raw), nil
}

// Raw turns `sha256:<hex>` back into the 32 bytes protobuf transports.
func Raw(spelled string) ([]byte, error) {
	if !strings.HasPrefix(spelled, "sha256:") || len(spelled) != 71 {
		return nil, refuse("malformed_digest", "%q is not a lowercase sha256: digest", spelled)
	}
	out, err := hex.DecodeString(spelled[7:])
	if err != nil {
		return nil, refuse("malformed_digest", "%q is not lowercase hex", spelled)
	}
	return out, nil
}

// Digest is the raw 32-byte SHA-256 of exactly these bytes — the only digest a
// receiver ever computes.
func Digest(data []byte) []byte {
	sum := sha256.Sum256(data)
	return sum[:]
}

// --------------------------------------------------------------------------- writer

func emitStr(s string, out *strings.Builder) error {
	for i := 0; i < len(s); i++ {
		if s[i] < 0x20 || s[i] > 0x7e {
			return refuse("non_ascii_field", "%q is not printable ASCII", s)
		}
	}
	out.WriteByte('"')
	out.WriteString(strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s))
	out.WriteByte('"')
	return nil
}

func writeValue(v Value, out *strings.Builder, depth int) error {
	if depth > depthMax {
		return refuse("depth_cap", "nesting deeper than %d", depthMax)
	}
	switch t := v.(type) {
	case bool:
		out.WriteString(strconv.FormatBool(t))
	case int64:
		out.WriteString(strconv.FormatInt(t, 10))
	case string:
		return emitStr(t, out)
	case []Value:
		out.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := writeValue(e, out, depth+1); err != nil {
				return err
			}
		}
		out.WriteByte(']')
	case map[string]Value:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys) // sorted at write; never by hand
		out.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				out.WriteByte(',')
			}
			if err := emitStr(k, out); err != nil {
				return err
			}
			out.WriteByte(':')
			if err := writeValue(t[k], out, depth+1); err != nil {
				return err
			}
		}
		out.WriteByte('}')
	case Doc:
		return writeValue(map[string]Value(t), out, depth)
	default:
		return refuse("wrong_type", "%T is not a canonical JSON value", v)
	}
	return nil
}

// Write is the canonical bytes of one value. These are what gets hashed.
func Write(v Value) ([]byte, error) {
	var sb strings.Builder
	if err := writeValue(v, &sb, 0); err != nil {
		return nil, err
	}
	if sb.Len() > DocMax {
		return nil, refuse("size_cap", "%d B over the %d B cap", sb.Len(), DocMax)
	}
	return []byte(sb.String()), nil
}
