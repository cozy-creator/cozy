// Package secret is the ONE carrier for a credential value in this binary (cl-011).
//
// A Value renders as `sha256:<12 hex>` — the same digest spelling tensorhub uses for
// its own redacted keys, so an operator can compare `cozy hub status` against
// `cozy hub config` and see whether the two ends hold the same token WITHOUT either
// end printing it. The raw string has exactly one reader, `Reveal`, and the `secret`
// fence family keeps that true: a Reveal call outside the client's request builder is
// CI-red, and a credential-shaped flag that takes an argv value is CI-red with it.
package secret

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// Unset is what an absent credential renders as, in text and in JSON. It is not the
// empty string: "the operator has configured nothing" and "the value is empty" are
// the same fact here, and it says so.
const Unset = "unset"

// Value holds a credential. The zero Value is absent.
type Value struct{ raw string }

// New trims and wraps a raw credential. Whitespace around a token pasted out of a
// secret manager is the single most common cause of a 401 that looks like a bug.
func New(raw string) Value { return Value{raw: strings.TrimSpace(raw)} }

func (v Value) Present() bool { return v.raw != "" }

// Digest is the rendering: sha256:<12 hex>, or "unset".
func (v Value) Digest() string {
	if v.raw == "" {
		return Unset
	}
	sum := sha256.Sum256([]byte(v.raw))
	return "sha256:" + hex.EncodeToString(sum[:])[:12]
}

// String makes the digest the DEFAULT rendering: `%v` on a Value, or on any struct
// containing one, prints the digest. There is no formatting verb that leaks it.
func (v Value) String() string { return v.Digest() }

// MarshalJSON does the same for `--json`.
func (v Value) MarshalJSON() ([]byte, error) { return json.Marshal(v.Digest()) }

// Reveal is the ONE raw read. Its only legitimate caller is the code that puts the
// value into an Authorization header; the fence enforces that.
func (v Value) Reveal() string { return v.raw }

// Equal compares a presented credential against this one in constant time, over the
// FULL sha256 of each rather than over the raw bytes.
//
// It exists so the local API server (cl-006) can authenticate a bearer without ever
// calling Reveal: the verifier holds a Value, the presented string is hashed, and the
// comparison is between two digests. A timing oracle over the raw token is impossible
// because the raw token is never one side of a comparison, and the `secret` fence keeps
// Reveal out of the server entirely.
func (v Value) Equal(presented string) bool {
	if v.raw == "" {
		return false // an unset credential authenticates nothing, including ""
	}
	mine := sha256.Sum256([]byte(v.raw))
	theirs := sha256.Sum256([]byte(strings.TrimSpace(presented)))
	return subtle.ConstantTimeCompare(mine[:], theirs[:]) == 1
}

// Mint generates a fresh high-entropy credential. 32 bytes of crypto/rand, hex-spelled:
// unguessable, URL-safe, and shaped so it can ride an Authorization header, a 0600 file
// or a URL FRAGMENT without escaping.
func Mint() Value {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic("cozy: the OS refused to produce randomness for a credential: " + err.Error())
	}
	return Value{raw: hex.EncodeToString(b[:])}
}

// EnvEntry is the ENV-VAR CARRIER: the one place a credential becomes a child process's
// environment entry (`NAME=value`). It exists so a launcher handing a per-spawn bootstrap
// credential to the child it just created (#449) never touches the raw value itself —
// same rule as the request-builder carrier in `api.Authorize`.
func EnvEntry(name string, v Value) string { return name + "=" + v.raw }

// GRPCMetadataPair is the METADATA CARRIER: the one place a credential becomes a gRPC
// metadata key/value (the worker echoing its #449 bootstrap credential at Register). Same
// rule as EnvEntry and api.Authorize — the raw value is read where it becomes a carrier.
func GRPCMetadataPair(key string, v Value) (string, string) { return key, v.raw }

// FileBody is the FILE CARRIER: the one place a credential becomes the bytes of an
// OS-protected 0600 handoff file (a rental's provisioned owner token, cl-015). The
// caller writes bytes it never looked at, which is the same rule EnvEntry keeps.
func FileBody(v Value) []byte { return []byte(v.raw + "\n") }
