// Package capability is the signed token in a machine URL's path, /v1/c/<capability>/…: an
// authorized key's grant to read a run's outputs or named objects on one machine until an
// expiry. A machine verifies it offline; it stores no bearer.
package capability

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"time"
)

const domain = "cozy-capability/1\x00"

// Grant is what a capability allows.
type Grant struct {
	Machine string   `json:"m"`           // the machine's worker id
	Run     string   `json:"r,omitempty"` // a run's outputs
	Objects []string `json:"d,omitempty"` // or these objects, sha256:<hex>
	Expires int64    `json:"e"`           // unix seconds
	Key     string   `json:"k"`           // the signer: KeyID of its public key
	Origins []string `json:"o,omitempty"` // browser origins it may be used from; empty: any
}

// KeyID names a public key compactly.
func KeyID(key ed25519.PublicKey) string {
	sum := sha256.Sum256(key)
	return base64.RawURLEncoding.EncodeToString(sum[:16])
}

// Mint signs a grant; the result is one URL path segment.
func Mint(key ed25519.PrivateKey, g Grant) (string, error) {
	g.Key = KeyID(key.Public().(ed25519.PublicKey))
	payload, err := json.Marshal(g)
	if err != nil {
		return "", err
	}
	signature := ed25519.Sign(key, append([]byte(domain), payload...))
	return base64.RawURLEncoding.EncodeToString(payload) + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

var (
	ErrInvalid = errors.New("the capability is malformed or not signed by an authorized key")
	ErrExpired = errors.New("the capability has expired")
	ErrScope   = errors.New("the capability does not grant this")
)

// Verify admits a token for machine at now, signed by one of keys.
func Verify(token, machine string, keys []ed25519.PublicKey, now time.Time) (Grant, error) {
	var g Grant
	encoded, sig, ok := strings.Cut(token, ".")
	payload, err1 := base64.RawURLEncoding.DecodeString(encoded)
	signature, err2 := base64.RawURLEncoding.DecodeString(sig)
	if !ok || err1 != nil || err2 != nil || len(signature) != ed25519.SignatureSize || json.Unmarshal(payload, &g) != nil {
		return g, ErrInvalid
	}
	signed := append([]byte(domain), payload...)
	if !slices.ContainsFunc(keys, func(k ed25519.PublicKey) bool { return KeyID(k) == g.Key && ed25519.Verify(k, signed, signature) }) {
		return g, ErrInvalid
	}
	if g.Machine != machine {
		return g, ErrScope
	}
	if now.Unix() >= g.Expires {
		return g, ErrExpired
	}
	return g, nil
}

// AllowsObject says whether the grant covers one object.
func (g Grant) AllowsObject(digest string) bool { return slices.Contains(g.Objects, digest) }

// AllowsOrigin says whether a browser origin may use the grant.
func (g Grant) AllowsOrigin(origin string) bool {
	return len(g.Origins) == 0 || origin == "" || slices.Contains(g.Origins, origin)
}
