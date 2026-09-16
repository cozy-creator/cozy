package packagepublish

import (
	_ "embed"
	"encoding/json"
	"sort"
	"strings"
)

// base-distributions.json is cozy-runtime's own export of the worker-image-owned
// families, generated there from `base_observation._PROTECTED_IMPORT_ROOTS` and fenced
// against it. It is vendored VERBATIM — the bytes, not a Go transcription of them — so
// this client and the worker cannot hold two opinions about what the image provides.
// The roster classifies framework artifacts for discovery, never for omission.
//
// TestVendoredBaseDistributions compares these bytes against a cozy-runtime peer when
// one is checked out, and says so by name when there is none.
//
//go:embed base-distributions.json
var baseDistributionsJSON []byte

// baseDistributions is a GENERATED CONSTANT, not a document: it carries no kind, is
// never stored, digested, or sent, and nothing versions it but the peer repository it
// is copied from.
type baseDistributions struct {
	Distributions []string `json:"distributions"`
	Prefixes      []string `json:"prefixes"`
}

// Framework families can be retained directly from their locked artifact facts
// without downloading their large wheels for callable App discovery.
var remoteBaseRoots = derivedBaseRoots()
var remoteBasePrefixes = derivedBasePrefixes()

// ImageOwnedDistribution says whether a normalized distribution name is owned
// by the worker image: a roster name or a member of a roster prefix family.
func ImageOwnedDistribution(name string) bool {
	if remoteBaseRoots[name] {
		return true
	}
	for _, prefix := range remoteBasePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func derivedBasePrefixes() []string {
	var document baseDistributions
	if err := json.Unmarshal(baseDistributionsJSON, &document); err != nil || len(document.Prefixes) == 0 {
		panic("vendored base-distributions.json is not cozy-runtime's exported roster")
	}
	for _, prefix := range document.Prefixes {
		if prefix == "" {
			panic("vendored base-distributions.json holds an empty prefix")
		}
	}
	return document.Prefixes
}

func derivedBaseRoots() map[string]bool {
	var document baseDistributions
	if err := json.Unmarshal(baseDistributionsJSON, &document); err != nil ||
		len(document.Distributions) == 0 || len(document.Prefixes) == 0 {
		panic("vendored base-distributions.json is not cozy-runtime's exported roster")
	}
	if !sort.StringsAreSorted(document.Distributions) {
		panic("vendored base-distributions.json does not list distributions in sorted order")
	}
	out := make(map[string]bool, len(document.Distributions))
	for _, name := range document.Distributions {
		if name == "" || out[name] {
			panic("vendored base-distributions.json repeats or omits a distribution name")
		}
		out[name] = true
	}
	return out
}

// BaseDistributions is the vendored roster, in the file's own order. It exists so the
// drift guard in tests/product can prove the Go side adds and drops nothing.
func BaseDistributions() []string {
	var document baseDistributions
	if err := json.Unmarshal(baseDistributionsJSON, &document); err != nil {
		panic("vendored base-distributions.json is unreadable")
	}
	return append([]string(nil), document.Distributions...)
}

// BaseDistributionsJSON is the vendored bytes, for the peer byte-comparison.
func BaseDistributionsJSON() []byte {
	return append([]byte(nil), baseDistributionsJSON...)
}
