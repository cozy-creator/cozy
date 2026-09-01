package packagepublish

import (
	_ "embed"
	"encoding/json"
	"sort"
)

// base-distributions.json is cozy-runtime's own export of the worker-image-owned
// families, generated there from `base_observation._PROTECTED_IMPORT_ROOTS` and fenced
// against it. It is vendored VERBATIM — the bytes, not a Go transcription of them — so
// this client and the worker cannot hold two opinions about what the image provides.
// A publish that pruned a name the worker does not own would strip a wheel the package
// needs; a name the worker owns and this file misses is a wheel uploaded to be refused.
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

// remoteBaseRoots is the worker image's own roster, DERIVED, never authored here.
// A rental cannot replace these families or their declared native closure; other
// libraries stay package-owned even when one image happens to carry a copy, and Runtime
// compares the package's requirements against the actual selected base before installing.
//
// The `cuda-`/`nvidia-` prefix families are deliberately not enumerated: they arrive only
// under torch, so `uv export --prune torch` already removes that whole subtree.
var remoteBaseRoots = derivedBaseRoots()

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

// PrunedDistributions is what `uv export --prune` strips from a publication: the whole
// roster and nothing else. Pruning a family removes its transitive subtree too, which is
// how the `cuda-`/`nvidia-` runtime wheels leave with torch.
func PrunedDistributions() []string {
	names := BaseDistributions()
	sort.Strings(names)
	return names
}
