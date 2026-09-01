// The worker-image roster drift guard.
//
// `internal/packagepublish/base-distributions.json` is cozy-runtime's file, copied. It
// decides which distributions `uv export --prune` strips from a publication, so a name
// this copy has and the worker does not strips a wheel the package needs, and a name the
// worker has and this copy does not uploads a wheel the worker will refuse. Those two
// lists WERE separately authored and had drifted by three names; this is what stops that
// happening again.
//
// Same shape as vendored_test.go: the half that needs no peer always runs, and the peer
// byte-comparison fails when it was asked for and cannot run. Nothing here skips.
package producttest

import (
	"flag"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// Defaults to the conventional sibling checkout, so an edit on either side is caught on
// a developer box without anyone remembering a flag.
var cozyRuntimeRepo = flag.String("cozy-runtime-repo", "",
	"a cozy-runtime checkout; the vendored base-distributions.json is compared against it")

var requireCozyRuntimePeer = flag.Bool("require-cozy-runtime-peer", false,
	"fail instead of reporting when no cozy-runtime checkout is found")

const baseDistributionsName = "base-distributions.json"

func TestVendoredBaseDistributionsAreDerived(t *testing.T) {
	names := packagepublish.BaseDistributions()
	if len(names) == 0 {
		t.Fatal("the vendored roster is empty")
	}
	if !sort.StringsAreSorted(names) {
		t.Fatalf("the vendored roster is not sorted: %v", names)
	}
	for i := 1; i < len(names); i++ {
		if names[i] == names[i-1] {
			t.Fatalf("the vendored roster repeats %q", names[i])
		}
	}
	// The publish path must prune exactly the roster: no Go-side addition, no omission.
	pruned := packagepublish.PrunedDistributions()
	if len(pruned) != len(names) {
		t.Fatalf("publish prunes %v, the roster is %v", pruned, names)
	}
	for i, name := range names {
		if pruned[i] != name {
			t.Fatalf("publish prunes %v, the roster is %v", pruned, names)
		}
	}
}

func TestVendoredBaseDistributionsMatchRuntime(t *testing.T) {
	repo := *cozyRuntimeRepo
	if repo == "" {
		home, err := os.UserHomeDir()
		if err == nil {
			candidate := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; the roster guarded here is generated there
			if _, err := os.Stat(filepath.Join(candidate, baseDistributionsName)); err == nil {
				repo = candidate
			}
		}
	}
	if repo == "" {
		if *requireCozyRuntimePeer {
			t.Fatal("-require-cozy-runtime-peer was set but no cozy-runtime checkout was found")
		}
		t.Log("peer comparison not wired (no -cozy-runtime-repo and no sibling checkout): " +
			"the derivation half above still ran, but nothing proved these bytes are current")
		return
	}
	peer, err := os.ReadFile(filepath.Join(repo, baseDistributionsName))
	if err != nil {
		t.Fatalf("cozy-runtime checkout %s has no %s: %v", repo, baseDistributionsName, err)
	}
	vendored := packagepublish.BaseDistributionsJSON()
	if string(peer) != string(vendored) {
		t.Fatalf("the vendored roster is not cozy-runtime's bytes\nvendored:\n%s\npeer (%s):\n%s\n"+
			"re-vendor: copy %s/%s over internal/packagepublish/%s",
			vendored, repo, peer, repo, baseDistributionsName, baseDistributionsName)
	}
}
