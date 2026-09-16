// The worker framework roster is vendored for artifact classification. Every
// dependency remains in the package closure regardless of this classification.
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
	"a cozy-runtime checkout for shared corpus and vendored roster comparisons")

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
