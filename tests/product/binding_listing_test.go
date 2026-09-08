package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
)

// Tensorhub removes obsolete default bindings. The ordinary list reports only
// the current rows, without exposing an operator cleanup state to invokers.
func TestBindingsListOnlyCurrentDefaults(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	h.mu.Lock()
	h.bindings = []hub.PackageBindingRow{goodLadder()}
	h.mu.Unlock()
	for _, flags := range [][]string{nil, {"--json"}} {
		args := append([]string{"package", "bindings", ladderPackage}, flags...)
		code, out := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, ladderSlot) {
			t.Fatalf("current binding is not listed [exit %d]: %s", code, out)
		}
		if strings.Contains(strings.ToLower(out), "standing") || strings.Contains(out, "orphaned") {
			t.Fatalf("operator cleanup state leaked into binding list: %s", out)
		}
	}
}
