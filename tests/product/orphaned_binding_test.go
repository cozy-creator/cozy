package producttest

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
)

// cl-184. `cozy package bindings paul/minimax-h3` returned FOUR rows for a package whose
// interface declares two model slots: `first_last_frame_to_video.models.model` and
// `reference_media_to_video.models.model` were the old spellings, left behind when the
// entrypoints were renamed to fl2va / ref2va. They still resolved and still carried a
// ladder nobody had touched in two revisions, and nothing in the read said so. `cozy
// package bind` verifies the slot against the latest published interface before writing,
// so a binding cannot be BORN orphaned; it becomes one under a rename.
func TestBindingsMarkASlotTheInterfaceNoLongerDeclares(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	live := goodLadder()
	stale := hub.PackageBindingRow{Slot: "old_generate.models.model", Model: ladderModel,
		Release: ladderRelease, Ladder: []hub.BindingRung{{GPU: "H100", Lane: "fp8-adaln-pruned"}},
		Revision: 2, Orphaned: true}
	h.mu.Lock()
	h.bindings = []hub.PackageBindingRow{live, stale}
	h.mu.Unlock()

	code, out := runCozy(t, root, "package", "bindings", ladderPackage)
	if code != 0 {
		t.Fatalf("bindings exited %d: %s", code, out)
	}
	if !strings.Contains(out, "orphaned") {
		t.Fatalf("the stale row is not marked: %s", out)
	}
	if !strings.Contains(out, "does not declare") {
		t.Fatalf("the reader is not told what the mark means: %s", out)
	}
	// The live row is not swept up in the mark.
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), ladderSlot) && strings.Contains(line, "orphaned") {
			t.Fatalf("the live slot is marked orphaned: %s", line)
		}
	}

	// With nothing orphaned there is no note at all — the column stays, the alarm does not.
	h.mu.Lock()
	h.bindings = []hub.PackageBindingRow{live}
	h.mu.Unlock()
	code, out = runCozy(t, root, "package", "bindings", ladderPackage)
	if code != 0 || strings.Contains(out, "orphaned") || strings.Contains(out, "does not declare") {
		t.Fatalf("a healthy package still warns [exit %d]: %s", code, out)
	}
}
