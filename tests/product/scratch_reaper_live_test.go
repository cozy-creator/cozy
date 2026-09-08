package producttest

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// TestScratchReaperSparesLiveRootsAndTakesDeadOnes drives the real reaper over a
// real directory holding the three cases that matter: a root owned by a live
// process, a root whose owner is gone, and a root whose contents are read-only
// the way a worker-environment tree is. The third is the one that has always
// failed silently.
func TestScratchReaperSparesLiveRootsAndTakesDeadOnes(t *testing.T) {
	dir := t.TempDir()

	mk := func(name string, owner string) string {
		root := filepath.Join(dir, name)
		must(t, os.MkdirAll(root, 0o755))
		if owner != "" {
			must(t, os.WriteFile(filepath.Join(root, scratchOwnerFile), []byte(owner), 0o644))
		}
		return root
	}

	// A live owner: this very process.
	live := mk("cozy-product-live", strconv.Itoa(os.Getpid()))

	// A dead owner. Pid 0 never names a live process, and the reaper must take
	// the root however new it is.
	dead := mk("cozy-product-dead", "0")

	// A dead owner whose tree carries a read-only content-addressed directory,
	// exactly as worker-environments/contents does on disk.
	readonly := mk("cozy-product-readonly", "0")
	contents := filepath.Join(readonly, "installs", "worker-environments", "contents", "deadbeef")
	must(t, os.MkdirAll(contents, 0o755))
	must(t, os.WriteFile(filepath.Join(contents, "package.json"), []byte("{}"), 0o444))
	must(t, os.Chmod(contents, 0o555))
	t.Cleanup(func() { _ = os.Chmod(contents, 0o755) })

	// An unmarked root, freshly created: no owner recorded, so the age floor
	// applies and it must survive.
	young := mk("cozy-product-young", "")

	// A root belonging to nobody we know. Never touched.
	foreign := mk("someone-elses-tree", "0")

	reaped, failed := reapAbandonedScratch(dir, scratchReapMinAge)
	if failed != 0 {
		t.Fatalf("reaper failed on %d roots; the read-only tree is the likely one", failed)
	}
	if reaped != 2 {
		t.Fatalf("reaped %d roots, want 2 (the dead one and the read-only one)", reaped)
	}
	for _, keep := range []string{live, young, foreign} {
		if _, err := os.Stat(keep); err != nil {
			t.Errorf("reaper took a root it must have spared: %s", keep)
		}
	}
	for _, gone := range []string{dead, readonly} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Errorf("reaper left an abandoned root behind: %s", gone)
		}
	}
}

// TestScratchReaperUsesCreationNotLastTouch pins the age floor to the root's
// creation time. A half-finished removal only ever makes a tree look newer, and
// an age floor read off the newest file would then treat an ancient abandoned
// root as fresh and spare it forever.
func TestScratchReaperUsesCreationNotLastTouch(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "cozy-product-old")
	must(t, os.MkdirAll(root, 0o755))
	old := time.Now().Add(-24 * time.Hour)
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte("{}"), 0o644))
	must(t, os.Chtimes(filepath.Join(root, "config.yaml"), old, old))
	// A file written just now, as a failed reap would leave behind.
	must(t, os.WriteFile(filepath.Join(root, "touched-just-now"), []byte("x"), 0o644))

	if got := createdAt(root); time.Since(got) < time.Hour {
		t.Fatalf("creation time read as %v ago; a recent write masked the real age", time.Since(got))
	}
	if reaped, _ := reapAbandonedScratch(dir, scratchReapMinAge); reaped != 1 {
		t.Fatalf("reaped %d, want 1: an unmarked root past the age floor must be taken", reaped)
	}
}
