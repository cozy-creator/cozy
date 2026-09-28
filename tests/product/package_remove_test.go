package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestPackageRemoveReportsSweptInstall: `cozy package remove` on a package whose pin is
// already gone reclaims the unpinned install through its sweep, and the verb must SAY so.
// Live (2026-09-02, paul/marco-polo): it printed the list's empty state, "No packages
// found." with changed: false, while deleting the install directory.
func TestPackageRemoveReportsSweptInstall(t *testing.T) {
	root := filepath.Join(scratchBase, "package-remove")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	l, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(l.DB)
	fatal(t, problem)
	unpinned := cleanupTestInstall(l, "eeeeeeeeeeeeeeee", "1.0.0")
	unpinned.BytesExcl = 4096
	_, problem = store.Activate(unpinned)
	fatal(t, problem)
	fatal(t, store.Unpin(unpinned.Package, unpinned.Major))
	store.Close()
	must(t, os.MkdirAll(filepath.Join(unpinned.Dir, "venv"), 0o700))

	code, out := runCozy(t, root, "package", "remove", unpinned.Package)
	if code != 0 || strings.Contains(out, "No packages found") ||
		!strings.Contains(out, unpinned.Package) || !strings.Contains(out, "changed: true") {
		t.Fatalf("swept install was not reported [exit %d]\n%s", code, out)
	}
	if _, err := os.Stat(unpinned.Dir); !os.IsNotExist(err) {
		t.Fatalf("unpinned install %s survived remove: %v", unpinned.Dir, err)
	}
}
