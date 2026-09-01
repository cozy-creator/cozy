package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestOutputRetention is the plan/perform split on the one verb that removes a user's
// bytes. Reclaiming a superseded package generation deletes a read-only tree the owner
// made unwritable on purpose, must not touch the ACTIVE generation beside it, must not
// follow a symlink out of the layout, and must refuse outright a recorded directory the
// layout does not own. Nothing here is mocked: a real home layout, a real records store,
// and the product's own install.Reclaim against real files.
func TestOutputRetention(t *testing.T) {
	l, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(l.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()

	retired := cleanupTestGeneration(l, "1111111111111111", "1.0.0")
	active := cleanupTestGeneration(l, "2222222222222222", "1.0.1")
	retired.BytesExcl = 123
	if _, problem = store.Activate(retired); problem != nil {
		t.Fatal(problem)
	}
	if superseded, activateProblem := store.Activate(active); activateProblem != nil || superseded != retired.ID {
		t.Fatalf("replacement = %q, %v", superseded, activateProblem)
	}

	retiredPackage := filepath.Join(retired.Dir, "venv", "lib", "python3.12", "site-packages", "hidiffusion")
	activePackage := filepath.Join(active.Dir, "venv", "lib", "python3.12", "site-packages", "hidiffusion")
	writeReadOnlyPackage(t, retiredPackage)
	writeReadOnlyPackage(t, activePackage)
	t.Cleanup(func() { restoreDirectoryWrites(activePackage) })

	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.MkdirAll(outside, 0o700); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(outside, "sentinel")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(retired.Dir, "outside-link")); err != nil {
		t.Logf("symlink safety fixture unavailable: %v", err)
	}

	reclaimed, problem := install.Reclaim(l, store, retired.ID)
	if problem != nil || reclaimed != retired.BytesExcl {
		t.Fatalf("reclaim = %d, %v", reclaimed, problem)
	}
	if _, err := os.Stat(retired.Dir); !os.IsNotExist(err) {
		t.Fatalf("retired generation remains: %v", err)
	}
	if got, err := os.ReadFile(sentinel); err != nil || string(got) != "keep" {
		t.Fatalf("out-of-tree symlink target changed: %q, %v", got, err)
	}
	if mode, err := packageMode(activePackage); err != nil || mode != 0o555 {
		t.Fatalf("active package environment mode = %o, %v", mode, err)
	}
	if generation, readProblem := store.Install(retired.ID); readProblem != nil || generation != nil {
		t.Fatalf("retired record = %+v, %v", generation, readProblem)
	}

	// A RECORDED DIRECTORY OUTSIDE THE GENERATION ROOT is refused whole: the row is a
	// pointer, not a licence, and reclaiming it would delete bytes this layout never owned.
	// The refusal keeps both the file and the record.
	foreignLayout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	foreignStore, problem := records.Open(foreignLayout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer foreignStore.Close()

	foreign := filepath.Join(t.TempDir(), "recorded-elsewhere")
	if err := os.MkdirAll(foreign, 0o700); err != nil {
		t.Fatal(err)
	}
	keeper := filepath.Join(foreign, "sentinel")
	if err := os.WriteFile(keeper, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	recorded := cleanupTestGeneration(foreignLayout, "3333333333333333", "1.0.0")
	recorded.Dir = foreign
	successor := cleanupTestGeneration(foreignLayout, "4444444444444444", "1.0.1")
	if _, problem = foreignStore.Activate(recorded); problem != nil {
		t.Fatal(problem)
	}
	if _, problem = foreignStore.Activate(successor); problem != nil {
		t.Fatal(problem)
	}
	if _, problem = install.Reclaim(foreignLayout, foreignStore, recorded.ID); problem == nil {
		t.Fatal("out-of-tree generation directory was accepted")
	}
	if got, err := os.ReadFile(keeper); err != nil || string(got) != "keep" {
		t.Fatalf("out-of-tree file changed: %q, %v", got, err)
	}
	if generation, readProblem := foreignStore.Install(recorded.ID); readProblem != nil || generation == nil {
		t.Fatalf("refused generation record = %+v, %v", generation, readProblem)
	}
}

func cleanupTestGeneration(l home.Layout, id, version string) records.PackageInstall {
	return records.PackageInstall{
		ID: id, Package: "cozy/example", Major: 1, Version: version,
		SourceKind: "tensorhub", SourceRef: "cozy/example@" + version,
		SourceDigest: "sha256:" + id + id + id + id, Verified: true,
		Dir: l.InstallDir(id),
	}
}

func writeReadOnlyPackage(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "utils.py"), []byte("pass\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(filepath.Join(dir, "utils.py"), 0o444); err != nil {
		t.Fatal(err)
	}
	for path := dir; filepath.Base(path) != "venv"; path = filepath.Dir(path) {
		if err := os.Chmod(path, 0o555); err != nil {
			t.Fatal(err)
		}
	}
}

func restoreDirectoryWrites(dir string) {
	for path := dir; filepath.Base(path) != "venv"; path = filepath.Dir(path) {
		_ = os.Chmod(path, 0o700)
	}
}

func packageMode(dir string) (os.FileMode, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return 0, err
	}
	return info.Mode().Perm(), nil
}
