package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

// TestInstallSweep is the startup sweep against the state that made cl-076 unreachable:
// install directories on disk whose records.db was lost or rebuilt. Nothing is mocked —
// a real home layout, a real records store, real read-only venv trees, and the product's
// own install.Sweep. What the sweep must prove is both halves of the ruling: a directory
// no record references goes, and a directory something still references does not.
func TestInstallSweep(t *testing.T) {
	l, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(l.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()

	pinned := cleanupTestGeneration(l, "aaaaaaaaaaaaaaaa", "1.0.0")
	retired := cleanupTestGeneration(l, "bbbbbbbbbbbbbbbb", "1.0.1")
	retired.Package, retired.BytesExcl = "cozy/retired", 4096
	serving := cleanupTestGeneration(l, "cccccccccccccccc", "1.0.0")
	serving.Package = "cozy/serving"
	for _, generation := range []records.PackageInstall{retired, serving, pinned} {
		if _, problem := store.Activate(generation); problem != nil {
			t.Fatal(problem)
		}
	}
	// The retired generation loses its pin the way `cozy package remove` drops it; the
	// serving one keeps a submitted request, which is the reference the claim must honour.
	if problem := store.Unpin(retired.Package, retired.Major); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.Unpin(serving.Package, serving.Major); problem != nil {
		t.Fatal(problem)
	}
	if _, _, problem := store.Submit(records.Request{
		ID: "req-serving", IdemKey: "idem-serving", BodyDigest: "sha256:" + serving.ID,
		Package: serving.Package, Entrypoint: "generate", Payload: []byte("{}"),
		InstallID: serving.ID,
	}); problem != nil {
		t.Fatal(problem)
	}

	// The defect itself: a directory whose row is GONE. No pin names it, no verb reaches
	// it, and before cl-076 nothing could remove it.
	stranded := l.InstallDir("dddddddddddddddd")
	for _, generation := range []string{pinned.Dir, retired.Dir, serving.Dir, stranded} {
		writeReadOnlyPackage(t, filepath.Join(generation, "venv", "lib", "python3.12",
			"site-packages", "hidiffusion"))
	}
	t.Cleanup(func() {
		for _, generation := range []string{pinned.Dir, serving.Dir} {
			restoreDirectoryWrites(filepath.Join(generation, "venv", "lib", "python3.12",
				"site-packages", "hidiffusion"))
		}
	})

	// A SIBLING of the install root is not the sweep's business, and neither is anything a
	// symlink under it merely names.
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.MkdirAll(outside, 0o700); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(outside, "sentinel")
	if err := os.WriteFile(sentinel, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(l.Installs, "linked-elsewhere")); err != nil {
		t.Logf("symlink safety fixture unavailable: %v", err)
	}

	swept, problem := install.Sweep(l, store)
	if problem != nil {
		t.Fatal(problem)
	}
	if swept.Scanned != 4 || swept.Removed != 2 {
		t.Fatalf("sweep = %+v, want 4 scanned and 2 removed", swept)
	}
	// Bytes freed is the RECORDED exclusive bytes plus what the stranded tree measured —
	// never a directory size, because an install tree is hardlinked against the uv cache.
	if swept.Bytes <= retired.BytesExcl {
		t.Fatalf("freed %d bytes, want more than the retired record's %d", swept.Bytes, retired.BytesExcl)
	}
	for _, gone := range []string{retired.Dir, stranded} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Fatalf("unreferenced install %s remains: %v", gone, err)
		}
	}
	for _, kept := range []string{pinned.Dir, serving.Dir} {
		if _, err := os.Stat(kept); err != nil {
			t.Fatalf("referenced install %s was swept: %v", kept, err)
		}
	}
	if generation, readProblem := store.Install(serving.ID); readProblem != nil || generation == nil {
		t.Fatalf("in-flight generation record = %+v, %v", generation, readProblem)
	}
	if got, err := os.ReadFile(sentinel); err != nil || string(got) != "keep" {
		t.Fatalf("symlinked directory outside the install root changed: %q, %v", got, err)
	}
	if _, err := os.Lstat(filepath.Join(l.Installs, "linked-elsewhere")); err != nil {
		t.Fatalf("symlink under the install root was removed: %v", err)
	}

	// The sweep is idempotent: a second pass over the survivors removes nothing.
	again, problem := install.Sweep(l, store)
	if problem != nil {
		t.Fatal(problem)
	}
	if again.Removed != 0 || again.Bytes != 0 {
		t.Fatalf("second sweep = %+v, want nothing removed", again)
	}
}
