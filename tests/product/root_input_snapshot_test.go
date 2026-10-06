package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/inputasset"
)

func TestRootInputSnapshotRejectsSymlinksAndCapacityBeforeManifest(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	must(t, os.Mkdir(source, 0700))
	outside := filepath.Join(root, "outside")
	must(t, os.WriteFile(outside, []byte("private"), 0600))
	must(t, os.Symlink(outside, filepath.Join(source, "link")))
	if _, problem := inputasset.CaptureTree(filepath.Join(root, "bad"), "files", source, 100); problem == nil {
		t.Fatal("symlink entered an input snapshot")
	}
	must(t, os.Remove(filepath.Join(source, "link")))
	must(t, os.WriteFile(filepath.Join(source, "file"), []byte("12345"), 0600))
	if _, problem := inputasset.CaptureTree(filepath.Join(root, "small"), "files", source, 4); problem == nil {
		t.Fatal("oversized input entered snapshot")
	}
	if _, problem := inputasset.CaptureTree(filepath.Join(source, "recursive"), "files", source, 100); problem == nil {
		t.Fatal("capture recursed into its own source")
	}
}
