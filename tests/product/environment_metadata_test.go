package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
)

func TestBaseInterpreterRefusesDependencyOwnedExecutable(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "pyvenv.cfg"), []byte("version_info = 3.12\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "bin", "python"), []byte("must not execute"), 0o700); err != nil {
		t.Fatal(err)
	}
	if _, problem := install.BasePython(root); problem == nil {
		t.Fatal("accepted an interpreter owned by the dependency environment")
	}
}
