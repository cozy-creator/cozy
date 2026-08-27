package home

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestOpenProtectsTheRecordsRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "cozy")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, problem := Open(root); problem != nil {
		t.Fatal(problem)
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(root)
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0o700 {
			t.Fatalf("root mode=%v", info.Mode().Perm())
		}
	}
}
