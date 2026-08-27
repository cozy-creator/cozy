package app

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCompositionOutputCannotAliasEditableSource(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "film.yaml")
	if err := os.WriteFile(source, []byte("source"), 0o600); err != nil {
		t.Fatal(err)
	}
	if !compositionOverwritesSource(source, source) {
		t.Fatal("lexically identical output accepted")
	}
	hardlink := filepath.Join(dir, "hardlink.yaml")
	if err := os.Link(source, hardlink); err == nil && !compositionOverwritesSource(source, hardlink) {
		t.Fatal("hardlink output accepted")
	}
	symlink := filepath.Join(dir, "symlink.yaml")
	if err := os.Symlink(source, symlink); err == nil && !compositionOverwritesSource(symlink, source) {
		t.Fatal("symlink source alias accepted")
	}
	if compositionOverwritesSource(source, filepath.Join(dir, "composition.json")) {
		t.Fatal("distinct output refused")
	}
}
