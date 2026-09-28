package producttest

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/resultfiles"
)

func TestJobMediaExportRejectsChangedCustodyAndSymlinkDestination(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "accepted")
	body := []byte("accepted bytes")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(body))
	must(t, os.WriteFile(source, body, 0600))
	directory := filepath.Join(root, "outputs")
	target, problem := resultfiles.Materialize(source, directory, digest, "video/mp4", int64(len(body)))
	fatal(t, problem)
	must(t, os.WriteFile(source, []byte("modified bytes"), 0600))
	if _, problem := resultfiles.Materialize(source, directory, digest, "video/mp4", int64(len(body))); problem == nil || problem.ErrName() != "output_export_source_changed" {
		t.Fatalf("changed accepted bytes were exported: %v", problem)
	}
	actual, err := os.ReadFile(target)
	must(t, err)
	if string(actual) != string(body) {
		t.Fatal("failed export replaced the previous complete copy")
	}
	entries, err := os.ReadDir(directory)
	must(t, err)
	if len(entries) != 1 {
		t.Fatal("failed export leaked its temporary file")
	}
	link := filepath.Join(root, "substituted")
	must(t, os.Symlink(directory, link))
	if _, problem := resultfiles.Materialize(source, link, digest, "video/mp4", int64(len(body))); problem == nil || problem.ErrName() != "output_export_directory_substituted" {
		t.Fatalf("symlink output directory was accepted: %v", problem)
	}
}
