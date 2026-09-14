package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

func TestRootInputSnapshotKeepsExactBytesAfterOriginalMutation(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source")
	must(t, os.MkdirAll(filepath.Join(source, "nested"), 0700))
	original := []byte(strings.Repeat("original", 10000))
	must(t, os.WriteFile(filepath.Join(source, "nested/report"), original, 0600))
	must(t, os.WriteFile(filepath.Join(source, "empty"), nil, 0600))
	stage := filepath.Join(root, "capture")
	captured, problem := inputasset.CaptureTree(stage, "files", source, inputasset.MaxRootInputBytes)
	fatal(t, problem)
	members, problem := resultfiles.ReadTreeManifest(captured.Snapshot.Path, captured.Snapshot.Manifest.Digest, captured.Snapshot.Manifest.Length, captured.Snapshot.ContentBytes)
	fatal(t, problem)
	if len(members) != 2 || captured.Snapshot.ContentBytes != int64(len(original)) || string(captured.Snapshot.Body) == "" {
		t.Fatal("snapshot omitted its bounded complete identity")
	}
	must(t, os.WriteFile(filepath.Join(source, "nested/report"), []byte("edited"), 0600))
	must(t, os.Remove(filepath.Join(source, "empty")))
	for _, member := range members {
		data, err := os.ReadFile(filepath.Join(stage+".files", strings.TrimPrefix(member.Digest, "sha256:")))
		must(t, err)
		if int64(len(data)) != member.Length || member.Length > 0 && string(data) != string(original) {
			t.Fatal("original edit changed captured input")
		}
	}
	asset, problem := inputasset.Bind(records.AssetBinding{FieldPath: "report", LocalPath: filepath.Join(source, "nested/report"), MaxBytes: 1024}, 1024)
	fatal(t, problem)
	file, problem := inputasset.CaptureFile(filepath.Join(root, "file-capture"), asset)
	fatal(t, problem)
	must(t, os.WriteFile(asset.LocalPath, []byte("changed again"), 0600))
	data, err := os.ReadFile(file.LocalPath)
	must(t, err)
	if string(data) != "edited" {
		t.Fatal("file snapshot borrowed mutable original bytes")
	}
}

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
