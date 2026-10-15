package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/records"
)

// An input file is hashed once, when it is attached. Dispatch reuses that digest while the
// file keeps its size and modification time, and hashes it again only when either changed —
// where changed bytes are still refused.
func TestInputAssetIsHashedOnceWhileUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reference.png")
	must(t, os.WriteFile(path, []byte("the attached bytes"), 0o600))
	bound, problem := inputasset.Bind(records.AssetBinding{FieldPath: "image", LocalPath: path}, 4096)
	fatal(t, problem)
	if bound.ModTime == 0 || bound.Length != int64(len("the attached bytes")) {
		t.Fatalf("attach did not record the file's size and modification time: %+v", bound)
	}

	// Unreadable but unchanged: the recorded digest is the answer, so nothing is read.
	if os.Geteuid() != 0 {
		must(t, os.Chmod(path, 0))
		if problem := inputasset.Verify(bound, 4096); problem != nil {
			t.Fatalf("an unchanged file was read again at dispatch: %s", problem.Message)
		}
		must(t, os.Chmod(path, 0o600))
	}

	// Touched, same bytes: hashed again and still the request's bytes.
	later := time.Unix(0, bound.ModTime).Add(time.Minute)
	must(t, os.Chtimes(path, later, later))
	if problem := inputasset.Verify(bound, 4096); problem != nil {
		t.Fatalf("a touched file with the same bytes was refused: %s", problem.Message)
	}

	// Different bytes: refused.
	must(t, os.WriteFile(path, []byte("different bytes entirely"), 0o600))
	if problem := inputasset.Verify(bound, 4096); problem == nil || problem.ErrName() != "input_asset_changed" {
		t.Fatalf("changed bytes were admitted: %v", problem)
	}
}

// Capture copies an input the command fingerprinted, and still unchanged, under that digest
// without hashing it again: an unchanged file's copy is named by the command's digest even
// when the two disagree (a machine verifies the bytes it receives). A touched file is hashed.
func TestCaptureReusesTheCommandsDigestWhileUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reference.png")
	must(t, os.WriteFile(path, []byte("the attached bytes"), 0o600))
	bound, problem := inputasset.Bind(records.AssetBinding{FieldPath: "image", LocalPath: path, MaxBytes: 4096}, 4096)
	fatal(t, problem)
	claimed := bound
	claimed.Digest = "sha256:" + strings.Repeat("ab", 32)
	frozen, problem := inputasset.CaptureFile(filepath.Join(t.TempDir(), "snapshot"), claimed)
	fatal(t, problem)
	if frozen.Snapshot == nil || filepath.Base(frozen.LocalPath) != strings.Repeat("ab", 32) {
		t.Fatalf("an unchanged input was hashed again at capture: %+v", frozen)
	}
	later := time.Unix(0, bound.ModTime).Add(time.Minute)
	must(t, os.Chtimes(path, later, later))
	if _, problem := inputasset.CaptureFile(filepath.Join(t.TempDir(), "snapshot"), claimed); problem == nil {
		t.Fatal("a touched input was captured under a digest its bytes do not have")
	}
	if _, problem := inputasset.CaptureFile(filepath.Join(t.TempDir(), "snapshot"), bound); problem != nil {
		t.Fatalf("a touched input with its own bytes was refused: %s", problem.Message)
	}
}
