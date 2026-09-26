package install

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublishedRequirementsNeedNoLocalPython(t *testing.T) {
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv is required for frozen export")
	}
	tools := t.TempDir()
	if err := os.Symlink(uv, filepath.Join(tools, filepath.Base(uv))); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", tools)
	exact := func(body string) ExactDocument {
		digest := sha256.Sum256([]byte(body))
		return ExactDocument{Bytes: []byte(body), Length: int64(len(body)), Digest: "sha256:" + hex.EncodeToString(digest[:])}
	}
	published := &PublishedSource{
		Package: "proof/export-proof", Release: "1.0.0", PythonVersion: "3.12.12",
		IndexURL:      "https://example.com/simple",
		PackageConfig: exact("[application]\nobject = 'proof:app'\n"),
		Selection:     Selection{PackageInterface: exact(`{"format":"cozy.package.interface/1","application":"proof:app","entrypoints":[],"jobs":[]}`)},
		Pyproject:     exact("[project]\nname = 'export-proof'\nversion = '1.0.0'\nrequires-python = '>=3.12'\n"),
		UVLock:        exact("version = 1\nrevision = 3\nrequires-python = '>=3.12'\n[[package]]\nname = 'export-proof'\nversion = '1.0.0'\nsource = { virtual = '.' }\n"),
		ProjectWheel:  PublishedWheel{Distribution: "export-proof", Version: "1.0.0", Filename: "export_proof-1.0.0-py3-none-any.whl", Digest: "sha256:" + strings.Repeat("1", 64)},
	}
	body, problem := PublishedRequirements(published)
	if problem != nil {
		t.Fatalf("frozen metadata export required a local Python: %v", problem)
	}
	if !strings.Contains(string(body), "export-proof==1.0.0 --hash="+published.ProjectWheel.Digest) {
		t.Fatalf("published wheel pin was lost: %s", body)
	}
}
