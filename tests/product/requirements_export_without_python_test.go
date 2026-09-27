package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
)

// The frozen export needs only uv, never a local Python. Repeated identical wheel rows are
// one wheel, and a local materialization wheel a newer Hub adds is pinned like the others
// rather than refusing the install.
func TestPublishedRequirementsDedupeRepeatedWheelsAndAcceptNewLocalWheels(t *testing.T) {
	uv, err := exec.LookPath("uv")
	if err != nil {
		t.Skip("uv is required for frozen export")
	}
	tools := t.TempDir()
	must(t, os.Symlink(uv, filepath.Join(tools, filepath.Base(uv))))
	t.Setenv("PATH", tools)
	exact := func(body string) install.ExactDocument {
		digest := sha256.Sum256([]byte(body))
		return install.ExactDocument{Bytes: []byte(body), Length: int64(len(body)), Digest: "sha256:" + hex.EncodeToString(digest[:])}
	}
	wheel := func(name string, digit string) install.PublishedWheel {
		return install.PublishedWheel{Distribution: name, Version: "1.0.0", Filename: name + "-1.0.0-py3-none-any.whl", Digest: "sha256:" + strings.Repeat(digit, 64)}
	}
	published := &install.PublishedSource{
		Package: "proof/export-proof", Release: "1.0.0",
		IndexURL:      "https://example.com/simple",
		PackageConfig: exact("[application]\nobject = 'proof:app'\n"),
		Selection:     install.Selection{PackageInterface: exact(`{"format":"cozy.package.interface/1","application":"proof:app","entrypoints":[],"jobs":[]}`)},
		Pyproject:     exact("[project]\nname = 'export-proof'\nversion = '1.0.0'\nrequires-python = '>=3.12'\n"),
		UVLock:        exact("version = 1\nrevision = 3\nrequires-python = '>=3.12'\n[[package]]\nname = 'export-proof'\nversion = '1.0.0'\nsource = { virtual = '.' }\n"),
		ProjectWheel:  wheel("export-proof", "1"),
		Wheels:        []install.PublishedWheel{wheel("helper", "2"), wheel("helper", "2")},
		LocalWheels:   []install.PublishedWheel{wheel("tensorfs", "4"), wheel("newer-hub-tool", "5")},
	}
	body, problem := install.PublishedRequirements(published)
	fatal(t, problem)
	if !strings.Contains(string(body), "export-proof==1.0.0 --hash="+published.ProjectWheel.Digest) {
		t.Fatalf("published wheel pin was lost: %s", body)
	}
	for _, name := range []string{"helper", "tensorfs", "newer-hub-tool"} {
		if strings.Count(string(body), name+"==1.0.0") != 1 {
			t.Fatalf("%s was not pinned exactly once: %s", name, body)
		}
	}
	published.Wheels = append(published.Wheels, wheel("helper", "9"))
	if _, problem := install.PublishedRequirements(published); problem == nil || problem.ErrName() != "package_wheel_ambiguous" {
		t.Fatalf("two different wheels for one distribution were accepted: %v", problem)
	}
}
