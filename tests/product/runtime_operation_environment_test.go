package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

func TestRuntimeOperationEnvironmentKeepsAnOwnedExactGeneration(t *testing.T) {
	root := t.TempDir()
	python := filepath.Join(root, "builtin", "contents", "generation", "bin", "python")
	must(t, os.MkdirAll(filepath.Dir(python), 0700))
	must(t, os.WriteFile(python, []byte("owned interpreter fixture"), 0700))
	digest := "sha256:" + strings.Repeat("1", 64)
	row := runtimeoperation.Environment{Python: python, EnvironmentDigest: digest, InterfaceDigest: digest, SourceDigest: digest}
	write := func(row runtimeoperation.Environment) {
		raw, err := json.Marshal(row)
		must(t, err)
		must(t, os.WriteFile(filepath.Join(root, runtimeoperation.EnvironmentFile), raw, 0600))
	}
	write(row)
	observed, problem := runtimeoperation.ReadEnvironment(root, digest, digest, digest)
	fatal(t, problem)
	if observed != python {
		t.Fatal("builtin changed its exact interpreter")
	}
	if _, problem := runtimeoperation.ReadEnvironment(root, "sha256:"+strings.Repeat("2", 64), digest, digest); problem == nil {
		t.Fatal("foreign environment identity admitted")
	}
	outside := filepath.Join(t.TempDir(), "bin")
	must(t, os.MkdirAll(outside, 0700))
	must(t, os.WriteFile(filepath.Join(outside, "python"), []byte("other interpreter"), 0700))
	link := filepath.Join(root, "foreign")
	must(t, os.Symlink(outside, link))
	row.Python = filepath.Join(link, "python")
	write(row)
	if _, problem := runtimeoperation.ReadEnvironment(root, digest, digest, digest); problem == nil {
		t.Fatal("foreign generation symlink admitted")
	}
	row.Python = "relative/python"
	write(row)
	if _, problem := runtimeoperation.ReadEnvironment(root, digest, digest, digest); problem == nil {
		t.Fatal("relative interpreter admitted")
	}
}
