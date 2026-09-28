package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestCapturedSourceDescribesItsVerifiedInstalledInterface(t *testing.T) {
	root := t.TempDir()
	installed := filepath.Join(root, "install")
	source := filepath.Join(installed, "source")
	must(t, os.MkdirAll(source, 0700))
	raw := []byte(`{"format":"cozy.package.interface/1","application":"missing_module:app","entrypoints":[],"jobs":[{"name":"main","models":[],"request":{"fields":[]},"result":{"fields":[]},"publishes":false}]}`)
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	if len(iface.Raw) == 0 {
		t.Fatal("fixture lost its canonical interface")
	}
	path := launch.PackageInterfacePath(installed)
	must(t, os.MkdirAll(filepath.Dir(path), 0700))
	must(t, os.WriteFile(path, raw, 0600))
	record := records.PackageInstall{ID: "captured", Package: "local/captured", Version: "1.0.0", Dir: installed, SourceKind: "local", SourceRef: source, ProjectDir: source}
	facts, problem := launch.Read(record, root, childEnv(t, root))
	fatal(t, problem)
	if facts.RuntimeCLI.PackageInterface != path {
		t.Fatal("immutable capture was treated as live editable source")
	}
	// Neither source code nor an environment Python exists. The actual Runtime
	// derives the descriptor from the verified installed interface, without import.
	job, problem := facts.Job("main")
	fatal(t, problem)
	if job.DescriptorID == "" {
		t.Fatal("Runtime returned no descriptor")
	}
	mutable := record
	mutable.SourceRef = filepath.Join(root, "author-checkout")
	mutable.ProjectDir = mutable.SourceRef
	live, problem := launch.Read(mutable, root, childEnv(t, root))
	fatal(t, problem)
	if live.RuntimeCLI.PackageInterface != "" {
		t.Fatal("live source incorrectly reused the captured interface")
	}

	must(t, os.WriteFile(path, []byte(`{}`), 0600))
	if _, problem := launch.Read(record, root, childEnv(t, root)); problem == nil {
		t.Fatal("malformed installed callable metadata was accepted")
	}
}
