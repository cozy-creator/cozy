package launch

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestReadRetainedFactsWithoutLocalPython(t *testing.T) {
	root := t.TempDir()
	t.Setenv("PATH", root)
	for _, sourceKind := range []string{"tensorhub", "wheel", "local"} {
		t.Run(sourceKind, func(t *testing.T) {
			dir := filepath.Join(root, sourceKind)
			path := PackageInterfacePath(dir)
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(`{"format":"cozy.package.interface/1","application":"example:app","entrypoints":[],"jobs":[]}`), 0600); err != nil {
				t.Fatal(err)
			}
			inst := records.PackageInstall{ID: sourceKind, Dir: dir, SourceKind: sourceKind, Python: "3.12.12"}
			facts, problem := Read(inst, root, nil)
			if problem != nil {
				t.Fatalf("retained facts required a local executor: %v", problem)
			}
			if facts.Install.Python != inst.Python || facts.PackageInterface.Application != "example:app" {
				t.Fatal("retained installation facts changed")
			}
		})
	}
}
