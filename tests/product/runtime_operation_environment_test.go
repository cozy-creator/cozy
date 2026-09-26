package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRuntimeOperationEnvironmentKeepsItsAcceptedInstallation(t *testing.T) {
	root := t.TempDir()
	install := records.PackageInstall{ID: "builtin-install", Dir: filepath.Join(root, "builtin")}
	expected := home.VenvPython(filepath.Join(install.Dir, "venv"))
	must(t, os.MkdirAll(filepath.Dir(expected), 0700))
	must(t, os.WriteFile(expected, []byte("owned interpreter fixture"), 0700))
	outside := filepath.Join(t.TempDir(), "python")
	must(t, os.WriteFile(outside, []byte("foreign interpreter"), 0700))
	link := filepath.Join(root, "foreign")
	must(t, os.Symlink(outside, link))
	// Interpreter ownership is the accepted installation layout. Legacy metadata
	// cannot redirect it to another generation, an external symlink or a relative path.
	for _, claimed := range []string{outside, link, "relative/python"} {
		raw, err := json.Marshal(map[string]string{"python": claimed, "environment_digest": "foreign"})
		must(t, err)
		must(t, os.WriteFile(filepath.Join(install.Dir, "environment.json"), raw, 0600))
		observed, problem := launch.EnvironmentPython(install)
		fatal(t, problem)
		if observed != expected {
			t.Fatalf("unowned metadata redirected interpreter: %s", observed)
		}
	}
	successor := install
	successor.ID = "next-install"
	successor.Dir = filepath.Join(root, "next")
	next, problem := launch.EnvironmentPython(successor)
	fatal(t, problem)
	if next == expected {
		t.Fatal("distinct accepted installation reused the prior interpreter path")
	}
}
