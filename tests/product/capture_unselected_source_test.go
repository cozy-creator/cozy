package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// A source project installed in the parent environment but outside every selected
// callable closure is not rebuilt during remote wheel capture.
func TestWheelCaptureSkipsUnselectedSourceBuild(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source-bound")
	must(t, os.Mkdir(source, 0o700))
	write := func(path, value string) {
		t.Helper()
		must(t, os.WriteFile(path, []byte(value), 0o600))
	}
	write(filepath.Join(root, "pyproject.toml"), "[project]\nname='capture-root'\nversion='1.0.0'\ndependencies=['source-bound>=1,<2']\n[tool.uv.sources]\nsource-bound={path='source-bound'}\n")
	write(filepath.Join(source, "pyproject.toml"), "[project]\nname='source-bound'\nversion='1.0.0'\n")
	write(filepath.Join(root, "uv.lock"), "version=1\n[[package]]\nname='capture-root'\nversion='1.0.0'\nsource={editable='.'}\n[[package]]\nname='source-bound'\nversion='1.0.0'\nsource={directory='source-bound'}\n")
	stage := t.TempDir()
	captured, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", "capture-root==1.0.0\nsource-bound==1.0.0", stage, t.TempDir(),
		map[string]map[string]string{"app": {"capture-root": "1.0.0"}}, "3.12.12")
	fatal(t, problem)
	if _, ok := captured["source-bound"]; ok {
		t.Fatalf("unselected source project was captured: %+v", captured)
	}
	if _, err := os.Stat(filepath.Join(stage, "dependencies")); !os.IsNotExist(err) {
		t.Fatalf("unselected source project created build output: %v", err)
	}
}
