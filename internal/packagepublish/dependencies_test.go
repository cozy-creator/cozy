package packagepublish

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestCollectLocalDependenciesForClosureSkipsUnselectedSourceBuild(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "source-bound")
	if err := os.Mkdir(source, 0o700); err != nil {
		t.Fatal(err)
	}
	write := func(path, value string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(value), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(root, "pyproject.toml"), "[project]\nname='capture-root'\nversion='1.0.0'\ndependencies=['source-bound>=1,<2']\n[tool.uv.sources]\nsource-bound={path='source-bound'}\n")
	write(filepath.Join(source, "pyproject.toml"), "[project]\nname='source-bound'\nversion='1.0.0'\n")
	document, problem := readProjectDocument(filepath.Join(root, "pyproject.toml"))
	if problem != nil {
		t.Fatal(problem)
	}
	stage := t.TempDir()
	wheels, _, _, problem := collectLocalDependenciesForClosure(context.Background(), root, document, stage, map[string]string{"capture-root": "1.0.0"}, "3.12.12")
	if problem != nil {
		t.Fatal(problem)
	}
	if len(wheels) != 0 {
		t.Fatalf("unselected source project was captured: %+v", wheels)
	}
	if _, err := os.Stat(filepath.Join(stage, "dependencies")); !os.IsNotExist(err) {
		t.Fatalf("unselected source project created build output: %v", err)
	}
}
