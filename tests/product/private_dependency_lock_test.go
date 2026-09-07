package producttest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPrivateDependencyRefusesSourceMutationDuringResolution(t *testing.T) {
	root := t.TempDir()
	project, library := filepath.Join(root, "operation"), filepath.Join(root, "library")
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.MkdirAll(library, 0o700))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name = "private-operation-mutation-proof"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["private-mutating-library"]
[tool.uv.sources]
private-mutating-library = {path = "../library"}
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = 'operation:app'\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "operation.py"), []byte("VALUE = 7\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(`[project]
name = "private-mutating-library"
version = "0.0.1"
dynamic = ["dependencies"]
[build-system]
requires = []
build-backend = "backend"
backend-path = ["."]
`), 0o600))
	marker := filepath.Join(project, "added_during_resolution.py")
	// uv runs this real PEP 517 backend to resolve dynamic dependency metadata.
	// Adding an authored file during resolution must invalidate the capture.
	backend := fmt.Sprintf(`from pathlib import Path
def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    Path(%q).write_text("CHANGED = True\n")
    name = "private_mutating_library-0.0.1.dist-info"
    target = Path(metadata_directory) / name
    target.mkdir()
    (target / "METADATA").write_text("Metadata-Version: 2.3\nName: private-mutating-library\nVersion: 0.0.1\n")
    return name
`, marker)
	must(t, os.WriteFile(filepath.Join(library, "backend.py"), []byte(backend), 0o600))
	pack, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	if pack != nil {
		pack.Close()
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("real uv resolution did not execute the mutation arm: %v (%v)", err, problem)
	}
	if problem == nil || problem.ErrName() != "conflict" {
		t.Fatalf("source changed during resolution without a capture refusal: %v", problem)
	}
}

func TestPrivateDependencyLocksOnlyCapturedSource(t *testing.T) {
	root := t.TempDir()
	project, library := filepath.Join(root, "operation"), filepath.Join(root, "library")
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.MkdirAll(library, 0o700))
	metadata := `[project]
name = "private-operation-lock-proof"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["private-lock-library"]
[tool.uv.sources]
private-lock-library = {path = "../library", editable = true}
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = 'operation:app'\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "operation.py"), []byte("VALUE = 7\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte("[project]\nname = 'private-lock-library'\nversion = '0.0.1'\n"), 0o600))
	implementation := filepath.Join(library, "library.py")
	must(t, os.WriteFile(implementation, []byte("VALUE = 11\n"), 0o600))
	if _, problem := packagepublish.PrepareFrom(project); problem == nil || problem.ErrName() != "package_source_required_file_missing" {
		t.Fatalf("publication no longer requires an authored lock: %v", problem)
	}
	first, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	fatal(t, problem)
	defer first.Close()
	firstIdentity, _, _, problem := first.SourceIdentity()
	fatal(t, problem)
	second, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	fatal(t, problem)
	defer second.Close()
	secondIdentity, _, _, problem := second.SourceIdentity()
	fatal(t, problem)
	if first.Tree == second.Tree || firstIdentity != secondIdentity {
		t.Fatalf("temporary capture path changed source identity: %s / %s", firstIdentity, secondIdentity)
	}
	dependencies, problem := packagepublish.LocalDependencyPaths(first.Tree)
	fatal(t, problem)
	if dependencies["private-lock-library"] != library {
		t.Fatalf("relative library path changed meaning: %v", dependencies)
	}
	for _, dir := range []string{project, library} {
		if _, err := os.Stat(filepath.Join(dir, "uv.lock")); !os.IsNotExist(err) {
			t.Fatalf("resolution changed authored lock at %s: %v", dir, err)
		}
	}
	actual, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	if string(actual) != metadata {
		t.Fatal("resolution rewrote authored dependency metadata")
	}
	if _, err := os.Stat(filepath.Join(first.Tree, "uv.lock")); err != nil {
		t.Fatalf("captured package has no resolved lock: %v", err)
	}
	must(t, os.WriteFile(implementation, []byte("VALUE = 12\n"), 0o600))
	edited, problem := packagepublish.PreparePrivateFrom(context.Background(), project)
	fatal(t, problem)
	defer edited.Close()
	editedIdentity, _, _, problem := edited.SourceIdentity()
	fatal(t, problem)
	if editedIdentity == firstIdentity {
		t.Fatal("same-version local library edit did not change captured identity")
	}
}
